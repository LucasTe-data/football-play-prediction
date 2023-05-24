# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:11:22 2023

@author: Lucas
"""


#%% Imports

import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import time



#%% Define Model Classes

class SimpleNN(nn.Module):
    
    def __init__(self, in_features, out_features = 2):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_features, 200),
            nn.ReLU(),
            nn.Linear(200, out_features))
    
    def forward(self, x):
        return self.model(x)
    

class SimpleLSTM(nn.Module):
    
    def __init__(self, in_features, out_features = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size = in_features, hidden_size = 248, batch_first = True)
        self.lin0 = nn.Linear(248, out_features)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1]
        
        output = self.lin0(output)
        return output
        

#%% Load Data
PATH = './Data/Processed/pbp-processed.csv'

pbp_df = pd.read_csv(PATH)

#%% Prepare Data for Modeling
to_scale = ['Down', 'ToGo', 'YardLine', 'GameTime']
std_scaler = StandardScaler()
pbp_df[to_scale] = std_scaler.fit_transform(pbp_df[to_scale])


n_previous = 3
for i in range(n_previous):
    if i > 0:
        name = 'PreviousPlay' + str(i + 1)
    else:
        name = 'PreviousPlay'
        
    prev_encoder = OrdinalEncoder()
    pbp_df[name] = prev_encoder.fit_transform(pbp_df[[name]]).astype('float64')


playtype_encoder = OrdinalEncoder()
pbp_df['PlayType'] = playtype_encoder.fit_transform(pbp_df[['PlayType']]).astype('float64')

#%% Add Drive Number feature

pbp_df['DriveId'] = ~ (pbp_df['OffenseTeam'] == pbp_df['OffenseTeam'].shift(1))
pbp_df['DriveId'] = pbp_df['DriveId'].cumsum()

pbp_df['DriveId'] = pbp_df.groupby('GameId')['DriveId'].transform(lambda x: pd.factorize(x)[0])

#%% Count length of drives

sorted_drive_counts = pbp_df.groupby(['GameId', 'DriveId']).size().value_counts().sort_index()

plt.bar(sorted_drive_counts.index, sorted_drive_counts)
plt.show()


#%% Train-Test Split for SimpleNN
variables = ['Down', 'ToGo', 'YardLine', 'GameTime', 'DriveId', 'NO HUDDLE', 
             'NO HUDDLE SHOTGUN', 'SHOTGUN', 'UNDER CENTER', 'PreviousPlay']

for i in range(1, n_previous):
    variables.append('PreviousPlay' + str(i + 1))
    
to_predict = 'PlayType'

X = pbp_df[variables].values
y = pbp_df[to_predict].values

#%%
X_train, X_test, y_train, y_test = train_test_split(X,y)

dataset_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

trainloader = DataLoader(dataset_train, shuffle=True, batch_size=256)


#%% Initialize Model
model = SimpleNN(len(variables))

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()

#%% Training Loop
epochs = 1
losses = []

for i in range(epochs):
    
    for batch in trainloader:
        X = batch[0]
        y = batch[1]

        pred = model.forward(X)
        
        loss = criterion(pred,y.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    loss = criterion(pred,y.long())
    losses.append(loss)

    print(F'{i}/{epochs} - Loss : {loss}')



#%% Plot Losses
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#%% Model Evaluation - Predictions
with torch.no_grad():
    y_pred = model.forward(torch.Tensor(X_test)).argmax(dim = 1).numpy()

#%% Model Evaluation - Scoring

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)

recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

print(F'''Accuracy: {100*accuracy.round(4)}%
Precision: {100*precision.round(4)}%
Recall: {100*recall.round(4)}%
F1-Score: {f1.round(4)}''')

cf_matrix = confusion_matrix(y_test, y_pred, normalize = 'all')

disp = ConfusionMatrixDisplay(confusion_matrix = cf_matrix,
                              display_labels = playtype_encoder.categories_[0])

disp.plot()
plt.title('Confusion Matrix')
plt.show()

#%% Reduce Drives to only include ones with more than three plays
pbp_df_time = pbp_df.groupby(['GameId', 'DriveId']).filter(lambda x: len(x) > 3).copy()


#%% For Drives with more plays only use last three
pbp_df_time = pbp_df_time.groupby(['GameId', 'DriveId'], group_keys=False).apply(
    lambda x: x.tail(4))

#%% Check How many drives are left
print('Drives Left:')
print(pbp_df_time.groupby(['GameId', 'DriveId']).size().value_counts().sort_index())

#%% Create Dataset for LSTM

X_stack = []
y_stack = []

for name, group in pbp_df_time.groupby(['GameId', 'DriveId']):
    X_stack.append(group[variables].head(3).to_numpy())
    y_stack.append(group['PlayType'].tail(1).values[0])

#%% Train-Test-Split for LSTM Dataset
X_train, X_test, y_train, y_test = train_test_split(X_stack,y_stack)

dataset_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

trainloader = DataLoader(dataset_train, shuffle=True, batch_size=256)

#%% Initialize Model
model = SimpleLSTM(len(variables))

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

#%% Training Loop for LSTM
epochs = 500
losses = []

start_time = time.time()
for i in range(epochs):
    
    for batch in trainloader:
        X = batch[0]
        y = batch[1]

        pred = model.forward(X)        
        
        loss = criterion(pred,y.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    loss = criterion(pred,y.long())
    losses.append(loss)

    runtime = time.time() - start_time
    print(F'{i}/{epochs} - Loss : {loss} - Runtime : {round(runtime/60, 2)}min')

#%% Plot Losses
plt.plot(range(len(losses)), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#%% Model Evaluation - Predictions
with torch.no_grad():
    y_pred = model.forward(torch.Tensor(X_test)).argmax(dim = 1).numpy()

#%% Model Evaluation - Scoring

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)

recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

print(F'''Accuracy: {100*accuracy.round(4)}%
Precision: {100*precision.round(4)}%
Recall: {100*recall.round(4)}%
F1-Score: {f1.round(4)}''')

cf_matrix = confusion_matrix(y_test, y_pred, normalize = 'all')

disp = ConfusionMatrixDisplay(confusion_matrix = cf_matrix,
                              display_labels = playtype_encoder.categories_[0])

disp.plot()
plt.title('Confusion Matrix')
plt.show()
