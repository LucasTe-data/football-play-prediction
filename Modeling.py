# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:11:22 2023

@author: Lucas
"""


#%% Imports

import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


#%% Define Model Classes

class SimpleNN(nn.Module):
    
    def __init__(self, in_features, out_features = 2):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 40),
            nn.LeakyReLU(),
            nn.Linear(40, out_features))
    
    def forward(self, x):
        return self.model(x)


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


encoder = OrdinalEncoder()
pbp_df['PlayType'] = encoder.fit_transform(pbp_df[['PlayType']]).astype('float64')


#%% Train-Test Split
variables = ['Down', 'ToGo', 'YardLine', 'GameTime', 'PreviousPlay', 
             'NO HUDDLE', 'NO HUDDLE SHOTGUN', 'SHOTGUN', 'UNDER CENTER']

for i in range(1, n_previous):
    variables.append('PreviousPlay' + str(i + 1))
    
to_predict = 'PlayType'

X = pbp_df[variables].values
y = pbp_df[to_predict].values

X_train, X_test, y_train, y_test = train_test_split(X,y)

dataset_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

trainloader = DataLoader(dataset_train, shuffle=True, batch_size=128)

#%% Initialize Model
model = SimpleNN(len(variables))

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()

#%% Training Loop
epochs = 100
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
    y_pred = model.forward(X_test).argmax(dim = 1)

y_test = encoder.inverse_transform(y_test.numpy().reshape(-1,1))
y_pred = encoder.inverse_transform(y_pred.numpy().reshape(-1,1))

#%% Model Evaluation - Scoring

# accuracy = accuracy_score(y_test,y_pred)
# precision = precision_score(y_test,y_pred)
# recall = recall_score(y_test,y_pred)
# f1 = f1_score(y_test,y_pred)

# print(F'''Accuracy: {100*accuracy.round(4)}%
# Precision: {100*precision.round(4)}%
# Recall: {100*recall.round(4)}%
# F1-Score: {f1.round(4)}''')


cf_matrix = confusion_matrix(y_test, y_pred,
                              labels = encoder.categories_[0],
                              normalize = 'all')

disp = ConfusionMatrixDisplay(confusion_matrix = cf_matrix,
                              display_labels = encoder.categories_[0])

disp.plot()
plt.title('Confusion Matrix')
plt.show()