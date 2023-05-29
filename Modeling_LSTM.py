# -*- coding: utf-8 -*-
"""
Created on Mon May 29 09:22:37 2023

@author: Lucas
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:11:22 2023

@author: Lucas
"""


#%% Imports

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, class_weight

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import time

import shap



#%% Define Model Classes
class SimpleLSTM(nn.Module):
    
    def __init__(self, in_features, hidden_size, num_layers, out_features = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size = in_features, hidden_size = hidden_size, 
                            num_layers = num_layers, batch_first = True)
        
        self.lin0 = nn.Linear(hidden_size, out_features)
        
        self.dropout = nn.Dropout(p = 0.5)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1]
        
        output = self.dropout(output)
        
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
plt.xlabel('Length of Drive')
plt.ylabel('Count')
plt.show()

#%% Reduce Drives to only include ones with more than three plays
pbp_df_time = pbp_df.groupby(['GameId', 'DriveId']).filter(lambda x: len(x) > 3).copy()


#%% For Drives with more plays only use last three
pbp_df_time = pbp_df_time.groupby(['GameId', 'DriveId'], group_keys=False).apply(
    lambda x: x.tail(4))

#%% Check How many drives are left
print('Drives Left:')
print(pbp_df_time.groupby(['GameId', 'DriveId']).size().value_counts().sort_index())

#%% Create LSTM Data Set
variables = ['Down', 'ToGo', 'YardLine', 'GameTime', 'DriveId', 'NO HUDDLE', 
             'NO HUDDLE SHOTGUN', 'SHOTGUN', 'UNDER CENTER', 'PreviousPlay']

for i in range(1, n_previous):
    variables.append('PreviousPlay' + str(i + 1))
    
to_predict = 'PlayType'

X_stack = []
y_stack = []

for name, group in pbp_df_time.groupby(['GameId', 'DriveId']):
    X_stack.append(group[variables].head(3).to_numpy())
    y_stack.append(group['PlayType'].tail(1).values[0])

#%% Check Distribution of Run/Pass in Training Data
print(F'{playtype_encoder.inverse_transform([[0]])[0][0]} : {y_stack.count(0)}')
print(F'{playtype_encoder.inverse_transform([[1]])[0][0]} : {y_stack.count(1)}')

X_stack = np.array(X_stack)
y_stack = np.array(y_stack)

#%% Create Validation Set
X_train, X_validation, y_train, y_validation = train_test_split(X_stack, y_stack, test_size = 0.1)

#%% Resample Data

group_0_data = X_train[y_train == 0]
group_1_data = X_train[y_train == 1]


# Determine the smaller group size
# min_group_size = min(len(group_0_data), len(group_1_data))
max_group_size = max(len(group_0_data), len(group_1_data))

#mean_group_size = int(sum([len(group_0_data), len(group_1_data)])/2)

# Resample the larger group to match the smaller group size
# resampled_group_0_data = resample(group_0_data, n_samples=min_group_size, replace=False, random_state=42)
# resampled_group_1_data = resample(group_1_data, n_samples=min_group_size, replace=False, random_state=42)


# Resample the groups to match the larger group size
resampled_group_0_data = resample(group_0_data, n_samples=max_group_size, replace=True, random_state=42)
resampled_group_1_data = resample(group_1_data, n_samples=max_group_size, replace=True, random_state=42)

# Resample the groups to match the mean group size
# resampled_group_0_data = resample(group_0_data, n_samples= mean_group_size, replace=True, random_state=42)
# resampled_group_1_data = resample(group_1_data, n_samples= mean_group_size, replace=True, random_state=42)

X_resampled = np.concatenate((resampled_group_0_data, resampled_group_1_data), axis=0)

# y_resampled = np.concatenate((np.zeros(min_group_size), np.ones(group_size)), axis=0)
y_resampled = np.concatenate((np.zeros(max_group_size), np.ones(max_group_size)), axis=0)
# y_resampled = np.concatenate((np.zeros(mean_group_size), np.ones(group_size)), axis=0)



#%% Train-Test-Split for LSTM Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)

dataset_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))

trainloader = DataLoader(dataset_train, shuffle=True, batch_size=256)


#%% Initialize Model
model = SimpleLSTM(len(variables), hidden_size = 124, num_layers = 3)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0007)
 
criterion = nn.CrossEntropyLoss()

#%% Training Loop for LSTM
epochs = 250
losses = []

model.train()

start_time = time.time()
for i in range(1, epochs + 1):
    
    for batch in trainloader:
        X = batch[0]
        y = batch[1]

        pred = model.forward(X)        
        
        loss = criterion(pred,y.long())

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
    
    losses.append(loss)

    runtime = time.time() - start_time
    
    hours, rem = divmod(runtime, 3600)
    minutes, seconds = divmod(rem, 60)
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    
    mean_rt = runtime/i * epochs
    mean_hours, rem = divmod(mean_rt, 3600)
    mean_minutes, mean_seconds = divmod(rem, 60)
    mean_hours, mean_minutes, mean_seconds = int(mean_hours), int(mean_minutes), int(mean_seconds)
    
    
    print(F'{i}/{epochs} - Loss : {round(float(loss), 6)} - Runtime : {hours}h:{minutes}min:{seconds}s / {mean_hours}h:{mean_minutes}min:{mean_seconds}s')

#%% Plot Losses
losses = [loss.detach().numpy() for loss in losses]

plt.plot(range(len(losses)), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#%% Model Evaluation - Train Data
with torch.no_grad():
    y_pred = model.forward(torch.Tensor(X_train)).argmax(dim = 1).numpy()


accuracy = accuracy_score(y_train,y_pred)
precision = precision_score(y_train,y_pred)
recall = recall_score(y_train,y_pred)
f1 = f1_score(y_train,y_pred)

print(F'''Accuracy: {100*accuracy.round(4)}%
Precision: {100*precision.round(4)}%
Recall: {100*recall.round(4)}%
F1-Score: {f1.round(4)}''')

cf_matrix = confusion_matrix(y_train ,y_pred, normalize = 'all')

disp = ConfusionMatrixDisplay(confusion_matrix = cf_matrix,
                              display_labels = playtype_encoder.categories_[0])

disp.plot()
plt.title('Confusion Matrix - Train Data')
plt.show()


#%% Model Evaluation - Test Data
with torch.no_grad():
    y_pred = model.forward(torch.Tensor(X_test)).argmax(dim = 1).numpy()

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
plt.title('Confusion Matrix - Test Data')
plt.show()

#%% Model Evaluation - Validation Data
with torch.no_grad():
    y_pred = model.forward(torch.Tensor(X_validation)).argmax(dim = 1).numpy()
    

accuracy = accuracy_score(y_validation,y_pred)
precision = precision_score(y_validation,y_pred)
recall = recall_score(y_validation,y_pred)
f1 = f1_score(y_validation,y_pred)

print(F'''Accuracy: {100*accuracy.round(4)}%
Precision: {100*precision.round(4)}%
Recall: {100*recall.round(4)}%
F1-Score: {f1.round(4)}''')

cf_matrix = confusion_matrix(y_validation ,y_pred, normalize = 'all')

disp = ConfusionMatrixDisplay(confusion_matrix = cf_matrix,
                              display_labels = playtype_encoder.categories_[0])

disp.plot()
plt.title('Confusion Matrix - Validation Data')
plt.show()

#%% SHAP Values

# TODO : Calculate SHAP values

#%% Save Model
PATH = './Models/'
name = input('Model Name:')

#check if name already exists
exists = os.path.exists(PATH + name)

if exists:
    print('Name already in use.')
else:    
    torch.save(model.state_dict(),  PATH + name)
    print(F'Model {name} Saved')

# Hyperparameters should also be saved
# TODO : Add logic for that

#%% Load Model
PATH = './Models/'
name = input('Model Name:')

model = SimpleLSTM(len(variables), hidden_size = 124, num_layers = 2)
model.load_state_dict(torch.load(PATH + name))
print('Model Loaded')

model.eval()

