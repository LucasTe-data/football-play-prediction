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
from sklearn.utils import resample

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import  matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import time
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier # for surogate

# TODO : Add meaningful comments

#%% Define Model Classes
class SimpleLSTM(nn.Module):
    
    def __init__(self, in_features, hidden_size, num_layers, out_features = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size = in_features, hidden_size = hidden_size, 
                            num_layers = num_layers, batch_first = True)
        
        self.dropout = nn.Dropout(p = 0.5)
        
        self.lin = nn.Linear(hidden_size, out_features)
        
        self.softmax = nn.Softmax(dim = 1)
        
        
    
    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1]
        
        output = self.dropout(output)
        
        output = self.lin(output)
        
        output = self.softmax(output)
        return output
    

#%% Define Functions

def model_evaluation(model, encoder, X, y, label = None):
    model.eval()
    
    if torch.cuda.is_available(): 
        device = "cuda:0" 
    else: 
        device = "cpu" 
    device = torch.device(device)
    
    with torch.no_grad():
        y_pred = model.forward(torch.Tensor(X).to(device)).argmax(dim = 1).cpu().numpy()
        

    accuracy = accuracy_score(y,y_pred)
    precision = precision_score(y,y_pred)
    recall = recall_score(y,y_pred)
    f1 = f1_score(y,y_pred)
    
    mcc = matthews_corrcoef(y, y_pred)

    print(F'''--- Metrics - {label} ---
    Accuracy: \t {100*accuracy.round(4)}%
    Precision: \t {100*precision.round(4)}%
    Recall: \t {100*recall.round(4)}%
    F1-Score: \t {100*f1.round(4)}%
    MCC: \t \t {mcc.round(2)}
    ''')

    cf_matrix = confusion_matrix(y,y_pred, normalize = 'all')

    disp = ConfusionMatrixDisplay(confusion_matrix = cf_matrix,
                                  display_labels = encoder.categories_[0])

    disp.plot()
    plt.title('Confusion Matrix - {label}')
    plt.show()
    
        

#%% Load Data
PATH = './Data/Processed/pbp-processed.csv'

pbp_df = pd.read_csv(PATH)

#%% Prepare Data for Modeling
to_scale = ['Down', 'ToGo', 'YardLine', 'GameTime', 'PointDifference']
std_scaler = StandardScaler()
pbp_df[to_scale] = std_scaler.fit_transform(pbp_df[to_scale])


playtype_encoder = OrdinalEncoder()
pbp_df['PlayType'] = playtype_encoder.fit_transform(pbp_df[['PlayType']]).astype('float64')

#%% Reduce Drives to only include ones with more than three plays
pbp_df_time = pbp_df.groupby(['GameId', 'DriveId']).filter(lambda x: len(x) > 3).copy()

#%% Create LSTM Data Set
variables = ['Down', 'ToGo', 'YardLine', 'GameTime', 'PointDifference', 'DriveId', 'NO HUDDLE', 
             'NO HUDDLE SHOTGUN', 'SHOTGUN', 'UNDER CENTER']

to_predict = 'PlayType'

X_stack = []
y_stack = []


for name, group in tqdm(pbp_df_time.groupby(['GameId', 'DriveId'])):
    X = [group[variables].shift(-x).values[:3] for x in range(0, len(group))][:len(group) -3]
    y = group['PlayType'].tolist()[3:] 
    
    X_stack += X
    y_stack += y
    
    
    # X_stack.append(group[variables].head(3).to_numpy())
    # y_stack.append(group['PlayType'].tail(1).values[0])

X_stack = np.array(X_stack)
y_stack = np.array(y_stack)


#%% Check Distribution of Run/Pass in Data
unique, counts = np.unique(y_stack, return_counts=True)
plt.bar(playtype_encoder.inverse_transform(unique.reshape(-1,1)).reshape(1,-1).tolist()[0], counts)
plt.show()


#%% Create Validation Set
X_train, X_validation, y_train, y_validation = train_test_split(X_stack, y_stack, test_size = 0.1)

#%% Resample Data

group_0_data = X_train[y_train == 0]
group_1_data = X_train[y_train == 1]


# Resample the larger group to match the smaller group size
min_group_size = min(len(group_0_data), len(group_1_data))

resampled_group_0_data = resample(group_0_data, n_samples=min_group_size, replace=False, random_state=42)
resampled_group_1_data = resample(group_1_data, n_samples=min_group_size, replace=False, random_state=42)

X_resampled = np.concatenate((resampled_group_0_data, resampled_group_1_data), axis=0)
y_resampled = np.concatenate((np.zeros(min_group_size), np.ones(min_group_size)), axis=0)

#%% Set available device

if torch.cuda.is_available(): 
    device = "cuda:0" 
else: 
    device = "cpu" 
device = torch.device(device) 


#%% Train-Test-Split for LSTM Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)

dataset_train = TensorDataset(torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device))

trainloader = DataLoader(dataset_train, shuffle=True, batch_size=256)


#%% Initialize Model
model = SimpleLSTM(len(variables), hidden_size = 256, num_layers = 2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.00007)

criterion = nn.CrossEntropyLoss().to(device)

#%% Training Loop for LSTM
epochs = 300
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
    
    losses.append(loss.cpu().detach().numpy())

    runtime = time.time() - start_time
    
    hours, rem = divmod(runtime, 3600)
    minutes, seconds = divmod(rem, 60)
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    
    mean_rt = runtime/i * epochs
    mean_hours, rem = divmod(mean_rt, 3600)
    mean_minutes, mean_seconds = divmod(rem, 60)
    mean_hours, mean_minutes, mean_seconds = int(mean_hours), int(mean_minutes), int(mean_seconds)
    
    
    print(F'{i}/{epochs} - Loss : {round(float(loss), 6)} - Runtime : {hours}h:{minutes}min:{seconds}s / {mean_hours}h:{mean_minutes}min:{mean_seconds}s')

#%% Save Model
PATH = './Models/'
name = input('Model Name:')

#check if name already exists
exists = os.path.exists(PATH + name)

if exists:
    print('Name already in use. Do you want to overwrite?')
    confirm = input('[y]/[n]').lower()
    overwrite = (confirm == 'y')
    if overwrite:
        torch.save(model.state_dict(),  PATH + name)
        print(F'Model {name} Saved')
else:    
    torch.save(model.state_dict(),  PATH + name)
    print(F'Model {name} Saved')

# Hyperparameters should also be saved
# TODO : Add logic for that

#%% Plot Losses
model.eval()

plt.plot(range(len(losses)), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

moving_loss = []
window_size = 20
i = window_size
while i < len(losses) - window_size:    
    window = losses[i - window_size:i + window_size]
    avg = sum(window)/len(window)
    moving_loss.append(avg)
    i += 1

plt.plot(range(len(moving_loss)), moving_loss)
plt.ylabel('Moving Loss')
plt.show()


#%% Model Evaluation - Train Data
model_evaluation(model, playtype_encoder, X_train, y_train, 'Train Data')

#%% Model Evaluation - Test Data
model_evaluation(model, playtype_encoder, X_test, y_test, 'Test Data')

#%% Model Evaluation - Validation Data
model_evaluation(model, playtype_encoder, X_validation, y_validation, 'Validation Data')

#%% Explainability - Surogate Model

#Unstack data
shape = X_stack.shape
X_unstacked = X_stack.reshape((shape[0], shape[1]*shape[2]))

unstack_var_names = [var + '(t-3)' for var in variables]
unstack_var_names += [var + '(t-2)' for var in variables]
unstack_var_names += [var + '(t-1)' for var in variables]

unstacked_df = pd.DataFrame(X_unstacked, columns=unstack_var_names)

model.eval()
with torch.no_grad():
    y_pred = model.forward(torch.Tensor(X_stack).to(device)).argmax(dim = 1).cpu().numpy()

#train classifier
surogate = DecisionTreeClassifier()
surogate.fit(unstacked_df, y_pred)

#%%
#extract variable importance
importances = surogate.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh([unstack_var_names[i] for i in indices], importances[indices])
plt.yticks(fontsize = 7)

plt.xlabel('Relative Importance')
plt.show()

indices = indices[::-1]
for feature, importance in zip([unstack_var_names[i] for i in indices], importances[indices]):
    print(F'{feature}: {importance.round(4)}')


#%% Load Model
PATH = './Models/'
name = input('Model Name:')

model = SimpleLSTM(len(variables), hidden_size = 124, num_layers = 2)
model.load_state_dict(torch.load(PATH + name))
model.eval()
print('Model Loaded')

