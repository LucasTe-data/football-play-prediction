from pathlib import Path

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

import time

from play_prediction import data_management, model, evaluation


# set device for using pytorch
if torch.cuda.is_available(): 
    device = "cuda:0" 
else: 
    device = "cpu" 
device = torch.device(device) 

# TODO : add meaningful comments

data_path = Path('./Data/Processed')

play_df = data_management.load_data(data_path / 'pbp-processed.csv')

play_df, playtype_encoder = data_management.encode_target(play_df, 'PlayType')

X_train, X_test, y_train, y_test = data_management.preprocess_data(play_df, 'PlayType',
                                                                   scale = True,
                                                                   lstm = True,
                                                                   train_test = True)

# TODO : add testloader to get validation loss
dataset_train = TensorDataset(torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device))
trainloader = DataLoader(dataset_train, shuffle=True, batch_size=256)

# TODO : remove need to have this list here
variables = ['Down', 'ToGo', 'YardLine', 'GameTime', 'PointDifference', 'DriveId', 'NO HUDDLE', 
             'NO HUDDLE SHOTGUN', 'SHOTGUN', 'UNDER CENTER'] 


playtype_model = model.PlayPredictionLSTM(len(variables), hidden_size = 248, num_layers = 2).to(device)

optimizer = torch.optim.Adam(playtype_model.parameters(), lr = 0.001)

criterion = nn.CrossEntropyLoss().to(device)

# TODO : add early stopping
# TODO : add validation loss ploting
epochs = 300
losses = []

playtype_model.train()

start_time = time.time()
for i in range(1, epochs + 1):
    
    for batch in trainloader:
        X = batch[0]
        y = batch[1]

        pred = playtype_model.forward(X)
        
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
    
    
    print(F'{i}/{epochs} - Loss : {float(loss):.4f} - Runtime : {hours}h:{minutes}min:{seconds}s / {mean_hours}h:{mean_minutes}min:{mean_seconds}s')


playtype_model.eval()

plt.plot(range(len(losses)), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


evaluation.model_evaluation(playtype_model, playtype_encoder, X_train, y_train, 'Train Data')
evaluation.model_evaluation(playtype_model, playtype_encoder, X_test, y_test, 'Test Data')