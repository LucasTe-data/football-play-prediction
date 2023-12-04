import torch
from torch import nn

# TODO : add meaningful comments

class PlayPredictionLSTM(nn.Module):

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



    def save(self):
        PATH = './Models/'
        name = input('Model Name:')
        
        # TODO : Add logic to check if model name already exists
        exists = True
    
        if exists:
            print('Name already in use. Do you want to overwrite?')
            confirm = input('[y]/[n]').lower()
            overwrite = (confirm == 'y')
            if overwrite:
                torch.save(self.state_dict(),  PATH + name)
                print(F'Model {name} Saved')
        else:    
            torch.save(self.state_dict(),  PATH + name)
            print(F'Model {name} Saved')
    
        # Hyperparameters should also be saved
        # TODO : Add logic for that


    def load(self):
        
        # TODO : Clean Up PATH variables
        PATH = './Models/'
        name = input('Model Name:')
        
        # TODO : Add Parameters for loaded model
        self.load_state_dict(torch.load(PATH + name))
        self.eval()
        
        print('Model Loaded')

