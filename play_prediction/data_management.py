import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split

from tqdm import tqdm

# TODO : add meaningful comments

def load_data(PATH):
    # create data frame
    df = pd.read_csv(PATH)
    
    return df


def encode_target(data, target):
    
    # make target numbers
    encoder = OrdinalEncoder()
    data[target] = encoder.fit_transform(data[[target]]).astype('float64')
    
    return data, encoder
    
    
    
def scale_data(data, to_scale):
    
    std_scaler = StandardScaler()
    if isinstance(data, pd.DataFrame):    
        data[to_scale] = std_scaler.fit_transform(data[to_scale].copy())
    else:
        data = std_scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        # TODO : Find better way to generalize slicing, and return the scaler

    return data



def construct_lstm_dataset(df, variables):
    
    # TODO : remove this line of code, when LSTM is able to handle varying length
    df_time = df.groupby(['GameId', 'DriveId']).filter(lambda x: len(x) > 3).copy()
    
    X_stack = []
    y_stack = []

    for name, group in tqdm(df_time.groupby(['GameId', 'DriveId'])):
        X = [group[variables].shift(-x).values[:3] for x in range(0, len(group))][:len(group) -3]
        y = group['PlayType'].tolist()[3:] 
        
        X_stack += X
        y_stack += y

    X_stack = np.array(X_stack)
    y_stack = np.array(y_stack)
    
    return X_stack, y_stack



def unstack_lstm_dataset(data, variables):
    
    shape = data.shape
    unstacked_data = data.reshape((shape[0], shape[1]*shape[2]))
    
    # only valid for 3 time steps, needs to be changed if varying length is implemented
    unstack_var_names = [var + '(t-3)' for var in variables]
    unstack_var_names += [var + '(t-2)' for var in variables]
    unstack_var_names += [var + '(t-1)' for var in variables]

    unstacked_df = pd.DataFrame(unstacked_data, columns=unstack_var_names)
    
    return unstacked_df



def resample(data, target, mode = 'min'):
    
    grouped_data = []
    for label in target.unique():
        grouped_data.append(data[target== label])
    
    if mode == 'min':
        # Determine the smaller group size
        group_size = min([len(group) for group in grouped_data])
        
        # Resample the larger group to match the smaller group size
        resampled_data = [resample(data, n_samples=group_size, replace=False, random_state=42) for data in grouped_data]
        
    elif mode == 'max':
        # Determine the larger group size
        group_size = max([len(group) for group in grouped_data])

        # Resample the both groups to match the larger group size --> Bootstraping happening
        resampled_data = [resample(data, n_samples=group_size, replace=True, random_state=42) for data in grouped_data]
    
    elif mode == 'mean':
        #Determine mean group size
        group_size = int(sum([len(group) for group in grouped_data])/len(grouped_data))
        resampled_data = [resample(data, n_samples=group_size, replace=True, random_state=42) for data in grouped_data]
        


    data_resampled = np.concatenate([resampled_data], axis = 0)        
    
    targets = [np.repeat(label, group_size) for label in target.unique()]
    target_resampled = np.concatenate(targets, axis=0)
    
    return_df = pd.DataFrame(np.concatenate((data_resampled, target_resampled), axis = 1),
                             columns=data.columns)
    
    return return_df

   
     
def preprocess_data(data, target, scale = False, resample_mode = False, lstm = False, train_test = False):

    variables = ['Down', 'ToGo', 'YardLine', 'GameTime', 'PointDifference', 'DriveId', 'NO HUDDLE', 
                 'NO HUDDLE SHOTGUN', 'SHOTGUN', 'UNDER CENTER']    
    
    if lstm:
        X_data, y_data = construct_lstm_dataset(data, variables)

    # TODO : scale after train test split
    # TODO : maybe two different functions for lstm/non-lstm preprocessing
    if train_test:
        
        if lstm:
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)
            
            if scale:
                to_scale = ['Down', 'ToGo', 'YardLine', 'GameTime', 'PointDifference']
                X_train = scale_data(X_train, to_scale)
            
            return X_train, X_test, y_train, y_test
        
        else:
            X_train, X_test, y_train, y_test = train_test_split(data.drop(target, axis = 1),
                                                                data[target]) 
            
            if scale:
                to_scale = ['Down', 'ToGo', 'YardLine', 'GameTime', 'PointDifference']
                X_train = scale_data(X_train, to_scale)
            
            return X_train, X_test, y_train, y_test
        
    else:
        if scale:
            to_scale = ['Down', 'ToGo', 'YardLine', 'GameTime', 'PointDifference']
            data[to_scale] = scale_data(data[to_scale], to_scale)
            
        return data.drop(target, axis = 1), data[target]
    
    
    
    
    
    




