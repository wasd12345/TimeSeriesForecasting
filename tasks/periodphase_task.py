# Generate some basic sinusoid time series data and store in .csv
#Easy time series (sinusoids) and prediction task is to predict either/both of 
#frequency/phase of the sine wave input.
#Simple scalar regression task as dummy test demo.


import torch
from torch.utils.data import Dataset
from tqdm import trange
import pandas as pd
import os
import sys
from math import pi


def create_dataset(
        train_size,
        val_size,
        data_len=100,
        x_max = 10.,
        period=5.,
        phase=1.,
        seed=None,        
        ):
    """
    Randomly generate a data set of sinusoids for a simple regression problem.
    
    X = vector of time series input vals.
    Y = (period, phase)
    Can regress on either or both.
    Just meant as a basic demo task.
    """
    
    TASKNAME = 'periodphase'
    
    data_dir = os.path.join('data',TASKNAME)
    
    if seed is not None:
        torch.manual_seed(seed)
    
    train_name = '{}-{}-len-{}-train.csv'.format(TASKNAME, train_size, data_len)
    val_name = '{}-{}-len-{}-val.csv'.format(TASKNAME, val_size, data_len)
    train_path = os.path.join(data_dir, train_name)
    val_path = os.path.join(data_dir, val_name)
    
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    else:
        if os.path.exists(train_path) and os.path.exists(val_path):
            return train_path, val_path
    
    
    
    for zz in zip([train_name, val_name], [train_size, val_size]):
        print('Creating data set for {}...'.format(zz[0]))
        X_all = []
        y_period = []
        y_phase = []
        for i in trange(zz[1]):
            period = torch.rand(1).item()
            phase = pi*torch.rand(1).item() #Only [0,pi] phases so well defined with only 1 possible offset 
            x = torch.linspace(0., x_max, data_len)
            x = torch.sin(2.*pi*x/period + phase).tolist()
            y_period.extend([period])
            y_phase.extend([phase])
            X_all.append(x)
        # print(y_period)
        # print(X_all)
        df = pd.DataFrame(data=X_all)
        df['period'] = y_period
        df['phase'] = y_phase
        out_path = os.path.join(data_dir, zz[0])
        df.to_csv(out_path, index=False)


    print(train_path)
    print(val_path)
    return train_path, val_path









class periodphaseDataset(Dataset):

    def __init__(self, dataset_path, data_len, y_colname):
        super(periodphaseDataset, self).__init__()
       
        print('Loading data')
        cols = [str(i) for i in range(data_len)] + [y_colname]
        self.data_set = torch.tensor(pd.read_csv(dataset_path, usecols=cols).values, dtype=float)
        self.size = self.data_set.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        XY = self.data_set[idx]
        #For this simple scalar regression problem, last column is regressor Y:
        X = XY[:-1]
        Y = XY[-1].view(-1) #EVen though this fake task predicts 1 scalar, standardizew/ other tasks which predict time series output as vector
        return (X,Y) #X and Y are both tensors
    
    
    def get_number_of_features(self):
        """
        Get number of features
        """
        self.number_features = 1 #For this fake data, just series of scalars, i.e. no multivariable time series, just univariate.
        return self.number_features
    
    def get_number_missing_features(self, idx):
        """
        For a given index, how many of the features are missing / present
        """
        pass

    def get_input_length(self):
        """
        Get length of input time series (assumes all input series are same length)
        """
        return self.data_set.shape[1] - 1
    
    def get_output_length(self):
        """
        Get length of output time series (assumes all output series are same length)
        """
        pass





if __name__ == '__main__':

    if int(sys.argv[1]) == 1:
        create_dataset(1000, 256, seed=123)