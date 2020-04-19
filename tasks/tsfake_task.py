# Generate some synthetic univariate time series data and store in .csv
# The Dataset class has various options for train/val chunking of time series:
# HISTORY and HORIZON sizes, starts, offset between history and horizon (usually 0),
# reverse = True or False, dilation/stride for history and horizon (usually 1 for both),
# max percents outside the history and horizon (since NaN pad when window goes outside)


import torch
from torch.utils.data import Dataset
from tqdm import trange
import numpy as np
import pandas as pd
import os
import sys



def create_dataset(
        train_size,
        val_size,
        data_len=400,
        x_max = 10.,
        period=5.,
        phase=1.,
        seed=None,        
        ):
    """
    Randomly generate a data set of sinusoids for a simple regression problem.
    
    X = vector of time series input vals.
    Y = vector of time series input vals.
    """
    
    TASKNAME = 'tsfake'
    
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
        for i in trange(zz[1]):
            period = torch.rand(1).item()
            phase = np.pi*torch.rand(1).item() #Only [0,pi] phases so well defined with only 1 possible offset 
            x = torch.linspace(0., x_max, data_len)
            x = torch.sin(2.*np.pi*x/period + phase).tolist()
            
            
            #!!!!!!!!!
            #for debugging simplicity during dev, just do monotonic integer seqs:
            x = [i for i in range(data_len)]
            
            
            
            #Simulate some missing data by randomly assigning NaNs to ~ 5pct of values:
            P = .05
            nan_inds = np.random.choice([0,1], data_len, replace=True, p=[1.-P,P])
            x = [x[vv] if nan_inds[vv]==0 else np.nan for vv in range(len(x))]
            X_all.append(x)
        #Intentionally putin examples of leading and trailing missing data NaNs:
        K1 = 10
        K2 = 5
        X_all[0][:K1] = [np.nan]*K1
        X_all[1][-K2:] = [np.nan]*K2
        df = pd.DataFrame(data=X_all)
        out_path = os.path.join(data_dir, zz[0])
        df.to_csv(out_path, index=False, header=None)

    print(train_path)
    print(val_path)
    return train_path, val_path








class TSFakeDataset(Dataset):
    """Load one of the dataset csv and do necessary slicing.
    
    params:
            
        history_span - 
        horizon_length -
        history_start,
        history_horizon_offset=1,#i.e. horizon immediately follows history just 1 timestep later (0 gap)
        history_reverse=False,
        horizon_reverse=False,
        history_stride=1,
        horizon_stride=1,
        history_max_pct_outside=100., #0 would mean exclude anything that isn't fully contained, i.e. anything that had to be NaN padded. 100 would include all time series even if completely empty NaN padded.
        horizon_max_pct_outside=100.,
        history_max_num_outside=0., #Absolute number of timesteps allowed to be NaN padded before tht time series would be excluded
        horizon_max_num_outside=0.
    
    """

    def __init__(self, dataset_path,
                 history_span,
                 horizon_span,
                 history_start,
                 history_horizon_offset=1,#i.e. horizon immediately follows history just 1 timestep later (0 gap)
                 history_reverse=False,
                 horizon_reverse=False,
                 history_stride=1,
                 horizon_stride=1,
                 
                 #!!!!!!!!below not implemented yet:
                 #Might want to exclude some series if the given history/horizon #windows would include too little data, either percentage-based or absolute:
                 history_max_pct_outside=100., #0 would mean exclude anything that isn't fully contained, i.e. anything that had to be NaN padded. 100 would include all time series even if completely empty NaN padded.
                 horizon_max_pct_outside=100.,
                 history_max_num_outside=0., #Absolute number of timesteps allowed to be NaN padded before tht time series would be excluded
                 horizon_max_num_outside=0.):
                 #!!!!!!!!!!!!!!!
                
        super(TSFakeDataset, self).__init__()
        
        self.dataset_path = dataset_path
        self.history_span = history_span
        self.horizon_span = horizon_span
        self.history_start = history_start #keeping in mind that this is 0 indexed
        self.history_horizon_offset = history_horizon_offset
        self.history_reverse = history_reverse
        self.horizon_reverse = horizon_reverse
        self.history_stride = history_stride
        self.horizon_stride = horizon_stride
        self.history_max_pct_outside = history_max_pct_outside
        self.horizon_max_pct_outside = horizon_max_pct_outside
        self.history_max_num_outside = history_max_num_outside
        self.horizon_max_num_outside = horizon_max_num_outside
        self.series_exclude_mask = None#torch.zeros(batchsize) #For those series which don't pass the max_num_outside and max_pct_outside checks

        #Some derived quantities:
        #Get number of history and horizon points that will actually be used, given the strides:        
        self.history_npts = int(np.floor((self.history_span - 1) / self.history_stride)) + 1
        self.horizon_npts = int(np.floor((self.horizon_span - 1) / self.horizon_stride)) + 1
        #The first time step to predict is [end of input] + [gap]:
        self.history_end = self.history_start + self.history_span #-1 for 0 indexing
        self.horizon_start = self.history_end + (self.history_horizon_offset-1) #i.e. when offset=1 (default typical case), horizon starts immediately afterhistory (immediate next timestep after history ends)
        self.horizon_end = self.horizon_start + self.horizon_span
        
        self.data_set = torch.tensor(pd.read_csv(dataset_path, header=None).values, dtype=float)
        self.size = self.data_set.shape[0]

        #Within a given batch, exclude those series which don't pass the max_num_outside and max_pct_outside checks
        #(too many timesteps i nthe series would be NaNs, on either abolsute or percentage basis)
        #...
        #Do some checks to make sure dimensions work out, otherwise need NaN padding:
        #!!!!!!!!!!!
        #...


        print('Loading data')
        
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        format is:
        
        ------ HISTORY --- | ----- HORIZON ------
        x1, x2, ...,   x_N,| x_N+1, ...,        x_N+M
        
        
        """
        XY = self.data_set[idx] #!!!!!!!!!!!!assuming univariate data
        
        X = XY[self.history_start : self.history_end : self.history_stride]
        Y = XY[self.horizon_start : self.horizon_end : self.horizon_stride]

        if self.history_reverse==True:
            X = torch.flip(X.reshape(1,-1),[1]).reshape(-1) #!!!!!!!!for now ok with univariate time series
        if self.horizon_reverse==True:
            Y = torch.flip(Y.reshape(1,-1),[1]).reshape(-1) #!!!!!!!!for now ok with univariate time series        

        #Do some checks on dimensionality
        assert(self.history_npts == X.size()[0]) #!!!!!!!ok for now univariate
        assert(self.horizon_npts == Y.size()[0]) #!!!!!!!ok for now univariate

        return (X,Y) #X and Y are both tensors
    
    
    def print_attributes(self):
        """
        Just summarize the attributes
        """
        attributes = vars(self)
        for kk, vv in attributes.items():
            print(kk, vv)
    
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