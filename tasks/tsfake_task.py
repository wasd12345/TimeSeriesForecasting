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
        train_n_examples,
        val_n_examples,
        data_len=400,
        x_max = 10.,
        n_multivariate=3,
        n_external_features=2,
        seed=None,        
        ):
    """
    Randomly generate a data set of sinusoids for a simple regression problem.
    
    X = vector of time series HISTORY vals.
    Y = vector of time series HORIZON vals.
    
    **ASSUMPTIONS:
        - during prediction (HORIZON), we will have the same features as during the training chunk (HISTORY)
        - in the multivariate case we are predicting an M dimensional multivariate output for each timestep
        - other than the multivariate time series, there are potentially other external features [e.g. could be features derived from timestamps if available, or other any relevant feature]
        - So the total input feature dimension is (n_multivariate + n_external_features)
        - and during prediction, we have access to the n_external_features features and will predict the n_multivariate-dimensional output at each timestep
        
    - For this synthetic data, when creating multivariate series, just take the original one and apply a random affine transofrmation to get the other dimensions of the series
    
    """
    
    TASKNAME = 'tsfake'
    MODE = 'sinusoid' #'ramp'
    # INSERT_NANS = False#True
    
    data_dir = os.path.join('data',TASKNAME)
    
    if seed is not None:
        torch.manual_seed(seed)
    
    train_name = '{}-{}-{}vars-{}feats-{}-len{}-train.pt'.format(TASKNAME, MODE, n_multivariate, n_external_features, train_n_examples, data_len)
    val_name = '{}-{}-{}vars-{}feats-{}-len{}-val.pt'.format(TASKNAME, MODE, n_multivariate, n_external_features, val_n_examples, data_len)
    train_path = os.path.join(data_dir, train_name)
    val_path = os.path.join(data_dir, val_name)
    
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    else:
        if os.path.exists(train_path) and os.path.exists(val_path):
            return train_path, val_path
    
    
    
    for zz in zip([train_name, val_name], [train_n_examples, val_n_examples]):
        print('Creating data set for {}...'.format(zz[0]))
        X_all = torch.zeros((zz[1], data_len, n_multivariate + n_external_features)) #[(all)batchsize x T x (Mvars + Nfeatures)]
        for i in trange(zz[1]):
            
            if MODE == 'sinusoid':
                period = torch.rand(1).item()
                phase = np.pi*torch.rand(1).item() #Only [0,pi] phases so well defined with only 1 possible offset 
                slope = (2*(torch.rand(1) - .5)).item()/(x_max) #Have a random slope in [-1/data_len, 1/data_len]
                x = torch.linspace(0., x_max, data_len)
                trend = slope*x
                x = torch.sin(2.*np.pi*x/period + phase)
                x = x + trend
                x += .05*torch.randn(x.shape[0]) #Additive noise
                
            elif MODE == 'ramp':
                #for debugging simplicity during dev, just do monotonic integer seqs:
                x = torch.arange(data_len)
            
            #Make the M total variable dimensions by doing random affine transformations on original series:
            for M in range(n_multivariate):
                b = (2*(torch.rand(1) - .5)).item()
                UU = 1.10
                LL = .90
                a = (UU-LL)*torch.rand(1) + LL
                x_i = a*x + b
                
                # # #Simulate some missing data by randomly assigning NaNs to ~ 5pct of values:
                # if INSERT_NANS:
                #     P = .05
                #     nan_inds = np.random.choice([0,1], data_len, replace=True, p=[1.-P,P])
                #     x_i = [x[vv] if nan_inds[vv]==0 else np.nan for vv in range(len(x))]
                
                X_all[i,:,M] = x_i
               
                
            # For real dataset, may use external features. Here, just add random noise to make useless features but to allow to work later on with real features:
            X_all[i,:,n_multivariate:] = torch.rand(1, data_len, n_external_features)                
            
            
        # #Intentionally put in examples of leading and trailing missing data NaNs:
        # if INSERT_NANS:
        #     K1 = 10
        #     K2 = 5
        #     X_all[0][:K1] = [np.nan]*K1
        #     X_all[1][-K2:] = [np.nan]*K2
         
        

            
        # print(X_all)
        # print(X_all.shape)
        # print(X_all[0,:])
        # df = pd.DataFrame(data=X_all)
        out_path = os.path.join(data_dir, zz[0])
        # df.to_csv(out_path, index=False, header=None)
        torch.save(X_all, out_path)

    print(train_path)
    print(val_path)
    return train_path, val_path








class TSFakeDataset(Dataset):
    """Load data and batch into chunks.
    
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
                 n_multivariate,
                 n_external_features,
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
        self.n_multivariate = n_multivariate
        self.n_external_features = n_external_features
        self.n_input_features = self.n_multivariate + self.n_external_features
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

        #Calculate the derived quantities based on those above:
        #Important to have this method e.g. if doing randomized chunking or 
        #curriculum learning or similar:
        self.calculate_derived()
        
        print(f'Loading data:\n{self.dataset_path}')
        #self.data_set = torch.tensor(pd.read_csv(self.dataset_path, header=None).values, dtype=float)
        self.data_set = torch.load(self.dataset_path)
        self.size = self.data_set.shape[0]
        
        
    def calculate_derived(self):
        """
        Some of te Dataset attributes are derived from other values which can 
        potentially change during training/validation. So recalculate those
        derived quantities here.
        Meant to be called after every BATCH.
        
        E.g. if you change self.history_size after a batch of training, then we
        need to recauclate things like the the start and end points of the history
        chunk of the ensuing training batches.
        """
        #Some derived quantities:
        #Get number of history and horizon points that will actually be used, given the strides:        
        self.history_npts = int(np.floor((self.history_span - 1) / self.history_stride)) + 1
        self.horizon_npts = int(np.floor((self.horizon_span - 1) / self.horizon_stride)) + 1
        #The first time step to predict is [end of input] + [gap]:
        self.history_end = self.history_start + self.history_span #-1 for 0 indexing
        self.horizon_start = self.history_end + (self.history_horizon_offset-1) #i.e. when offset=1 (default typical case), horizon starts immediately afterhistory (immediate next timestep after history ends)
        self.horizon_end = self.horizon_start + self.horizon_span
        

        #Within a given batch, exclude those series which don't pass the max_num_outside and max_pct_outside checks
        #(too many timesteps i nthe series would be NaNs, on either abolsute or percentage basis)
        #...
        #Do some checks to make sure dimensions work out, otherwise need NaN padding:
        #!!!!!!!!!!!
        #...



        
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        format is:
        
        ------ HISTORY --- | ----- HORIZON ------
        x1, x2, ...,   x_N,| x_N+1, ...,        x_N+M
        
        where each given history side x_i is a vector with  #features = n_input_features,
        but each horizon side x_j does not have the multivariate series itself known
        
        so final shape of X is:
            [T_history x n_input_features]
        
        and final shape of Y is:
            [T_horizon x n_multivariate]
        """
        XY = self.data_set[idx]
        X = XY[self.history_start : self.history_end : self.history_stride]
        Y = XY[self.horizon_start : self.horizon_end : self.horizon_stride]
        # (For this dataset, the first M features are the multivariable series to predict; 
        # (the last F features are the external features [in this dataset just noise] )
        assert(X.shape[1] == self.n_input_features)
        assert(Y.shape[1] == self.n_input_features)
        # Both X and Y should have same number of features.
        # During training and inference, Y will always include the external features (e.g. known future information like timestamps)
        # However, the first n_multivariate values are not know since this is the multivariate distribution we are trying to predict.
        # But if we use teacher forcing during training, then for prediction we do use the multivariate series itself.
        # Either way, for train/validation, will pass in the full Y tensor and just use the relevant slices.

        if self.history_reverse==True:
            X = torch.flip(X, [0])
        if self.horizon_reverse==True:
            Y = torch.flip(Y, [0])
        
        #Do some checks on dimensionality
        assert(self.history_npts == X.shape[0])
        assert(self.horizon_npts == Y.shape[0])
        # print('X.shape', X.shape)
        # print('Y.shape', Y.shape)
        
        return (X,Y) #X and Y are both tensors
    
    
    def print_attributes(self):
        """
        Just summarize the attributes
        """
        attributes = vars(self)
        for kk, vv in attributes.items():
            print(kk, vv)
    
    # def get_n_multivariate(self):
    #     """
    #     Get number of multivariate dimensions for the prediction task
        
        
    #     For multivariate forecasting, some assumptions:
            
    #         - Of the input features, some subset of them are the actual multivariate variables we need to predict
    #         - the remaining features are exogenous variables, or e.g. timestamp features derived from the inputs
    #         - For the prediction task, if predicting an M dimensional multivariate output distribution,
    #           we have all M multivariate series as input features.
              
    #         - With this synthetic dataset, not doing timestamp features yet, so just manually say how many
    #           of the features will be treated as the multiavriate series.
    #     """
    #     return self.n_multivariate    
    
    # def get_n_future_features(self):
    #     """
    #     Get number of features
    #     """
    #     #For this fake data, no features used in future (e.g. no future
    #     #timestamp features), so just = 0
    #     self.n_future_features = 0
    #     return self.n_future_features
    
    
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

    # def update_timespans(self, **kwargs):
    #     """
    #     Change a few different properties: history_span, horizon_span, history_start
        
    #     Meant to be called between each training batch, in order to have random
    #     size history and horizons, instead of just training with a single fixed history
    #     and horizon. This randomization over sizes whould act as a form of regularization
    #     by causing the models to peerform well over a range of input/output sizes.
    #     Or can also be thought of as data augmentation since we are randomly chunking
    #     the data in different ways.
    #     """
    #     for kk,vv in kwargs.items():
    #         self.kk = vv
    #         # print(kk, self.kk)

    #     #And now since some parameters may have changed, must recalculate all
    #     #the derived parameters:
    #     self.calculate_derived()


if __name__ == '__main__':

    # if int(sys.argv[1]) == 1:
        # create_dataset(1000, 256, seed=123)
        
    create_dataset(1000, 256, seed=123)