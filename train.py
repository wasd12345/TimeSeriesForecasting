# Main training and inference script for comparing time series forecasting models

import os
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

#Other evaluation metrics
from metrics import SMAPE, MAPE, bias

#Logging / tracker class
from logging_util import Logger

#Pytorch models
from models.DummyMLP import DummyMLP
# from models.DSRNN import sssssssss

#Tasks / problem applications
#(defined in specific way with Dataset class to be used by DataLoader)
import tasks.tsfake_task as tsfake_task
import tasks.periodphase_task as periodphase_task
#... other tasks






# =============================================================================
# PARAMETERS
# =============================================================================

# Task params
TASK = 'tsfake' #'periodphase' #'stocks' 'rainfall' 'energy' #which prediction task to do [which dataset]
TRAINING_METRICS_TRACKED = [F.mse_loss, SMAPE, MAPE, bias] #'SMAPE' 'MAAPE' #List of metrics to track, one of which is actually optimized (OPTIMIZATION_FUNCTION)
OPTIMIZATION_FUNCTION_IND = 0 #The actual function used for optimizing the parameters. Provide the index within the list TRAINING_METRICS_TRACKED
VALIDATION_METRICS_TRACKED = [F.mse_loss, SMAPE, MAPE, bias] #!!!!!!!!!!!!In general doesn't have to be same as training, but for now, loss plotting scode assumes it is
#!!!!!!!! HISTORY_PARAMS = {} #e.g. min,max of allowed range during training, then random sample          vs. fixed
#!!!!!!!! HORIZON_PARAMS = {}


# Model params
MODEL = 'DummyMLP' #'dsrnn' #or try all models, to do direct comparison      MODEL_LIST = ...
#!!!!!!! vs. MODEL_LIST = ['dsrnn', 'Cho',...] and then track stats for each model, at same time, using individual optimizers but exact same training / val batches
#!!!!!!! ENSEMBLE_N_MODELS = 1 #Number of models to average together in bagged ensemble
#!!!!!!! ENSEMBLE_N_CHECKPOINTS = 1 #Number of checkpoints over which to do weight averaging [EMA weighting]


# Training params
MAX_EPOCHS = 19
#!!!!!!!! EARLY_STOPPING = False
# BATCHSIZE_SCHEDULE = ppppppp #how to adjust batchszie with training, e.g. random btwn min max,    vs.    decreasing batchsizes, etc.
BS_0__train = 200 #Initial training batchsize (since batchsize may vary)
BS_0__val = 70 #Initial validation batchsize
LOGS_DIR = 'logs'
OUTPUT_DIR = 'output'
TRACK_INPUT_STATS = False #True #Whether to trackbasic descriptive stats in input batches (feature-wise statistics)
NUM_WORKERS = 4 #For DataLoader, the num_workers. Just setting equal to number of cores on my CPU


# Analysis params
HISTORY_SIZES = [7,10,50,100]
HORIZON_SIZES = [3,5,10]


#!!!!! for now just use fixed size
# SEQ_LENGTH = 100 #HISTORY_SIZES[0] #can do for i for j loops over histories and horizons
#Some kinds of models implemented from papers can only train on a fixed HISTORY size, need to individually train different sizes (bad)
#so for now, just use fixed 30 timestep history size as example







# =============================================================================
# INITIALIZATION
# =============================================================================
# CPU vs. GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True

# DataLoaders
if TASK == 'periodphase':
    history_span = 100
    Y_COLNAME = 'period' #For this dataset, can predict period or phase
    TRAIN_PATH = os.path.join('data', 'periodphase', f'periodphase-1000-len-{history_span}-train.csv') #!!!!!!!!!!!!!!!!!
    VAL_PATH = os.path.join('data', 'periodphase', f'periodphase-256-len-{history_span}-val.csv')#!!!!!!!!!!!!!!!!!
    train_set = periodphase_task.periodphaseDataset(TRAIN_PATH, history_span, Y_COLNAME)
    train_dl = DataLoader(train_set, batch_size=BS_0__train, shuffle=True, num_workers=NUM_WORKERS)
    val_set = periodphase_task.periodphaseDataset(VAL_PATH, history_span, Y_COLNAME)
    val_dl = DataLoader(val_set, batch_size=BS_0__val)
    INPUT_SIZE = train_set.get_number_of_features() #Feature dimension of input is just 1, since we just have a scalar time series for this fake example data




elif TASK == 'tsfake':
    TRAIN_PATH = os.path.join('data', 'tsfake', 'train.csv')#f'tsfake-1000-len-400-train.csv') #!!!!!!!!!!!!!!!!!
    VAL_PATH = os.path.join('data', 'tsfake', 'val.csv')#f'tsfake-256-len-400-val.csv')#!!!!!!!!!!!!!!!!!
    history_span = 66 #!!!!!should randomly vary over training
    horizon_span = 7 #!!!!!rand
    history_start = 2 #!!!!!!keeping in mind 0 indexing, so start=K means K+1 th timestep, i.e. it is legit to have start=0
    train_set = tsfake_task.TSFakeDataset(TRAIN_PATH, history_span, horizon_span, history_start)
    train_dl = DataLoader(train_set, batch_size=BS_0__train, shuffle=True, num_workers=NUM_WORKERS)
    #val_set = tsfake_task.TSFakeDataset(VAL_PATH, 70, 15, 333)
    val_set = tsfake_task.TSFakeDataset(VAL_PATH, 66, 7, 2) #!!!!!!!!!For now testing use same val sizes as training
    val_dl = DataLoader(val_set, batch_size=BS_0__val)
    INPUT_SIZE = train_set.get_number_of_features() #Feature dimension of input is just 1, since we just have a scalar time series for this fake example data

else:
    raise Exception(f'"{TASK}" TASK not implemented yet')
    #INPUT_SIZE = number of features for this task


    

#Model
if MODEL == 'DummyMLP':
    model = DummyMLP(history_span, INPUT_SIZE)
# elif MODEL == 'dsrnn':
#     model = dsrnn(...)
# elif MODEL == 'cinar': #!!!!!!!!!
#     model = aaaa(...)
# elif MODEL == 'transformer': ##!!!!!!!! 
#     model = ddddddddd(...)
else:
    raise Exception(f'"{MODEL}" MODEL not implemented yet')




#!!!!!!!!!!!!!!!!!!!!
#just make sure training and metrics work when switched over to forecasting task:
import torch.nn as nn
class test_sequential(nn.Module):
    def __init__(self, history_span, horizon_span):
        super(test_sequential, self).__init__()
        self.history_span = history_span #test using fixed size input chunks
        self.MLP_block1 = nn.Sequential(
            nn.Linear(self.history_span, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, horizon_span)
            )
    def forward(self, x):
        return self.MLP_block1(x)
MODEL = 'test_sequential'
model = test_sequential(history_span, horizon_span)    
#!!!!!!!!!!!!!!!!!!!!!




#Optimizer
opt = torch.optim.Adam(model.parameters())
#scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, ***)

#Training Metric [metric which actually gets optimized]
optim_function = TRAINING_METRICS_TRACKED[OPTIMIZATION_FUNCTION_IND]
# train_criterion = TRAINING_METRICS_TRACKED()
# val_criterion = VALIDATION_METRICS_TRACKED()


START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_text = f'START_TIME = {START_TIME}\n' + \
    f'TASK = {TASK}\n' + \
    f'INPUT_SIZE (#features) = {INPUT_SIZE}\n' + \
    f'MODEL = {MODEL}\n' + \
    f'opt = {opt.__class__.__name__}\n' + \
    f'TRAINING_METRICS_TRACKED = {[of.__name__ for of in TRAINING_METRICS_TRACKED]}\n' + \
    f'VALIDATION_METRICS_TRACKED = {[of.__name__ for of in VALIDATION_METRICS_TRACKED]}\n' + \
    f'optim_function = {optim_function.__class__.__name__}\n'
print(summary_text)


#Init the logger to track things / plot things, do metalearning
logger = Logger(TASK, START_TIME)


#Save summary text to log file
#...

#Save parameters to txt file
#...


# =============================================================================
# TRAINING LOOP
# =============================================================================
train_batchsize_this_epoch = BS_0__train
val_batchsize_this_epoch = BS_0__val
for epoch in range(MAX_EPOCHS):
    print(f'------------- Starting epoch {epoch} -------------\n')
    
    t0 = time.perf_counter()
    
    # Training
    print('Training...\n')
    model.train()
    

    #Other per-batch training params:
    train_batchsize_this_epoch = BS_0__train #For now just use fixed batchsize but could adjust dynamically as necessary
    #...
    
    #Learning rate scheduler adjustment
    #scheduler.step()

    #Potentially change batchsize, other things, by reinitializing DataLoader:
    #if also want to randomize over history and horizon sizes during training, then re-init Dataset each time
    #train_set = tsfake_task.TSFakeDataset(TRAIN_PATH, history_span, horizon_span, history_start)
    train_dl = DataLoader(train_set, batch_size=BS_0__train, shuffle=True)#, num_workers=NUM_WORKERS)
    #!!!!!!!!!! put log transform / box-cox etc. in Dataset transform method 

    for bb, sample in enumerate(train_dl):
        # print(f'training batch {bb}')
        X = sample[0].float()
        Y = sample[1].float()
        
        # Transfer to GPU, if available:
        X, Y = X.to(device), Y.to(device)
        print(X.shape)
        print(Y.shape)        

        bsize = X.shape[0]
        logger.n_exs_cumulative_per_batch += bsize #Do this per training batch, since potentially have variable batchsize
        logger.batchsizes['training'].extend([bsize])
        
        #Get basic descriptive stats on the batch INPUTS
        #(e.g. if want to do use this info to intentionally change input
        #feature distributions [for curriculum learning, metalearning, etc.])
        #...
        
        
        # Run the forward and backward passes:
        opt.zero_grad()
        y_pred = model(X)
        # print(y_pred)
        #!!!!!!!!!! for now just optimize on single loss function even when tracking multiple.
        #could do as combined loss of the diff loss functions, or change dynamically during training [random, RL, meta, etc.]
        for train_criterion in TRAINING_METRICS_TRACKED:
            train_loss = train_criterion(y_pred, Y)
            print(f'training batch {bb}, {train_criterion.__name__}: {train_loss.item()}')
            logger.losses_dict['training'][train_criterion.__name__].extend([train_loss.item()])
            #If this is the single function that needs to be optimized
            if train_criterion.__name__ == optim_function.__name__:
                # print('loss is train_criterion.__name__, so backward-----------')
                train_loss.backward()
                
        #Gradient clipping, etc.
        GRADIENT_CLIPPING = False#True
        if GRADIENT_CLIPPING:
            print('Doing gradient clipping')
            #torch.nn.utils.clip_grad_norm_(model.parameters(), ...)
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.)
        
        opt.step()


    
    elapsed_train_time = time.perf_counter() - t0
    logger.n_exs_per_epoch += [logger.n_exs_cumulative_per_batch] #done at the epoch level
    print(f'elapsed_train_time = {elapsed_train_time}')



    # Validation
    print('\n\n')
    print('Validation...\n')
    model.eval()
    val_batchsize_this_epoch = BS_0__val #For now just use fixed batchsize but could adjust dynamically as necessary
    
    #with torch.set_grad_enabled(False):
    with torch.no_grad():
        for bb, sample in enumerate(val_dl):
            # print(f'validation batch {bb}')
            
            # Transfer to GPU, if available:
            X = sample[0].float()
            Y = sample[1].float()
            X, Y = X.to(device), Y.to(device)

            bsize = X.shape[0]
            logger.batchsizes['validation'].extend([bsize])
            logger.n_exs_cumulative_per_epoch.extend([logger.n_exs_per_epoch[-1]]) #To use for validation loss plots. Value is repeated for each validation batch.
            
            
            # Run the forward pass:
            y_pred = model(X)
            for val_criterion in VALIDATION_METRICS_TRACKED:
                val_loss = val_criterion(y_pred, Y)
                print(f'validation batch {bb}, {val_criterion.__name__}: {val_loss.item()}')
                logger.losses_dict['validation'][val_criterion.__name__].extend([val_loss.item()])     
        
        
        #Print a few examples to compare
        # print(y_pred[:10])
        # print(Y[:10])
            
    elapsed_val_time = time.perf_counter() - elapsed_train_time
    print(f'elapsed_val_time = {elapsed_val_time}')

    # Save model / optimizer checkpoints:
    #...
        
    
    # Plot training/validation loss
    # Right now this plots losses vs. num training examples which is appended per batch.
    # So could run this after each batch but may as well just do at end of each epoch
    logger.plot_metrics()
    
    
    # Weight / bias stats
    #...
    # logger.weights_biases ...
    
    
    # Stats on gradient updates
    #...
    # logger.gradient_magnitudes[...] = ...
    # logger.gradient_angles[...] = ...
    # logger.plot_gradients(...)
    
    
    # Save text logs of stats / pickle save the instance of the Logger class:
    #...
    
    
    logger.n_epochs_completed += 1
    train_and_val_time = time.perf_counter() - t0
    logger.epoch_time.extend([train_and_val_time])
    print(f'time this epoch = {train_and_val_time}')
    print(f'------------- Finished epoch {epoch} -------------\n\n\n\n\n\n\n\n\n\n')


print('finished training + validation')