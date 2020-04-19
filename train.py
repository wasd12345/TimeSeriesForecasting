# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:00:33 2020

@author: GK
"""

import os
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader

#Other evaluation metrics
from metrics import SMAPE, MAAPE

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
TRAINING_METRIC = torch.nn.MSELoss #'SMAPE' 'MAAPE'
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
    SEQ_LENGTH = 100
    Y_COLNAME = 'period' #For this dataset, can predict period or phase
    TRAIN_PATH = os.path.join('data', 'periodphase', f'periodphase-1000-len-{SEQ_LENGTH}-train.csv') #!!!!!!!!!!!!!!!!!
    VAL_PATH = os.path.join('data', 'periodphase', f'periodphase-256-len-{SEQ_LENGTH}-val.csv')#!!!!!!!!!!!!!!!!!
    train_set = periodphase_task.periodphaseDataset(TRAIN_PATH, SEQ_LENGTH, Y_COLNAME)
    train_dl = DataLoader(train_set, batch_size=BS_0__train, shuffle=True, num_workers=NUM_WORKERS)
    val_set = periodphase_task.periodphaseDataset(VAL_PATH, SEQ_LENGTH, Y_COLNAME)
    val_dl = DataLoader(val_set, batch_size=BS_0__val)
    INPUT_SIZE = train_set.get_number_of_features() #Feature dimension of input is just 1, since we just have a scalar time series for this fake example data




elif TASK == 'tsfake':
    TRAIN_PATH = os.path.join('data', 'tsfake', 'train.csv')#f'tsfake-1000-len-400-train.csv') #!!!!!!!!!!!!!!!!!
    VAL_PATH = os.path.join('data', 'tsfake', 'val.csv')#f'tsfake-256-len-400-val.csv')#!!!!!!!!!!!!!!!!!
    history_span = 10 #!!!!!should randomly vary over training
    horizon_span = 7 #!!!!!rand
    history_start = 2 #!!!!!!keeping in mind 0 indexing, so start=K means K+1 th timestep, i.e. it is legit to have start=0
    train_set = tsfake_task.TSFakeDataset(TRAIN_PATH, history_span, horizon_span, history_start)
    train_dl = DataLoader(train_set, batch_size=BS_0__train, shuffle=True, num_workers=NUM_WORKERS)
    val_set = tsfake_task.TSFakeDataset(VAL_PATH, 70, 15, 333)
    val_dl = DataLoader(val_set, batch_size=BS_0__val)
    INPUT_SIZE = train_set.get_number_of_features() #Feature dimension of input is just 1, since we just have a scalar time series for this fake example data

else:
    raise Exception(f'"{TASK}" TASK not implemented yet')
    #INPUT_SIZE = number of features for this task


    







print('\n'*5)
history_span = 10 #!!!!!should randomly vary over training
horizon_span = 7 #!!!!!rand
history_start = 2
train_set = tsfake_task.TSFakeDataset(TRAIN_PATH, history_span, horizon_span, history_start)
train_set.print_attributes()
print()
train_dl = DataLoader(train_set, batch_size=400)#shuffle=True)
for bb, sample in enumerate(train_dl):
    print(f'training batch {bb}')
    X = sample[0].float()
    Y = sample[1].float()
    
    # Transfer to GPU, if available:
    X, Y = X.to(device), Y.to(device)

    print(X)
    print(Y)
    print(X.shape)
    print(Y.shape)
    print()


print('\n'*5)
history_span = 22 #!!!!!should randomly vary over training
horizon_span = 17 #!!!!!rand
history_start = 4
train_set = tsfake_task.TSFakeDataset(TRAIN_PATH, history_span, horizon_span, history_start)
train_set.print_attributes()
print()
train_dl = DataLoader(train_set, batch_size=400)#shuffle=True)
for bb, sample in enumerate(train_dl):
    print(f'training batch {bb}')
    X = sample[0].float()
    Y = sample[1].float()
    
    # Transfer to GPU, if available:
    X, Y = X.to(device), Y.to(device)

    print(X)
    print(Y)
    print(X.shape)
    print(Y.shape)
    print()









c=cccccccccccbbbbbbbbbbbbbb











#Model
if MODEL == 'DummyMLP':
    model = DummyMLP(SEQ_LENGTH, INPUT_SIZE)
# elif MODEL == 'dsrnn':
#     model = dsrnn(...)
# elif MODEL == 'cinar': #!!!!!!!!!
#     model = aaaa(...)
# elif MODEL == 'transformer': ##!!!!!!!! 
#     model = ddddddddd(...)
else:
    raise Exception(f'"{MODEL}" MODEL not implemented yet')


#Optimizer
opt = torch.optim.Adam(model.parameters())
#scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, ***)

#Training Metric [metric which actually gets optimied]
criterion = TRAINING_METRIC()



START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_text = f'START_TIME = {START_TIME}\n' + \
    f'TASK = {TASK}\n' + \
    f'INPUT_SIZE (#features) = {INPUT_SIZE}\n' + \
    f'MODEL = {MODEL}\n' + \
    f'opt = {opt.__class__.__name__}\n' + \
    f'criterion = {criterion}\n'
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
    
    t0 = time.clock()
    
    # Training
    print('Training...\n')
    model.train()
    

    #Other per-batch training params:
    train_batchsize_this_epoch = BS_0__train #For now just use fixed batchsize but could adjust dynamically as necessary
    logger.batchsizes['training'][epoch] = train_batchsize_this_epoch
    #...
    
    #Learning rate scheduler adjustment
    #scheduler.step()

    #Potentially change batchsize, other things, by reinitializing DataLoader:
    train_dl = DataLoader(train_set, batch_size=BS_0__train, shuffle=True)
    #!!!!!!!!!! put log transform / box-cox etc. in Dataset transform method 

    for bb, sample in enumerate(train_dl):
        # print(f'training batch {bb}')
        X = sample[0].float()
        Y = sample[1].float()
        
        # Transfer to GPU, if available:
        X, Y = X.to(device), Y.to(device)
        # print(X.shape)
        # print(Y.shape)        
        
        #Get basic descriptive stats on the batch INPUTS
        #(e.g. if want to do use this info to intentionally change input
        #feature distributions [for curriculum learning, metalearning, etc.])
        #...
        
        
        # Run the forward and backward passes:
        opt.zero_grad()
        y_pred = model(X)
        train_loss = criterion(y_pred, Y)
        print(f'training batch {bb}, train_loss: {train_loss.item()}')
        train_loss.backward()
        
        #Gradient clipping, etc.
        GRADIENT_CLIPPING = False#True
        if GRADIENT_CLIPPING:
            print('Doing gradient clipping')
            #torch.nn.utils.clip_grad_norm_(model.parameters(), ...)
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.)
        
        opt.step()


    elapsed_train_time = time.clock() - t0
    print(f'elapsed_train_time = {elapsed_train_time}')


    # Validation
    print('\n\n')
    print('Validation...\n')
    model.eval()
    val_batchsize_this_epoch = BS_0__val #For now just use fixed batchsize but could adjust dynamically as necessary
    logger.batchsizes['validation'][epoch] = val_batchsize_this_epoch
    
    #with torch.set_grad_enabled(False):
    with torch.no_grad():
        for bb, sample in enumerate(val_dl):
            # print(f'validation batch {bb}')
            
            # Transfer to GPU, if available:
            X = sample[0].float()
            Y = sample[1].float()
            X, Y = X.to(device), Y.to(device)

            # Run the forward pass:
            y_pred = model(X)
            val_loss = criterion(y_pred, Y)
            print(f'validation batch {bb}, val_loss: {val_loss.item()}')
        
        #Print a few examples to compare
        # print(y_pred[:10])
        # print(Y[:10])
            
    elapsed_val_time = time.clock() - elapsed_train_time
    print(f'elapsed_val_time = {elapsed_val_time}')

    # Save model / optimizer checkpoints:
    #...
        
    
    # Plot training/validation loss
    #...
    #logger.plot_losses(...)
    
    
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
    train_and_val_time = time.clock() - t0
    logger.epoch_time.extend([train_and_val_time])
    print(f'time this epoch = {train_and_val_time}')
    print(f'------------- Finished epoch {epoch} -------------\n\n\n\n\n\n\n\n\n\n')


print('finished training + validation')