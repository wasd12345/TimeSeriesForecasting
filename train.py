# Main training and inference script for comparing time series forecasting models

import os
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

#Other evaluation metrics
from metrics import SMAPE, MAPE, bias, pearson_r, mutual_information


#Logging / tracker class
from utils import Logger, plot_regression_scatterplot, plot_predictions

#Pytorch models
from models.DummyMLP import DummyMLP
import models.RecurrentEncoderDecoder as RecEncDec
# from models.DSRNN import sssssssss

#Tasks / problem applications
#(defined in specific way with Dataset class to be used by DataLoader)
import tasks.tsfake_task as tsfake_task
import tasks.periodphase_task as periodphase_task
#... other tasks







# =============================================================================
# PARAMETERS
# =============================================================================
#Repeatability
TORCH_SEED = 12345

# Task params
TASK = 'tsfake' #'periodphase' #'stocks' 'rainfall' 'energy' #which prediction task to do [which dataset]
TRAINING_METRICS_TRACKED = [F.mse_loss, SMAPE, MAPE, bias, pearson_r, mutual_information] #'SMAPE' 'MAAPE' #List of metrics to track, one of which is actually optimized (OPTIMIZATION_FUNCTION)
OPTIMIZATION_FUNCTION_IND = 0 #The actual function used for optimizing the parameters. Provide the index within the list TRAINING_METRICS_TRACKED
VALIDATION_METRICS_TRACKED = [F.mse_loss, SMAPE, MAPE, bias, pearson_r, mutual_information] #!!!!!!!!!!!!In general doesn't have to be same as training, but for now, loss plotting scode assumes it is
HISTORY_SIZE_TRAINING_MIN_MAX = [7,90] #[min,max] of allowed range during training. For each batch, an int in [min,max] is chosen u.a.r. as the history size
HORIZON_SIZE_TRAINING_MIN_MAX = [7,50] #same idea but for the horizon size
HISTORY_SIZE_VALIDATION_MIN_MAX = [30,70] #Same idea but the HISTORY for validation
HORIZON_SIZE_VALIDATION_MIN_MAX = [30,40] #HORIZON for validation. E.g. 


# Model params
MODEL = 'RecurrentEncoderDecoder' #'DummyMLP' ########'dsrnn' #or try all models, to do direct comparison      MODEL_LIST = ...
#!!!!!!! vs. MODEL_LIST = ['dsrnn', 'Cho',...] and then track stats for each model, at same time, using individual optimizers but exact same training / val batches
#!!!!!!! ENSEMBLE_N_MODELS = 1 #Number of models to average together in bagged ensemble
#!!!!!!! ENSEMBLE_N_CHECKPOINTS = 1 #Number of checkpoints over which to do weight averaging [EMA weighting]
#!!!!!!! QUANTILES_LIST = [.05, .25, .45, .5, .55, .75, .95] #Which quantiles for which to predict vals


# Pre-processing optionss
NORMALIZATION = 'once' #'windowed' #how to do normalization: one-time, or in a moving sliding window for each chunk of input
# BOX_COX - standard 1var Box-Cox power transform
# DESEASONED - 
# LEARNED - learn the parameters of the transform via backprop
# META - learn the params via metalearning



# Training params
MAX_EPOCHS = 200#89
EARLY_STOPPING_K = 3 #None #If None, don't use early stopping. If int, stop if validation loss in at least one of the most recent K epochs is not less than the K-1th last epoch
#!!!!!!!!!!!! for now assuming the loss metric is the one being optimized


# BATCHSIZE_SCHEDULE = ppppppp #how to adjust batchszie with training, e.g. random btwn min max,    vs.    decreasing batchsizes, etc.
BS_0__train = 200 #Initial training batchsize (since batchsize may vary)
BS_0__val = 70 #Initial validation batchsize
LOGS_DIR = 'logs'
OUTPUT_DIR = 'output'
TRACK_INPUT_STATS = False #True #Whether to trackbasic descriptive stats in input batches (feature-wise statistics)
NUM_WORKERS = 4 #For DataLoader, the num_workers. Just setting equal to number of cores on my CPU


# Analysis params (for making heatmaps of metrics as f(history,horizon))
# HISTORY_SIZES = [7,10,50,100]
# HORIZON_SIZES = [3,5,10]


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
torch.manual_seed(TORCH_SEED)


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
    TRAIN_PATH = os.path.join('data', 'tsfake', 'train_sinusoid_noisy.csv')#f'tsfake-1000-len-400-train.csv') #!!!!!!!!!!!!!!!!!
    VAL_PATH = os.path.join('data', 'tsfake', 'val_sinusoid_noisy.csv')#f'tsfake-256-len-400-val.csv')#!!!!!!!!!!!!!!!!!
    history_span = 66 #!!!!!should randomly vary over training
    horizon_span = 14 #!!!!!rand
    history_start = 2 #!!!!!!keeping in mind 0 indexing, so start=K means K+1 th timestep, i.e. it is legit to have start=0
    train_set = tsfake_task.TSFakeDataset(TRAIN_PATH, history_span, horizon_span, history_start)
    train_dl = DataLoader(train_set, batch_size=BS_0__train, shuffle=True, num_workers=NUM_WORKERS)
    #val_set = tsfake_task.TSFakeDataset(VAL_PATH, 70, 15, 333)
    val_set = tsfake_task.TSFakeDataset(VAL_PATH, 66, 14, 4) #!!!!!!!!!For now testing use same val sizes as training, since testing using fixed size dummy MLP needs always same dims as training
    val_dl = DataLoader(val_set, batch_size=BS_0__val, shuffle=True)
    INPUT_SIZE = train_set.get_number_of_features() #Feature dimension of input is just 1, since we just have a scalar time series for this fake example data

else:
    raise Exception(f'"{TASK}" TASK not implemented yet')
    #INPUT_SIZE = number of features for this task


    

#Model (for now assuming you must choose a single model) !!!!!!!!
if MODEL == 'DummyMLP':
    model = DummyMLP(history_span, INPUT_SIZE)
elif MODEL == 'RecurrentEncoderDecoder':
    N_LAYERS = 2
    D_HIDDEN = 32
    D_OUTPUT = 1 #Since right now just test with univariate regression
    enc = RecEncDec.Encoder(INPUT_SIZE, D_HIDDEN, N_LAYERS)
    dec = RecEncDec.Decoder(D_OUTPUT, INPUT_SIZE, D_HIDDEN, N_LAYERS)
    model = RecEncDec.RecurrentEncoderDecoder(enc, dec).to(device)    
    
# elif MODEL == 'dsrnn':
#     model = dsrnn(...)
# elif MODEL == 'cinar': #!!!!!!!!!
#     model = aaaa(...)
# elif MODEL == 'transformer': ##!!!!!!!! 
#     model = ddddddddd(...)
else:
    raise Exception(f'"{MODEL}" MODEL not implemented yet')




#Optimizer
# Assuming use same optimizer for all parameters:
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
    
    
    # =============================================================================
    # Training
    # =============================================================================
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
        print(f'training batch {bb}')
        print('train_set.history_span', train_set.history_span)
        print('train_set.horizon_span', train_set.horizon_span)
        print('train_set.history_start', train_set.history_start)
        X = sample[0].float()
        Y = sample[1].float()
        
        # Transfer to GPU, if available:
        X, Y = X.to(device), Y.to(device)
        
        #!!!!!!!!!!! for now without changing the datasets themselves, since univariate, unsqueeze to be multivariate but with feature dim 1, to generalize:
        #and use unsqueeze dim is 2 so that shape is LSTM format:
        # batch x length x features
        X = torch.unsqueeze(X, 2)
        Y = torch.unsqueeze(Y, 2)
        # print(X.shape)
        # print(Y.shape)        

        bsize = X.shape[0]
        logger.n_exs_cumulative_per_batch += bsize #Do this per training batch, since potentially have variable batchsize
        logger.batchsizes['training'].extend([bsize])
        
        #Get basic descriptive stats on the batch INPUTS
        #(e.g. if want to do use this info to intentionally change input
        #feature distributions [for curriculum learning, metalearning, etc.])
        #...
        
        # Run the forward and backward passes:
        opt.zero_grad()
        #y_pred = model(X) #For models like DummyMLP which do direct forecast, i.e. don't rely on recurrent decoder predictions, so don't need Y vals:
        # vs. for recurrent deocders, which may use teacher forcing.
        #If don't need teacher forcing(not implemented yet anyway), then can ignore Y
        y_pred = model(X, Y)
        
        #!!!!!!!!!! for now just optimize on single loss function even when tracking multiple.
        #could do as combined loss of the diff loss functions, or change dynamically during training [random, RL, meta, etc.]
        for train_criterion in TRAINING_METRICS_TRACKED:
            train_loss = train_criterion(y_pred, Y)
            logger.losses_dict['training'][train_criterion.__name__].extend([train_loss.item()])
            print(f'training batch {bb}, {train_criterion.__name__}: {train_loss.item()}')
            #If this is the single function that needs to be optimized
            if train_criterion.__name__ == optim_function.__name__:
                train_loss.backward()
        
        #Gradient clipping, etc.
        GRADIENT_CLIPPING = False#True
        if GRADIENT_CLIPPING:
            print('Doing gradient clipping')
            #torch.nn.utils.clip_grad_norm_(model.parameters(), ...)
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.)
        
        opt.step()
        
        #To do per batch changes, like diff history and horizon sizes, update for the next batch:
        next_history = torch.randint(HISTORY_SIZE_TRAINING_MIN_MAX[0], HISTORY_SIZE_TRAINING_MIN_MAX[1], [1], dtype=int).item()
        next_horizon = torch.randint(HORIZON_SIZE_TRAINING_MIN_MAX[0], HORIZON_SIZE_TRAINING_MIN_MAX[1], [1], dtype=int).item()
        next_start = torch.randint(0, 10, [1], dtype=int).item() #!!!!!!!!this number is constraind by training size, history size, horizon size. Put in the daatet class to derive this valid range....
        #!!!!!!!!!!! not updating properly #train_set.update_timespans(history_span=next_history, horizon_span=next_horizon, history_start=next_start)
        train_set = tsfake_task.TSFakeDataset(TRAIN_PATH, next_history, next_horizon, next_start)
        print()
    
    elapsed_train_time = time.perf_counter() - t0
    logger.n_exs_per_epoch += [logger.n_exs_cumulative_per_batch] #done at the epoch level
    print(f'elapsed_train_time = {elapsed_train_time}')



    # =============================================================================
    # Validation
    # =============================================================================
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
            X = torch.unsqueeze(X, 2)
            Y = torch.unsqueeze(Y, 2)


            bsize = X.shape[0]
            logger.batchsizes['validation'].extend([bsize])
            logger.n_exs_cumulative_per_epoch.extend([logger.n_exs_per_epoch[-1]]) #To use for validation loss plots. Value is repeated for each validation batch.
            
            
            # Run the forward pass:
            y_pred = model(X, Y)
            for val_criterion in VALIDATION_METRICS_TRACKED:
                val_loss = val_criterion(y_pred, Y)
                print(f'validation batch {bb}, {val_criterion.__name__}: {val_loss.item()}')
                logger.losses_dict['validation'][val_criterion.__name__].extend([val_loss.item()])     
        
        
            #Print a few examples to compare
            plot_regression_scatterplot(y_pred.view(-1), Y.view(-1), logger.output_dir, logger.n_epochs_completed)
            
            #Plot ONE randomly chosen time series. History, and prediction along with ground truth future:
            INDEX = torch.randint(0,X.shape[0],[1],dtype=int).item()
            plot_predictions(X[INDEX], Y[INDEX], y_pred[INDEX], logger.output_dir, logger.n_epochs_completed)
            # print(y_pred[:10])
            # print(Y[:10])
            
            #other example metrics like r^2 or mutual information:
            # r, p_val = pearsonr(Y.view(-1), y_pred.view(-1))
            # print(r, p_val)
            print()
            
            
            
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