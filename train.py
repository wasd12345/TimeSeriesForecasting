# Main training and inference script for comparing time series forecasting models

import os
import time
from datetime import datetime
from shutil import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

#Other evaluation metrics
from metrics import SMAPE, MAPE, bias, pearson_r, mutual_information, quantile_loss

#Logging / tracker class
from utils import Logger, plot_regression_scatterplot, plot_predictions

#Pytorch models
from models.DummyMLP import DummyMLP
import models.RecurrentEncoderDecoder as EncDec

#Tasks / problem applications
#(defined in specific way with Dataset class to be used by DataLoader)
import tasks.tsfake_task as tsfake_task
import tasks.periodphase_task as periodphase_task






# =============================================================================
# PARAMETERS
# =============================================================================

#Repeatability
TORCH_SEED = 12345
NUMPY_SEED = TORCH_SEED+1

# Task params
TASK = 'tsfake' #'periodphase' #'stocks' 'rainfall' 'energy' #which prediction task to do [which dataset]

#Performance Metrics / Optimization metric
#The actual function used for optimizing the parameters. Provide the name of one key in the dict TRAINING_METRICS_TRACKED                            
OPTIMIZATION_FUNCTION_NAME = 'SMAPE'#'quantile_loss'#'SMAPE'#'q45_point_est'#'SMAPE'
QUANTILES_LIST = [.5, .05, .25, .75, .95] #[.05, .25, .45, .5, .55, .75, .95] #!!!!!!!!!!!!for now just use same quantiles for all quantile metrics but in general can differ
QUANTILES_INDS = [0,1,2,3,4]
#Format is: f'{metric name}' : ({function}, {loss indices list}, {dict of kwargs})
TRAINING_METRICS_TRACKED = {'mse_loss':(F.mse_loss, [0], {}),
                            'SMAPE':(SMAPE, [0], {}),
                            'MAPE':(MAPE, [0], {}),
                            'bias':(bias, [0], {}),
                            'pearson_r':(pearson_r, [0], {}),
                            'mutual_information':(mutual_information, [0], {}),
                            'quantile_loss':(quantile_loss, QUANTILES_INDS, {'quantiles':QUANTILES_LIST}), #!!!!!!!!!must have indices list in order corresponding to the listy of quantiles
                            # 'q50_point_est':(quantile_loss, {'quantiles':[.50], 'loss_inds':ssssssssss})
                            # 'l1_loss':(F.l1_loss, [], {}) #Just to compare to 50q pinball loss to make sure is same
                            }
#In general doesn't have to be same as training, but for now, loss plotting scode assumes it is
VALIDATION_METRICS_TRACKED = {'mse_loss':(F.mse_loss, [0], {}),
                            'SMAPE':(SMAPE, [0], {}),
                            'MAPE':(MAPE, [0], {}),
                            'bias':(bias, [0], {}),
                            'pearson_r':(pearson_r, [0], {}),
                            'mutual_information':(mutual_information, [0], {}),
                            'quantile_loss':(quantile_loss, QUANTILES_INDS, {'quantiles':QUANTILES_LIST}),
                            }

# Model params
MODEL = 'RecurrentEncoderDecoder' #'DummyMLP' ########'dsrnn' #or try all models, to do direct comparison      MODEL_LIST = ...
#!!!!!!! vs. MODEL_LIST = ['dsrnn', 'Cho',...] and then track stats for each model, at same time, using individual optimizers but exact same training / val batches
#!!!!!!! ENSEMBLE_N_MODELS = 1 #Number of models to average together in bagged ensemble
#!!!!!!! ENSEMBLE_N_CHECKPOINTS = 1 #Number of checkpoints over which to do weight averaging [EMA weighting]



# Pre-processing optionss
# NORMALIZATION = 'once' #'windowed' #how to do normalization: one-time, or in a moving sliding window for each chunk of input
# BOX_COX - standard 1var Box-Cox power transform
# DESEASONED - 
# LEARNED - learn the parameters of the transform via backprop
# META - learn the params via metalearning

# Training params
HISTORY_SIZE_TRAINING_MIN_MAX = [20,75] #[min,max] of allowed range during training. For each batch, an int in [min,max] is chosen u.a.r. as the history size
HORIZON_SIZE_TRAINING_MIN_MAX = [7,30] #same idea but for the horizon size
HISTORY_SIZE_VALIDATION_MIN_MAX = [20,70] #Same idea but the HISTORY for validation
HORIZON_SIZE_VALIDATION_MIN_MAX = [7,25] #HORIZON for validation
MAX_EPOCHS = 300 #89
# EARLY_STOPPING_K = 3 #None #If None, don't use early stopping. If int, stop if validation loss in at least one of the most recent K epochs is not less than the K-1th last epoch
#!!!!!!!!!!!! for now assuming the loss metric is the one being optimized
# BATCHSIZE_SCHEDULE = ppppppp #how to adjust batchszie with training, e.g. random btwn min max,    vs.    decreasing batchsizes, etc.
BS_0__train = 200 #Initial training batchsize (since batchsize may vary)
BS_0__val = 256 #Initial validation batchsize
# TRACK_INPUT_STATS = False #True #Whether to trackbasic descriptive stats in input batches (feature-wise statistics)
NUM_WORKERS = 4 #For DataLoader, the num_workers. Just setting equal to number of cores on my CPU

# !!!!!!!!! Analysis params (for making heatmaps of metrics as f(history,horizon))
# HISTORY_SIZES = [7,10,50,100]
# HORIZON_SIZES = [3,5,10]









# =============================================================================
# INITIALIZATION
# =============================================================================
# CPU vs. GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True
torch.manual_seed(TORCH_SEED)
np.random.seed(NUMPY_SEED)

OUTPUT_DIR = 'output'

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
    INPUT_SIZE = train_set.get_n_input_features() #Feature dimension of input is just 1, since we just have a scalar time series for this fake example data
elif TASK == 'tsfake':
    n_multiv = 3 #Just for this synthetic task, make sure is same as created
    n_ext_feats = 2
    data_len = 400
    TRAIN_PATH = os.path.join('data', f'{TASK}', f'tsfake-sinusoid-{n_multiv}vars-{n_ext_feats}feats-1000-len{data_len}-train.pt')
    VAL_PATH = os.path.join('data', f'{TASK}', f'tsfake-sinusoid-{n_multiv}vars-{n_ext_feats}feats-256-len{data_len}-val.pt')
    history_span = 66 #!!!!!should randomly vary over training
    horizon_span = 14 #!!!!!rand
    history_start = 2 #!!!!!!keeping in mind 0 indexing, so start=K means K+1 th timestep, i.e. it is legit to have start=0
    train_set = tsfake_task.TSFakeDataset(TRAIN_PATH, n_multiv, n_ext_feats, history_span, horizon_span, history_start)
    train_dl = DataLoader(train_set, batch_size=BS_0__train, shuffle=True, num_workers=NUM_WORKERS)
    val_set = tsfake_task.TSFakeDataset(VAL_PATH, n_multiv, n_ext_feats, 55, 12, 4)
    val_dl = DataLoader(val_set, batch_size=BS_0__val, shuffle=True)
else:
    raise Exception(f'"{TASK}" TASK not implemented yet')
#Regardless of dataset, should have these variables to track them:
INPUT_SIZE = train_set.n_input_features #Feature dimension of input is just 1, since we just have a scalar time series for this fake example data
D_FUTURE_FEATURES = train_set.n_external_features #Number of future features. E.g. timestamp related features on horizon timesteps
N_MULTIVARIATE = train_set.n_multivariate
assert(INPUT_SIZE == N_MULTIVARIATE + D_FUTURE_FEATURES)

       
#Model (for now assuming you must choose a single model) !!!!!!!!
if MODEL == 'DummyMLP':
    model = DummyMLP(history_span, INPUT_SIZE)
elif MODEL == 'RecurrentEncoderDecoder':
    N_LAYERS = 2#3
    D_HIDDEN = 32
    #!!!!!!!!!!!!!!!!!!!! get number of quantiles
    #if 0 in QUANTILES_INDS it means one of the quantiles was used as point estimate
    #otherwise there was separately another index used as the point estimate
    #*** 0 is always assumed as point estimate, i.e. we ALWAYS make a point estimate!!!!!!!!!
    Q_QUANTILES = len(QUANTILES_INDS) if (0 in QUANTILES_INDS) else len(QUANTILES_INDS)+1
    BIDIRECTIONAL_ENC = False #False #True #Use bidirectional encoder
    P_DROPOUT_ENCODER = 0.#.25
    P_DROPOUT_DECODER = 0.#.25
    enc_dec_params = {'architecture':'LSTM-LSTM',
                      'M':N_MULTIVARIATE,
                      'Q':Q_QUANTILES,
                      'encoder_params':{'architecture':'recurrent',
                                        'd_input':INPUT_SIZE,
                                        'n_layers':N_LAYERS,
                                        'd_hidden':D_HIDDEN,
                                        'bidirectional':BIDIRECTIONAL_ENC,
                                        'p_dropout_encoder':P_DROPOUT_ENCODER,
                                        },
                      'decoder_params':{'architecture':'recurrent',
                                        'n_layers':N_LAYERS,
                                        'd_hidden':D_HIDDEN,
                                        'p_dropout_decoder':P_DROPOUT_DECODER,
                                        'd_future_features':D_FUTURE_FEATURES,
                                        'attention_type':None
                                        }
                      }
    model = EncDec.RecurrentEncoderDecoder(**enc_dec_params).to(device)  
    model_run_params = {'horizon':train_set.horizon_span,
                        'teacher_forcing':True,
                        'teacher_forcing_prob':.25
                        }
    
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
optim_function = TRAINING_METRICS_TRACKED[OPTIMIZATION_FUNCTION_NAME][0]

START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_text = f'START_TIME = {START_TIME}\n' + \
    f'TORCH_SEED = {TORCH_SEED}\n' + \
    f'NUMPY_SEED = {NUMPY_SEED}\n' + \
    f'TASK = {TASK}\n' + \
    f'N_MULTIVARIATE (#dimensions of the time series) = {N_MULTIVARIATE}\n' + \
    f'D_FUTURE_FEATURES (#features) = {D_FUTURE_FEATURES}\n' + \
    f'INPUT_SIZE (total #features of time series) = {INPUT_SIZE}\n' + \
    f'MODEL = {MODEL}\n' + \
    f'opt = {opt.__class__.__name__}\n' + \
    f'TRAINING_METRICS_TRACKED = {[of for of in TRAINING_METRICS_TRACKED]}\n' + \
    f'VALIDATION_METRICS_TRACKED = {[of for of in VALIDATION_METRICS_TRACKED]}\n' + \
    f'optim_function = {optim_function.__name__}\n'

#Init the logger to track things / plot things, do metalearning
logger = Logger(TASK, START_TIME, TRAINING_METRICS_TRACKED,
                VALIDATION_METRICS_TRACKED, N_MULTIVARIATE, Q_QUANTILES)

#Save summary text to log file
print(summary_text)
with open(os.path.join(logger.output_dir, 'summary_text.txt'), 'w') as gg:
    gg.write(summary_text)

#Save parameters to txt file
#...
    
#For repeatability, copy this script to output dir
THIS_SCRIPT_NAME = 'train.py'
copy(THIS_SCRIPT_NAME, os.path.join(logger.output_dir, THIS_SCRIPT_NAME))



# =============================================================================
# TRAINING LOOP
# =============================================================================
train_batchsize_this_epoch = BS_0__train
val_batchsize_this_epoch = BS_0__val
t0 = time.perf_counter()
for epoch in range(MAX_EPOCHS):
    print(f'------------- Starting epoch {epoch} -------------\n')
    
    
    # =============================================================================
    # Training
    # =============================================================================
    print('Training...\n')
    t_train_start = time.perf_counter()
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
        batch_inds = sample[0]
        X = sample[1].float()
        Y = sample[2].float()
        transform_params = sample[3]
        
        # PRE_PROCESSING
        # Transform this chunk of data. May want to learn the  preprocessing
        # method instead of using a deterministic transformation like 
        # standard scaling. (E.g. can learn the parameter lambda in Box-Cox).
        # So, do pre-processing here in the train script instead of in the 
        # Dataset and DataLoader.
        # 
        # E.g. below, just do standard scaling, but allowing for future improvements:
        X, Y = tsfake_task.standard_scale(X, Y, transform_params)
        
        
        # Transfer to GPU, if available:
        X = X.to(device)
        # Y = Y.to(device)
        Y_teacher = Y[:,:,:N_MULTIVARIATE].to(device)
        future_features = Y[:,:,N_MULTIVARIATE:].to(device)
        
        # batch x length x features
        # print('X.shape', X.shape)
        # print('Y.shape', Y.shape)        
        # print('Y_teacher.shape', Y_teacher.shape)  
        # print('future_features.shape', future_features.shape)          

        bsize = X.shape[0]
        logger.n_exs_cumulative_per_batch += bsize #Do this per training batch, since potentially have variable batchsize
        logger.batchsizes['training'].extend([bsize])
        
        #Get basic descriptive stats on the batch INPUTS
        #(e.g. if want to do use this info to intentionally change input
        #feature distributions [for curriculum learning, metalearning, etc.])
        #...
        
        # Run the forward and backward passes:
        opt.zero_grad()
        #Y_pred = model(X) #For models like DummyMLP which do direct forecast, i.e. don't rely on recurrent decoder predictions, so don't need Y vals:
        # vs. for recurrent deocders, which may use teacher forcing.
        #If don't need teacher forcing(not implemented yet anyway), then can ignore Y
        #if the model works for variable size horizons, must specify what horizon size to use:
        model_run_params = {}
        if MODEL == 'RecurrentEncoderDecoder':
            model_run_params = {'horizon_span':train_set.horizon_span,
                                'teacher_forcing':True,
                                'teacher_forcing_prob':.25
                                }            
        Y_pred = model(X, future_features, Y_teacher, **model_run_params)
        # print('Y_pred.shape', Y_pred.shape)
    
        #format is [batch x T_horizon x multivar x 1]
        Y_teacher = torch.unsqueeze(Y_teacher, 3)
        # print('Y_teacher.shape', Y_teacher.shape)
        
        #For now just optimize on single loss function even when tracking multiple functions.
        #Can combine functions by just defining a new combined function in the metrics.py script, then add that function to TRAINING_METRICS_TRACKED
        #If want to have combined function but with dynamically changing parameters, or e.g. 
        #completely switching loss functions during training [via random, RL, meta, etc.],
        #then would have to modify below ....
        for name, function_tuple in TRAINING_METRICS_TRACKED.items():
            # print(name)
            function = function_tuple[0]
            loss_inds = function_tuple[1]
            kwargs = function_tuple[2]
            #When dealing with multivariate output, and quantile forecasting, may not use all outputs for all losees.
            #E.g. maybe use 0 index for the point estimate but other indices for quantile loss
            #Also, indices can be REused. If e.g. use a given index for both a point estimate and a 50th percentile estimate,
            #or if a point estimate index is resued for multiple point estimate loss functions (e.g. both SMAPE and MSELoss).
            #However, in the latter case, better to define a weighted combination of the different losses, and use it as one
            #of the cutom loss functions in the TRAINING_METRICS_TRACKED dict (and also set as the OPTIMIZATION_FUNCTION_NAME)
            #format is [batch x T_horizon x multivar x quantiles]
            Y_pred_this_loss = Y_pred[:,:,:,loss_inds]
            # print('Y_pred_this_loss.shape', Y_pred_this_loss.shape)
            

            #For each of the M multivariate output dimensions, get the loss:
            # (track this individually for each dimension for analsysis, but
            # use a weighted average for the actual loss function.
            # E.g. with uniform weights for each of the M dimensions:
            dim_weights = torch.ones(logger.n_multivariate)
            dim_weights = dim_weights/torch.sum(dim_weights)
            train_loss_combined = torch.zeros(1)
            train_loss_tracker = []#torch.Tensor()
            for ii, dim in enumerate(range(logger.n_multivariate)):
                loss_this_dim = function(Y_pred_this_loss[:,:,dim,:], Y_teacher[:,:,dim,:], **kwargs)
                loss_this_dim = loss_this_dim.reshape((-1))
                
                #Reduce the loss function to a scalar. The point estimate losses
                #will already be scalars but quantile_loss with Q>1 will be a 
                #vector, so reduce here by doing weighted average.
                #E.g. doing uniform weights:
                q_weights = torch.ones_like(loss_this_dim)
                q_weights = q_weights/torch.sum(q_weights)
                
                train_loss_combined += dim_weights[ii] * torch.dot(loss_this_dim, q_weights)
                train_loss_tracker.append(loss_this_dim.tolist())
                # train_loss_tracker = torch.cat((train_loss_tracker, loss_this_dim.data),dim=0)
                print(f'training batch {bb}, {name}, dim{dim}: {loss_this_dim.tolist()}')
                # print()

            # train_loss_tracker is [batch x M x Q] in a nested lists format
            logger.losses_dict['training'][name].append(train_loss_tracker)

            #If this is the single function that needs to be optimized
            if name == optim_function.__name__:
                #use this ones arg so will work even if doing:
                #    - multivariate output
                #    - and/or quantile_loss
                #(all 1's means equal weight to each variable/quantile. Can adjust weights as necessary if desired)
                #(and if only doing univariate point estimates, this is equivalent to having no arg as in usual loss.backward() )
                train_loss_combined.backward(torch.ones(train_loss_combined.shape))
        
        #Gradient clipping, etc.
        GRADIENT_CLIPPING = False#True
        if GRADIENT_CLIPPING:
            print('Doing gradient clipping')
            #torch.nn.utils.clip_grad_norm_(model.parameters(), ...)
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.)
        
        #Add a small amount of noise to the gradients as a form of regularization:
        # GRADIENT_NOISE = False
        # if GRADIENT_NOISE:
        #     ...
            
        opt.step()
        print('\n'*2)
        

    #To do per EPOCH changes, like diff history and horizon sizes, update for the next EPOCH:
    next_history = torch.randint(HISTORY_SIZE_TRAINING_MIN_MAX[0], HISTORY_SIZE_TRAINING_MIN_MAX[1], [1], dtype=int).item()
    next_horizon = torch.randint(HORIZON_SIZE_TRAINING_MIN_MAX[0], HORIZON_SIZE_TRAINING_MIN_MAX[1], [1], dtype=int).item()
    next_start = torch.randint(0, 10, [1], dtype=int).item() #!!!!!!!!this number is constraind by training size, history size, horizon size. Put in the daatet class to derive this valid range....
    #!!!!!!!!!!! not updating properly #train_set.update_timespans(history_span=next_history, horizon_span=next_horizon, history_start=next_start)
    train_set = tsfake_task.TSFakeDataset(TRAIN_PATH, train_set.n_multivariate, train_set.n_external_features, next_history, next_horizon, next_start)
    train_dl = DataLoader(train_set, batch_size=BS_0__train, shuffle=True)#, num_workers=NUM_WORKERS)

    t_train_end = time.perf_counter()
    elapsed_train_time = t_train_end - t_train_start
    cumulative_train_val_time = t_train_end - t0
    logger.n_exs_per_epoch += [logger.n_exs_cumulative_per_batch] #done at the epoch level
    print(f'elapsed_train_time = {elapsed_train_time}')
    print(f'cumulative_train_val_time = {cumulative_train_val_time}')



    # =============================================================================
    # Validation
    # =============================================================================
    print('\n\n\n\n')
    print('Validation...\n')
    t_val_start = time.perf_counter()
    model.eval()
    val_batchsize_this_epoch = BS_0__val #For now just use fixed batchsize but could adjust dynamically as necessary
    
    #with torch.set_grad_enabled(False):
    with torch.no_grad():
        for bb, sample in enumerate(val_dl):
            # print(f'validation batch {bb}')
            
            # Load this batch:
            batch_inds = sample[0]
            X = sample[1].float()
            Y = sample[2].float()
            transform_params = sample[3]
            #In validation, don't transform the Y (horizon), only the X (history).
            #But to have actual MSE / SMAPE / etc. metrics, invert the pre-processing on the model's output
            X, Y = X.to(device), Y.to(device)
            
            
            # PRE_PROCESSING
            # Do same transformations as during training phase.
            # E.g. below, just do standard scaling:         
            X, Y = tsfake_task.standard_scale(X, Y, transform_params)
            
            
            Y_gt = Y[:,:,:N_MULTIVARIATE]
            future_features = Y[:,:,N_MULTIVARIATE:]

            bsize = X.shape[0]
            logger.batchsizes['validation'].extend([bsize])
            logger.n_exs_cumulative_per_epoch.extend([logger.n_exs_per_epoch[-1]]) #To use for validation loss plots. Value is repeated for each validation batch.
            
            # Run the forward pass:
            model_run_params = {}
            if MODEL == 'RecurrentEncoderDecoder':
                model_run_params = {'horizon_span':val_set.horizon_span,
                                    'teacher_forcing':False,
                                    'teacher_forcing_prob':None
                                    }            
            
            Y_gt = torch.unsqueeze(Y_gt, 3)
            Y_pred = model(X, future_features, Y_gt, **model_run_params)
            # Invert the transformation process on Y_pred to get the final predictions:
            # Y_pred = tsfake_task.standard_scale_inverse(...)


            for name, function_tuple in VALIDATION_METRICS_TRACKED.items():
                function = function_tuple[0]
                loss_inds = function_tuple[1]
                kwargs = function_tuple[2]
                Y_pred_this_loss = Y_pred[:,:,:,loss_inds]
                
                #For each of the M multivariate output dimensions, get the loss:
                # dim_weights = torch.ones(logger.n_multivariate)
                # dim_weights = dim_weights/torch.sum(dim_weights)
                # val_loss_combined = torch.zeros(1)
                val_loss_tracker = []#torch.Tensor()
                for ii, dim in enumerate(range(logger.n_multivariate)):
                    loss_this_dim = function(Y_pred_this_loss[:,:,dim,:], Y_gt[:,:,dim,:], **kwargs)
                    loss_this_dim = loss_this_dim.reshape((-1))
                    q_weights = torch.ones_like(loss_this_dim)
                    q_weights = q_weights/torch.sum(q_weights)
                    # val_loss_combined += dim_weights[ii] * torch.dot(loss_this_dim, q_weights)
                    val_loss_tracker.append(loss_this_dim.tolist())
                    # val_loss_tracker = torch.cat((val_loss_tracker, loss_this_dim.data),dim=0)
                    print(f'validation batch {bb}, {name}, dim{dim}: {loss_this_dim.tolist()}')
                    # print()
    
                # val_loss_tracker is [batch x M x Q] in a nested lists format
                logger.losses_dict['validation'][name].append(val_loss_tracker)


            #Plot ONE randomly chosen time series. History, and prediction along with ground truth future:
            #For multivariate case, just treat each variable independently for scatterplots and correlations:
            INDEX = torch.randint(0,X.shape[0],[1],dtype=int).item()
            
            # Invert the transformation process on X, Y_pred, Y_gt to get the final history, and horizon predictions:
            X, Y_pred, Y_gt = tsfake_task.standard_scale_inverse(X, Y_pred, Y_gt, transform_params)

            #**The [batch x T x M x Q][0] indices are the point estimates (0th index along last axis)
            # so use Y_gt[:,:,MM,0] and Y_pred[:,:,MM,0]
            multivar_inds = [mm for mm in range(logger.n_multivariate)]
            for nn, MM in enumerate(multivar_inds):
                plot_regression_scatterplot(Y_pred[:,:,MM,0].view(-1), Y_gt[:,:,MM,0].view(-1), logger.output_dir, logger.n_epochs_completed, nn)
                plot_predictions(X[INDEX,:,MM], Y_gt[INDEX,:,MM,0], Y_pred[INDEX,:,MM], logger.output_dir, logger.n_epochs_completed, nn, QUANTILES_LIST, QUANTILES_INDS)
                # print(Y_pred[:10])
                # print(Y[:10])
            
            print()            
            

    #To do per batch changes, like diff history and horizon sizes, update for the next batch:
    #however this will greatly increase variance over validation metrics.
    #better to do a full range of history/horizon sizes, at each valifation epoch. !!!!!!!!!!!!!!!!!
    #or if the particular application has a single horizon size of interest then obviously use that....
    next_history = torch.randint(HISTORY_SIZE_VALIDATION_MIN_MAX[0], HISTORY_SIZE_VALIDATION_MIN_MAX[1], [1], dtype=int).item()
    next_horizon = torch.randint(HORIZON_SIZE_VALIDATION_MIN_MAX[0], HORIZON_SIZE_VALIDATION_MIN_MAX[1], [1], dtype=int).item()
    next_start = torch.randint(0, 10, [1], dtype=int).item() #!!!!!!!!this number is constraind by training size, history size, horizon size. Put in the daatet class to derive this valid range....
    val_set = tsfake_task.TSFakeDataset(VAL_PATH, val_set.n_multivariate, val_set.n_external_features, next_history, next_horizon, next_start)
    val_dl = DataLoader(val_set, batch_size=BS_0__val, shuffle=True)
            
    t_val_end = time.perf_counter()
    elapsed_val_time = t_val_end - t_val_start    
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
    train_and_val_time = elapsed_train_time + elapsed_val_time
    logger.epoch_time.extend([train_and_val_time])
    print(f'time this epoch = {train_and_val_time}')
    print(f'------------- Finished epoch {epoch} -------------\n\n\n\n\n\n\n\n\n\n')


print('finished training + validation')