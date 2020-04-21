# Logger class for tracking different things during training/validation.
# Useful for:
#       - collecting statistics on training
#       - plotting + analysis
#       - optimizing training via hyperparameter tuning, metalearning, and/or RL


import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from scipy.stats import pearsonr


class Logger():
    """
    Just basic structure to organize training/validation related metrics
    and evaluate training progress
    """
    
    def __init__(self, task_name, start_time):
        self.task_name = task_name
        self.start_time = start_time
        self.total_elapsed_time = 0
        self.n_epochs_completed = 0
        self.n_exs_cumulative_per_batch = 0 #Updated after every batch
        self.n_exs_per_epoch = [] #Updated after every epoch(is sum of num examples over all training batches in that epoch)
        self.epoch_time = [] #track elapsed time per epoch (train and val combined)
        self.losses_dict = {'training':defaultdict(list), 'validation':defaultdict(list)} #!!!!! for now assuming only 1 model instead of ensemble of models
        self.batchsizes = {'training':[], 'validation':[]} #the list of batchsizes used during each epoch #!!!!!! assuming 1 model
        self.n_exs_cumulative_per_epoch = [] #Within an epoch, the cumulative number of training exs. Used for validation los plots. Should have repetition if have more than one validation batch.
        self.learning_rates = {}  #per epoch, the learning rate used
        self.weights_biases = {} #per epoch
        self.gradient_magnitudes = {}
        self.gradient_angles = {}
        
        self.output_dir = os.path.join('output', f'{self.start_time}__{self.task_name}')
        os.makedirs(self.output_dir)
        
        
        
    def plot_metrics(self):
        """
        Plot training and validation loss metrics [SMAPE, MAPE, MAAPE, MSE, etc.]
        vs. cumulative examples #or could also do [epoch / wallclock time]
        
        Each save will just append new data by overwrite previous fig
        
        self.losses_dict format is:
            {'training':{'SMAPE':[...], 'MAPE':[...]},
             'validation':{'SMAPE':[...], 'MAPE':[...]}}
            i.e. for each metric, there is a list of the value of that metric, for each batch (ordered)
        """
        
        #Get training losses: 
        #!!!!!!!!!!! assumes same set of metrics are recorded for both training and validation
        loss_names_list = list(self.losses_dict['training'].keys())
        
        for loss_name in loss_names_list:
            
            x_train = np.cumsum(self.batchsizes['training'])
            y_train = self.losses_dict['training'][loss_name]
            
            x_val = self.n_exs_cumulative_per_epoch
            y_val = self.losses_dict['validation'][loss_name]
            
            plt.figure()
            plt.title(f'Train & Validation {loss_name}', fontsize=20)
            plt.plot(x_train, y_train, marker='o', color='k', label='Train')
            plt.plot(x_val, y_val, marker='s', color='r', label='Validation')
            plt.xlabel('Cumulative Examples', fontsize=20)
            plt.ylabel(f'{loss_name}', fontsize=20)
            plt.legend(numpoints=1)
            
            savepath = os.path.join(self.output_dir, f'loss__{loss_name}.png')
            plt.savefig(savepath)
            plt.close()





    def plot_weights(self):
        """
        Plot histograms etc. of weights / biases as functions of epoch
        """
        pass
    
    def plot_activations(self):
        """
        Plot histograms etc. of activations as functions of epoch
        """
        pass
    
    def plot_gradients(self):
        """
        Plot histograms etc. of gradient magntudes, anngles as functions of epoch
        """
        pass       
    
    
    
def plot_predictions(x_history, y_true, y_pred, output_dir, epoch):
    """
    Visualize predictions
    
    **for now assuming univariate
    """
    x = torch.arange(x_history.numel() + y_true.numel())
    hist_end = x_history.numel()
    
    plt.figure()
    plt.title(f'Predictions at epoch {epoch}', fontsize=20)
    plt.plot(x[:hist_end], x_history, marker='o', color='k', linestyle='-', label='history')
    plt.plot(x[hist_end:], y_true, marker='o', color='b', linestyle='-', label='y_true')
    plt.plot([hist_end-1, hist_end], [x_history[-1], y_true[0]], marker='None', color='k', linestyle='-')
    plt.plot(x[hist_end:], y_pred, marker='x', color='r', linestyle='--', label='y_pred')
    plt.xlabel('Timestep', fontsize=20)
    plt.ylabel('Y', fontsize=20)
    plt.legend(numpoints=1)
    savepath = os.path.join(output_dir, f'predictions__epoch{epoch}__random.png') #just for single random time series
    plt.savefig(savepath)
    plt.close()


    
def plot_regression_scatterplot(pred, true, output_dir, epoch):
    """
    Flattened over all timesteps in entire batch
    """
    r, p_val = pearsonr(true,pred)
    n_dec = 4
    r_rounded = round(r, n_dec)
    p_rounded = round(p_val, n_dec)
    plt.figure()
    plt.title(f'Scatterplot at epoch {epoch}\nr={r_rounded}, p-val={p_rounded}', fontsize=16)
    plt.plot(true, pred, marker='o', color='b', linestyle='None')
    plt.xlabel('True', fontsize=20)
    plt.ylabel('Predicted', fontsize=20)
    
    minval = torch.min( torch.min(true), torch.min(pred) )
    maxval = torch.max( torch.max(true), torch.max(pred) )
    plt.plot([minval, maxval], [minval, maxval], linestyle='--', color='k')
    
    savepath = os.path.join(output_dir, f'scatterplot__epoch{epoch}.png')
    plt.savefig(savepath)
    plt.close()