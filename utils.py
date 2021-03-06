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
import pandas as pd


TIGHT_LAYOUT = .2

class Logger():
    """
    Just basic structure to organize training/validation related metrics
    and evaluate training progress
    """
    
    def __init__(self, task_name, start_time, training_metrics_tracked, validation_metrics_tracked, n_multivariate, n_quantiles):
        self.task_name = task_name
        self.start_time = start_time
        self.training_metrics_tracked = training_metrics_tracked
        self.validation_metrics_tracked = validation_metrics_tracked
        self.n_multivariate = n_multivariate
        self.n_quantiles = n_quantiles
        
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
            i.e. for each metric, there is a list of the value of that metric, for each batch (ordered).
            Since doing multivariate and quantiles in tracking, the format is 3-nested list, i.e.:
                    [batch x M x Q] -> [batch [M [Q]]]
                
        """
        loss_names_set = set(self.losses_dict['training'].keys()).union(self.losses_dict['validation'].keys())
        
        #Quantile forecasting losses implemented in this repo:
        QUANTILE_FORECASTING_LOSS_NAMES = set(['quantile_loss', 'huber_quantile_loss', 'MSIS'])
        
        quantile_losses_used = loss_names_set.intersection(QUANTILE_FORECASTING_LOSS_NAMES)
        point_est_losses_used = loss_names_set.difference(QUANTILE_FORECASTING_LOSS_NAMES)
        
        #Deal with the point estimate losses, and other single valued losses: (all non-quantile losses):
        for loss_name in point_est_losses_used:
            print(loss_name)
            for dim in range(self.n_multivariate): #For each dimension, in the multivariate case:
                plt.figure()
                plt.title(f'Train & Validation {loss_name}, dim{dim}', fontsize=20)            
                #if loss_name in self.losses_dict['training']:
                if loss_name in self.training_metrics_tracked:
                    x_train = np.cumsum(self.batchsizes['training'])
                    y_train = self.losses_dict['training'][loss_name]
                    y_train = [batch[dim][0] for batch in y_train]
                    #Moving average over metric during training:
                    #!!!!!!!!!!! since graphing per batch, (so last batch of epoch could have fewer exs), should really first weight by number of samples in batch....
                    y_train_MA = pd.DataFrame(y_train).ewm(span=5,adjust=False).mean().values.flatten()
                    plt.plot(x_train, y_train, marker='o', color='k', linestyle='None', alpha=.3, label='Train')
                    plt.plot(x_train, y_train_MA, color='k', label='Train Ave')            
                if loss_name in self.validation_metrics_tracked:
                    x_val = self.n_exs_cumulative_per_epoch
                    y_val = self.losses_dict['validation'][loss_name]
                    y_val = [batch[dim][0] for batch in y_val]
                    #Moving average over metric during validation....
                    #since all batches of validation done after same amoutn of training, must merge values first 1 get single value per epoch, then do EMA
                    plt.plot(x_val, y_val, marker='s', color='r', linestyle='None', alpha=.3, label='Validation')
                plt.xlabel('Cumulative Examples', fontsize=20)
                plt.ylabel(f'{loss_name}', fontsize=20)
                plt.legend(numpoints=1)
                plt.grid()
                plt.tight_layout(TIGHT_LAYOUT)
                savepath = os.path.join(self.output_dir, f'metric__{loss_name}_dim{dim}.png')
                plt.savefig(savepath)
                plt.close()


        #All quantile losses:
        for loss_name in quantile_losses_used:
            
            #For this quantile loss, get the quantiles used for training, validation:
            q_train = []
            if loss_name in self.training_metrics_tracked:
                q_train = self.training_metrics_tracked[loss_name][2]['quantiles']
                
            q_val = []
            if loss_name in self.validation_metrics_tracked:
                q_val = self.validation_metrics_tracked[loss_name][2]['quantiles']
                
            all_q = set(q_train).union(set(q_val))
            # print(all_q)
            for qq in all_q:
                for dim in range(self.n_multivariate): #For each dimension, in the multivariate case:
                    plt.figure()
                    plt.title(f'Train & Validation {loss_name},\ndim {dim}, q={qq}', fontsize=20)            
                    if (loss_name in self.training_metrics_tracked) and (qq in self.training_metrics_tracked[loss_name][2]['quantiles']):
                        x_train = np.cumsum(self.batchsizes['training'])
                        gg = self.losses_dict['training'][loss_name]
                        ind = q_train.index(qq)
                        y_train = [batch[dim] for batch in gg]
                        y_train = [ii[ind] for ii in y_train]
                        #Moving average over metric during training:
                        #!!!!!!!!!!! since graphing per batch, (so last batch of epoch could have fewer exs), should really first weight by number of samples in batch....
                        y_train_MA = pd.DataFrame(y_train).ewm(span=5,adjust=False).mean().values.flatten()
                        plt.plot(x_train, y_train, marker='o', color='k', linestyle='None', alpha=.3, label='Train')
                        plt.plot(x_train, y_train_MA, color='k', label='Train Ave')            
                    if (loss_name in self.validation_metrics_tracked) and (qq in self.validation_metrics_tracked[loss_name][2]['quantiles']):
                        x_val = self.n_exs_cumulative_per_epoch
                        gg = self.losses_dict['validation'][loss_name]
                        ind = q_val.index(qq)
                        y_val = [batch[dim] for batch in gg]
                        y_val = [ii[ind] for ii in y_val]
                        #Moving average over metric during validation....
                        #since all batches of validation done after same amoutn of training, must merge values first 1 get single value per epoch, then do EMA
                        plt.plot(x_val, y_val, marker='s', color='r', linestyle='None', alpha=.3, label='Validation')
                    plt.xlabel('Cumulative Examples', fontsize=20)
                    plt.ylabel(f'{loss_name}', fontsize=20)
                    plt.legend(numpoints=1)
                    plt.grid()
                    plt.tight_layout(TIGHT_LAYOUT)
                    savepath = os.path.join(self.output_dir, f'metric__{loss_name}_dim{dim}_q{qq}.png')
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
    
    
    
def plot_predictions(x_history, y_true, y_pred, output_dir, epoch, m_var, quantiles, quantiles_inds):
    """
    Visualize predictions
    
    x_history - [T_history]
    
    y_true - [T_horizon]
    
    y_pred - [T_horizon x Q], where y_pred[:,0] is the series of point estimates
            (the other 1:Q are quantiles in order of quantile_list)
    
    m_var - for the multivariate output case, m_var is which dimension of the M
            (just for labels in plots, NOT slicing arrays)
    
    quantiles - list of the quantiles used
    
    quantiles_inds - indices within y_pred corresponding to each of the values in quantiles
    """
    
    POINT_EST_IND = 0
    
    x = torch.arange(x_history.numel() + y_true.numel())
    hist_end = x_history.numel()
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.title(f'Predictions at epoch {epoch}, dim {m_var}', fontsize=20)
    plt.plot(x[:hist_end], x_history, marker='o', color='k', linestyle='-', label='history')
    plt.plot(x[hist_end:], y_true, marker='o', color='b', linestyle='-', label='y_true')
    plt.plot([hist_end-1, hist_end], [x_history[-1], y_true[0]], marker='None', color='k', linestyle='-') #line between last history and 1st horizon
    plt.plot(x[hist_end:], y_pred[:,POINT_EST_IND], marker='x', color='r', linestyle='--', label='y_pred')
    
    Q = len(quantiles)
    if Q > 0:
        lines = ['--']*Q
        colors = [str(ii) for ii in np.linspace(.90,.10,Q)]#np.random.choice(['r','g','y','c'],Q)
        for qq in range(Q):
            qq = Q - 1 - qq
            # if not (quantiles_inds[qq]] == POINT_EST_IND):
            plt.plot(x[hist_end:], y_pred[:,quantiles_inds[qq]], color=colors[qq], linestyle=lines[qq], label=quantiles[qq])
            ax.fill_between(x[hist_end:], y_pred[:,quantiles_inds[qq]], y_pred[:,POINT_EST_IND], color=colors[qq], alpha=.2)
            
    plt.xlabel('Timestep', fontsize=20)
    plt.ylabel('Y', fontsize=20)
    plt.legend(numpoints=1)
    plt.grid()
    plt.tight_layout(TIGHT_LAYOUT)
    savepath = os.path.join(output_dir, f'predictions_epoch{epoch}_dim{m_var}__random.png') #just for single random time series
    plt.savefig(savepath)
    plt.close()


    
def plot_regression_scatterplot(pred, true, output_dir, epoch, m_var):
    """
    Flattened over all timesteps in entire batch
    
    m_var - for the multivariate output case, m_var is which dimension of the M
    """
    r, p_val = pearsonr(true,pred)
    n_dec = 4
    r_rounded = round(r, n_dec)
    p_rounded = round(p_val, n_dec)
    plt.figure()
    plt.title(f'Scatterplot at epoch {epoch}, dim {m_var}\nr={r_rounded}, p-val={p_rounded}', fontsize=16)
    plt.plot(true, pred, marker='o', color='b', alpha=.5, linestyle='None')
    plt.xlabel('True', fontsize=20)
    plt.ylabel('Predicted', fontsize=20)
    
    minval = torch.min( torch.min(true), torch.min(pred) )
    maxval = torch.max( torch.max(true), torch.max(pred) )
    plt.plot([minval, maxval], [minval, maxval], linestyle='--', color='k')
    plt.grid()
    plt.tight_layout(TIGHT_LAYOUT)
    
    savepath = os.path.join(output_dir, f'scatterplot_epoch{epoch}_dim{m_var}.png')
    plt.savefig(savepath)
    plt.close()