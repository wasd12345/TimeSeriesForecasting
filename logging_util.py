# Logger class for tracking different things during training/validation.
# Useful for:
#       - collecting statistics on training
#       - plotting + analysis
#       - optimizing training via hyperparameter tuning, metalearning, and/or RL


import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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
        
        self.output_dir = os.path.join('output', self.start_time)
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