# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:31:28 2020

@author: GK
"""




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
        self.epoch_time = [] #track elapsed time per epoch (train and val combined)
        self.losses_dict = {'training':{}, 'validation':{}} #!!!!! for now assuming only 1 model instead of ensemble of models
        self.batchsizes = {'training':{}, 'validation':{}} #for train/val, the batchsize used for each epoch #!!!!!! assuming 1 model
        self.learning_rates = {}  #per epoch, the learning rate used
        self.weights_biases = {} #per epoch
        self.gradient_magnitudes = {}
        self.gradient_angles = {}
        
    def plot_losses(self):
        """
        Plot training and validation losses vs. epoch / wallclock time
        """
        pass

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