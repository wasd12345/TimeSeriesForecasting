# A few basic forecasting metrics.
# For optimization itself, and/or performance analysis

import torch
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

#def SMAPE_200(pred, true):
def SMAPE(pred, true):    
    """
    mean unweughted SMAPE. I.e. all timesteps of all series in batch are weighted equally.
    Using the *2 definition. So range is [0, 200] pct.
    """
    denom = torch.abs(true) + torch.abs(pred)
    smape = torch.where(denom == 0., torch.zeros_like(true), torch.abs(pred - true) / denom)
    mean_smape = smape.mean()
    return mean_smape * 200.


def bias(pred, true):
    """
    Check if the forecasts are biased up or down
    """
    return torch.sum(true - pred) / torch.sum(true + pred)


def MAPE(pred, true):
    """
    mean unweughted MAPE. I.e. all timesteps of all series in batch are weighted equally.
    """
    EPSILON = 10e-5
    denom = torch.abs(true)
    #!!!!!!!!!! due to definition of MAPE, this can give infs if time series has true=0, so consider using other metric
    mape = torch.abs(pred - true) / (denom + EPSILON)
    mean_mape = mape.mean()
    return mean_mape * 100.




def pearson_r(pred, true):
    """
    Wrapper around scipy stats pearsonr
    
    to put in same format as other matrix shaped loss functions, in case want to 
    have non-uniform weights later, i.e. weight different training examples differently
    
    returns (r, pval)
    
    """
    
    r, p_val = pearsonr(true.detach().numpy().flatten(), pred.detach().numpy().flatten())
    #return (r, p_val)
    return r

def mutual_information(pred, true):
    """
    Wrapper around sklearn.feature_selection.mutual_info_regression
    
    to put in same format as other matrix shaped loss functions, in case want to 
    have non-uniform weights later, i.e. weight different training examples differently
    
    returns MI
    
    """
    
    #for now , only for univariate forecasting. So reshapes entire batch of K timesteps into vector as if single feature
    MI = mutual_info_regression(true.detach().numpy().flatten().reshape(-1,1), pred.detach().numpy().flatten())
    return MI




    
def MAAPE(pred, true):
    """
    Arctangent MAPE

    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    
    
    
def QuantileLoss():
    pass

def QuantileHuberLoss():
    pass

def MSIS():
    pass