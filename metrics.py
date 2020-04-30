# A few basic forecasting metrics.
# For optimization itself, and/or performance analysis

import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression



# =============================================================================
# POINT ESTIMATES
# =============================================================================

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

def MAAPE(pred, true):
    pass




# =============================================================================
# QUANTILES (and/or POINT ESTIMATES, TOO)
# =============================================================================

def quantile_loss(pred, true, quantiles):#, reduction='mean'):
    """
    Pinball loss = 
        q * (true - pred)           if    true >= pred
        (1-q) * (pred - true)       if    true < pred
        
    returns a list of the mean pinball loss for each quantile q
    (mean is over all BATCHES and TIMESTEPS)
    
    Don't directlyoptimize this function. Instead
    """
    mean_pinball = []
    for nn, q in enumerate(quantiles):
        pinball = q*F.relu(true - pred) + (1.-q)*F.relu(pred - true)
        mean_pinball.append(pinball.mean())
    mean_pinball = torch.cat([i.reshape(-1) for i in mean_pinball])
    return mean_pinball

def huber_quantile_loss():
    pass

def MSIS():
    pass







# =============================================================================
# ADDITIONAL TRACKING METRICS FOR TRAINING (NOT OPTIMIZATION)
# =============================================================================

def pearson_r(pred, true):
    """
    Wrapper around scipy stats pearsonr
    
    to put in same format as other matrix shaped loss functions, in case want to 
    have non-uniform weights later, i.e. weight different training examples differently
    
    **NOT intended to be optimized directly. Just a useful metric to track.
    
    returns (r, pval)
    
    """
    
    r, p_val = pearsonr(true.detach().numpy().flatten(), pred.detach().numpy().flatten())
    #return (r, p_val)
    return torch.tensor(r)

def mutual_information(pred, true):
    """
    Wrapper around sklearn.feature_selection.mutual_info_regression
    
    to put in same format as other matrix shaped loss functions, in case want to 
    have non-uniform weights later, i.e. weight different training examples differently
    
    **NOT intended to be optimized directly. Just a useful metric to track.
    
    returns MI
    
    """
    
    #for now , only for univariate forecasting. So reshapes entire batch of K timesteps into vector as if single feature
    MI = mutual_info_regression(true.detach().numpy().flatten().reshape(-1,1), pred.detach().numpy().flatten())[0]
    return torch.tensor(MI)