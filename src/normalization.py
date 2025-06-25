import numpy as np

from .sizeFactor import size_factor_expressed


def log_size_factor(x, size_factors):
    """
    function to calculate the log-transformed size-factor-normalized of the count matrix (gene x samples) = (p x n)
    """

    x_log = np.log((x+1)/size_factors[np.newaxis, :])
    
    return x_log



"""
we need the log-transformed size-factor-normalized counts
return bias 
"""

def size_factor_normalization(x, factors):
    
    x_log = log_size_factor(x, size_factors=factors)
    
    bias = np.mean(x_log, axis =1, keepdims=True)
    
    x_norm = x_log - bias
    
    return x_norm, bias