import numpy as np
from scipy.stats import gmean

def size_factor_expressed(expressed):
    """
    calculate the size factor for each sample
    
    if a gene is 0 in any sample, it is not considered in the calculation
    
    implementation of DESeq2 size factor calculation
    
    output: vector of size factors for each sample
    """
    
    
    non_zero_id = np.all(expressed > 0 , axis=1)
    
    non_zero_genes= expressed[non_zero_id]
    
    
    geo_means = gmean(non_zero_genes, axis=1)

    ratios = non_zero_genes/geo_means[:, np.newaxis]
    
    size_factors = np.median(ratios, axis=0)
    
    return size_factors
    
    
    
    
    
    