#injecting outliers 
import numpy as np

def inject_outliers(counts, size_factors, freq=0.01, z_mean=np.log(3), z_sd=np.log(1.6)):
    """
    Injects artificial outliers into a count matrix.
    counts: (genes, samples) raw counts
    size_factors: (samples,) size factors
    freq: fraction of entries to corrupt
    Returns: corrupted_counts, outlier_mask """
    n_genes, n_samples = counts.shape
    
    # matrix of the size of the raw counts each either 0, 1 or -1
    mask = np.random.choice([-1, 0, 1], size=(n_genes, n_samples),
                            p=[1-freq, freq/2, freq/2])
    
    
    norm_counts = np.log2(counts  / size_factors[np.newaxis, :] + 1 )
    

    gene_sd = np.std(norm_counts, axis=1, ddof=1)  # (n_genes, 1)
    
    

    # z amplitude from the log normal distribution
    z = np.random.lognormal(z_mean, z_sd, size=(n_genes, n_samples))
    
   
    corrupted_log = norm_counts + mask * z * gene_sd[:, np.newaxis]


    corrupted_counts = counts.copy()
    
    corrupted_counts[mask != 0] = np.round(
        size_factors[np.newaxis, :] * 2 ** corrupted_log
    )[mask != 0]
    
    mask[mask!=0] = 1

    return corrupted_counts, mask

# p-value 
from scipy.stats import nbinom
"""
Even though we call theta the dispersion parameter, 
it is actually equal to 1/dispersion in the NB
"""
def compute_pvalues(counts, mu, theta):
    """
    counts: (genes, samples) observed counts
    mu: (genes, samples) expected mean
    theta: (genes,) dispersion per gene
    Returns: p-values (genes, samples)
    """
    theta = theta[:, np.newaxis]
    
    p = theta / (theta+mu)
    
    cdf = nbinom.cdf(counts, theta, p)
    sf = nbinom.sf(counts-1, theta, p)
    
    pvals = 2*np.minimum(0.5, cdf, sf)
    pvals = np.minimum(pvals, 1.0)
    return pvals

# AUC-PR
from sklearn.metrics import precision_recall_curve, auc

def compute_auc_pr(true_outlier_mask, outlier_scores):
    """
    true_outlier_mask: (genes x samples,) 1 if outlier, 0 otherwise
    outlier_scores: (genes x samples, ) 
    Returns: AUC-PR (float)
    """
    
    scores = - outlier_scores
    
    precission, recall, _ = precision_recall_curve(true_outlier_mask, scores)
    auc_pr = auc(recall, precission)
    return auc_pr
    