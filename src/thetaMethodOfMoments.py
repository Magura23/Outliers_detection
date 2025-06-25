import numpy as np

def robust_method_of_moments_theta(cts, max_theta=1000, min_theta=0.01, min_mu=0.01, trim_frac=0.125):
    

    def trimmed_mean(a, trim_frac):
        n = a.shape[0]
        trim_n = int(n * trim_frac)
        a_sorted = np.sort(a)
        return np.mean(a_sorted[trim_n:n-trim_n]) if n > 2*trim_n else np.mean(a_sorted)

    # Step 1: Trimmed mean for each gene
    mue = np.apply_along_axis(trimmed_mean, 1, cts, trim_frac)
    mue = np.maximum(mue, min_mu)

    # Step 2: Trimmed variance for each gene
    se = (cts - mue[:, None])**2
    see = np.apply_along_axis(trimmed_mean, 1, se, trim_frac)
    ve = 1.51 * see

    # Step 3: Method of moments estimate
    theta = mue**2 / (ve - mue)
    theta[theta < 0] = max_theta
    theta = np.clip(theta, min_theta, max_theta)
    return theta