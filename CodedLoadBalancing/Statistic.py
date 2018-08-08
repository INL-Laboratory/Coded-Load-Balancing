# This module implement some statistical functions and classes

import numpy as np
import scipy.stats as stats


# This function returns smpl_nmbr samples of a random variable X \in {0,...,n-1}
# with Zipf distribution. The distribution is defined over the set {1,...,n}

def bounded_zipf(n, gamma, smpl_nmbr):
    gamma = float(gamma)
    x = np.arange(1, n+1)
    weights = x ** (-gamma)
    weights /= weights.sum()
    #print weights
    bndd_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))

    sample = bndd_zipf.rvs(size=smpl_nmbr)
    return (sample - 1)
