import numpy as np

def crosstalk(M, inds=None):
    '''
    Compute crosstalk for a given set of responses of n_CT cell types
    to n_LC ligand combinations, where crosstalk is defined as the ratio
    of maximum off-target activation to minimum on-target activation.

    Parameters
    ----------
    M: numpy.ndarray, shape (n_LC, n_CT)
        Matrix where element (i, j) represents the pathway activity of
        cell type j when exposed to ligand combination i.
    inds: numpy.ndarray, shape (n_LC, n_CT)
        Boolean matrix where element (i, j) indicates whether cell type j
        is on-target (True) or off-target (False) for ligand combination i.
        Assumed to be identity matrix if None.
    '''
    n = M.shape[0]
    if (n == 1):
        return 0
    if inds is not None:
        assert(isinstance(inds, np.ndarray))
    else:
        inds = np.eye(n, dtype=bool)
    assert(inds.shape == M.shape)

    max_off = np.max(M[np.where(inds == 0)])
    min_on = np.min(M[np.where(inds == 1)])
    if not min_on:
        return np.inf
    return max_off / min_on
