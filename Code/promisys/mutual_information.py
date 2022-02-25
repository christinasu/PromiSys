import numpy as np
import itertools
from scipy.spatial.distance import pdist, squareform
import promisys.bmp as psb

def MI_from_responses(resp_mat, sigma_sq):
    '''
    Compute mutual information between a set of n_LC ligand combinations
    and the resulting responses of n_CT cell types.

    Parameters
    ----------
    resp_mat: numpy.ndarray, shape (n_LC, n_CT)
        Matrix where element (i, j) represents the pathway activity of
        cell type j when exposed to ligand combination i.
    sigma_sq: float, positive
        Variance for Gaussian distributions capturing fluctuations in
        pathway activity.
    '''
    # Store key parameters
    n_LC = len(resp_mat)

    # Exponentiate distance matrix
    D = squareform(pdist(resp_mat, metric='sqeuclidean'))
    D_exp = np.exp(- D / (2 * sigma_sq))

    # Compute estimate of mutual information
    MI = np.log2(n_LC) - (1 / n_LC) * np.sum(np.log2(np.sum(D_exp, axis=1)))
    return MI

def MI_from_parameters(model_size, L0, R0, K, e, model='onestep', fixed=None,
                       Ke_comb=True, sigma_sq=0.5):
    '''
    Compute mutual information (MI) for specified set(s) of parameters.
    Note that responses are normalized by maximum for each cell type.

    Parameters
    ----------
    model_size: tuple of ints, shape (3, )
        Specification of model parameters, given as (nL, nA, nB) or (number of
        ligands, number of type I receptors, number of type II receptors).
    L0: array_like, shape (nL, ) or (n_envs, nL)
        Starting ligand concentrations for each environment, with each
        combination following order L(1), ..., L(nL).
    R0: array_like, shape (nA + nB, ) or (n_cells, nA + nB)
        Starting receptor levels for each cell type, with each set following
        order A(1), ..., A(nA), B(1), ..., B(nB).
    K: array_like, last dimension (nL*nA*nB, ) or (nL*nA + nL*nA*nB, )
        Affinity parameters. These parameters should correspond to complexes
        (optional dimeric intermediates followed by trimeric complexes)
        with last index increasing first. For a model with nL=2, nA=2, and nB=2,
        complex order is D_11, D_12, D_21, D_22, T_111, T_112, T_121, T_122,
        T_211, T_212, T_221, T_222 (D_ij included only for two-step model).
        If this array is two-dimensional, each row should correspond to one
        such set of parameters.
    e: array_like, last dimension (nL * nA * nB, )
        Efficiency parameters. These parameters should correspond to complexes
        with last index increasing first. For a model with nL=2, nA=2, and nB=2,
        complex order is T_111, T_112, T_121, T_122, T_211, T_212, T_221, T_222.
        If this array is two-dimensional, each row should correspond to one
        such set of parameters.
    model: string
        Indicator of desired model. Currently, 'onestep' is supported.
    fixed: None or array_like, shape (nL + nA + nB, )
        Indicator of whether each component has fixed concentration,
        specified in the order [L(1), L(2), ..., L(nL),
        A(1), ..., A(nA), B(1), ..., B(nB)]. Default is None. In this
        case, ligand concentrations remain constant, while receptors are
        depleted.
    Ke_comb: bool
        Indicator whether all possible combinations of affinity and efficiency
        parameters should be tested (relevant only if both are two-dimensional
        with the same length).
    sigma_sq: float, positive
        Variance for Gaussian distributions capturing fluctuations in
        pathway activity.
    '''
    nL, nA, nB = model_size
    L0_all = np.atleast_2d(np.array(L0))
    R0_all = np.atleast_2d(np.array(R0))
    K_all = np.atleast_2d(np.array(K))
    e_all = np.atleast_2d(np.array(e))

    # Choose appropriate function for specified model
    if model == 'onestep':
        sim_func = psb.sim_LAB_onestep

    # Set up parameter lists
    if len(K_all) == 1 and len(e_all) > 1:
        Ke_comb = True
    elif len(K_all) > 1 and len(e_all) == 1:
        e_all = np.tile(e_all, (len(K_all), 1))
        Ke_comb = False
    elif len(K_all) > 1 and len(e_all) > 1:
        if len(K_all) != len(e_all):
            Ke_comb = True
    if not Ke_comb:
        MI = np.zeros(len(K_all))
    else:
        MI = np.zeros((len(K_all), len(e_all)))

    # Compute MI for each parameter set
    for i in range(len(K_all)):
        if not Ke_comb:
            # Compute response matrix and MI
            resp_mat = psb.sim_S_LAB(model_size, L0, R0, K_all[i], e_all[i],
                                     model=model, fixed=fixed, norm=False).T
            resp_mat = resp_mat / np.max(resp_mat, axis=0)
            MI[i] = MI_from_responses(resp_mat, sigma_sq=sigma_sq)
        else:
            # Compute signaling complex levels to avoid redundant calculations
            C = psb.sim_S_LAB(model_size, L0, R0, K_all[i], None,
                              model=model, fixed=fixed, norm=False)
            C = C[:, :, -nL*nA*nB:]

            # Find resulting responses and MIs with each set of efficiencies
            for j in range(len(e_all)):
                resp_mat = np.dot(C, e_all[j]).T
                resp_mat = resp_mat / np.max(resp_mat, axis=0)
                MI[i, j] = MI_from_responses(resp_mat, sigma_sq=sigma_sq)

    return MI

def sim_resp_mat(model_size, L0, R0, K, e, model='onestep', fixed=None):
    '''
    Compute response matrix for specified set of parameters.
    Note that responses are normalized by maximum for each cell type.

    Parameters
    ----------
    model_size: tuple of ints, shape (3, )
        Specification of model parameters, given as (nL, nA, nB) or (number of
        ligands, number of type I receptors, number of type II receptors).
    L0: array_like, shape (nL, ) or (n_envs, nL)
        Starting ligand concentrations for each environment, with each
        combination following order L(1), ..., L(nL).
    R0: array_like, shape (nA + nB, ) or (n_cells, nA + nB)
        Starting receptor levels for each cell type, with each set following
        order A(1), ..., A(nA), B(1), ..., B(nB).
    K: array_like, last dimension (nL*nA*nB, ) or (nL*nA + nL*nA*nB, )
        Affinity parameters. These parameters should correspond to complexes
        (optional dimeric intermediates followed by trimeric complexes)
        with last index increasing first. For a model with nL=2, nA=2, and nB=2,
        complex order is D_11, D_12, D_21, D_22, T_111, T_112, T_121, T_122,
        T_211, T_212, T_221, T_222 (D_ij included only for two-step model).
        If this array is two-dimensional, each row should correspond to one
        such set of parameters.
    e: array_like, last dimension (nL * nA * nB, )
        Efficiency parameters. These parameters should correspond to complexes
        with last index increasing first. For a model with nL=2, nA=2, and nB=2,
        complex order is T_111, T_112, T_121, T_122, T_211, T_212, T_221, T_222.
        If this array is two-dimensional, each row should correspond to one
        such set of parameters.
    model: string
        Indicator of desired model. Currently, 'onestep' and 'twostep' (but not
        'onestep_hexameric') are supported.
    fixed: None or array_like, shape (nL + nA + nB, )
        Indicator of whether each component has fixed concentration,
        specified in the order [L(1), L(2), ..., L(nL),
        A(1), ..., A(nA), B(1), ..., B(nB)]. Default is None. In this
        case, ligand concentrations remain constant, while receptors are
        depleted.
    '''
    resp_mat = psb.sim_S_LAB(model_size, L0, R0, K, e,
                             model=model, fixed=fixed).T
    resp_mat = resp_mat / np.max(resp_mat, axis=0)
    return resp_mat

def compute_pairwise_addressability(a, b):
    '''
    Return maximum pairwise addressability between activation profiles
    [a] and [b].
    '''
    return np.exp(np.max(np.abs(np.log(a / b))))

def find_n_words(n, addressability_mat):
    '''
    Return maximum addressability for [n] words (defined as minimum
    pairwise addressability of different inputs), as well as indices of
    corresponding ligand combinations, given matrix [addressability_mat]
    where element [i, j] corresponds to pairwise addressability of
    inputs i and j.
    '''
    # Ignore diagonal entries corresponding to identical inputs
    addr_mat = np.array(addressability_mat).copy()
    np.fill_diagonal(addr_mat, np.inf)

    # Scan all subsets of n distinct ligand combinations
    subsets = list(itertools.combinations(np.arange(len(addr_mat)), n))

    # Compute addressability of all subsets
    addr_subsets = np.array([np.min(addr_mat[np.ix_(subset, subset)])
                             for subset in subsets])

    # Find best result
    ind = np.argmax(addr_subsets)
    return addr_subsets[ind], subsets[ind]

def addressability(n_in_all, model_size, lig_combos, cell_types,
                   K, e, model='onestep', fixed=None):
    '''
    Compute addressability for each number of inputs given in [n_in_all].
    '''
    # Compute response matrix and pairwise addressability matrix
    resp_mat = sim_resp_mat(model_size, lig_combos, cell_types, K, e,
                            model=model, fixed=fixed)
    addr_mat = squareform(pdist(resp_mat,
                                metric=compute_pairwise_addressability))

    # Compute addressability for each number of inputs
    return np.array([find_n_words(n_in, addr_mat)[0]
                     for n_in in n_in_all])
