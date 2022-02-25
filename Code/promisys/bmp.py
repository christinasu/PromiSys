import numpy as np
import itertools
import re

import eqtk

def titrate_ligand(nL=2, L_min=-3, L_max=3, n_conc=15):
    '''
    Generate a titration of ligand(s).

    Parameters
    ----------
    nL: int
        Number of ligands. Default is 2.
    L_min: array_like, shape (nL, ) or (1, )
        Lower limit of the ligand range for each ligand, given in log10 units.
        Scalar values set the same limit for all ligands. Default is -3,
        corresponding to a lower limit of 10^-3.
    L_max: array_like, shape (nL, ) or (1, )
        Upper limit of the ligand range for each ligand, given in log10 units.
        Scalar values set the same limit for all ligands. Default is 3,
        corresponding to an upper limit of 10^3.
    n_conc: array_like, shape (nL, ) or (1, )
        Number of ligand levels for each ligand, sampled logarithmically.
        Scalar values set the same limit for all ligands. Default is 15.

    Returns
    -------
    L: array_like, shape (numpy.product(n_conc), nL) or (n_conc ** nL, nL)
        Ligand values for each titration.

    Raises
    ------
    ValueError
        Incorrect size of L_min, L_max, or n_conc.
    '''
    if np.size(L_min) == 1:
        L_min = np.repeat(L_min, nL)
    if np.size(L_max) == 1:
        L_max = np.repeat(L_max, nL)
    if np.size(n_conc) == 1:
        n_conc = np.repeat(n_conc, nL)

    if (np.size(L_min) != nL or np.size(L_max) != nL
        or np.size(n_conc) != nL):
        raise ValueError('Incorrect size of L_min, L_max, or n_conc.')

    L = np.array(list(itertools.product(
        *[np.logspace(L_min[i], L_max[i], n_conc[i]) for i in range(nL)])))

    return L

def pick_parameters_rand(model_size=(2, 2, 2), n_draws=1, R_min=-3, R_max=3,
                         pick_e=False):
    '''
    Pick random parameter set(s) in a one-step model.

    Parameters
    ----------
    model_size: tuple of ints, shape (3, )
        Specification of model parameters, given as (nL, nA, nB).
        Default is (2, 2, 2).
    n_draws: int
        Number of parameter sets to generate. Default is 1.
    R_min: array_like, shape (nA + nB, ) or (1, )
        Lower limit of the range for each receptor, given in log10 units.
        Scalar values set the same limit for all receptors. Default is -3,
        corresponding to a lower limit of 10^-3.
    R_max: array_like, shape (nA + nB, ) or (1, )
        Upper limit of the range for each receptor, given in log10 units.
        Scalar values set the same limit for all receptors. Default is 3,
        corresponding to an upper limit of 10^3.
    pick_e: bool
        Argument specifying whether or not to return random values for
        efficiency parameters.

    Returns
    -------
    R0: array_like, shape (n_draws, nA + nB)
        Starting receptor levels, sampled in a log-uniform distribution.
    K: array_like, shape (n_draws, nL * nA * nB)
        Affinity parameters, sampled in a uniform distribution over [0, 1).
    e: array_like, shape (n_draws, nL * nA * nB)
        (Optional, if pick_e) Efficiency parameters, sampled in a uniform
        distribution over [0, 1).
    '''
    nL, nA, nB = model_size
    R0 = 10 ** (np.random.uniform(R_min, R_max, (n_draws, nA + nB)))
    K = np.random.rand(n_draws, nL * nA * nB)
    e = np.random.rand(n_draws, nL * nA * nB)
    if pick_e:
        return R0, K, e
    return R0, K

def pick_parameters_tier(model_size=(2, 2, 2), n_tiers=3, R_fold=10, K_fold=10):
    '''
    Pick dimensionless parameter set(s) in a tiered sampling structure
    in a one-step model.

    Parameters
    ----------
    model_size: tuple of ints, shape (3, )
        Specification of model parameters, given as (nL, nA, nB).
        Default is (2, 2, 2).
    n_tiers: int
        Number of tiers to sample for initial receptor levels and affinity
        parameters. Default is 3.
    R_fold: int
        Fold difference between successive tiers of receptor levels.
        Default is 10.
    k_fold: int
        Fold difference between successive tiers of affinity parameters.
        Default is 10.

    Returns
    -------
    R0: array_like, shape (n, nA + nB)
        Starting receptor levels, sampled in logarithmically spaced tiers.
    K: array_like, shape (n, nL * nA * nB)
        Affinity parameters, sampled in logarithmically spaced tiers.
    '''
    nL, nA, nB = model_size

    # Define possible values for receptors
    x_R = [R_fold ** i for i in range(n_tiers)]

    # Find combinations that include all possible values
    A = itertools.combinations_with_replacement(x_R, nA)
    B = itertools.combinations_with_replacement(x_R, nB)
    R0_all = [i + j for i, j in itertools.product(A, B)]

    # Find subset of combinations that do not include lowest possible value
    A = itertools.combinations_with_replacement(x_R[1:], nA)
    B = itertools.combinations_with_replacement(x_R[1:], nB)
    R0_sub = [i + j for i, j in itertools.product(A, B)]

    # Compute and scale unique combinations of initial receptor levels
    s = set(R0_sub)
    R0 = np.array([r for r in R0_all if r not in s])
    R0 = R0 / np.sum(R0, axis=1, keepdims=True)

    # Define possible values for affinities
    x_K = [K_fold ** i for i in range(n_tiers)]

    # Compute and scale unique combinations of affinity parameters by ligand
    K_all = [tuple(i) for i in itertools.product(x_K, repeat=nA*nB)]
    K_sub = [tuple(i) for i in itertools.product(x_K[1:], repeat=nA*nB)]
    s = set(K_sub)
    K_L = [k for k in K_all if k not in s]
    K_L = [tuple(K_L[i] / np.sum(K_L[i])) for i in range(len(K_L))]

    # Combine affinity parameters for all ligands
    K = np.array([sum([K_L[j] for j in i], ()) for i in
        itertools.combinations_with_replacement(np.arange(len(K_L)), nL)])

    # Find possible combinations of receptor levels and affinities
    R0, K = zip(*itertools.product(R0, K))
    R0 = np.array(R0)
    K = np.array(K)
    return R0, K

def make_reactions_onestep(model_size=(2, 2, 2)):
    '''
    Define reactions for a one-step trimeric model.

    Parameters
    ----------
    model_size: tuple of ints, shape (3, )
        Specification of model parameters, given as (nL, nA, nB) or
        (number of ligands, number of type I receptors, number of type II
        receptors). Default is (2, 2, 2).

    Outputs
    -------
    reactions: string
        Set of reactions for specified numbers of ligands and receptors, one
        per line.
    '''
    nL, nA, nB = model_size
    reactions = ''

    # Enumerate all possible trimeric signaling complexes
    for i in range(nL):
        for j in range(nA):
            for k in range(nB):
                r = f'L_{i+1} + A_{j+1} + B_{k+1} <=> T_{i+1}_{j+1}_{k+1}\n'
                reactions += r

    return reactions

def make_reactions(model='onestep', model_size=(2, 2, 2)):
    '''
    Define reactions for a specified model.

    Parameters
    ----------
    model: string
        Indicator of desired model. Currently, 'onestep' is supported.
    model_size: tuple of ints
        Specification of model parameters. Default is (2, 2, 2).

    Outputs
    -------
    reactions: string
        Set of reactions for specified numbers of ligands and receptors, one
        per line.
    '''
    if model == 'onestep':
        return make_reactions_onestep(model_size)

def list_names(species, n_variants):
    '''
    Enumerate column names corresponding to possible forms of a given species.

    Parameters
    ----------
    species: string
        Name of species of interest.
    n_variants: tuple of ints
        Number of variants for each element of the species. For example,
        enumerating all possiblities of a trimeric complex would require
        providing three values corresponding to the three components.

    Outputs
    -------
    names: list, length product(n_variants)
        List where each element represents a possible species.
    '''
    # Generate all possible indices, based on number of variants of each element
    inds = itertools.product(*[np.arange(n) + 1 for n in n_variants])

    # Define names for all species, given the set of indices
    names = [f'{species}_' + '_'.join(str(i) for i in ind) for ind in inds]
    return names

def make_names_onestep(model_size=(2, 2, 2)):
    '''
    Enumerate names corresponding to possible species for one-step
    trimeric model.

    Parameters
    ----------
    model_size: tuple of ints, shape (3, )
        Specification of model parameters, given as (nL, nA, nB) or
        (number of ligands, number of type I receptors, number of type II
        receptors). Default is (2, 2, 2).

    Outputs
    -------
    names: list, length (nL + nA + nB + nL*nA*nB)
        List where each element represents a species (ligand, type I
        receptor, type II receptor, trimeric complex) in the one-step model.
    '''
    nL, nA, nB = model_size
    names = list_names('L', (nL, )) + list_names('A', (nA, )) + \
        list_names('B', (nB, )) + list_names('T', (nL, nA, nB))
    return names

def make_names(model='onestep', model_size=(2, 2, 2)):
    '''
    Enumerate species names for a specified model.

    Parameters
    ----------
    model: string
        Indicator of desired model. Currently, 'onestep' is supported.
    model_size: tuple of ints
        Specification of model parameters. Default is (2, 2, 2).

    Outputs
    -------
    names: list, length dependent on model choice
        List where each element represents a species in the specified model.
    '''
    if model == 'onestep':
        return make_names_onestep(model_size)

def make_N(model='onestep', model_size=(2, 2, 2)):
    '''
    Define stoichiometric matrix for a specified model.

    Parameters
    ----------
    model: string
        Indicator of desired model. Currently, 'onestep' is supported.
    model_size: tuple of ints
        Specification of model parameters. Default is (2, 2, 2).

    Outputs
    -------
    N: numpy array, shape dependent on model choice
        Stoichiometric matrix where each row represents the stoichiometry
        of a given reaction and each column denotes a species.
    '''
    # Create stoichiometric matrix using list of reactions
    reactions = make_reactions(model, model_size)
    N = eqtk.parse_rxns(reactions)

    # Generate corresponding column names
    names = make_names(model, model_size)
    N = N[names]

    return N.to_numpy(copy=True, dtype=float)

def _Kijk_to_Kjk(model_size, L0, K_ijk):
    '''
    Convert affinity parameters K (for a one-step trimeric model) to
    ligand-independent values. This utility function is used in
    simplified model where ligand levels are fixed.
    '''
    # Reshape quantities for indexing
    K_ijk = K_ijk.reshape(model_size)

    # Calculate modified affinity constants
    M_ijk = 1 / (K_ijk * L0[:, np.newaxis, np.newaxis])
    K_jk = np.sum(1 / M_ijk, axis=0)

    return K_jk.flatten()

def _Tjk_to_Tijk(model_size, L0, K_ijk, T_jk):
    '''
    Convert ligand-independent complex levels to trimeric complex levels.
    This utility function is used in simplified model where ligand levels
    are fixed.
    '''
    T_ijk = np.zeros(model_size)

    # Reshape quantities for indexing
    T_jk = T_jk.reshape(model_size[1:])
    K_ijk = K_ijk.reshape(model_size)

    # Calculate ligand-dependent term
    M_ijk = 1 / (K_ijk * L0[:, np.newaxis, np.newaxis])
    W_jk = np.sum(M_ijk[0:1, :, :] / M_ijk, axis=0)

    # Calculate T_ijk
    T_ijk = T_jk / W_jk * M_ijk[0, :, :] / M_ijk[:, :, :]

    return T_ijk.flatten()

def sim_LAB_onestep_reduction(model_size, L0, R0, K, e=None, S_only=True):
    '''
    Simulate a one-step model at steady state for specified ligand and
    receptor levels, using simplification when ligand levels are fixed.

    Parameters
    ----------
    model_size: tuple of ints, shape (3, )
        Specification of model parameters, given as (nL, nA, nB) or (number of
        ligands, number of type I receptors, number of type II receptors).
    L0: array_like, shape (nL, ) or (n_titrations_L, nL)
        Starting ligand concentrations for each titration.
    R0: array_like, shape (nA + nB, ) or (n_titrations_R, nA + nB)
        Starting receptor levels for each titration.
    K: array_like, shape (nL * nA * nB, )
        Affinity parameters. These parameters should correspond to complexes
        with last index increasing first. For a model with nL=2, nA=2, and nB=2,
        complex order is T_111, T_112, T_121, T_122, T_211, T_212, T_221, T_222.
    e: None or array_like, shape (nL * nA * nB, )
        Efficiency parameters. These parameters should correspond to complexes
        with last index increasing first. For a model with nL=2, nA=2, and nB=2,
        complex order is T_111, T_112, T_121, T_122, T_211, T_212, T_221, T_222.
        Default is None, where only steady-state levels of each component will
        be returned. If specified, steady-state signal can also be returned.
    S_only: bool
        Indicator of whether only signal S should be returned (omitting
        steady-state levels of each species). This argument is considered
        only if e is not None.

    Returns
    -------
    c: array_like, shape (n_titrations, nL + nA + nB + nL*nA*nB)
        Steady-state levels of each species at each set of initial conditions.
        This value is omitted if e is provided and S_only is set to True.
        Columns are ordered as elemental particles followed by
        complexes, which are ordered with last index increasing
        first. Specifically, a model with nL=2, nA=2, and nB=2 and fixed
        ligand concentrations would have columns L1, L2, A1, A2, B1, and B2,
        followed by complexes (indexed as LAB) in the following order:
        111, 112, 121, 122, 211, 212, 221, 222.
    S: array_like, shape (n_titrations, )
        Steady-state signal at each set of initial conditions.
        This value is returned if e is provided.

    Raises
    ------
    ValueError
        Incorrect size of L0, R0, K, e, and fixed.
    '''
    nL, nA, nB = model_size

    # Validate L0
    L0 = np.atleast_2d(np.array(L0))
    if L0.shape[1] != nL:
        raise ValueError('Shape of L0 is not consistent with model size.')

    # Validate R0
    R0 = np.atleast_2d(np.array(R0))
    if R0.shape[1] != nA + nB:
        raise ValueError('Shape of R0 is not consistent with model size.')

    # Validate K
    K = np.array(K).squeeze()
    if K.shape != (nL * nA * nB, ):
        raise ValueError('Length of K is not consistent with model size.')

    # Validate e
    if e is not None:
        e = np.array(e).squeeze()
        if e.shape != (nL * nA * nB, ):
            raise ValueError('Length of e is not consistent with model size.')

    # Reshape L0 if needed
    if L0.shape[0] < R0.shape[0]:
        L0 = np.tile(L0, (R0.shape[0] / L0.shape[0], 1))

    # Compute ligand-independent affinity parameters
    K_jk = np.array([_Kijk_to_Kjk(model_size, L, K) for L in L0])

    # Define set of initial conditions
    c0_jk = np.zeros((max(L0.shape[0], R0.shape[0]), nA + nB + nA*nB))
    c0_jk[:, :nA+nB] = R0

    # Define stoichiometric matrix
    N_jk = make_N(model='onestep', model_size=(1, nA, nB))[:, 1:]

    # Solve for steady-state levels of all species
    c_jk = np.array([eqtk.solve(c0_jk[i], N=N_jk, K=K_jk[i])
                     for i in range(len(K_jk))])

    # Convert steady-state levels from model reduction back to full model
    c = np.zeros((c_jk.shape[0], np.sum(model_size) + np.prod(model_size)))
    c[:, :nL] = L0
    c[:, nL:nL+nA+nB] = c_jk[:, :nA+nB]
    c[:, nL+nA+nB:] = np.array(
        [_Tjk_to_Tijk(model_size, L0[i], K, c_jk[i, nA+nB:])
         for i in range(c_jk.shape[0])])

    if e is None:
        return c

    # Compute steady-state signal
    T = np.atleast_2d(c[:, nL+nA+nB:])
    S = np.dot(T, e)

    if S_only:
        return S
    return c, S

def sim_LAB_onestep(model_size, L0, R0, K, e=None, fixed=None, S_only=True):
    '''
    Simulate a one-step model at steady state for specified ligand and
    receptor levels.

    Parameters
    ----------
    model_size: tuple of ints, shape (3, )
        Specification of model parameters, given as (nL, nA, nB) or (number of
        ligands, number of type I receptors, number of type II receptors).
    L0: array_like, shape (nL, ) or (n_titrations_L, nL)
        Starting ligand concentrations for each titration.
    R0: array_like, shape (nA + nB, ) or (n_titrations_R, nA + nB)
        Starting receptor levels for each titration.
    K: array_like, shape (nL * nA * nB, )
        Affinity parameters. These parameters should correspond to complexes
        with last index increasing first. For a model with nL=2, nA=2, and nB=2,
        complex order is T_111, T_112, T_121, T_122, T_211, T_212, T_221, T_222.
    e: None or array_like, shape (nL * nA * nB, )
        Efficiency parameters. These parameters should correspond to complexes
        with last index increasing first. For a model with nL=2, nA=2, and nB=2,
        complex order is T_111, T_112, T_121, T_122, T_211, T_212, T_221, T_222.
        Default is None, where only steady-state levels of each component will
        be returned. If specified, steady-state signal can also be returned.
    fixed: None or array_like, shape (nL + nA + nB, )
        Indicator of whether each component has fixed concentration,
        specified in the order [L(1), L(2), ..., L(nL),
        A(1), ..., A(nA), B(1), ..., B(nB)]. Default is None. In this
        case, ligand concentrations remain constant, while receptors are
        depleted.
    S_only: bool
        Indicator of whether only signal S should be returned (omitting
        steady-state levels of each species). This argument is considered
        only if e is not None.

    Returns
    -------
    c: array_like, shape (n_titrations, nL + nA + nB + nL*nA*nB)
        Steady-state levels of each species at each set of initial conditions.
        This value is omitted if e is provided and S_only is set to True.
        Columns are ordered as elemental particles followed by
        complexes, which are ordered with last index increasing
        first. Specifically, a model with nL=2, nA=2, and nB=2 and fixed
        ligand concentrations would have columns L1, L2, A1, A2, B1, and B2,
        followed by complexes (indexed as LAB) in the following order:
        111, 112, 121, 122, 211, 212, 221, 222.
    S: array_like, shape (n_titrations, )
        Steady-state signal at each set of initial conditions.
        This value is returned if e is provided.

    Raises
    ------
    ValueError
        Incorrect size of L0, R0, K, e, and fixed.
    '''
    nL, nA, nB = model_size

    # Validate L0
    L0 = np.atleast_2d(np.array(L0))
    if L0.shape[1] != nL:
        raise ValueError('Shape of L0 is not consistent with model size.')

    # Validate R0
    R0 = np.atleast_2d(np.array(R0))
    if R0.shape[1] != nA + nB:
        raise ValueError('Shape of R0 is not consistent with model size.')

    # Validate K
    K = np.array(K).squeeze()
    if K.shape != (nL * nA * nB, ):
        raise ValueError('Length of K is not consistent with model size.')

    # Validate e
    if e is not None:
        e = np.array(e).squeeze()
        if e.shape != (nL * nA * nB, ):
            raise ValueError('Length of e is not consistent with model size.')

    # Validate fixed
    if fixed is None:
        fixed = [True] * nL + [False] * (nA + nB)
    fixed = np.array(fixed, dtype=bool).squeeze()
    if fixed.size != (nL + nA + nB):
        raise ValueError('Length of fixed is not consistent with model size.')
    fixed_cols = np.where(fixed)[0]

    # Use model reduction if applicable
    if np.all(fixed_cols == np.arange(nL)):
        return sim_LAB_onestep_reduction(model_size, L0, R0, K,
                                         e=e, S_only=S_only)

    # Define set of initial conditions
    c0 = np.zeros((max(L0.shape[0], R0.shape[0]),
                   np.sum(model_size) + np.prod(model_size)))
    c0[:, :nL] = L0
    c0[:, nL:nL+nA+nB] = R0

    # Define fixed values
    fixed_c = -np.ones(c0.shape)
    fixed_c[:, fixed_cols] = c0[:, fixed_cols]

    # Define stoichiometric matrix
    N = make_N(model='onestep', model_size=model_size)

    # Solve for steady-state levels of all species
    c = eqtk.fixed_value_solve(c0, fixed_c, N=N, K=K)

    if e is None:
        return c

    # Compute steady-state signal
    T = np.atleast_2d(c[:, nL+nA+nB:])
    S = np.dot(T, e)

    if S_only:
        return S
    return c, S

def sim_S_LAB(model_size, L0, R0, K, e, model='onestep',
              fixed=None, norm=False):
    '''
    Simulate a trimeric model at steady state for specified ligand and
    receptor levels, allowing for multiple sets of ligand and receptor vectors.

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
    K: array_like, shape (nL*nA*nB, ) or (nL*nA + nL*nA*nB, )
        Affinity parameters. These parameters should correspond to complexes
        (optional dimeric intermediates followed by trimeric complexes)
        with last index increasing first. For a model with nL=2, nA=2, and nB=2,
        complex order is D_11, D_12, D_21, D_22, T_111, T_112, T_121, T_122,
        T_211, T_212, T_221, T_222 (D_ij included only for two-step model).
    e: array_like, shape (nL * nA * nB, )
        Efficiency parameters. These parameters should correspond to complexes
        with last index increasing first. For a model with nL=2, nA=2, and nB=2,
        complex order is T_111, T_112, T_121, T_122, T_211, T_212, T_221, T_222.
        Default is None, where only steady-state levels of each component will
        be returned. If specified, steady-state signal can also be returned.
    model: string
        Indicator of desired model. Currently, 'onestep' is supported.
    fixed: None or array_like, shape (nL + nA + nB, )
        Indicator of whether each component has fixed concentration,
        specified in the order [L(1), L(2), ..., L(nL),
        A(1), ..., A(nA), B(1), ..., B(nB)]. Default is None. In this
        case, ligand concentrations remain constant, while receptors are
        depleted.
    norm: bool
        Indicator of whether signal S should be normalized by maximum value.

    Returns
    -------
    S: array_like, shape (n_cells, n_envs)
        Steady-state signal at each set of initial conditions, representing
        response of each cell type in each ligand environment.

    Raises
    ------
    ValueError
        Incorrect size of L0, R0, K, e, and fixed.
    '''
    L0 = np.atleast_2d(np.array(L0))
    R0 = np.atleast_2d(np.array(R0))

    # Choose appropriate function for specified model
    if model == 'onestep':
        sim_func = sim_LAB_onestep

    # Compute steady-state signal for each cell type
    S = np.array([sim_func(model_size, L0, R0[i], K, e=e, fixed=fixed)
                  for i in range(R0.shape[0])])

    # Normalize results as needed
    if norm:
        if S.max() > 0:
            S /= S.max()

    return S
