import numpy as np

import os
import pickle
import contextlib

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

import promisys.bmp as psb
import promisys.combinatorial_addressing as psc

def read_pickle_configs(data_dir, data_file, version='v1'):
    '''
    Read simulation results from directory [data_dir] with
    file name [data_file].

    Specify version under which data were generated (likely 'v1' for data
    generated through 2019 and 'v2' for data generated from 2020 onward).
    '''
    with open(os.path.join(data_dir, data_file), 'rb') as f:
        # Read simulation parameters
        if version == 'v1':
            model_size, L, all_configs = pickle.load(f)
        elif version == 'v2':
            model_size, model, L, all_configs = pickle.load(f)

        # Read simulation results
        all_opts = pickle.load(f)
        while True:
            try:
                all_opts.append(pickle.load(f))
            except EOFError:
                break

    if version == 'v1':
        return model_size, L, all_configs.astype(int), all_opts
    elif version == 'v2':
        return model_size, model, L, all_configs.astype(int), all_opts

def index_configs_by_channels(all_configs):
    '''
    Group configurations by number of channels as [inds],
    where inds[i] gives the indices of the configurations
    in [all_configs] that have i channels.
    '''
    channels = np.sum(all_configs, axis=1)
    max_channels = np.max(channels)
    inds = [[] for i in range(max_channels + 1)]
    for i in range(max_channels + 1):
        inds[i] = np.where(channels == i)[0]
    return inds

def save_all_configs(all_configs, all_opts, version='v1'):
    '''
    Save all identified parameters and errors for each configuration
    in [all_configs] from simulation results given in [all_opts].

    Specify version under which data were generated (likely 'v1' for data
    generated through 2019 and 'v2' for data generated from 2020 onward).
    '''
    params = [[] for i in range(len(all_configs))]
    errs = [[] for i in range(len(all_configs))]
    if version == 'v2':
        chis = [[] for i in range(len(all_configs))]
    for opt in all_opts:
        if version == 'v1':
            config, (err, param) = opt
        elif version == 'v2':
            config, (param, err, chi) = opt
            chis[config].append(chi)
        params[config].append(param)
        errs[config].append(err)

    if version == 'v1':
        return params, errs
    elif version == 'v2':
        return params, errs, chis

def save_opt_configs(all_configs, all_opts, version='v1'):
    '''
    Find parameters yielding lowest error for each configuration
    in [all_configs] from simulation results given in [all_opts].

    Specify version under which data were generated (likely 'v1' for data
    generated through 2019 and 'v2' for data generated from 2020 onward).
    '''
    params = [[] for i in range(len(all_configs))]
    errs = np.inf * np.ones(len(all_configs))
    chis = [[] for i in range(len(all_configs))]
    for opt in all_opts:
        if version == 'v1':
            config, (err, param) = opt
        elif version == 'v2':
            config, (param, err, chi) = opt
        if err < errs[config]:
            params[config] = param
            errs[config] = err
            if version == 'v2':
                chis[config] = chi

    if version == 'v1':
        return params, errs
    elif version == 'v2':
        return params, errs, chis

def compute_crosstalk(model_size, L, all_configs, all_params,
                      crosstalk_func=psc.crosstalk):
    '''
    Compute crosstalk values [chis] for the configurations in
    [all_configs], evaluating the corresponding parameter set(s)
    in [params] at ligand combinations in [L].
    '''
    nL, nA, nB = model_size
    nT = nL * nA * nB
    nR = nA + nB

    chis = [[] for i in range(len(all_configs))]
    for i, params in enumerate(all_params):
        # Skip configurations that have not been optimized
        if not len(params):
            continue

        # Account for cases with a single optimization
        if not isinstance(params, list) or len(params) == 1:
            params = [np.array(params).flatten()]

        # Iterate over all parameters
        for param in params:
            # Unpack parameters
            K = param[:nT]
            e = param[nT:nT+nT]
            R = param[nT+nT:].reshape(-1, nR)

            # Find indices of ligand environments defining channels
            inds = np.where(all_configs[i] == 1)[0]

            # Simulate responses at given ligand combinations
            S = psb.sim_S_LAB(model_size, L[inds, :], R, K, e)

            # Compute associated crosstalk
            if len(S):
                chis[i].append(crosstalk_func(S))

    return chis

def read_pickle_channels(data_dir, data_file, version='v1'):
    '''
    Read simulation results from directory [data_dir] with
    file name [data_file].

    Specify version under which data were generated (likely 'v1' for data
    generated through 2019 and 'v2' for data generated from 2020 onward).
    '''
    with open(os.path.join(data_dir, data_file), 'rb') as f:
        # Read simulation parameters
        if version == 'v1':
            model_size, L = pickle.load(f)
        elif version == 'v2':
            model_size, model, L = pickle.load(f)
        all_channels = np.arange(len(L))

        # Read simulation results
        all_opts = pickle.load(f)
        while True:
            try:
                all_opts.append(pickle.load(f))
            except EOFError:
                break

    if version == 'v1':
        return model_size, L, all_channels.astype(int), all_opts
    elif version == 'v2':
        return model_size, model, L, all_channels.astype(int), all_opts

def save_all_channels(all_channels, all_opts, version='v1'):
    '''
    Save all identified parameters and errors for each bandwidth
    in [all_channels] from simulation results given in [all_opts].

    Specify version under which data were generated (likely 'v1' for data
    generated through 2019 and 'v2' for data generated from 2020 onward).
    '''
    params = [[] for i in range(len(all_channels))]
    errs = [[] for i in range(len(all_channels))]
    configs = [[] for i in range(len(all_channels))]
    if version == 'v2':
        chis = [[] for i in range(len(all_channels))]
    for opt in all_opts:
        if version == 'v1':
            channel, (param, err, config) = opt
        elif version == 'v2':
            channel, (config, param, err, chi) = opt
            chis[channel].append(chi)
        params[channel].append(param)
        errs[channel].append(err)
        configs[channel].append(config)

    if version == 'v1':
        return params, errs, configs
    elif version == 'v2':
        return configs, params, errs, chis

def save_opt_channels(all_channels, all_opts, version='v1'):
    '''
    Find parameters yielding lowest error for each bandwidth
    in [all_channels] from simulation results given in [all_opts].

    Specify version under which data were generated (likely 'v1' for data
    generated through 2019 and 'v2' for data generated from 2020 onward).
    '''
    params = [[] for i in range(len(all_channels))]
    errs = np.inf * np.ones(len(all_channels))
    configs = [[] for i in range(len(all_channels))]
    if version == 'v2':
        chis = [[] for i in range(len(all_channels))]
    for opt in all_opts:
        if version == 'v1':
            channel, (param, err, config) = opt
        elif version == 'v2':
            channel, (config, param, err, chi) = opt
        if err < errs[channel]:
            params[channel] = param
            errs[channel] = err
            configs[channel] = config
            if version == 'v2':
                chis[channel] = chi

    if version == 'v1':
        return params, errs, configs
    elif version == 'v2':
        return configs, params, errs, chis

def add_config_data(all_params, all_errs, all_configs, all_orders, all_chis,
                    model_size, model_ind, data_dir, data_file, version, L):
    '''
    Append additional results from configuration-based optimization.
    '''
    _, _, _, new_configs, new_opts = read_pickle_configs(data_dir, data_file, version=version)
    new_params, new_errs, new_chis = save_all_configs(new_configs, new_opts, version=version)

    for i, config in enumerate(new_configs):
        # Check for optimized configuration
        order = config.copy()
        inds = np.where((all_configs[model_ind] == config).all(axis=1))[0]

        # Check for transpose
        if not len(inds):
            n = int(np.sqrt(len(config))) # Assumes 2-ligand case
            config_T = np.reshape(config, (n, n)).T.flatten()
            inds = np.where((all_configs[model_ind] == config_T).all(axis=1))[0]

        # Recompute crosstalk values if needed
        for c, chi in enumerate(new_chis[i]):
            if np.isclose(chi, 1):
                if np.isclose(np.sum(new_params[i][c]), 0):
                    continue
                S = psb.sim_S_LAB(model_size, L[np.where(order)[0], :],
                                  *unpack_params(new_params[i][c], model_size))
                new_chis[i][c] = psc.crosstalk(S)

        # Append to data
        ind = inds[0]
        all_params[model_ind][ind] += new_params[i]
        all_errs[model_ind][ind] += new_errs[i]
        all_orders[model_ind][ind] += [order for o in range(len(new_params[i]))]
        all_chis[model_ind][ind] += new_chis[i]

    return all_params, all_errs, all_configs, all_orders, all_chis

def add_channel_data(all_params, all_errs, all_configs, all_orders, all_chis,
                     model_size, model_ind, data_dir, data_file, version, L):
    '''
    Append additional results from channel-based optimization.
    '''
    _, _, _, new_channels, new_opts = read_pickle_channels(data_dir, data_file, version=version)
    new_configs, new_params, new_errs, new_chis = save_all_channels(new_channels, new_opts, version=version)

    for b, b_configs in enumerate(new_configs):
        for i, config in enumerate(b_configs):
            # Check for optimized configuration
            order = config.copy()
            inds = np.where((all_configs[model_ind] == config).all(axis=1))[0]

            # Check for transpose
            if not len(inds):
                n = int(np.sqrt(len(config))) # Assumes 2-ligand case
                config_T = np.reshape(config, (n, n)).T.flatten()
                inds = np.where((all_configs[model_ind] == config_T).all(axis=1))[0]

            # Recompute crosstalk value if needed
            if np.isclose(new_chis[b][i], 1):
                if np.isclose(np.sum(new_params[b][i]), 0):
                    continue
                S = psb.sim_S_LAB(model_size, L[np.where(order)[0], :],
                                  *unpack_params(new_params[b][i], model_size))
                new_chis[b][i] = psc.crosstalk(S)

            ind = inds[0]
            all_params[model_ind][ind].append(new_params[b][i])
            all_errs[model_ind][ind].append(new_errs[b][i])
            all_orders[model_ind][ind].append(order)
            all_chis[model_ind][ind].append(new_chis[b][i])

    return all_params, all_errs, all_configs, all_orders, all_chis

def compute_crosstalk_channels(model_size, L, all_channels,
                               all_params, all_configs,
                               crosstalk_func=psc.crosstalk):
    '''
    Compute crosstalk values [chis] for the bandwidths in
    [all_channels], evaluating the corresponding parameter set(s)
    in [params] at ligand combinations in [L] according to the
    configurations given in [all_configs].
    '''
    nL, nA, nB = model_size
    nT = nL * nA * nB
    nR = nA + nB

    chis = [[] for i in range(len(all_channels))]
    mean_chis = [[] for i in range(len(all_channels))]
    for i, params in enumerate(all_params):
        # Skip configurations that have not been optimized
        if not len(params):
            continue

        configs = all_configs[i]

        # Account for cases with a single optimization
        if not isinstance(params, list) or len(params) == 1:
            params = [np.array(params).flatten()]
            configs = [np.array(configs).flatten()]

        # Iterate over all parameters
        for j, param in enumerate(params):
            # Unpack parameters
            K = param[:nT]
            e = param[nT:nT+nT]
            R = param[nT+nT:].reshape(-1, nR)

            # Find indices of ligand environments defining channels
            inds = np.where(configs[j] == 1)[0]

            # Simulate responses at given ligand combinations
            S = psb.sim_S_LAB(model_size, L[inds, :], R, K, e)

            # Compute associated crosstalk
            if len(S):
                chis[i].append(crosstalk_func(S))

    return chis

def unpack_params(param, model_size=(2, 2, 2), model='onestep', T=False):
    '''
    Unpack parameters from optimization results.
    '''
    nL, nA, nB = model_size
    nD = nL * nA
    nT = nL * nA * nB
    nR = nA + nB

    # Unpack parameters assuming one-step model
    if model == 'onestep':
        # Extract parameters for optimization of complex level
        if T:
            K = param[:nT]
            e = np.zeros(nT)
            e[0] = 1
            R = param[nT:].reshape(-1, nR)
        # Extract parameters for optimization of pathway activity
        else:
            K = param[:nT]
            e = param[nT:nT+nT]
            R = param[nT+nT:].reshape(-1, nR)

    # Unpack parameters assuming two-step model
    if model == 'twostep':
        # Extract parameters for optimization of complex level
        if T:
            K = param[:nD+nT]
            e = np.zeros(nT)
            e[0] = 1
            R = param[nD+nT:].reshape(-1, nR)
        # Extract parameters for optimization of pathway activity
        else:
            K = param[:nD+nT]
            e = param[nD+nT:nD+nT+nT]
            R = param[nD+nT+nT:].reshape(-1, nR)

    return R, K, e

def count_params(model_size=(2, 2, 2), model='onestep', n_cells=1):
    '''
    Return number of parameters for specified model.
    '''
    nL, nA, nB = model_size
    nT = nL * nA * nB
    nR = nA + nB
    if model == 'onestep':
        n_param = 2 * nT + n_cells * nR
    if model == 'twostep':
        nD = nL * nA
        n_param = nD + 2 * nT + n_cells * nR
    return n_param

def find_log_bins(start=0, stop=3, num=10):
    '''
    Compute bin edges on logarithmic scale for use with
    pcolormesh or the like, using [num] bins uniformly
    spaced between 10^[start] and 10^[stop].
    '''
    bin_width = (stop - start) / (num - 1)
    bin_edges = np.logspace(start - bin_width / 2,
                            stop + bin_width / 2, num + 1)
    x = bin_edges.repeat(num + 1).reshape((num + 1, num + 1))
    y = x.T
    return x, y

def plot_log_square(x_min, x_max, y_min, y_max, axis=None,
                    **kwargs):
    '''
    Plot square bounded by given coordinates, with optional
    [axis] and [kwargs], on logarithmic scale.
    '''
    if axis is None:
        fig, axis = plt.subplots()
    axis.fill_between([x_min, x_max],
                      [y_max, y_max],
                      [y_min, y_min], **kwargs)
    axis.set_xscale('log')
    axis.set_yscale('log')
    return axis

def map_colors(X, cmap='viridis'):
    '''
    Map values in [X] to colors using colormap [cmap].
    - X: values to be translated to color scale
    - cmap: colormap (default: viridis)
    '''
    # Normalize between 0 and maximum
    norm = mpl.colors.Normalize(vmin=0, vmax=X.max())
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    return mappable.to_rgba(X)

def plot_targets(config, L_min=0, L_max=3, L_res=3,
                 axes=None, sp_size=2,
                 xticks_on=True, xticks=None,
                 xlabels_on=True, xlabels=None,
                 yticks_on=True, yticks=None,
                 ylabels_on=True, ylabels=None,
                 cmap='viridis',
                 compact=False):
    '''
    Plot target configuration for orthogonal specificity, given
    by [config].
    - L_min, L_max: logarithm (base 10) of axis limits
    - L_res: number of ligand levels to consider
    - axes: plotting on given Axes (2D array) if provided
    - sp_size: subplot size if Axes are not provided
    - xticks_on: tick marks on x-axis if True
    - xticks: manual placement of ticks on x-axis if provided
    - xlabels_on: tick labels on x-axis if True
    - xlabels: manual setting of tick labels on x-axis if provided
    - yticks_on: tick marks on y-axis (left subplot) if True
    - yticks: manual placement of ticks on y-axis if provided
    - ylabels_on: tick labels on y-axis (left subplot) if True
    - ylabels: manual setting of tick labels on y-axis if provided
    - cmap: colormap for ligand combinations considered in optimization
    - compact: representation on a single plot (rather than separated by
        cell type) if True
    '''
    # Initialize plot
    n = np.sum(config)
    if axes is None:
        figsize = (sp_size * n, sp_size)
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        axes = axes[0]

    # Set plotting parameters
    kwargs_on = {'facecolor': map_colors(np.ones(1), cmap=cmap),
                 'edgecolor': 'k'}
    kwargs_off = {'facecolor': map_colors(np.zeros(1), cmap=cmap),
                  'edgecolor': 'k'}
    kwargs_na = {'facecolor': 'w', 'hatch': '///', 'cmap': 'Greys'}

    # Find ligand environments of orthogonal channels
    ticks = np.logspace(L_min, L_max, L_res)
    tick_labels = np.linspace(L_min, L_max, L_res)
    edges = np.unique(find_log_bins(L_min, L_max, L_res))
    env_inds = np.where(config == 1)[0]
    env_subs = np.unravel_index(env_inds, (L_res, L_res))

    # Plot each targeted response type
    for i in range(n):
        if hasattr(axes, '__len__'):
            axis = axes[i]
        else:
            axis = axes

        plot_log_square(edges[0], edges[-1], edges[0], edges[-1],
                        axis, **kwargs_na)
        for j in range(len(env_subs[0])):
            x = env_subs[0][j]
            y = env_subs[1][j]
            if i == j or compact:
                plot_log_square(edges[x], edges[x + 1],
                                edges[y], edges[y + 1],
                                axis=axis, **kwargs_on)
            else:
                plot_log_square(edges[x], edges[x + 1],
                                edges[y], edges[y + 1],
                                axis=axis, **kwargs_off)
        # Settings for x-axis
        axis.set_xscale('log')
        axis.set_xlim([edges[0], edges[-1]])
        # Ticks for x-axis
        if xticks_on:
            if xticks:
                axis.set_xticks(xticks)
            else:
                axis.set_xticks(ticks)
        else:
            axis.tick_params(axis='x', bottom=False)
        # Labels for x-axis
        if xlabels_on:
            if xlabels:
                axis.set_xticklabels(xlabels)
            else:
                axis.set_xticklabels([r'$10^{{{:g}}}$'.format(l) for l in tick_labels])
            axis.set_xlabel(r'$L_1$')
        else:
            axis.tick_params(axis='x', labelbottom=False)
        # Settings for y-axis
        axis.set_yscale('log')
        axis.set_ylim([edges[0], edges[-1]])
        # Ticks for y-axis
        if yticks_on:
            if yticks:
                axis.set_yticks(yticks)
            else:
                axis.set_yticks(ticks)
        else:
            axis.tick_params(axis='y', left=False)
        # Labels for y-axis
        if ylabels_on:
            if i == 0:
                if ylabels:
                    axis.set_yticklabels(ylabels)
                else:
                    axis.set_yticklabels([r'$10^{{{:g}}}$'.format(l) for l in tick_labels])
                axis.set_ylabel(r'$L_2$')
            else:
                axis.tick_params(axis='y', labelleft=False)
        else:
            axis.tick_params(axis='y', labelleft=False)
        # Settings for plot
        axis.minorticks_off()
        axis.set_aspect('equal')

        if compact:
            break

    return axes

def plot_receptors(model_size, R, axes=None, sp_size=2,
                   xticks_on=True, xticks=None,
                   xlabels_on=True, xlabels=None,
                   yticks_on=True, yticks=None,
                   ylabels_on=True, ylabels=None):
    '''
    Visualize set of receptor expression profiles, given
    by [R], in [model_size] system.
    - axes: plotting on given Axes (2D array) if provided
    - sp_size: subplot size if Axes are not provided
    - xticks_on: tick marks on x-axis if True
    - xticks: manual placement of ticks on x-axis if provided
    - xlabels_on: tick labels on x-axis if True
    - xlabels: manual setting of tick labels on x-axis if provided
    - yticks_on: tick marks on y-axis (left subplot) if True
    - yticks: manual placement of ticks on y-axis if provided
    - ylabels_on: tick labels on y-axis (left subplot) if True
    - ylabels: manual setting of tick labels on y-axis if provided
    '''
    # Initialize plot
    b = R.shape[0]
    n_row = 1
    n_col = b
    if axes is None:
        fig, axes = plt.subplots(n_row, n_col,
                                 figsize=(sp_size * n_col, sp_size * n_row),
                                 squeeze=False)
        axes = axes[0]
    y_max = 1.05 * np.max(R)

    # Plot each receptor expression profile
    for i in range(b):
        if hasattr(axes, '__len__'):
            axis = axes[i]
        else:
            axis = axes
        axis.bar(np.arange(R.shape[1]), R[i], color='k')
        if i == 0:
            label_y = 0
        if np.max(R[i]) < (0.1 * np.max(R)):
            label_y = 2
        else:
            axis.set_ylim([0, y_max])
            label_y = np.max((label_y - 1, 0))

        # Ticks for x-axis
        if xticks_on:
            if xticks:
                axis.set_xticks(xticks)
            else:
                axis.set_xticks(np.arange(R.shape[1]))
        else:
            axis.tick_params(axis='x', bottom=False)
        # Labels for x-axis
        if xlabels_on:
            if xlabels:
                axis.set_xticklabels(xlabels)
            else:
                xlabels = [r'$A_{{{:d}}}$'.format(r + 1) for r in range(model_size[1])]
                xlabels += ([r'$B_{{{:d}}}$'.format(r + 1) for r in range(model_size[2])])
                axis.set_xticklabels(xlabels)
            axis.set_xlabel('Receptor')
        else:
            axis.tick_params(axis='x', labelbottom=False)
        # Ticks for y-axis
        if yticks_on:
            if yticks:
                axis.set_yticks(yticks)
        else:
            axis.tick_params(axis='y', left=False)
        # Labels for y-axis
        if ylabels_on:
            if i == 0 or label_y:
                if ylabels:
                    axis.set_yticklabels(ylabels)
                if i == 0:
                    axis.set_ylabel('Expression')
            else:
                axis.tick_params(axis='y', labelleft=False)
        else:
            axis.tick_params(axis='y', labelleft=False)

    return axes

def plot_combinations(L, axes=None, sp_size=2,
                      marker_on='o', mec_on='r', mew_on=2,
                      mfc_on='None', ms_on=5, kwargs_on=None,
                      marker_off='o', mec_off='w', mew_off=1,
                      mfc_off='None', ms_off=5, kwargs_off=None):
    '''
    Visualize ligand combinations given by [L].
    - axes: plotting on given Axes (2D array) if provided
    - sp_size: subplot size if Axes are not provided
    - marker_[on/off]: marker for activating or nonactivating combinations
    - mec_[on/off]: marker edge color for activating or nonactivating
        combinations
    - mew_[on/off]: marker edge width for activating or nonactivating
        combinations
    - mfc_[on/off]: marker face color for activating or nonactivating
        combinations
    - ms_[on/off]: marker size for activating or nonactivating
        combinations
    - kwargs_[on/off]: additional keyword arguments for activating or
        nonactivating combinations (overrides previous settings)
    '''
    # Initialize plot
    b = L.shape[0]
    n_row = 1
    n_col = b
    if axes is None:
        fig, axes = plt.subplots(n_row, n_col,
                                 figsize=(sp_size * n_col, sp_size * n_row),
                                 squeeze=False)
        axes = axes[0]

    # Set keyword arguments
    if kwargs_on is None:
        kwargs_on = {}
    if kwargs_off is None:
        kwargs_off = {}
    if 'marker' not in kwargs_on:
        kwargs_on['marker'] = marker_on
    if 'marker' not in kwargs_off:
        kwargs_off['marker'] = marker_off
    if 'mec' not in kwargs_on:
        kwargs_on['mec'] = mec_on
    if 'mec' not in kwargs_off:
        kwargs_off['mec'] = mec_off
    if 'mew' not in kwargs_on:
        kwargs_on['mew'] = mew_on
    if 'mew' not in kwargs_off:
        kwargs_off['mew'] = mew_off
    if 'mfc' not in kwargs_on:
        kwargs_on['mfc'] = mfc_on
    if 'mfc' not in kwargs_off:
        kwargs_off['mfc'] = mfc_off
    if 'ms' not in kwargs_on:
        kwargs_on['ms'] = ms_on
    if 'ms' not in kwargs_off:
        kwargs_off['ms'] = ms_off

    # Plot each ligand combination
    for i in range(b):
        if hasattr(axes, '__len__'):
            axis = axes[i]
        else:
            axis = axes

        for j in range(b):
            # Plot activating ligand combination
            if i == j:
                axis.plot(*L[j], **kwargs_on)
            else:
                axis.plot(*L[j], **kwargs_off)

    return axes

def plot_responses(model_size, matrices=None,
                   R=None, K=None, e=None,
                   L=None, L_min=-1.5, L_max=1.5, L_res=10,
                   fig=None, axes=None, sp_size=2,
                   vmin=0, vmax_shared=True,
                   cb_size=0, cb_label='Signal',
                   cmap='viridis', plot_rgb=False, plot_cmy=False,
                   xticks_on=True, xticks=None,
                   xlabels_on=True, xlabels=None,
                   yticks_on=True, yticks=None,
                   ylabels_on=True, ylabels=None,
                   zticks_on=True, zticks=None,
                   zlabels_on=True, zlabels=None):
    '''
    Plot responses for [model_size] system, either specified
    directly in [matrices] or indirectly through parameters
    [R], [K], and [e].  If using parameters, ligand combinations
    can be given in [L] or by specifying bounds [L_min] and
    [L_max] as well as resolution [L_res].
    - model_size: model size, given as (nL, nA, nB) in a tuple
        (not used if matrix provided)
    - matrices: values to plot, where each row represents a
        flattened square matrix corresponding to ligand combinations
        given by L or by L_min, L_max, and L_res
        (takes precedence over values specified by R, K, and e)
    - R, K, e: parameters for computing responses
        corresponding to ligand combinations
        given by L or by L_min, L_max, and L_res
        (not used if matrices is provided)
    - L: matrix specifying ligand combinations, where ith row gives
        ligand combination for ith column in matrices
        (takes precedence over values specified by L_min, L_max, and L_res)
    - L_min, L_max: logarithm (base 10) of axis limits (must be
        provided to use default ticks and labels), which can be given
        as matrix of size [matrices.shape[0], nL] (only for 2 ligands)
    - L_res: number of ligand levels to plot (must be square root of
        number of columns in matrices if matrices is provided)
    - fig, axes: plotting on given Figure (three-ligand plotting only)
        and Axes (as 2D array for two-ligand plotting, 1D array for
        three-ligand plotting) if provided
    - sp_size: subplot size if Axes are not provided
    - vmin: lower limit for colorbar
    - vmax_shared: parameter indicating whether all plots should share
        same upper limit for colorbar (currently valid only for 2 ligands)
    - cb_size: colorbar size if positive
    - cb_label: colorbar label
    - cmap: colormap
    - plot_rgb: indicator for whether to plot responses of 3 cell types
        as red, green, and blue (currently valid only for 2 ligands)
    - [xyz]ticks_on: tick marks on [xyz]-axis if True
    - [xyz]ticks: manual placement of ticks on [xyz]-axis if provided
    - [xyz]labels_on: tick labels on [xyz]-axis if True
    - [xyz]labels: manual setting of tick labels on [xyz]-axis if provided
    '''
    if model_size is None or model_size[0] == 2:
        return plot_responses_2d(model_size=model_size, matrices=matrices,
                                 R=R, K=K, e=e,
                                 L=L, L_min=L_min, L_max=L_max, L_res=L_res,
                                 axes=axes, sp_size=sp_size,
                                 vmin=vmin, vmax_shared=vmax_shared,
                                 cb_size=cb_size, cb_label=cb_label,
                                 cmap=cmap, plot_rgb=plot_rgb, plot_cmy=plot_cmy,
                                 xticks_on=xticks_on, xticks=xticks,
                                 xlabels_on=xlabels_on, xlabels=xlabels,
                                 yticks_on=yticks_on, yticks=yticks,
                                 ylabels_on=ylabels_on, ylabels=ylabels)

def plot_responses_2d(model_size, matrices=None,
                      R=None, K=None, e=None,
                      L=None, L_min=-1.5, L_max=1.5, L_res=10,
                      axes=None, sp_size=2,
                      vmin=0, vmax_shared=True,
                      cb_size=0, cb_label='Signal',
                      cmap='viridis', plot_rgb=False, plot_cmy=False,
                      xticks_on=True, xticks=None,
                      xlabels_on=True, xlabels=None,
                      yticks_on=True, yticks=None,
                      ylabels_on=True, ylabels=None):
    '''
    Plot responses for two-ligand system.
    See plot_responses() for full documentation.
    '''
    # Compute ligand titration if needed
    if L is None:
        L = psb.titrate_ligand(nL=model_size[0], L_min=L_min,
                               L_max=L_max, n_conc=L_res)

    # Simulate responses if needed
    if matrices is None:
        matrices = psb.sim_S_LAB(model_size, L, R, K, e, norm=True)

    # Initialize plot
    b = matrices.shape[0]
    n_row = 1
    n_col = b
    if axes is None:
        fig, axes = plt.subplots(n_row, n_col,
                                 figsize=(sp_size * n_col + cb_size, sp_size * n_row),
                                 squeeze=False)
        axes = axes[0]

    # Define axis limits
    if not hasattr(L_min, '__len__'):
        L_min = np.tile(L_min, (matrices.shape[0], model_size[0]))
    elif len(L_min.shape) == 1:
        if L_min.shape[0] == matrices.shape[0]:
            L_min = np.tile(L_min[:, np.newaxis], (1, model_size[0]))
        elif L_min.shape[0] == model_size[0]:
            L_min = np.tile(L_min[np.newaxis, :], (matrices.shape[0], 1))
    if not hasattr(L_max, '__len__'):
        L_max = np.tile(L_max, (matrices.shape[0], model_size[0]))
    elif len(L_max.shape) == 1:
        if L_max.shape[0] == matrices.shape[0]:
            L_max = np.tile(L_max[:, np.newaxis], (1, model_size[0]))
        elif L_max.shape[0] == model_size[0]:
            L_max = np.tile(L_max[np.newaxis, :], (matrices.shape[0], 1))
    vmax = matrices.max()

    # Visualize responses
    for i in range(b):
        # Check for single AxesSubplot
        if hasattr(axes, '__len__'):
            axis = axes[i]
        else:
            axis = axes

        # Compute matrix- and ligand-specific bins
        bins_x, _ = find_log_bins(start=L_min[i, 0], stop=L_max[i, 0],
                                  num=L_res)
        _, bins_y = find_log_bins(start=L_min[i, 1], stop=L_max[i, 1],
                                  num=L_res)
        bins = (bins_x, bins_y)

        # Compute ticks and tick labels
        if xticks is None:
            xticks = np.logspace(L_min[i, 0], L_max[i, 0], 3)
        if xlabels is None:
            xtick_labels = np.linspace(L_min[i, 0], L_max[i, 0], 3)
        if yticks is None:
            yticks = np.logspace(L_min[i, 1], L_max[i, 1], 3)
        if ylabels is None:
            ytick_labels = np.linspace(L_min[i, 1], L_max[i, 1], 3)

        # Reshape matrix
        matrix = np.reshape(matrices[i], (L_res, L_res))

        # Set color scale
        if not vmax_shared:
            vmax = matrix.max()

        # Plot matrix
        if plot_rgb:
            rgb = np.zeros((*matrix.shape, 3))
            rgb[:, :, i%3] = matrix / vmax
            color_tuple = rgb.reshape((rgb.shape[0] * rgb.shape[1], rgb.shape[2]))
            im = axis.pcolormesh(*bins, matrix, color=color_tuple)
            im.set_array(None)
        elif plot_cmy:
            rgb = np.zeros((*matrix.shape, 3))
            rgb[:, :, i%3] = matrix / vmax
            rgb[:, :, (i-1)%3] = matrix / vmax
            color_tuple = rgb.reshape((rgb.shape[0] * rgb.shape[1], rgb.shape[2]))
            im = axis.pcolormesh(*bins, matrix, color=color_tuple)
            im.set_array(None)
        else:
            im = axis.pcolormesh(*bins, matrix, vmin=vmin, vmax=vmax,
                                 cmap=cmap)

        # Settings for plot
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_aspect('equal')
        # Ticks for x-axis
        if xticks_on:
            axis.set_xticks(xticks)
        else:
            axis.tick_params(axis='x', bottom=False)
        # Labels for x-axis
        if xlabels_on:
            if xlabels is not None:
                axis.set_xticklabels(xlabels)
            else:
                axis.set_xticklabels([r'$10^{{{:g}}}$'.format(l) for l in xtick_labels])
            axis.set_xlabel(r'$L_1$')
        else:
            axis.tick_params(axis='x', labelbottom=False)
        # Ticks for y-axis
        if yticks_on:
            axis.set_yticks(yticks)
        else:
            axis.tick_params(axis='y', left=False)
        # Labels for y-axis
        if ylabels_on:
            if i == 0:
                if ylabels is not None:
                    axis.set_yticklabels(ylabels)
                else:
                    axis.set_yticklabels([r'$10^{{{:g}}}$'.format(l) for l in ytick_labels])
                axis.set_ylabel(r'$L_2$')
            else:
                axis.tick_params(axis='y', labelleft=False)
        else:
            axis.tick_params(axis='y', labelleft=False)

    # Add colorbar if specified
    if cb_size:
        frac = cb_size / (sp_size * n_col + cb_size)
        n_digits = 2
        factor = 10 ** (-np.floor(np.log10(vmax)) + (n_digits - 1))
        vmax_tick = np.floor(factor * vmax) / factor
        cb = plt.colorbar(im, ax=axes, fraction=frac,
                          shrink=0.9, pad=0.05, ticks=[vmin, vmax])
        cb.set_label(cb_label, rotation=-90)

    return axes
