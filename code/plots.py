from pathlib import Path
import copy
import pickle as pkl
from mmap import mmap
from scipy import stats as st
from scipy.stats._continuous_distns import FitDataError
import torch
from sklearn import svm
from sklearn import linear_model
import pandas as pd
import seaborn as sns
import warnings
import numpy as np
import os
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes
from matplotlib.ticker import MaxNLocator
from joblib import Memory
import math
import lyap
import model_loader_utils as loader
import initialize_and_train as train
import utils

sns.set_palette('colorblind')


memory = Memory(location='./memoization_cache', verbose=2)

# memory.clear()
## Functions for computing means and error bars for the plots. 68% confidence
# intervals and means are currently
# implemented in this code. The commented out code is for using a gamma
# distribution to compute these, but uses a
# custom version of seaborn plotting library to plot.

def orth_proj(v):
    n = len(v)
    vv = v.reshape(-1, 1)
    return torch.eye(n) - (vv@vv.T)/(v@v)


USE_ERRORBARS = True
# USE_ERRORBARS = False
LEGEND = False
# LEGEND = True

folder_root = '.'


def ci_acc(vals):
    median, bounds = median_and_bound(vals, perc_bound=0.75, loc=1.,
                                      shift=-.0001, reflect=True)
    return bounds[1], bounds[0]


# ci_acc = 68
# ci_acc = 95

def est_acc(vals):
    median, bounds = median_and_bound(vals, perc_bound=0.75, loc=1.,
                                      shift=-.0001, reflect=True)
    return median


# est_acc = "mean"

def ci_dim(vals):
    median, bounds = median_and_bound(vals, perc_bound=0.75, loc=.9999)
    return bounds[1], bounds[0]


# ci_dim = 68
# ci_dim = 95


def est_dim(vals):
    median, bounds = median_and_bound(vals, perc_bound=0.75, loc=.9999)
    return median


# est_dim = "mean"

def point_replace(a_string):
    a_string = str(a_string)
    return a_string.replace(".", "p")


def get_color(x, cmap=plt.cm.plasma):
    """Get normalized color assignments based on input data x and colormap
    cmap."""
    mag = torch.max(x) - torch.min(x)
    x_norm = (x.float() - torch.min(x))/mag
    return cmap(x_norm)


def median_and_bound(samples, perc_bound, dist_type='gamma', loc=0., shift=0,
                     reflect=False):
    """Get median and probability mass intervals for a gamma distribution fit
    of samples."""
    samples = np.array(samples)

    def do_reflect(x, center):
        return -1*(x - center) + center

    if dist_type == 'gamma':
        if np.sum(samples[0] == samples) == len(samples):
            median = samples[0]
            interval = [samples[0], samples[0]]
            return median, interval

        if reflect:
            samples_reflected = do_reflect(samples, loc)
            shape_ps, loc_fit, scale = st.gamma.fit(samples_reflected,
                                                    floc=loc + shift)
            median_reflected = st.gamma.median(shape_ps, loc=loc, scale=scale)
            interval_reflected = np.array(
                st.gamma.interval(perc_bound, shape_ps, loc=loc, scale=scale))
            median = do_reflect(median_reflected, loc)
            interval = do_reflect(interval_reflected, loc)
        else:
            shape_ps, loc, scale = st.gamma.fit(samples, floc=loc + shift)
            median = st.gamma.median(shape_ps, loc=loc, scale=scale)
            interval = np.array(
                st.gamma.interval(perc_bound, shape_ps, loc=loc, scale=scale))
    else:
        raise ValueError("Distribution option (dist_type) not recognized.")

    return median, interval


## Set parameters for figure aesthetics
plt.rcParams['font.size'] = 6
plt.rcParams['font.size'] = 6
plt.rcParams['lines.markersize'] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.titlesize'] = 8

# Colormaps
class_style = 'color'
cols11 = np.array([90, 100, 170])/255
cols12 = np.array([37, 50, 120])/255
cols21 = np.array([250, 171, 62])/255
cols22 = np.array([156, 110, 35])/255
cmap_activation_pnts = mcolors.ListedColormap([cols11, cols21])
cmap_activation_pnts_edge = mcolors.ListedColormap([cols12, cols22])

rasterized = False
dpi = 800
ext = 'pdf'

# Default figure size
figsize = (1.5, 1.2)
ax_pos = (0, 0, 1, 1)


def make_fig(figsize=figsize, ax_pos=ax_pos):
    """Create figure."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_pos)
    return fig, ax


def out_fig(fig, figname, subfolder='', show=False, save=True, axis_type=0,
            name_order=0, data=None):
    """ Save figure."""
    folder = Path(folder_root)
    figname = point_replace(figname)
    # os.makedirs('../results/figs/', exist_ok=True)
    os.makedirs(folder, exist_ok=True)
    ax = fig.axes[0]
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_rasterized(rasterized)
    if axis_type == 1:
        ax.tick_params(axis='both', which='both',
                       # both major and minor ticks are affected
                       bottom=False,  # ticks along the bottom edge are off
                       left=False, top=False,
                       # ticks along the top edge are off
                       labelbottom=False,
                       labelleft=False)  # labels along the bottom edge are off
    elif axis_type == 2:
        ax.axis('off')
    if name_order == 0:
        fig_path = folder/subfolder/figname
    else:
        fig_path = folder/subfolder/figname
    if save:
        os.makedirs(folder/subfolder, exist_ok=True)
        fig_file = fig_path.with_suffix('.' + ext)
        print(f"Saving figure to {fig_file}")
        fig.savefig(fig_file, dpi=dpi, transparent=True, bbox_inches='tight')

    if show:
        fig.tight_layout()
        fig.show()

    if data is not None:
        os.makedirs(folder/subfolder/'data/', exist_ok=True)
        with open(folder/subfolder/'data/{}_data'.format(figname),
                  'wb') as fid:
            pkl.dump(data, fid, protocol=4)
    plt.close('all')


def autocorrelation(train_params, figname='autocorrelation'):
    train_params_loc = train_params.copy()
    model, params, run_dir = train.initialize_and_train(**train_params_loc)
    model.to("cpu")

    class_datasets = params['datasets']
    # val_loss = params['history']['losses']['val']
    # val_losses[i0, i1] = val_loss
    # val_acc = params['history']['accuracies']['val']
    # val_accs[i0, i1] = val_acc

    train_samples_per_epoch = len(class_datasets['train'])
    class_datasets['train'].max_samples = 10
    torch.manual_seed(params['model_seed'])
    X = class_datasets['train'][:][0]
    T = 0
    if T > 0:
        X = utils.extend_input(X, T)
        X0 = X[:, 0]
    elif train_params_loc['network'] != 'feedforward':
        X0 = X[:, 0]
    else:
        X0 = X
    # X = utils.extend_input(X, 10)
    loader.load_model_from_epoch_and_dir(model, run_dir, -1)
    model.to("cpu")
    hid = []
    hid += model.get_post_activations(X)[:-1]
    # auto_corr_mean = []
    # auto_corr_var = []
    auto_corr_table = pd.DataFrame(columns=['t_next', 'autocorr'])
    h = hid[0]
    for i0 in range(len(hid)):
        h_next = hid[i0]
        overlap = torch.sum(h*h_next, dim=1)
        norms_h = torch.sqrt(torch.sum(h**2, dim=1))
        norms_h_next = torch.sqrt(torch.sum(h_next**2, dim=1))
        corrs = overlap/(norms_h*norms_h_next)
        avg_corr = torch.mean(corrs)
        d = {'t_next': i0, 'autocorr': corrs}
        auto_corr_table = auto_corr_table.append(pd.DataFrame(d),
                                                 ignore_index=True)
    fig, ax = make_fig(figsize)
    sns.lineplot(ax=ax, x='t_next', y='autocorr', data=auto_corr_table)
    out_fig(fig, figname)


def snapshots_through_time(train_params, figname="snap", subdir="snaps",
                           multiprocess_lock=None):
    """
    Plot PCA snapshots of the representation through time.

    Parameters
    ----------
    train_params : dict
        Dictionary of training parameters that specify the model and dataset
        to use for training.

    """
    subdir = Path(subdir)
    X_dim = train_params['X_dim']
    FEEDFORWARD = train_params['network'] == 'feedforward'

    num_pnts_dim_red = 800
    num_plot = 600

    train_params_loc = copy.deepcopy(train_params)

    model, params, run_dir = train.initialize_and_train(
        **train_params_loc, multiprocess_lock=multiprocess_lock)
    model.to("cpu")

    class_datasets = params['datasets']
    class_datasets['train'].max_samples = num_pnts_dim_red
    torch.manual_seed(train_params_loc['model_seed'])
    X, Y = class_datasets['train'][:]

    if FEEDFORWARD:
        T = 10
        y = Y
        X0 = X
    else:
        T = 30
        # T = 100
        X = utils.extend_input(X, T + 2)
        X0 = X[:, 0]
        y = Y[:, -1]
    loader.load_model_from_epoch_and_dir(model, run_dir, 0, 0)
    model.to("cpu")
    hid_0 = [X0]
    hid_0 += model.get_post_activations(X)[:-1]
    loader.load_model_from_epoch_and_dir(model, run_dir,
                                         train_params_loc['num_epochs'], 0)
    model.to("cpu")
    hid = [X0]
    hid += model.get_post_activations(X)[:-1]

    if FEEDFORWARD:
        r = model.layer_weights[-1].detach().clone().T
    else:
        r = model.Wout.detach().T.clone()
    # r0_n = r[0] / torch.norm(r[0])
    # r1_n = r[1] / torch.norm(r[1])
    #
    # r0_n_v = r0_n.reshape(r0_n.shape[0], 1)
    # r1_n_v = r1_n.reshape(r1_n.shape[0], 1)
    # r0_orth = torch.eye(len(r0_n)) - r0_n_v @ r0_n_v.T
    # r1_orth = torch.eye(len(r1_n)) - r1_n_v @ r1_n_v.T
    # h = hid[10]
    # # h_proj = h @ r_orth
    # u, s, v = torch.svd(h)
    # v0 = v[:, 0]
    # def orth_projector(v):
    #     n = len(v)
    #     return (torch.eye(n) - v.reshape(n, 1)@v.reshape(1, n))/(v@v)
    # v0_orth = (torch.eye(n) - v0.reshape(n,1)@v0.reshape(1,n))/(v0@v0)
    # h_v0_orth = h @ v0_orth
    # r0_e_p = orth_projector(r0_e)
    # r1_e_p = orth_projector(r1_e)
    # h_r0_e_p0 = h[y] @ r0_e_p
    # h_r0_e_p1 = h[y] @ r1_e_p

    coloring = get_color(y, cmap_activation_pnts)[:num_plot]
    edge_coloring = get_color(y, cmap_activation_pnts_edge)[:num_plot]

    ## Now get principal components (pcs) and align them from time point to
    # time point
    pcs = []
    p_track = 0
    norm = np.linalg.norm
    projs = []
    for i1 in range(1, len(hid)):
        # pc = utils.get_pcs_covariance(hid[i1], [0, 1])
        out = utils.get_pcs_covariance(hid[i1], [0, 1], return_extra=True)
        pc = out['pca_projection']
        mu = out['mean']
        proj = out['pca_projectors']
        mu_proj = mu@proj[:, :2]
        if i1 > 0:
            # Check for the best alignment
            pc_flip_x = pc.clone()
            pc_flip_x[:, 0] = -pc_flip_x[:, 0]
            pc_flip_y = pc.clone()
            pc_flip_y[:, 1] = -pc_flip_y[:, 1]
            pc_flip_both = pc.clone()
            pc_flip_both[:, 0] = -pc_flip_both[:, 0]
            pc_flip_both[:, 1] = -pc_flip_both[:, 1]
            difference0 = norm(p_track - pc)
            difference1 = norm(p_track - pc_flip_x)
            difference2 = norm(p_track - pc_flip_y)
            difference3 = norm(p_track - pc_flip_both)
            amin = np.argmin(
                [difference0, difference1, difference2, difference3])
            if amin == 1:
                pc[:, 0] = -pc[:, 0]
                proj[:, 0] = -proj[:, 0]
            elif amin == 2:
                pc[:, 1] = -pc[:, 1]
                proj[:, 1] = -proj[:, 1]
            elif amin == 3:
                pc[:, 0] = -pc[:, 0]
                pc[:, 1] = -pc[:, 1]
                proj[:, 0] = -proj[:, 0]
                proj[:, 1] = -proj[:, 1]
        pc = pc + mu_proj
        p_track = pc.clone()
        pcs.append(pc[:num_plot])
        projs.append(proj)

    def take_snap(i0, scats, fig, dim=2, border=False):
        # ax = fig.axes[0]
        hid_pcs_plot = pcs[i0][:, :dim].numpy()

        xm = np.min(hid_pcs_plot[:, 0])
        xM = np.max(hid_pcs_plot[:, 0])
        ym = np.min(hid_pcs_plot[:, 1])
        yM = np.max(hid_pcs_plot[:, 1])
        xc = (xm + xM)/2
        yc = (ym + yM)/2
        hid_pcs_plot[:, 0] = hid_pcs_plot[:, 0] - xc
        hid_pcs_plot[:, 1] = hid_pcs_plot[:, 1] - yc
        v = projs[i0]
        # u, s, v = torch.svd(h)
        if r.shape[0] == 2:
            r0_p = r[0]@v
            r1_p = r[1]@v
        else:
            r0_p = r.flatten()@v
            r1_p = -r.flatten()@v
        if class_style == 'shape':
            scats[0][0].set_offsets(hid_pcs_plot)
        else:
            if dim == 3:
                scat._offsets3d = juggle_axes(*hid_pcs_plot[:, :dim].T, 'z')
                scat._offsets3d = juggle_axes(*hid_pcs_plot[:, :dim].T, 'z')
            else:
                scats[0].set_offsets(hid_pcs_plot)
                scats[1].set_offsets(r0_p[:2].reshape(1, 2))
                scats[2].set_offsets(r1_p[:2].reshape(1, 2))

        xm = np.min(hid_pcs_plot[:, 0])
        xM = np.max(hid_pcs_plot[:, 0])
        ym = np.min(hid_pcs_plot[:, 1])
        yM = np.max(hid_pcs_plot[:, 1])
        max_extent = max(xM - xm, yM - ym)
        max_extent_arg = xM - xm > yM - ym

        if dim == 2:
            x_factor = .4
            if max_extent_arg:
                ax.set_xlim(
                    [xm - x_factor*max_extent, xM + x_factor*max_extent])
                ax.set_ylim([xm - .1*max_extent, xM + .1*max_extent])
            else:
                ax.set_xlim(
                    [ym - x_factor*max_extent, yM + x_factor*max_extent])
                ax.set_ylim([ym - .1*max_extent, yM + .1*max_extent])
        else:
            if max_extent_arg:
                ax.set_xlim([xm - .1*max_extent, xM + .1*max_extent])
                ax.set_ylim([xm - .1*max_extent, xM + .1*max_extent])
                ax.set_zlim([xm - .1*max_extent, xM + .1*max_extent])
            else:
                ax.set_xlim([ym - .1*max_extent, yM + .1*max_extent])
                ax.set_ylim([ym - .1*max_extent, yM + .1*max_extent])
                ax.set_zlim([ym - .1*max_extent, yM + .1*max_extent])

        # ax.plot([r0_p[0]], [r0_p[1]], 'x', markersize=3, color='black')
        # ax.plot([r1_p[0]], [r1_p[1]], 'x', markersize=3, color='black')
        # ax.set_ylim([-4, 4])

        if dim == 3:
            out_fig(fig, f"{figname}_{i0}",
                    subfolder=subdir, axis_type=0,
                    name_order=1)
        else:
            out_fig(fig, f"{figname}_{i0}",
                    subfolder=subdir, axis_type=0,
                    name_order=1)
        return scats,

    dim = 2
    hid_pcs_plot = pcs[0]
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
    else:
        fig, ax = make_fig()
    ax.grid(False)

    scat1 = ax.scatter(*hid_pcs_plot[:num_plot, :dim].T, c=coloring,
                       edgecolors=edge_coloring, s=10, linewidths=.65)

    ax.plot([0], [0], 'x', markersize=7)
    scat2 = ax.scatter([0], [0], marker='x', s=3, c='black')
    scat3 = ax.scatter([0], [0], marker='x', s=3, color='black')
    scats = [scat1, scat2, scat3]
    # ax.plot([0], [0], 'o', markersize=10)

    if FEEDFORWARD:
        snap_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    else:
        snap_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 21, 26,
                             31])  # snap_idx = list(range(T + 1))
    for i0 in snap_idx:
        take_snap(i0, scats, fig, dim=dim, border=False)

    print


def _cluster_holdout_test_acc_stat_fun(h, y, clust_identity,
                                       classifier_type='logistic_regression',
                                       num_repeats=5, train_ratio=0.8, seed=11):
    np.random.seed(seed)
    num_clusts = np.max(clust_identity) + 1
    num_clusts_train = int(round(num_clusts*train_ratio))
    num_samples = h.shape[0]
    test_accs = np.zeros(num_repeats)
    train_accs = np.zeros(num_repeats)

    for i0 in range(num_repeats):
        permutation = np.random.permutation(np.arange(len(clust_identity)))
        perm_inv = np.argsort(permutation)
        clust_identity_shuffled = clust_identity[permutation]
        train_idx = clust_identity_shuffled <= num_clusts_train
        test_idx = clust_identity_shuffled > num_clusts_train
        hid_train = h[train_idx[perm_inv]]
        y_train = y[train_idx[perm_inv]]
        y_test = y[test_idx[perm_inv]]
        hid_test = h[test_idx[perm_inv]]
        if classifier_type == 'svm':
            classifier = svm.LinearSVC(random_state=3*i0 + 1)
        else:
            classifier = linear_model.LogisticRegression(random_state=3*i0 + 1,
                                                         solver='lbfgs')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.fit(hid_train, y_train)
        train_accs[i0] = classifier.score(hid_train, y_train)
        test_accs[i0] = classifier.score(hid_test, y_test)

    return train_accs, test_accs


def get_stats(stat_fun, train_params_list, seeds=None, hue_key=None,
              style_key=None, *args, **kwargs):

    if seeds is None:
        seeds = [train_params_list[0]['model_seed']]

    table = pd.DataFrame()
    if hue_key is not None:
        table.reindex(columns=table.columns.tolist() + [hue_key])
    if style_key is not None:
        table.reindex(columns=table.columns.tolist() + [style_key])

    for params in train_params_list:
        table_piece = stat_fun(params, hue_key, style_key, seeds, *args,
                               **kwargs)
        table = table.append(table_piece, ignore_index=True)

    if hue_key is not None:
        table[hue_key] = table[hue_key].astype('category')
    if style_key is not None:
        table[style_key] = table[style_key].astype('category')

    return table


def dim_over_training(train_params_list, seeds=None, hue_key=None,
                      style_key=None, figname='', x_axis='epoch', y_lim=None,
                      style_agg=None, hue_agg=None, subdir=None,
                      multiprocess_lock=None, **plot_kwargs):
    if subdir is None:
        subdir = train_params_list[0][
                     'network'] + '/' + 'dim_over_training' + '/'

    @memory.cache
    def compute_dim_over_training(params, hue_key, style_key, seeds):
        num_pnts_dim_red = 500
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    **params, multiprocess_lock=multiprocess_lock)
                model.to("cpu")
                class_datasets = returned_params['datasets']
                class_datasets['train'].max_samples = num_pnts_dim_red
                torch.manual_seed(int(params['model_seed']))
                np.random.seed(int(params['model_seed']))
                X, Y = class_datasets['train'][:]
                T = 0
                if T > 0:
                    X = utils.extend_input(X, T)
                    X0 = X[:, 0]
                elif params['network'] != 'feedforward':
                    X0 = X[:, 0]
                else:
                    X0 = X
                epochs, saves = loader.get_epochs_and_saves(run_dir)
                epochs = [epoch for epoch in epochs if
                          epoch <= params['num_epochs']]
                for i_epoch, epoch in enumerate(epochs):
                    loader.load_model_from_epoch_and_dir(model, run_dir, epoch)
                    model.to("cpu")
                    hid = [X0]
                    hid += model.get_post_activations(X)[:-1]
                    try:
                        dim = utils.get_effdim(hid[-1],
                                               preserve_gradients=False).item()
                    except RuntimeError:
                        print("Dim computation didn't converge.")
                        dim = np.nan
                    num_updates = int(
                        params['num_train_samples_per_epoch']/params[
                            'batch_size'])*epoch
                    d = {
                        'effective_dimension': dim, 'seed': seed,
                        'epoch_index': i_epoch, 'epoch': epoch,
                        'num_updates': num_updates
                        }
                    # if epoch == epochs[-1]:
                        # breakpoint()
                    if hue_key is not None:
                        d.update({hue_key: hue_value})
                    if style_key is not None:
                        d.update({style_key: style_value})
                    # casting d to DataFrame necessary to preserve type
                    table_piece = table_piece.append(pd.DataFrame(d, index=[0]),
                                                     ignore_index=True)
        return table_piece

    table = get_stats(compute_dim_over_training, train_params_list,
                      seeds, hue_key, style_key)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    key_list = [hue_key, style_key]
    key_list = [key for key in key_list if key is not None]
    if key_list == []:
        xlimit = table[x_axis].max()
    else:
        # xlimit = table.groupby(key_list).max()[x_axis].min()
        xlimit = table.groupby(key_list).max()[x_axis].max()
    if style_agg is not None:
        if style_agg == 'diff':
            style_vals = pd.unique(table[style_key])
            ed_1 = table[table[style_key]==style_vals[0]]['effective_dimension']
            ed_1 = ed_1.reset_index(drop=True)
            ed_2 = table[table[style_key]==style_vals[1]]['effective_dimension']
            ed_2 = ed_2.reset_index(drop=True)
            ed_diff = ed_1 - ed_2
            table = table[table[style_key]==style_vals[0]].drop(
                columns=[style_key, 'effective_dimension'])
            table['ed_diff'] = ed_diff
            yvar = 'ed_diff'
        else:
            raise ValueError('This choice of style_agg is not recognized.')
        style_key = None
    elif hue_agg is not None:
        if hue_agg == 'diff':
            hue_vals = pd.unique(table[hue_key])
            ed_1 = table[table[hue_key]==hue_vals[0]]['effective_dimension']
            ed_1 = ed_1.reset_index(drop=True)
            ed_2 = table[table[hue_key]==hue_vals[1]]['effective_dimension']
            ed_2 = ed_2.reset_index(drop=True)
            ed_diff = ed_1 - ed_2
            table = table[table[hue_key]==hue_vals[0]].drop(
                columns=[hue_key, 'effective_dimension'])
            table['ed_diff'] = ed_diff
            yvar = 'ed_diff'
        else:
            raise ValueError('This choice of style_agg is not recognized.')
        hue_key = None
    else:
        yvar = 'effective_dimension'
    fig, ax = make_fig((1.5, 1.2))
    if USE_ERRORBARS:
        g = sns.lineplot(ax=ax, x=x_axis, y=yvar,
                         data=table, estimator=est_dim, ci=ci_dim, hue=hue_key,
                         style=style_key, **plot_kwargs)
        if not LEGEND and g.legend_ is not None:
            g.legend_.remove()
    else:
        g1 = sns.lineplot(ax=ax, x=x_axis, y=yvar,
                          data=table, estimator=None, units='seed', hue=hue_key,
                          style=style_key, alpha=.6, **plot_kwargs, legend=None)
        g2 = sns.lineplot(ax=ax, x=x_axis, y=yvar,
                          data=table, estimator='mean', ci=None, hue=hue_key,
                          style=style_key, **plot_kwargs)
        if not LEGEND and g2.legend_ is not None:
            g2.legend_.remove()

    ax.set_ylim([0, y_lim])
    ax.set_xlim([None, xlimit])
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)
    plt.close('all')


def dim_over_layers(train_params_list,
                    seeds=None, hue_key=None, style_key=None,
                    figname="dim_over_layers", subdir=None, T=0,
                    multiprocess_lock=None, use_error_bars=None, **plot_kwargs):
    """
    Effective dimension measured over layers (or timepoints if looking at an
    RNN) of the network, before and after training.

    Parameters
    ----------
    seeds : List[int]
        List of random number seeds to use for generating instantiations of
        the model and dataset. Variation over
        these seeds is used to plot error bars.
    gs : List[float]
        Values of g_radius to iterate over.
    train_params : dict
        Dictionary of training parameters that specify the model and dataset
        to use for training. Value of g_radius
        is overwritten by values in gs.
    figname : str
        Name of the figure to save.
    T : int
        Final timepoint to plot (if looking at an RNN). If 0, disregard this
        parameter.
    """
    if subdir is None:
        subdir = train_params_list[0]['network'] + '/dim_over_layers/'
    if use_error_bars is None:
        use_error_bars = USE_ERRORBARS
    hue_bool = train_params_list is not None
    if hue_bool:
        train_params_list = [copy.deepcopy(t) for t in train_params_list]
    # breakpoints()
    

    @memory.cache
    def compute_dim_over_layers(params, hue_key, style_key, seeds):
        num_pnts_dim_red = 500
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    **params, multiprocess_lock=multiprocess_lock)
                model.to("cpu")
                class_datasets = returned_params['datasets']

                class_datasets['train'].max_samples = num_pnts_dim_red
                torch.manual_seed(int(params['model_seed']))
                np.random.seed(int(params['model_seed']))
                X, Y = class_datasets['train'][:]
                if T > 0:
                    X = utils.extend_input(X, T)
                if params['network'] != 'feedforward':
                    X0 = X[:, 0]
                else:
                    X0 = X
                # epochs, saves = loader.get_epochs_and_saves(run_dir)
                # for i_epoch, epoch in enumerate([0, -1]):
                loader.load_model_from_epoch_and_dir(model, run_dir,
                                                     params['num_epochs'])
                model.to("cpu")
                hid = [X0]
                hid += model.get_post_activations(X)[:-1]
                dims = []
                for h in hid:
                    try:
                        dims.append(utils.get_effdim(h,
                                                     preserve_gradients=False).item())
                    except RuntimeError:
                        dims.append(np.nan)

                d = {
                    'effective_dimension': dims,
                    'layer': list(range(len(dims))), 'seed': seed
                    }
                # breakpoint()
                if hue_key is not None:
                    d.update({hue_key: hue_value})
                if style_key is not None:
                    d.update({style_key: style_value})
                # casting d to DataFrame necessary to preserve type
                table_piece = table_piece.append(pd.DataFrame(d),
                                                 ignore_index=True)
        return table_piece
    table = get_stats(compute_dim_over_layers, train_params_list,
                      seeds, hue_key, style_key)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    # print(table)
    fig, ax = make_fig((1.5, 1.2))
    # table['g_radius'] = table['g_radius'].astype('float64')
    # norm = plt.Normalize(table['g_radius'].min(), table['g_radius'].max())
    # sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    # sm.set_array([])
    # try:
    if use_error_bars:
        g = sns.lineplot(ax=ax, x='layer', y='effective_dimension',
                         data=table, estimator=est_dim, ci=ci_dim,
                         style=style_key, hue=hue_key, **plot_kwargs)
        # g.figure.colorbar(sm)
        if not LEGEND and g.legend_ is not None:
            g.legend_.remove()
    else:
        g1 = sns.lineplot(ax=ax, x='layer', y='effective_dimension',
                          data=table, estimator=None, units='seed',
                          style=style_key, hue=hue_key, alpha=0.6,
                          **plot_kwargs, legend=None)
        g2 = sns.lineplot(ax=ax, x='layer', y='effective_dimension',
                          data=table, estimator='mean', ci=None,
                          style=style_key, hue=hue_key, **plot_kwargs)
        if not LEGEND and g2.legend_ is not None:
            g2.legend_.remove()
    # except FitDataError:
    #     print("Plotting data invalid.")
    layers = set(table['layer'])
    if len(layers) < 12:
        ax.set_xticks(range(len(layers)))
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(
            integer=True))  # ax.xaxis.set_major_locator(plt.MaxNLocator(10))

    # ax.set_ylim([0, None])
    # ax.set_ylim([0, 15])

    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)

    plt.close('all')


def orth_compression_through_layers(train_params_list,
                                     seeds=None,
                                    hue_key=None, style_key=None,
                                    figname="orth_compression_through_layers",
                                    subdir=None, multiprocess_lock=None,
                                    **plot_kwargs):
    """
    """
    # if train_params_list[0]['loss'] != 'mse_scalar':
    #     raise ValueError("Expected scalar mse loss.")

    if subdir is None:
        subdir = train_params_list[0][
                     'network'] + '/orth_compression_through_layers/'

    train_params_list = [copy.deepcopy(t) for t in train_params_list]
    

    @memory.cache
    def compute_orth_compression_through_layers(params, hue_key, style_key,
                                                seeds):
        num_pnts = 500
        # num_dims = 2
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    **params, multiprocess_lock=multiprocess_lock)
                model.to("cpu")
                num_dims = int(returned_params['X_dim'])

                class_datasets = returned_params['datasets']

                def pca(v):
                    out = utils.get_pcs(v, list(range(num_dims)),
                                        return_extra=True)
                    h_pcs = out['pca_projection']
                    v = out['pca_projectors'][:, :num_dims]
                    return h_pcs, v

                class_datasets['train'].max_samples = num_pnts
                torch.manual_seed(int(params['model_seed']))
                np.random.seed(int(params['model_seed']))
                X, Y = class_datasets['train'][:]
                T = 0
                # T = 20
                if T > 0:
                    X = utils.extend_input(X, T)
                    X0 = X[:, 0]
                elif params['network'] != 'feedforward':
                    X0 = X[:, 0]
                else:
                    X0 = X
                epochs, saves = loader.get_epochs_and_saves(run_dir)
                epochs = [epoch for epoch in epochs if
                          epoch <= params['num_epochs']]
                r0s = []
                r1s = []
                for save in saves[-2][:]:
                    loader.load_model_from_epoch_and_dir(model, run_dir,
                                                         epochs[-1], save)
                    model.to("cpu")
                    if params['network'] == 'feedforward':
                        r = model.layer_weights[-1].detach().clone().T
                    else:
                        r = model.Wout.detach().clone()
                    r0s.append(r[0].double())
                    if params['loss'] != 'mse_scalar':
                        r1s.append(r[1].double())
                r0 = torch.mean(torch.stack(r0s), dim=0)
                if params['loss'] != 'mse_scalar':
                    r1 = torch.mean(torch.stack(r1s), dim=0)
                if params['network'] == 'feedforward':
                    y = Y.flatten()
                else:
                    y = Y[:, -1]
                # for i_epoch, epoch in enumerate([0, -1]):
                loader.load_model_from_epoch_and_dir(model, run_dir, 0)
                model.to("cpu")
                hid0 = [X0]
                hid0 += model.get_post_activations(X)[:-1]
                loader.load_model_from_epoch_and_dir(model, run_dir,
                                                     params['num_epochs'])
                model.to("cpu")
                hid = [X0]
                hid += model.get_post_activations(X)[:-1]
                rs = []
                avg_ratios = []
                for i0, (h, h0) in enumerate(zip(hid, hid0)):
                    h = h.double()
                    h_pcs, v = pca(h)
                    h0 = h0.double()
                    h0_pcs, v0 = pca(h0)
                    if params['loss'] == 'mse_scalar':
                        h_proj = h_pcs@orth_proj(r0@v).T
                        h0_proj = h0_pcs@orth_proj(r0@v0).T
                        h_norms = torch.norm(h_proj, dim=1)
                        h0_norms = torch.norm(h0_proj, dim=1)
                        ratios = h_norms/h0_norms
                        avg_ratio = torch.mean(ratios).item()
                    else:
                        h_proj = h_pcs[y == 0]@orth_proj(
                            r0@v).T  # todo: maybe need to use yh (net
                        # prediction)
                        h0_proj = h0_pcs[y == 0]@orth_proj(r0@v0).T
                        h_norms = torch.norm(h_proj, dim=1)
                        h0_norms = torch.norm(h0_proj, dim=1)
                        ratios = h_norms/h0_norms
                        avg_ratio1 = torch.mean(ratios).item()

                        h_proj = h_pcs[y == 1]@orth_proj(r1@v).T
                        h0_proj = h0_pcs[y == 1]@orth_proj(r1@v).T
                        h_norms = torch.norm(h_proj, dim=1)
                        h0_norms = torch.norm(h0_proj, dim=1)
                        ratios = h_norms/h0_norms
                        avg_ratio2 = torch.mean(ratios).item()

                        avg_ratio = (avg_ratio1 + avg_ratio2)/2

                    avg_ratios.append(avg_ratio)
                # u, s, v = torch.svd(h)
                # proj_mags = [(h @ r_orth.T)]
                # def get_shrink(r, h, h0):

                d = {
                    'projections_magnitude': avg_ratios,
                    'layer': list(range(len(avg_ratios))), 'seed': seed
                    }
                if hue_key is not None:
                    d.update({hue_key: hue_value})
                if style_key is not None:
                    d.update({style_key: style_value})
                # casting d to DataFrame necessary to preserve type
                table_piece = table_piece.append(pd.DataFrame(d),
                                                 ignore_index=True)
        return table_piece

    table = get_stats(compute_orth_compression_through_layers,
                      train_params_list, seeds,
                      hue_key, style_key)
    print(table)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    fig, ax = make_fig((1.5, 1.2))
    try:
        if USE_ERRORBARS:
            g = sns.lineplot(ax=ax, x='layer', y='projections_magnitude',
                             data=table, estimator='mean', ci=68,
                             style=style_key, hue=hue_key)
        else:
            g = sns.lineplot(ax=ax, x='layer', y='projections_magnitude',
                             data=table, estimator=None, units='seed',
                             style=style_key, hue=hue_key)
        if not LEGEND and g.legend_ is not None:
            g.legend_.remove()
    except FitDataError:
        print("Invalid data.")
    layers = set(table['layer'])
    if len(layers) < 12:
        ax.set_xticks(range(len(layers)))
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_ylim([-.05, None])
    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)

    plt.close('all')


def orth_compression_through_training(train_params_list,
                                       seeds=None,
                                      hue_key=None, style_key=None,
                                      figname="orth_compression_through_training",
                                      x_axis='epoch',
                                      subdir=None, multiprocess_lock=None,
                                      **plot_kwargs):
    """
    """
    # if train_params_list[0]['loss'] != 'mse_scalar':
    #     raise ValueError("Expected scalar mse loss.")

    if subdir is None:
        subdir = train_params_list[0][
                     'network'] + '/orth_compression_through_training/'

    train_params_list = [copy.deepcopy(t) for t in train_params_list]
    

    @memory.cache
    def compute_orth_compression_through_training(params, hue_key, style_key,
                                                  seeds):
        num_pnts = 500
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    **params, multiprocess_lock=multiprocess_lock)
                model.to("cpu")
                num_dims = int(returned_params['X_dim'])

                class_datasets = returned_params['datasets']

                def pca(v):
                    out = utils.get_pcs(v, list(range(num_dims)),
                                        return_extra=True)
                    h_pcs = out['pca_projection']
                    v = out['pca_projectors'][:, :num_dims]
                    return h_pcs, v

                class_datasets['train'].max_samples = num_pnts
                torch.manual_seed(int(params['model_seed']))
                np.random.seed(int(params['model_seed']))
                X, Y = class_datasets['train'][:]
                if params['network'] == 'feedforward':
                    y = Y
                else:
                    y = Y[:, -1]
                T = 0
                if T > 0:
                    X = utils.extend_input(X, T)
                    X0 = X[:, 0]
                elif params['network'] != 'feedforward':
                    X0 = X[:, 0]
                else:
                    X0 = X
                # epochs, saves = loader.get_epochs_and_saves(run_dir)
                # for i_epoch, epoch in enumerate([0, -1]):
                loader.load_model_from_epoch_and_dir(model, run_dir, 0)
                model.to("cpu")
                # hid0 = [X0]
                h0 = model.get_post_activations(X)[:-1][-1].double()
                h0_pcs, v0 = pca(h0)
                # avg_ratios = []
                epochs, saves = loader.get_epochs_and_saves(run_dir)
                epochs = [epoch for epoch in epochs if
                          epoch <= params['num_epochs']]
                # saves = saves[params['num_epochs']-1]
                for epoch_idx, epoch in enumerate(epochs):
                    loader.load_model_from_epoch_and_dir(model, run_dir, epoch)
                    model.to("cpu")
                    h = model.get_post_activations(X)[:-1][-1].double()
                    r0s = []
                    r1s = []
                    num_updates = int(
                        params['num_train_samples_per_epoch']/params[
                            'batch_size'])*epoch
                    for save in saves[-2][:]:
                        loader.load_model_from_epoch_and_dir(model, run_dir,
                                                             epoch, save)
                        model.to("cpu")
                        if params['network'] == 'feedforward':
                            r = model.layer_weights[
                                -1].detach().clone().double().T
                        else:
                            r = model.Wout.detach().double().T.clone()
                        r0s.append(r[0].double())
                        if params['loss'] != 'mse_scalar':
                            r1s.append(r[1].double())
                    r0 = torch.mean(torch.stack(r0s), dim=0)
                    if params['loss'] != 'mse_scalar':
                        r1 = torch.mean(torch.stack(r1s), dim=0)

                    h_pcs, v = pca(h)

                    if params['loss'] == 'mse_scalar':
                        h_proj = h_pcs@orth_proj(r0@v).T
                        h0_proj = h0_pcs@orth_proj(r0@v0).T
                        h_norms = torch.norm(h_proj, dim=1)
                        h0_norms = torch.norm(h0_proj, dim=1)
                        ratios = h_norms/h0_norms
                        avg_ratio = torch.mean(ratios).item()
                    else:
                        h_proj = h_pcs[y == 0]@orth_proj(
                            r0@v).T  # todo: maybe need to use yh (net
                        # prediction)
                        h0_proj = h0_pcs[y == 0]@orth_proj(r0@v0).T
                        h_norms = torch.norm(h_proj, dim=1)
                        h0_norms = torch.norm(h0_proj, dim=1)
                        ratios = h_norms/h0_norms
                        avg_ratio1 = torch.mean(ratios).item()
                        h_proj = h_pcs[y == 1]@orth_proj(r1@v).T
                        h0_proj = h0_pcs[y == 1]@orth_proj(r1@v).T
                        h_norms = torch.norm(h_proj, dim=1)
                        h0_norms = torch.norm(h0_proj, dim=1)
                        ratios = h_norms/h0_norms
                        avg_ratio2 = torch.mean(ratios).item()

                        avg_ratio = (avg_ratio1 + avg_ratio2)/2

                    d = {
                        'projections_magnitude': avg_ratio, 'epoch': epoch,
                        'num_updates': num_updates, 'epoch_idx': epoch_idx,
                        'seed': seed
                        }
                    if hue_key is not None:
                        d.update({hue_key: hue_value})
                    if style_key is not None:
                        d.update({style_key: style_value})
                    # casting d to DataFrame necessary to preserve type
                    table_piece = table_piece.append(pd.DataFrame(d, index=[0]),
                                                     ignore_index=True)
        return table_piece

    table = get_stats(compute_orth_compression_through_training,
                      train_params_list, seeds,
                      hue_key, style_key)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    # print(table)
    key_list = [hue_key, style_key]
    key_list = [key for key in key_list if key is not None]
    if key_list == []:
        xlimit = table[x_axis].max()
    else:
        # xlimit = table.groupby(key_list).max()[x_axis].min()
        xlimit = table.groupby(key_list).max()[x_axis].max()
    fig, ax = make_fig((1.5, 1.2))
    if USE_ERRORBARS:
        g = sns.lineplot(ax=ax, x=x_axis, y='projections_magnitude',
                         data=table, estimator='mean', ci=68, style=style_key,
                         hue=hue_key)
        if not LEGEND and g.legend_ is not None:
            g.legend_.remove()
    else:
        g1 = sns.lineplot(ax=ax, x=x_axis, y='projections_magnitude',
                          data=table, estimator=None, units='seed',
                          style=style_key, hue=hue_key, alpha=0.6,
                          legend=None)
        g2 = sns.lineplot(ax=ax, x=x_axis, y='projections_magnitude',
                          data=table, estimator='mean', ci=None,
                          style=style_key, hue=hue_key)
        if not LEGEND and g2.legend_ is not None:
            g2.legend_.remove()
    ax.set_ylim([-0.05, None])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)

    plt.close('all')


def orth_compression_through_training_input_sep(train_params_list,
                                                
                                                seeds=None, hue_key=None,
                                                style_key=None,
                                                figname="orth_compression_through_training_input_sep",
                                                x_axis='epoch',
                                                subdir=None,
                                                multiprocess_lock=None,
                                                **plot_kwargs):
    """
    """
    # if train_params_list[0]['loss'] != 'mse_scalar':
    #     raise ValueError("Expected scalar mse loss.")

    if subdir is None:
        subdir = train_params_list[0][
                     'network'] + \
                 '/orth_compression_through_training_input_sep/'

    train_params_list = [copy.deepcopy(t) for t in train_params_list]
    

    @memory.cache
    def compute_orth_compression_through_training_input_sep(params, hue_key,
                                                            style_key, seeds):
        num_pnts = 500
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    **params, multiprocess_lock=multiprocess_lock)
                model.to("cpu")
                num_dims = int(returned_params['X_dim'])

                class_datasets = returned_params['datasets']

                def pca(v):
                    out = utils.get_pcs(v, list(range(num_dims)),
                                        return_extra=True)
                    h_pcs = out['pca_projection']
                    v = out['pca_projectors'][:, :num_dims]
                    return h_pcs, v

                class_datasets['train'].max_samples = num_pnts
                torch.manual_seed(int(params['model_seed']))
                np.random.seed(int(params['model_seed']))
                X, Y = class_datasets['train'][:]
                if params['network'] == 'feedforward':
                    y = Y
                else:
                    y = Y[:, -1]
                T = 0
                if T > 0:
                    X = utils.extend_input(X, T)
                    X0 = X[:, 0]
                elif params['network'] != 'feedforward':
                    X0 = X[:, 0]
                else:
                    X0 = X
                # epochs, saves = loader.get_epochs_and_saves(run_dir)
                # for i_epoch, epoch in enumerate([0, -1]):
                loader.load_model_from_epoch_and_dir(model, run_dir, 0)
                model.to("cpu")
                # hid0 = [X0]
                h0 = model.get_post_activations(X)[:-1][-1].double()
                h0_pcs, v0 = pca(h0)
                # avg_ratios = []
                epochs, saves = loader.get_epochs_and_saves(run_dir)
                epochs = [epoch for epoch in epochs if
                          epoch <= params['num_epochs']]
                # saves = saves[params['num_epochs']-1]
                for epoch_idx, epoch in enumerate(epochs):
                    loader.load_model_from_epoch_and_dir(model, run_dir, epoch)
                    model.to("cpu")
                    h = model.get_post_activations(X)[:-1][-1].double()
                    # h_pcs, v = pca(h)
                    # class_diff = torch.mean(h_pcs[y == 0], dim=0) -
                    # torch.mean(
                    #     h_pcs[y == 1], dim=0)
                    class_diff = torch.mean(h[y == 0], dim=0) - torch.mean(
                        h[y == 1], dim=0)

                    num_updates = int(
                        params['num_train_samples_per_epoch']/params[
                            'batch_size'])*epoch
                    h_proj = h@orth_proj(class_diff).T
                    h0_proj = h0@orth_proj(class_diff).T
                    h_norms = torch.norm(h_proj, dim=1)
                    h0_norms = torch.norm(h0_proj, dim=1)
                    ratios = h_norms/h0_norms
                    avg_ratio = torch.mean(ratios).item()

                    # if params['loss'] == 'mse_scalar':
                    #     # h_proj = h_pcs@orth_proj(class_diff@v).T
                    #     # h0_proj = h0_pcs@orth_proj(class_diff@v).T
                    #
                    # else:
                    #     h_proj = h_pcs[y == 0]@orth_proj(
                    #         r0@v).T  # todo: maybe need to use yh (net
                    #     # prediction. Doesn't matter if net is perfectly )
                    #     h0_proj = h0_pcs[y == 0]@orth_proj(r0@v0).T
                    #     h_norms = torch.norm(h_proj, dim=1)
                    #     h0_norms = torch.norm(h0_proj, dim=1)
                    #     ratios = h_norms/h0_norms
                    #     avg_ratio1 = torch.mean(ratios).item()
                    #     h_proj = h_pcs[y == 1]@orth_proj(r1@v).T
                    #     h0_proj = h0_pcs[y == 1]@orth_proj(r1@v).T
                    #     h_norms = torch.norm(h_proj, dim=1)
                    #     h0_norms = torch.norm(h0_proj, dim=1)
                    #     ratios = h_norms/h0_norms
                    #     avg_ratio2 = torch.mean(ratios).item()
                    #
                    #     avg_ratio = (avg_ratio1 + avg_ratio2)/2

                    d = {
                        'projections_magnitude': avg_ratio, 'epoch': epoch,
                        'num_updates': num_updates,
                        'epoch_idx': epoch_idx, 'seed': seed
                        }
                    if hue_key is not None:
                        d.update({hue_key: hue_value})
                    if style_key is not None:
                        d.update({style_key: style_value})
                    # casting d to DataFrame necessary to preserve type
                    table_piece = table_piece.append(pd.DataFrame(d, index=[0]),
                                                     ignore_index=True)
        return table_piece

    table = get_stats(compute_orth_compression_through_training_input_sep,
                      train_params_list, seeds,
                      hue_key, style_key)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    key_list = [hue_key, style_key]
    key_list = [key for key in key_list if key is not None]
    if key_list == []:
        xlimit = table[x_axis].max()
    else:
        # xlimit = table.groupby(key_list).max()[x_axis].min()
        xlimit = table.groupby(key_list).max()[x_axis].max()
    # print(table)
    fig, ax = make_fig((1.5, 1.2))
    if USE_ERRORBARS:
        g = sns.lineplot(ax=ax, x=x_axis, y='projections_magnitude',
                         data=table, estimator='mean', ci=68, style=style_key,
                         hue=hue_key)
        if not LEGEND and g.legend_ is not None:
            g.legend_.remove()
    else:
        g1 = sns.lineplot(ax=ax, x=x_axis, y='projections_magnitude',
                          data=table, estimator=None, units='seed',
                          style=style_key, hue=hue_key, alpha=0.6,
                          **plot_kwargs, legend=None)
        g2 = sns.lineplot(ax=ax, x='epoch_idx', y='projections_magnitude',
                          data=table, estimator='mean', ci=None,
                          style=style_key, hue=hue_key, **plot_kwargs)
        if not LEGEND and g2.legend_ is not None:
            g2.legend_.remove()
    ax.set_ylim([-0.05, None])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)

    plt.close('all')


def clust_holdout_over_layers(train_params_list, seeds=None, hue_key=None,
                              style_key=None, figname="dim_over_layers",
                              subdir=None, T=0, multiprocess_lock=None,
                              use_error_bars=None, **plot_kwargs):
    """
    Effective dimension measured over layers (or timepoints if looking at an
    RNN) of the network, before and after
    training.

    Parameters
    ----------
    seeds : List[int]
        List of random number seeds to use for generating instantiations of
        the model and dataset. Variation over
        these seeds is used to plot error bars.
    gs : List[float]
        Values of g_radius to iterate over.
    train_params : dict
        Dictionary of training parameters that specify the model and dataset
        to use for training. Value of g_radius
        is overwritten by values in gs.
    figname : str
        Name of the figure to save.
    T : int
        Final timepoint to plot (if looking at an RNN). If 0, disregard this
        parameter.
    """
    if subdir is None:
        subdir = train_params_list[0]['network'] + '/dim_over_layers/'
    if use_error_bars is None:
        use_error_bars = USE_ERRORBARS
    hue_bool = train_params_list is not None
    
    if hue_bool:
        train_params_list = [copy.deepcopy(t) for t in train_params_list]

    @memory.cache
    def compute_clust_holdout_over_layers(params, hue_key, style_key, seeds):
        num_pnts_dim_red = 500
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    **params, multiprocess_lock=multiprocess_lock)
                model.to("cpu")

                class_datasets = returned_params['datasets']

                class_datasets['train'].max_samples = num_pnts_dim_red
                torch.manual_seed(int(params['model_seed']))
                np.random.seed(int(params['model_seed']))
                X, Y = class_datasets['train'][:]
                T = 0
                if T > 0:
                    X = utils.extend_input(X, T)
                    X0 = X[:, 0]
                elif params['network'] != 'feedforward':
                    X0 = X[:, 0]
                else:
                    X0 = X
                # epochs, saves = loader.get_epochs_and_saves(run_dir)
                # for i_epoch, epoch in enumerate([0, -1]):
                loader.load_model_from_epoch_and_dir(model, run_dir,
                                                     params['num_epochs'])
                model.to("cpu")
                hid = [X0]
                hid += model.get_post_activations(X)[:-1]
                dims = []

                if len(Y.shape) > 1:
                    Y = Y[:, -1]
                cluster_identity = class_datasets['train'].cluster_identity
                for lay, h in enumerate(hid):
                    stat = _cluster_holdout_test_acc_stat_fun(h.numpy(),
                                                              Y.numpy(),
                                                              cluster_identity)
                    d = {
                        'LR_training': np.mean(stat[0]),
                        'LR_testing': np.mean(stat[1]),
                        'layer': lay, 'seed': seed
                        }
                    if hue_key is not None:
                        d.update({hue_key: hue_value})
                    if style_key is not None:
                        d.update({style_key: style_value})
                    # casting d to DataFrame necessary to preserve type
                    table_piece = table_piece.append(
                        pd.DataFrame(d, index=[0]),
                        ignore_index=True)
                    # ds.extend([{
                    #     'seed': seed, 'g_radius': g,
                    #     'training': epoch_label, layer_label: lay,
                    #     'LR_training': stat[0][k], 'LR_testing': stat[1][k]
                    #     } for k in range(len(stat[0]))])

        return table_piece
    table = get_stats(compute_clust_holdout_over_layers, train_params_list,
                      seeds, hue_key, style_key)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    # print(table)
    fig, ax = make_fig((1.5, 1.2))
    # table['g_radius'] = table['g_radius'].astype('float64')
    # norm = plt.Normalize(table['g_radius'].min(), table['g_radius'].max())
    # sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    # sm.set_array([])
    # try:
    layers = set(table['layer'])

    for stage in ['LR_training', 'LR_testing']:
        if stage == 'LR_training':
            clust_acc_table_stage = table.drop(columns=['LR_testing'])
        else:
            clust_acc_table_stage = table.drop(
                columns=['LR_training'])
        fig, ax = make_fig((1.5, 1.2))
        if USE_ERRORBARS:
            g = sns.lineplot(ax=ax, x='layer', y=stage,
                             data=clust_acc_table_stage, estimator=est_acc,
                             ci=ci_acc, hue=hue_key, style=style_key,
                             **plot_kwargs)
            # g = sns.lineplot(ax=ax, x='layer', y=stage,
            #                  data=clust_acc_table_stage, estimator='mean',
            #                  ci=95, hue=hue_key, style=style_key)
        else:  # This code probably doesn't work
            g1 = sns.lineplot(ax=ax, x='layer', y=stage,
                              data=clust_acc_table_stage, estimator=None,
                              units='seed', style='training',
                              style_order=['after', 'before'], hue='g_radius',
                              alpha=0.6, legend=None)
            g2 = sns.lineplot(ax=ax, x='layer', y=stage,
                              data=clust_acc_table_stage, estimator='mean',
                              ci=None, style='training',
                              style_order=['after', 'before'], hue='g_radius')
            if not LEGEND and g2.legend_ is not None:
                g2.legend_.remove()
        if not LEGEND and g.legend_ is not None:
            g.legend_.remove()
        ax.set_ylim([-.01, 1.01])
        ax.set_xticks(range(len(layers)))

        if len(layers) < 12:
            ax.set_xticks(range(len(layers)))
        else:
            ax.xaxis.set_major_locator(plt.MaxNLocator(
                integer=True))  # ax.xaxis.set_major_locator(plt.MaxNLocator(
            # 10))

        # ax.set_ylim([0, None])
        #
        out_fig(fig, figname + '_' + stage, subfolder=subdir, show=False,
                save=True, axis_type=0,
                data=table)
    #
    # plt.close('all')


def acc_over_training(train_params_list, seeds=None, hue_key=None,
                      style_key=None, figname="acc_over_training",
                      x_axis='epoch', subdir=None, multiprocess_lock=None,
                      **plot_kwargs):
    """
    Parameters
    ----------
    seeds : List[int]
        List of random number seeds to use for generating instantiations of the
        model and dataset. Variation over these seeds is used to plot error
        bars.
    gs : List[float]
        Values of g_radius to iterate over.
    train_params : dict
        Dictionary of training parameters that specify the model and dataset to
        use for training. Value of g_radius is overwritten by values in gs.
    figname : str
        Name of the figure to save.
    T : int
        Final timepoint to plot (if looking at an RNN). If 0, disregard this
        parameter.
    """
    if subdir is None:
        subdir = Path('acc_over_training/')

    train_params_list = [copy.deepcopy(t) for t in train_params_list]
    

    @memory.cache
    def compute_acc_over_training(params, hue_key, style_key, seeds):
        num_pnts = 1000
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    multiprocess_lock=multiprocess_lock, **params)
                model.to("cpu")

                class_datasets = returned_params['datasets']

                class_datasets['train'].max_samples = num_pnts
                torch.manual_seed(int(params['model_seed']))
                np.random.seed(int(params['model_seed']))
                X, Y = class_datasets['train'][:]
                if params['network'] == 'feedforward':
                    y = Y
                else:
                    y = Y[:, -1]
                epochs, saves = loader.get_epochs_and_saves(run_dir)
                epochs = [epoch for epoch in epochs if
                          epoch <= params['num_epochs']]
                with torch.no_grad():
                    for i0, epoch in enumerate(epochs):
                        for save in [0]:
                            num_updates = int(
                                params['num_train_samples_per_epoch']/params[
                                    'batch_size'])*epoch
                            # for i_epoch, epoch in enumerate([0, -1]):
                            loader.load_model_from_epoch_and_dir(model, run_dir,
                                                                 epoch)
                            model.to("cpu")
                            out = model(X).detach()
                            if not params['network'] == 'feedforward':
                                out = out[:, -1]
                            if params['loss'] == 'mse_scalar':
                                out_cat = out.flatten() > 0
                                y_class = y > 0
                                acc = torch.mean((out_cat == y_class).type(
                                    torch.float)).item()
                            else:
                                out_cat = torch.argmax(out, dim=1)
                                acc = torch.mean(
                                    (out_cat == y).type(torch.float)).item()
                            # accs.append(acc)

                            d = {
                                'seed': seed, 'accuracy': acc, 'epoch': epoch,
                                'num_updates': num_updates, 'epoch_idx': i0
                                }
                            if hue_key is not None:
                                d.update({hue_key: hue_value})
                            if style_key is not None:
                                d.update({style_key: style_value})
                            table_piece = table_piece.append(
                                pd.DataFrame(d, index=[0]), ignore_index=True)
        return table_piece

    table = get_stats(compute_acc_over_training, train_params_list,
                      seeds, hue_key, style_key)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    fig, ax = make_fig((1.5, 1.2))
    if USE_ERRORBARS:
        g = sns.lineplot(ax=ax, x=x_axis, y='accuracy', data=table,
                         estimator=est_acc, ci=ci_acc, style=style_key,
                         hue=hue_key, **plot_kwargs)
        if not LEGEND and g.legend_ is not None:
            g.legend_.remove()
    else:
        g1 = sns.lineplot(ax=ax, x=x_axis, y='accuracy', data=table,
                          estimator=None, units='seed', style=style_key,
                          hue=hue_key, alpha=0.6, **plot_kwargs, legend=None)
        g2 = sns.lineplot(ax=ax, x=x_axis, y='accuracy', data=table,
                          estimator='mean', ci=None, style=style_key,
                          hue=hue_key, **plot_kwargs)
        if not LEGEND and g2.legend_ is not None:
            g2.legend_.remove()

    key_list = [hue_key, style_key]
    key_list = [key for key in key_list if key is not None]
    if key_list == []:
        xlimit = table[x_axis].max()
    else:
        # xlimit = table.groupby(key_list).max()[x_axis].min()
        xlimit = table.groupby(key_list).max()[x_axis].max()
    ax.set_xlim([None, xlimit])
    ax.set_ylim([-.01, 1.01])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)

    plt.close('all')


def loss_over_training(train_params_list, seeds=None, hue_key=None,
                       style_key=None, figname="loss_over_training",
                       x_axis='epoch', y_scale='linear', subdir=None,
                       multiprocess_lock=None, **plot_kwargs):
    """
    Parameters
    ----------
    seeds : List[int]
        List of random number seeds to use for generating instantiations of the
        model and dataset. Variation over these seeds is used to plot error
        bars.
    gs : List[float]
        Values of g_radius to iterate over.
    train_params : dict
        Dictionary of training parameters that specify the model and dataset to
        use for training. Value of g_radius is overwritten by values in gs.
    figname : str
        Name of the figure to save.
    y_scale : str
        Scale of the y axis ('linear' or 'log')
    T : int
        Final timepoint to plot (if looking at an RNN). If 0, disregard this
        parameter.
    """
    if subdir is None:
        subdir = Path('loss_over_training/')

    train_params_list = [copy.deepcopy(t) for t in train_params_list]
    

    @memory.cache
    def compute_loss_over_training(params, hue_key, style_key, seeds):
        num_pnts = 1000
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    multiprocess_lock=multiprocess_lock, **params)
                model.to("cpu")

                class_datasets = returned_params['datasets']

                class_datasets['train'].max_samples = num_pnts
                torch.manual_seed(int(params['model_seed']))
                np.random.seed(int(params['model_seed']))
                X, Y = class_datasets['train'][:]
                if params['network'] == 'feedforward':
                    y = Y
                else:
                    y = Y[:, -1]
                epochs, saves = loader.get_epochs_and_saves(run_dir)
                epochs = [epoch for epoch in epochs if
                          epoch <= params['num_epochs']]
                loss_function = returned_params['loss_function']
                with torch.no_grad():
                    for i0, epoch in enumerate(epochs):
                        for save in [0]: # Todo: have this work over saves
                            # for i_epoch, epoch in enumerate([0, -1]):
                            loader.load_model_from_epoch_and_dir(model, run_dir,
                                                                 epoch)
                            model.to("cpu")
                            out = model(X).detach()
                            loss = loss_function(out, Y).item()
                            # accs.append(acc)
                            num_updates = int(
                                params['num_train_samples_per_epoch']/params[
                                    'batch_size'])*epoch

                            d = {
                                'seed': seed, 'loss': loss, 'epoch': epoch,
                                'epoch_idx': i0, 'num_updates': num_updates
                                }
                            if hue_key is not None:
                                d.update({hue_key: hue_value})
                            if style_key is not None:
                                d.update({style_key: style_value})
                            table_piece = table_piece.append(
                                pd.DataFrame(d, index=[0]), ignore_index=True)
        return table_piece

    table = get_stats(compute_loss_over_training, train_params_list,
                      seeds, hue_key, style_key)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    key_list = [hue_key, style_key]
    key_list = [key for key in key_list if key is not None]
    if key_list == []:
        xlimit = table[x_axis].max()
    else:
        # xlimit = table.groupby(key_list).max()[x_axis].min()
        xlimit = table.groupby(key_list).max()[x_axis].max()
    if y_scale == 'log':
        table['loss'] = table['loss'].apply(np.log10)
    fig, ax = make_fig((1.5, 1.2))
    if USE_ERRORBARS:
        g = sns.lineplot(ax=ax, x=x_axis, y='loss', data=table,
                         estimator='mean', ci=68, style=style_key,
                         hue=hue_key, **plot_kwargs)
        if not LEGEND and g.legend_ is not None:
            g.legend_.remove()
    else:
        g1 = sns.lineplot(ax=ax, x=x_axis, y='loss', data=table,
                          estimator=None, units='seed', style=style_key,
                          hue=hue_key, alpha=0.6, **plot_kwargs, legend=None)
        g2 = sns.lineplot(ax=ax, x=x_axis, y='loss', data=table,
                          estimator='mean', ci=None, style=style_key,
                          hue=hue_key, **plot_kwargs)
        if not LEGEND and g2.legend_ is not None:
            g2.legend_.remove()

    if y_scale != 'log':
        ax.set_ylim([-.01, None])
    ax.set_xlim([None, xlimit])
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)

    plt.close('all')


def lyaps(seeds, train_params, epochs, figname="lyaps", subdir="lyaps",
         multiprocess_lock=None):
    """
    Lyapunov exponent plots.

    Parameters
    ----------
    seeds : List[int]
        List of random number seeds to use for generating instantiations of
        the model and dataset. Variation over
        these seeds is used to plot error bars.
    train_params : dict
        Dictionary of training parameters that specify the model and dataset
        to use for training. Value of g_radius
        is overwritten by values in gs.
    epochs : List[int]
        Epochs at which to plot the accuracy.
    figname : str
        Name of the figure to save.
    """
    ICs = 'random'

    train_params_loc = copy.deepcopy(train_params)
    lyap_table = pd.DataFrame(
        columns=['seed', 'epoch', 'lyap', 'lyap_num', 'sem', 'chaoticity'])
    k_LE = 10

    for i0, seed in enumerate(seeds):
        train_params_loc['model_seed'] = seed

        model, params, run_dir = train.initialize_and_train(
            **train_params_loc, multiprocess_lock=multiprocess_lock)
        model.to("cpu")

        torch.manual_seed(train_params_loc['model_seed'])

        for epoch in epochs:
            loader.load_model_from_epoch_and_dir(model, run_dir, epoch)
            model.to("cpu")
            Wrec = model.Wrec.detach().T.numpy()
            Win = model.Win.detach().T.numpy()
            brec = model.brec.detach().numpy()

            if isinstance(ICs, str):
                if ICs == 'random':
                    ICs_data = None
                else:
                    ICs_data = None
            else:
                ICs_data = ICs

            LEs, sem, trajs = lyap.getSpectrum(Wrec, brec, Win, x=0, k_LE=k_LE,
                                               max_iters=1000, max_ICs=10,
                                               ICs=ICs_data, tol=2e-3,
                                               verbose=True)
            LEs = np.sort(LEs)[::-1]
            chaoticity = np.sum(LEs[:3]/np.arange(1, len(LEs[:3]) + 1))
            d = [{
                'seed': seed, 'epoch': epoch, 'lyap': LEs[k], 'lyap_num': k,
                'sem': sem, 'chaoticity': chaoticity
                } for k in range(len(LEs))]
            lyap_table = lyap_table.append(d, ignore_index=True)

    lyap_table['seed'] = lyap_table['seed'].astype('category')
    lyap_table['epoch'] = lyap_table['epoch'].astype('category')

    fig, ax = make_fig((2, 1.2))
    lyap_table_plot = lyap_table.drop(columns=['sem', 'chaoticity'])
    g = sns.pointplot(ax=ax, x='lyap_num', y='lyap', data=lyap_table_plot,
                      style='training', hue='epoch', ci=68, scale=0.5)
    ax.set_xticks(sorted(list(set(lyap_table['lyap_num']))))
    ax.axhline(y=0, color='black', linestyle='--')
    out_fig(fig, figname, subdir, show=False, save=True, axis_type=0,
            data=lyap_table)  #


def ashok_compression_metric(train_params_list, seeds=None, hue_key=None,
                             style_key=None,
                             figname="ashok_compression_metric", subdir=None,
                             multiprocess_lock=None, **plot_kwargs):
    """
    Figures tracking linear separability of representation through time and
    through training.
    """

    if subdir is None:
        subdir = train_params_list[0][
                     'network'] + '/ashok_compression_metric/'

    train_params_list = [copy.deepcopy(t) for t in train_params_list]
    

    @memory.cache
    def compute_ashok_compression_metric(params, hue_key, style_key, seeds):
        num_pnts = 1000
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    **params, multiprocess_lock=multiprocess_lock)
                model.to("cpu")

                class_datasets = returned_params['datasets']

                class_datasets['train'].max_samples = num_pnts
                torch.manual_seed(int(params['model_seed']))
                np.random.seed(int(params['model_seed']))
                X, Y = class_datasets['train'][:]
                if params['network'] == 'feedforward':
                    y = Y
                else:
                    y = Y[:, -1]
                T = 0
                if T > 0:
                    X = utils.extend_input(X, T)
                    X0 = X[:, 0]
                elif params['network'] != 'feedforward':
                    X0 = X[:, 0]
                else:
                    X0 = X
                loader.load_model_from_epoch_and_dir(model, run_dir,
                                                     params['num_epochs'])
                model.to("cpu")
                hid = [X0]
                hid += model.get_post_activations(X)[:-1]
                for layer, h in enumerate(hid):
                    hy0 = h[y == 0]
                    hy1 = h[y == 1]
                    d0 = torch.pdist(hy0)
                    d1 = torch.pdist(hy1)
                    d_within = torch.mean(torch.cat((d0, d1), dim=0)).item()
                    # d1_mu = np.mean(d1)
                    # d_within = (d0_mu + d1_mu) / 2
                    d01 = torch.cdist(hy0, hy1)
                    d_across = torch.mean(d01).item()
                    d = {
                        'seed': seed, 'layer': layer, 'd_across': d_across,
                        'd_within': d_within
                        }
                    if hue_key is not None:
                        d.update({hue_key: hue_value})
                    if style_key is not None:
                        d.update({style_key: style_value})
                    table_piece = table_piece.append(pd.DataFrame(d, index=[0]),
                                                     ignore_index=True)
        return table_piece
    table = get_stats(compute_ashok_compression_metric, train_params_list,
                      seeds, hue_key, style_key)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()

    fig, ax = make_fig((1.5, 1.2))
    ax2 = ax.twinx()
    if USE_ERRORBARS:
        g1 = sns.lineplot(ax=ax, x='layer', y='d_across', data=table,
                          estimator='mean', ci=68, style=style_key, hue=hue_key,
                          **plot_kwargs, legend=None)
        # with sns.color_palette("husl", 9):
        g2 = sns.lineplot(ax=ax2, x='layer', y='d_within', data=table,
                          estimator='mean', ci=68, style=style_key, hue=hue_key,
                          palette=sns.color_palette("hls", 8), **plot_kwargs)
    else:
        g1 = sns.lineplot(ax=ax, x='layer', y='d_across', data=table,
                          estimator=None, units='seed', style=style_key,
                          hue=hue_key, **plot_kwargs)
        with sns.color_palette("husl", 9):
            g2 = sns.lineplot(ax=ax2, x='layer', y='d_within', data=table,
                              estimator=None, units='seed', style=style_key,
                              hue=hue_key, **plot_kwargs)
    if g2.legend_ is not None:
        g2.legend_.remove()
    # if not LEGEND and g1.legend_ is not None:
    #     g1.legend_.remove()
    ax.set_ylim([-0.05, None])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)

    plt.close('all')


def weight_var_through_training(train_params_list, seeds=None, hue_key=None,
                                style_key=None,
                                figname="weight_var_through_training",
                                subdir=None, weight_type='readout',
                                multiprocess_lock=None, **plot_kwargs):
    """
    Parameters
    ----------
    seeds : List[int]
        List of random number seeds to use for generating instantiations of the
        model and dataset. Variation over these seeds is used to plot error
        bars.
    gs : List[float]
        Values of g_radius to iterate over.
    train_params : dict
        Dictionary of training parameters that specify the model and dataset to
        use for training. Value of g_radius is overwritten by values in gs.
    figname : str
        Name of the figure to save.
    T : int
        Final timepoint to plot (if looking at an RNN). If 0, disregard this
        parameter.
    """
    if subdir is None:
        subdir = ''
    train_params_list = [copy.deepcopy(t) for t in train_params_list]
    

    @memory.cache
    def compute_weight_var_through_training(params, hue_key, style_key, seeds,
                                            weight_type):
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    multiprocess_lock=multiprocess_lock, **params)
                model.to("cpu")

                epochs, saves = loader.get_epochs_and_saves(run_dir)
                epochs = [epoch for epoch in epochs if
                          epoch <= params['num_epochs']]
                with torch.no_grad():
                    for i0, epoch in enumerate(epochs[:-1]):
                        weights = []
                        for save in saves[i0]:
                            loader.load_model_from_epoch_and_dir(model, run_dir,
                                                                 epoch, save)
                            model.to("cpu")
                            if weight_type == 'readout':
                                if params['network'] == 'feedforward':
                                    r = model.layer_weights[
                                        -1].detach().clone().T
                                else:
                                    r = model.Wout.detach().T.clone()
                                weights.append(r)
                            elif weight_type == 'product':
                                if params['network'] == 'feedforward':
                                    raise NotImplementedError()
                                else:
                                    r = model.Wout.detach().T.clone()
                                    W = model.Wrec.detach().T.clone()
                                weights.append(torch.matrix_power(
                                    W.T, int(returned_params['n_lag']))@r.T
                                               )
                            else:
                                raise NotImplementedError()
                        var = torch.var(torch.stack(weights), dim=0)
                        var_mu = torch.mean(var).item()
                        d = {'seed': seed, 'epoch': epoch, 'var': var_mu}
                        if hue_key is not None:
                            d.update({hue_key: hue_value})
                        if style_key is not None:
                            d.update({style_key: style_value})
                        table_piece = table_piece.append(
                            pd.DataFrame(d, index=[0]), ignore_index=True)
        return table_piece

    table = get_stats(compute_weight_var_through_training,
                      train_params_list, seeds,
                      hue_key, style_key, weight_type)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    fig, ax = make_fig((1.5, 1.2))
    if style_key is not None or hue_key is not None:
        print("Noise integration not implemented with style_key and hue_key")
    var_sums = torch.tensor(
        [table[table['seed'] == seed]['var'].sum() for seed in seeds])
    avg_var = '{0:1.2g}'.format(torch.mean(var_sums).item())
    cs_var = '{0:1.2g}'.format(1.96*torch.std(var_sums).item())
    if USE_ERRORBARS:
        g = sns.lineplot(ax=ax, x='epoch', y='var', data=table,
                         estimator='mean', ci=68, style=style_key, hue=hue_key,
                         **plot_kwargs)
        ax.text(.4, .9, avg_var + '±' + cs_var, transform=ax.transAxes)
        if not LEGEND and g.legend_ is not None:
            g.legend_.remove()
    else:
        g1 = sns.lineplot(ax=ax, x='epoch', y='var', data=table, estimator=None,
                          units='seed', style=style_key, hue=hue_key, alpha=0.6,
                          **plot_kwargs, legend=None)
        g2 = sns.lineplot(ax=ax, x='epoch', y='var', data=table,
                          estimator='mean', ci=None, style=style_key,
                          hue=hue_key, **plot_kwargs)
        if not LEGEND and g2.legend_ is not None:
            g2.legend_.remove()
        ax.text(.4, .9, avg_var + '±' + cs_var, transform=ax.transAxes)

    # ax.set_ylim([-.01, 1.01])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(1, 10))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)

    plt.close('all')


def weight_var_through_training_2(train_params_list, seeds=None, hue_key=None,
                                  style_key=None,
                                  figname="weight_var_through_training",
                                  subdir=None, weight_type='readout',
                                  multiprocess_lock=None, n_bins=1,
                                  x_axis='epoch', **plot_kwargs):
    """
    Parameters
    ----------
    seeds : List[int]
        List of random number seeds to use for generating instantiations of the
        model and dataset. Variation over these seeds is used to plot error
        bars.
    gs : List[float]
        Values of g_radius to iterate over.
    train_params : dict
        Dictionary of training parameters that specify the model and dataset to
        use for training. Value of g_radius is overwritten by values in gs.
    figname : str
        Name of the figure to save.
    T : int
        Final timepoint to plot (if looking at an RNN). If 0, disregard this
        parameter.
    """
    if subdir is None:
        subdir = ''
    train_params_list = [copy.deepcopy(t) for t in train_params_list]
    
    @memory.cache
    def compute_weight_var_through_training_2(params, hue_key, style_key, seeds,
                                            weight_type, n_bins):
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    multiprocess_lock=multiprocess_lock, **params)
                model.to("cpu")

                epochs, saves = loader.get_epochs_and_saves(run_dir)
                epochs = [epoch for epoch in epochs if
                          epoch <= params['num_epochs']]
                n_epochs_bin = int(params['num_epochs'] / n_bins)
                with torch.no_grad():
                    weights = []
                    update_cnt = 0
                    # updates_per_epoch = int(
                        # params['num_train_samples_per_epoch']/params[
                            # 'batch_size'])*epoch
                    for i0, epoch in enumerate(epochs):
                        for save in saves[i0]:
                            loader.load_model_from_epoch_and_dir(model, run_dir,
                                                                 epoch, save)
                            model.to("cpu")
                            if weight_type == 'readout':
                                if params['network'] == 'feedforward':
                                    r = model.layer_weights[
                                        -1].detach().clone().T
                                else:
                                    r = model.Wout.detach().T.clone()
                                weights.append(r)
                            elif weight_type == 'product':
                                if params['network'] == 'feedforward':
                                    raise NotImplementedError()
                                else:
                                    r = model.Wout.detach().T.clone()
                                    W = model.Wrec.detach().T.clone()
                                    Wpow = torch.matrix_power(
                                        W.T, int(returned_params['n_lag'])-2)
                                    weight_eff = Wpow @ r.T
                                    weights.append(weight_eff)
                            else:
                                raise NotImplementedError()

                        if epoch >= n_epochs_bin:
                            num_updates = int(
                                params['num_train_samples_per_epoch']/params[
                                    'batch_size'])*epoch
                            # print("Number of updates used for variance compute:",
                                  # "{}.".format(len(weights)))
                            var = torch.var(torch.stack(weights), dim=0)
                            var_mu = torch.mean(var).item()
                            d = {'seed': seed, 'epoch': epoch, 'var': var_mu,
                                 'num_updates': num_updates}
                            if hue_key is not None:
                                d.update({hue_key: hue_value})
                            if style_key is not None:
                                d.update({style_key: style_value})
                            table_piece = table_piece.append(
                                pd.DataFrame(d, index=[0]), ignore_index=True)
                            weights.pop(0)
        return table_piece

    table = get_stats(compute_weight_var_through_training_2,
                      train_params_list, seeds,
                      hue_key, style_key, weight_type, n_bins)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    fig, ax = make_fig((1.5, 1.2))
    if style_key is not None or hue_key is not None:
        print("Noise integration not implemented with style_key and hue_key")
    var_sums = torch.tensor(
        [table[table['seed'] == seed]['var'].sum() for seed in seeds])
    avg_var = '{0:1.2g}'.format(torch.mean(var_sums).item())
    cs_var = '{0:1.2g}'.format(1.96*torch.std(var_sums).item())
    if USE_ERRORBARS:
        g = sns.lineplot(ax=ax, x=x_axis, y='var', data=table,
                         estimator='mean', ci=68, style=style_key, hue=hue_key,
                         **plot_kwargs)
        ax.text(.4, .9, avg_var + '±' + cs_var, transform=ax.transAxes)
        if not LEGEND and g.legend_ is not None:
            g.legend_.remove()
    else:
        g1 = sns.lineplot(ax=ax, x=x_axis, y='var', data=table, estimator=None,
                          units='seed', style=style_key, hue=hue_key, alpha=0.6,
                          **plot_kwargs, legend=None)
        g2 = sns.lineplot(ax=ax, x=x_axis, y='var', data=table,
                          estimator='mean', ci=None, style=style_key,
                          hue=hue_key, **plot_kwargs)
        if not LEGEND and g2.legend_ is not None:
            g2.legend_.remove()
        ax.text(.4, .9, avg_var + '±' + cs_var, transform=ax.transAxes)

    # ax.set_ylim([-.01, 1.01])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(1, 10))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)

    plt.close('all')


def saturation_through_layers(train_params_list, seeds=None, hue_key=None,
                              style_key=None,
                              figname="saturation_through_layers", subdir=None,
                              multiprocess_lock=None, use_error_bars=None,
                              **plot_kwargs):

    if subdir is None:
        subdir = train_params_list[0]['network'] + '/saturation_through_layers/'
    if use_error_bars is None:
        use_error_bars = USE_ERRORBARS
    hue_bool = train_params_list is not None
    
    if hue_bool:
        train_params_list = [copy.deepcopy(t) for t in train_params_list]

    @memory.cache
    def compute_saturation_through_layers(params, hue_key, style_key, seeds):
        num_pnts = 1000
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    **params, multiprocess_lock=multiprocess_lock)
                model.to("cpu")

                class_datasets = returned_params['datasets']

                class_datasets['train'].max_samples = num_pnts
                torch.manual_seed(int(params['model_seed']))
                np.random.seed(int(params['model_seed']))
                X, Y = class_datasets['train'][:]
                if params['network'] != 'feedforward':
                    T = 15
                else:
                    T=0
                if T > 0:
                    X = utils.extend_input(X, T)
                    X0 = X[:, 0]
                elif params['network'] != 'feedforward':
                    X0 = X[:, 0]
                else:
                    X0 = X
                # epochs, saves = loader.get_epochs_and_saves(run_dir)
                # for i_epoch, epoch in enumerate([0, -1]):
                loader.load_model_from_epoch_and_dir(model, run_dir,
                                                     params['num_epochs'])
                model.to("cpu")
                hid = [X0]
                hid += model.get_post_activations(X)[:-1]
                col = ('|h|', 'layer', 'seed')
                if hue_key is not None:
                    col = col + ('hue_key',)
                if style_key is not None:
                    col = col + ('style_key',)
                # df = pd.DataFrame(columns=col)
                abs_means = []
                abs_stds = []
                for layer_idx, h in enumerate(hid[1:]):
                    abs_means.append(torch.abs(h).mean().item())
                    abs_stds.append(torch.abs(h).std().item())

                d = {
                    '|h|': abs_means,
                    'abs_std': abs_stds,
                    'layer': list(range(len(abs_means))),
                    'seed': seed
                    }
                if hue_key is not None:
                    d.update({hue_key: hue_value})
                if style_key is not None:
                    d.update({style_key: style_value})
                table_piece = table_piece.append(pd.DataFrame(d),
                                                 ignore_index=True)

        return table_piece

    table = get_stats(compute_saturation_through_layers, train_params_list,
                      seeds, hue_key, style_key)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    table_means = table.drop(columns=['abs_std'])
    # std
    # layers = set(table['layer'])
    # abs_means = [table[table['layer']==k]['abs_mean'].mean() for k in
    #              layers]
    # abs_stds = [table[table['layer']==k]['abs_std'].mean() for k in
    #             layers]

    fig, ax = make_fig((1.5, 1.2))
    fig, ax = plt.subplots()
    # g = ax.errorbar(list(layers), abs_means, abs_stds)
    # print
    # if use_error_bars:
    # # g = ax.err_plot(
    # g = ax.errorbar(layers, abs_means, abs_stds)
    g = sns.lineplot(ax=ax, x='layer', y='|h|',
    data=table_means, style=style_key, hue=hue_key,
    **plot_kwargs)
    ax.set_ylim([-.05, 1.05])
    # # g.figure.colorbar(sm)
    # if not LEGEND and g.legend_ is not None:
    # g.legend_.remove()
    # else:
    # g1 = sns.lineplot(ax=ax, x='layer', y='|h|',
    # data=table, estimator=None, units='seed',
    # style=style_key, hue=hue_key, alpha=0.6,
    # **plot_kwargs)
    # g2 = sns.lineplot(ax=ax, x='layer', y='|h|',
    # data=table, estimator='mean', ci=None,
    # style=style_key, hue=hue_key, **plot_kwargs)
    # if g1.legend_ is not None:
    # g1.legend_.remove()
    # if not LEGEND and g2.legend_ is not None:
    # g2.legend_.remove()

    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)

def dim_var_scatter(train_params_list, seeds=None, hue_key=None,
                    style_key=None, figname='', weight_type='readout',
                    update=6000, subdir=None, multiprocess_lock=None,
                    **plot_kwargs):
    if subdir is None:
        subdir = train_params_list[0]['network'] + '/' \
                + 'dim_var_scatter' + '/'

    @memory.cache
    def compute_dim_var_scatter(params, hue_key, style_key, seeds, update):
        num_pnts_dim_red = 500
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                params['model_seed'] = seed
                model, returned_params, run_dir = train.initialize_and_train(
                    **params, multiprocess_lock=multiprocess_lock)
                model.to("cpu")
                class_datasets = returned_params['datasets']
                class_datasets['train'].max_samples = num_pnts_dim_red
                torch.manual_seed(int(params['model_seed']))
                np.random.seed(int(params['model_seed']))
                X, Y = class_datasets['train'][:]
                T = 0
                if T > 0:
                    X = utils.extend_input(X, T)
                    X0 = X[:, 0]
                elif params['network'] != 'feedforward':
                    X0 = X[:, 0]
                else:
                    X0 = X
                epochs, saves = loader.get_epochs_and_saves(run_dir)
                epochs = [epoch for epoch in epochs if
                          epoch <= params['num_epochs']]
                epoch = int(params['batch_size'] * update \
                        / params['num_train_samples_per_epoch'])
                epoch_dist = np.abs(np.array(epochs) - epoch)
                m_idx = np.argmin(epoch_dist)
                epoch = epochs[m_idx]
                loader.load_model_from_epoch_and_dir(model, run_dir, epoch)
                model.to("cpu")
                hid = [X0]
                hid += model.get_post_activations(X)[:-1]
                try:
                    dim = utils.get_effdim(hid[-1],
                                           preserve_gradients=False).item()
                except RuntimeError:
                    print("Dim computation didn't converge.")
                    dim = np.nan

                weights = []
                for i0, epoch in enumerate(epochs[:m_idx+1]):
                    for save in saves[i0]:
                        loader.load_model_from_epoch_and_dir(model, run_dir,
                                                             epoch, save)
                        model.to("cpu")
                        if weight_type == 'readout':
                            if params['network'] == 'feedforward':
                                r = model.layer_weights[
                                    -1].detach().clone().T
                            else:
                                r = model.Wout.detach().T.clone()
                            weights.append(r)
                        elif weight_type == 'product':
                            if params['network'] == 'feedforward':
                                raise NotImplementedError()
                            else:
                                r = model.Wout.detach().T.clone()
                                W = model.Wrec.detach().T.clone()
                            weights.append(torch.matrix_power(
                                W.T, int(returned_params['n_lag']))@r.T
                                           )
                        else:
                            raise NotImplementedError()

                num_updates = int(
                    params['num_train_samples_per_epoch']/params[
                        'batch_size'])*epoch
                # print("Number of updates used for variance compute:",
                      # "{}.".format(len(weights)))
                var = torch.var(torch.stack(weights), dim=0)
                var_mu = torch.mean(var).item()
                num_updates = int(
                    params['num_train_samples_per_epoch']/params[
                        'batch_size'])*epoch
                d = {
                    'effective_dimension': dim,
                    'var': var_mu,
                    'seed': seed,
                    'epoch': epoch,
                    'num_updates': num_updates
                    }
                if hue_key is not None:
                    d.update({hue_key: hue_value})
                if style_key is not None:
                    d.update({style_key: style_value})
                # casting d to DataFrame necessary to preserve type
                table_piece = table_piece.append(pd.DataFrame(d, index=[0]),
                                                 ignore_index=True)
        return table_piece

    table = get_stats(compute_dim_var_scatter, train_params_list, seeds,
                      hue_key, style_key, update)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    key_list = [hue_key, style_key]
    key_list = [key for key in key_list if key is not None]
    fig, ax = make_fig((1.5, 1.2))
    g = sns.scatterplot(ax=ax, x='var', y='effective_dimension',
                        data=table, hue=hue_key,
                        style=style_key)
    g.legend_.remove()
    # if USE_ERRORBARS:
        # g = sns.lineplot(ax=ax, x=x_axis, y='effective_dimension',
                         # data=table, estimator=est_dim, ci=ci_dim, hue=hue_key,
                         # style=style_key, **plot_kwargs)
        # if not LEGEND and g.legend_ is not None:
            # g.legend_.remove()
    # else:
        # g1 = sns.lineplot(ax=ax, x=x_axis, y='effective_dimension',
                          # data=table, estimator=None, units='seed', hue=hue_key,
                          # style=style_key, alpha=.6, **plot_kwargs, legend=None)
        # g2 = sns.lineplot(ax=ax, x=x_axis, y='effective_dimension',
                          # data=table, estimator='mean', ci=None, hue=hue_key,
                          # style=style_key, **plot_kwargs)
        # if not LEGEND and g2.legend_ is not None:
            # g2.legend_.remove()

    # ax.set_ylim([0, None])
    # ax.set_xlim([None, xlimit])
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)
    plt.close('all')

def dim_depth_frozen_scatter(train_params_list, seeds=None, hue_key=None,
                             style_key=None, figname='', weight_type='readout',
                             update=6000, max_depth=3, subdir=None,
                             multiprocess_lock=None, **plot_kwargs):
    if subdir is None:
        subdir = train_params_list[0]['network'] + '/' \
                + 'dim_over_training' + '/'

    @memory.cache
    def dim_depth_frozen_scatter(params, hue_key, style_key, seeds, update,
                                max_depth):
        num_pnts_dim_red = 500
        table_piece = pd.DataFrame()
        if params is not None:
            if hue_key is not None:
                hue_value = params[hue_key]
            if style_key is not None:
                style_value = params[style_key]
            for seed in seeds:
                for depth in range(1, max_depth+1):
                    params['model_seed'] = seed
                    params['n_lag'] = depth-1
                    params['train_output_weights'] = False
                    # params['train_output_bias'] = False
                    model, returned_params, run_dir = train.initialize_and_train(
                        **params, multiprocess_lock=multiprocess_lock)
                    model.to("cpu")
                    class_datasets = returned_params['datasets']
                    class_datasets['train'].max_samples = num_pnts_dim_red
                    torch.manual_seed(int(params['model_seed']))
                    np.random.seed(int(params['model_seed']))
                    X, Y = class_datasets['train'][:]
                    T = 0
                    if T > 0:
                        X = utils.extend_input(X, T)
                        X0 = X[:, 0]
                    elif params['network'] != 'feedforward':
                        X0 = X[:, 0]
                    else:
                        X0 = X
                    epochs, saves = loader.get_epochs_and_saves(run_dir)
                    epochs = [epoch for epoch in epochs if
                              epoch <= params['num_epochs']]
                    epoch = int(params['batch_size'] * update \
                            / params['num_train_samples_per_epoch'])
                    epoch_dist = np.abs(np.array(epochs) - epoch)
                    m_idx = np.argmin(epoch_dist)
                    epoch = epochs[m_idx]
                    loader.load_model_from_epoch_and_dir(model, run_dir, epoch)
                    model.to("cpu")
                    hid = [X0]
                    hid += model.get_post_activations(X)[:-1]
                    try:
                        dim = utils.get_effdim(hid[-1],
                                               preserve_gradients=False).item()
                    except RuntimeError:
                        print("Dim computation didn't converge.")
                        dim = np.nan

                    num_updates = int(
                        params['num_train_samples_per_epoch']/params[
                            'batch_size'])*epoch
                    # print("Number of updates used for variance compute:",
                          # "{}.".format(len(weights)))
                    var = torch.var(torch.stack(weights), dim=0)
                    var_mu = torch.mean(var).item()
                    num_updates = int(
                        params['num_train_samples_per_epoch']/params[
                            'batch_size'])*epoch
                    d = {
                        'effective_dimension': dim,
                        'depth': depth,
                        'seed': seed,
                        'epoch': epoch,
                        'num_updates': num_updates
                        }
                    if hue_key is not None:
                        d.update({hue_key: hue_value})
                    if style_key is not None:
                        d.update({style_key: style_value})
                    # casting d to DataFrame necessary to preserve type
                    table_piece = table_piece.append(pd.DataFrame(d, index=[0]),
                                                     ignore_index=True)
        return table_piece

    table = get_stats(dim_depth_frozen_scatter, train_params_list, seeds,
                      hue_key, style_key, update, max_depth)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna()
    key_list = [hue_key, style_key]
    key_list = [key for key in key_list if key is not None]
    fig, ax = make_fig((1.5, 1.2))
    g = sns.scatterplot(ax=ax, x='depth', y='effective_dimension',
                        data=table, hue=hue_key,
                        style=style_key)
    g.legend_.remove()
    # if USE_ERRORBARS:
        # g = sns.lineplot(ax=ax, x=x_axis, y='effective_dimension',
                         # data=table, estimator=est_dim, ci=ci_dim, hue=hue_key,
                         # style=style_key, **plot_kwargs)
        # if not LEGEND and g.legend_ is not None:
            # g.legend_.remove()
    # else:
        # g1 = sns.lineplot(ax=ax, x=x_axis, y='effective_dimension',
                          # data=table, estimator=None, units='seed', hue=hue_key,
                          # style=style_key, alpha=.6, **plot_kwargs, legend=None)
        # g2 = sns.lineplot(ax=ax, x=x_axis, y='effective_dimension',
                          # data=table, estimator='mean', ci=None, hue=hue_key,
                          # style=style_key, **plot_kwargs)
        # if not LEGEND and g2.legend_ is not None:
            # g2.legend_.remove()

    # ax.set_ylim([0, None])
    # ax.set_xlim([None, xlimit])
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    out_fig(fig, figname, subfolder=subdir, show=False, save=True, axis_type=0,
            data=table)
    plt.close('all')
if __name__ == '__main__':
    base_params = dict(N=200, num_epochs=5, num_train_samples_per_epoch=800,
                       num_test_samples_per_epoch=100, X_clusters=60, X_dim=200,
                       num_classes=2, n_lag=10,  # g_radius=1,
                       g_radius=20, wout_scale=1., clust_sig=.02,
                       input_scale=1.0, input_style='hypercube', model_seed=1,
                       n_hold=1, n_out=1, use_biases=False,  # use_biases=True,
                       loss='mse', optimizer='sgd', momentum=0, dt=.01,
                       learning_rate=1e-4, batch_size=1, freeze_input=False,
                       network='vanilla_rnn',  # network='feedforward',
                       Win='orthog', patience_before_stopping=6000,
                       hid_nonlin='linear', saves_per_epoch=20, )

    weight_var_through_training([base_params], None, [0])
