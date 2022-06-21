'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
from typing import *
import os
import sys
import time
import math
import pickle as pkl
from pathlib import Path
import warnings

from matplotlib import pyplot as plt
import matplotlib.ticker
import scipy.stats as st
import numpy as np

import torch.nn as nn
import torch.nn.init as init
import torch

# def get_pcs(X, pcs, original_shape=True, return_projectors=False):
#     """
#         Return principal components of X (Stefan's version).
#         Args:
#             X ([num_samples, ambient space dimension]): Data matrix of samples where each sample corresponds to a row of
#                 X.
#             pcs ([num_pcs,]): List of principal components to return.
#
#         Returns:
#             pca_proj: ([num_pcs, ambient space dimension]): Projection of X onto principal components given by pcs.
#
#         """
#     X_shape = X.shape
#     N = X_shape[0]
#     X_centered = X - torch.mean(X, dim=0)
#     # if X.shape[0] < X.shape[1]:
#     #     X_centered = X_centered.T
#     C = X_centered.T@X_centered/(N - 1)
#     eigs, ev = torch.symeig(C, eigenvectors=True)
#     idx = torch.argsort(eigs, descending=True)
#     ev = ev[:, idx]
#     # pca_proj = np.dot(ev[:, pcs].T, X.T)
#     pca_proj = np.dot(X, ev[:, pcs])
#     if original_shape:
#         pca_proj = pca_proj.reshape(*X_shape[:-1], pca_proj.shape[-1])
#     if return_projectors:
#         return pca_proj, ev
#     else:
#         return pca_proj

def svd_flip(u, vt, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u, vt : ndarray
        u and vt are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, vt)`.

    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of vt. The choice of which variable to base the
        decision on is generally algorithm dependent.


    Returns
    -------
    u_adjusted, vt_adjusted : arrays with the same dimensions as the input.

    """
    u = u.copy()
    vt = vt.copy()
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        vt *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(vt), axis=1)
        signs = np.sign(vt[range(vt.shape[0]), max_abs_rows])
        u *= signs
        vt *= signs[:, np.newaxis]
    return u, vt

def extend_input(X, T, T_before=None):
    s = X.shape
    if T_before is None or T_before < 1:
        Xn = torch.zeros((s[0], T, s[2]))
        Xn[:, :s[1]] = X
    else:
        # Xn = np.zeros((s[0], T+T_before, s[2]))
        Xn = torch.zeros((s[0], T+T_before, s[2]))
        Xn[:, T_before:T_before+s[1]] = X

    return Xn

def get_pcs_covariance(X, pcs, original_shape=True, return_extra=False):
    """
        Return principal components of X (using the covariance matrix).
        Args:
            X ([num_samples, ambient space dimension]): Data matrix of samples where each sample corresponds to a row of
                X.
            pcs ([num_pcs,]): List of principal components to return.

        Returns:
            pca_proj: ([num_pcs, ambient space dimension]): Projection of X onto principal components given by pcs.

        """
    N = X.shape[0]
    if X.ndim > 2:
        X = X.reshape(-1, X.shape[-1])
        print("Warning: concatenated first however many dimensions to get square data array")
    mu = torch.mean(X, dim=0)
    X = X - mu
    if X.shape[0] < X.shape[1]:
        X = X.T
    cov = X.T @ X / (N - 1)
    # eig, ev = torch.symeig(cov, eigenvectors=True)
    # ind = torch.argsort(torch.abs(eig), descending=True)
    eig, ev = torch.linalg.eigh(cov)
    ind = torch.argsort(eig.abs(), descending=True)
    ev = ev[:, ind]
    # pca_proj = np.dot(ev[:, pcs].T, X.T)
    pca_proj = X @ ev[:, pcs]
    if original_shape:
        pca_proj = pca_proj.reshape(*X.shape[:-1], pca_proj.shape[-1])
    if return_extra:
        return {'pca_projection': pca_proj, 'pca_projectors': ev, 'mean': mu}
    else:
        return pca_proj

def get_pcs(X, pcs, original_shape=True, return_extra=False):
    """
    Return principal components of X.
    Args:
        X ([num_samples, ambient space dimension]): Data matrix of samples where each sample corresponds to a row of X.
        pcs ([num_pcs,]): List of principal components to return.

    Returns:
        pca_proj: ([len(pcs), num_samples]): Projection of X onto principal components given by pcs.

    """
    # X = X.copy()

    X_shape = X.shape
    if X.ndim > 2:
        X = X.reshape(-1, X.shape[-1])
        print("Warning: concatenated first however many dimensions to get square data array")
    mu = torch.mean(X, dim=0)
    X_centered = X - mu
    U, s, V = torch.svd(X_centered)
    # U, Vt = svd_flip(U, V.T)
    # V = Vt.T
    # pca_proj = (s[pcs] * U[:, pcs]).T
    pca_proj = s[pcs] * U[:, pcs]
    if original_shape:
        pca_proj = pca_proj.reshape(*X_shape[:-1], pca_proj.shape[-1])
    if return_extra:
        return {'pca_projection': pca_proj, 'pca_projectors': V, 'mean': mu}
    return pca_proj

def get_effdim(X, preserve_gradients=True):
    # X_centered = X - np.mean(X, axis=0)
    # C = X_centered.T @ X_centered / (X.shape[0]-1)
    N = X.shape[0]
    X_centered = X - torch.mean(X, dim=0)
    if X.shape[0] < X.shape[1]:
        X_centered = X_centered.T
    C = X_centered.T @ X_centered / (N - 1)
    # eigs = torch.linalg.eigvalsh(C, eigenvectors=preserve_gradients)
    eigs = torch.linalg.eigvalsh(C)
    return torch.sum(eigs) ** 2 / torch.sum(eigs ** 2)

def get_effcca(X, Y):
    Nx = X.shape[0]
    Ny = Y.shape[0]
    # X = X[:, :Nsample]
    # Y = Y[:, :Nsample]
    X_centered = X - torch.mean(X, dim=0)
    Y_centered = Y - torch.mean(Y, dim=0)
    if X.shape[0] < X.shape[1]:
        X_centered = X_centered.T
    if Y.shape[0] < Y.shape[1]:
        Y_centered = Y_centered.T
    Cxx = X_centered.T @ X_centered
    Cyy = Y_centered.T @ Y_centered
    Cxy = X_centered.T @ Y_centered
    Cyx = Cxy.T
    cuda0 = torch.device('cuda:0')
    A = torch.cat(
        (torch.cat((torch.zeros([Nx, Nx], device=cuda0), Cxy), dim=1),
         torch.cat((Cyx, torch.zeros([Nx, Nx], device=cuda0)), dim=1)), dim=0)

    B = torch.cat(
        (torch.cat((Cxx, torch.zeros([Nx, Nx], device=cuda0)), dim=1),
         torch.cat((torch.zeros([Nx, Nx], device=cuda0), Cyy), dim=1)))

    # eigs = torch.linalg.eigh(torch.inverse(B) @ A, eigenvectors=True)
    eigs = torch.linalg.eigvalsh(torch.inverse(B) @ A)

    return torch.sum(eigs) / (2 * Nx)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width=40

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def median_and_bound(samples, perc_bound, dist_type='gamma', loc=0, shift=0, reflect=False, show_fit=False):
    def do_reflect(x, center):
        return -1 * (x - center) + center

    if dist_type == 'gamma':
        if np.sum(samples[0] == samples) == len(samples):
            median = samples[0]
            interval = [samples[0], samples[0]]
            return median, interval

        if reflect:
            # reflect_point = loc + shift
            samples_reflected = do_reflect(samples, loc)
            shape_ps, loc_fit, scale = st.gamma.fit(samples_reflected, floc=loc + shift)
            median_reflected = st.gamma.median(shape_ps, loc=loc, scale=scale)
            interval_reflected = np.array(st.gamma.interval(perc_bound, shape_ps, loc=loc, scale=scale))
            median = do_reflect(median_reflected, loc)
            interval = do_reflect(interval_reflected, loc)
        else:
            shape_ps, loc, scale = st.gamma.fit(samples, floc=loc + shift)
            median = st.gamma.median(shape_ps, loc=loc, scale=scale)
            interval = np.array(st.gamma.interval(perc_bound, shape_ps, loc=loc, scale=scale))

        if np.isnan(np.sum(median)) or np.isnan(np.sum(interval)):
            return -1

        if show_fit:
            fig, ax = plt.subplots()
            ax.hist(samples, density=True)
            xx = np.linspace(np.min(samples), np.max(samples))
            if reflect:
                yy_hat = st.gamma.pdf(do_reflect(xx, loc), shape_ps, loc=loc, scale=scale)
                ax.plot(xx, yy_hat)
                ax.axvline(x=median)
                ax.axvline(x=interval[0], linestyle='--')
                ax.axvline(x=interval[1], linestyle='--')
                plt.show()
            else:
                yy_hat = st.gamma.pdf(xx, shape_ps, loc=loc, scale=scale)
                ax.plot(xx, yy_hat)
                ax.axvline(x=median)
                ax.axvline(x=interval[0], linestyle='--')
                ax.axvline(x=interval[1], linestyle='--')
                plt.show()

    return median, interval

def point_replace(a_string):
    a_string = str(a_string)
    return a_string.replace(".", "p")

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=None, useLocale=None):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText, useLocale=useLocale)

    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

def make_fig(figsize, ax_pos=(1, 1, 1, 1)):
    """

    Args:
        figsize ():
        ax_pos ():

    Returns:

    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_pos)
    return fig, ax

def print_gpu_usage():
    from inspect import currentframe, getframeinfo
    frameinfo = getframeinfo(currentframe().f_back)
    print(frameinfo.filename, frameinfo.lineno)
    import os; os.system('nvidia-smi')

def save_fig(fig, filename, show=False, save=True, axis_type=0, data=None):
    filename = Path(filename)
    Path.mkdir(filename.parents[0], exist_ok=True)
    ax = fig.axes[0]
    if axis_type == 1:
        ax.tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off
    elif axis_type == 2:
        ax.axis('off')
    # fig_file = (file_name).with_suffix('.pdf')
    if save:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fig.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')
    if data is not None:
        data_dir = filename / 'data'
        Path.mkdir(data_dir, exist_ok=True)
        with open(data_dir / 'data.pkl', 'wb') as fid:
            pkl.dump(data, fid, protocol=4)
    if show:
        fig.tight_layout()
        fig.show()

def shade_plot(ax, x_pnts, middle, intervals, color, label, alpha=0.2, **kwargs):
    lines, = ax.plot(x_pnts, middle, color=color, label=label, **kwargs)
    ax.fill_between(x_pnts,
                    intervals[:, 0],
                    intervals[:, 1],
                    color=color,
                    alpha=alpha, lw=0)
    return lines

def chaos_before_after_training_shadeplots(fig_name, g_vals, x_pnts, center_pnts, intervals):
    # axes_labels = ['g', 'epochs', 'time']

    train_test_lines = ['--', '-']
    train_label = ['before training', 'after training']
    g_label = ['edge of chaos', 'strongly chaotic']
    fig, ax = make_fig(figsize_small)
    for i0 in range(center_pnts.shape[0]):
        for i1 in range(center_pnts.shape[1]):
            shade_plot(ax, x_pnts, center_pnts[i0, i1], intervals[i0, i1], chaos_colors[i0], '', 0.2,
                       linestyle=train_test_lines[i1])

    return fig, ax

def saves_to_partial_epochs(epochs, saves):
    """epochs is a list of epochs, and saves[k] is a list of all the saves for epoch k. This utility converts these two
    lists into a single list that shows all of the (fractional) epochs at which saves occur. For instance,
    epochs = [0,1,2,3] and saves=[[0,1],[0,1],[0,1],[0]] would result in a return value of [0, 0.5, 1, 1.5, 2, 2.5, 3]"""
    partial_epochs = []
    for epoch_id, epoch in enumerate(epochs):
        num_saves = len(saves[epoch_id])
        epoch_frac = 1/num_saves
        partial_epochs.append(epoch)
        for save in saves[epoch_id]:
            if save != 0:
                partial_epochs.append(epoch + save*epoch_frac)
    return partial_epochs

def compute_saves(saves_per_epoch: Union[int, float, Sequence[int]],
                  num_epochs: int,
                  start_epoch: int = 0) -> Union[List[int], List[float]]:
    """Compute the saves for the network in terms of epochs. Returns a list saves_in_epochs where
    each entry saves_in_epochs[k] is the number of epochs that have transpired before save k,
    including fractions of epochs. When saves_per_epoch<1, this is interpreted as specifying the
    "epochs per save" which is computed as epochs_per_save = round(1/saves_per_epoch).
    Here are some examples:

    compute_saves(1, 3) = [0, 1, 2, 3]  # save 1 occurs after epoch 1, save 2 after epoch 2, etc.
    compute_saves(2, 3) = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # save 1 occurs after 0.5 epochs, etc.
    compute_saves(1/2, 6) = [0, 1, 3, 5]  # save 1 occurs after epoch 1, save 2 after epoch 3, etc.

    If saves_per_epoch is a Sequence such as a list, then saves_per_epoch[k] holds the number of saves
    that should be saved in the interval of epochs (k, k+1]. As an example:
    compute_saves([2,0,1], 3) = [0, 0.5, 1.0, 3.0]

    start_epoch is the epoch to consider as the starting point. Examples with this nonzero:
    compute_saves(1, 3, 1) = [1, 2, 3, 4]
    compute_saves(2, 3, 1) = [1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    compute_saves(1/2, 6, 1) = [1, 2, 4, 6]
    compute_saves([2,0,1], 3, 1) = [1, 1.5, 2.0, 4.0]
    """

    SPE_is_numeric = not hasattr(saves_per_epoch, '__len__')
    if SPE_is_numeric and saves_per_epoch <= 1:
        epochs_per_save = round(1 / saves_per_epoch)
        saves_in_epochs = [start_epoch] + list(range(start_epoch + 1, start_epoch + num_epochs + 1, epochs_per_save))
    elif SPE_is_numeric:
        saves_in_epochs = [start_epoch]
        for epoch in range(start_epoch, start_epoch + num_epochs):
            delta = 1 / saves_per_epoch
            saves_for_epoch = epoch + torch.arange(delta, 1 + delta / 2, delta)
            saves_in_epochs.extend([x.item() for x in saves_for_epoch])
    else:
        saves_in_epochs = [start_epoch]
        epochs = range(start_epoch, start_epoch + num_epochs)
        if len(epochs) != num_epochs:
            print("Warning: num_epochs does not match saves_per_epoch. Setting num_epochs=len(saves_per_epoch).")
        for epoch, saves_this_epoch in zip(epochs, saves_per_epoch):
            if saves_this_epoch > 0:
                delta = 1 / saves_this_epoch
                saves_for_epoch = epoch + torch.arange(delta, 1 + delta / 2, delta)
                saves_in_epochs.extend([x.item() for x in saves_for_epoch])
    return saves_in_epochs

def test_dataset_dim():
    import torchvision
    import torch
    import time
    from matplotlib import pyplot as plt
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    X = torch.stack([x[0].flatten() for x in testset])

    labels = [x[1] for x in testset]

    # dims = [get_effdim(X[:k]) for k in range(X.shape[0]+1)]
    tic = time.time()
    dims = [get_effdim(X[:k]) for k in range(10, 1000)]
    toc = time.time()
    print(toc - tic)
    plt.plot(range(10, 1000), dims)
    plt.show()
    # tic = time.time()
    # dims = [get_effdim(X[-k:]) for k in range(100)]
    # toc = time.time()
    # print(toc - tic)

    # N = X.shape[0]
    # X_centered = X - torch.mean(X, dim=0)
    # if X.shape[0] < X.shape[1]:
    #     X_centered = X_centered.T
    # C = X_centered.T @ X_centered / (N-1)
    # eigs = torch.symeig(C, eigenvectors=True)[0]

    # dims = [torch.sum(eigs[:k])**2 / torch.sum(eigs[:k]**2) for k in range(C.shape[0])]
    # return torch.sum(eigs)**2 / torch.sum(eigs**2)


class CheckConvergence:
    """Handles determining if a sequence is converging to a given value, or if a sequence has stopped
    increasing.
    """

    def __init__(self, memory_length=10, tolerance=1e-5, tolerance_below=None, tolerance_mode='absolute',
                 min_length=5):
        """

        Args:
            memory_length (int or None): Length of entries of sequence to use for measuring convergence. If None,
                this is set to 1e6.
            tolerance (float): Tolerance with which sequence can deviate from a limiting value and still be considered
                converging.
            tolerance_below (float): Not yet implemented.
            tolerance_mode (str): Type of tolerance to use. Options are 'absolute' and 'relative'. The interpretation is
                the same as in the documentation for np.allclose.
            min_length (int): The minimum number of sequence values seen before allowing for positive convergence.
                Default is 5. Must be at least 2.
        """
        if memory_length is None:
            memory_length = 1e6
        self.memory_length = memory_length
        self.tolerance = tolerance
        self.tolerance_mode = tolerance_mode
        if tolerance_below is None:
            tolerance_below = tolerance
        self.tolerance_below = tolerance_below
        if min_length < 2:
            raise TypeError('Invalid option for min_length. You need to have min_length >=2.')
        self.min_length = min_length
        self.cur_length = 0

        self.memory = []

    def reset(self):
        self.memory.clear()
        self.cur_length = 0

    def check_convergence(self, single_value, comparison_mode='sup', verbose=False, ret_dev=False):
        """
        Take most recent value of sequence single_value and add it to a list of previous sequence inputs of length
        at most memory_length. Check to see if the sequence has converged to a limit within a specified tolerance.

        Args:
            single_value (float or array of floats): Most recent value of sequence.
            comparison_mode (str): 'sup', 'avg', 'l2', or 'sem', for using maximum or average or l2 norm of last
                memory_length values to determine convergence. 'sem' is or standard error of the mean of the sequence.
            verbose (bool): Verbose mode.
            ret_dev (bool): Return (True) the measured error dev or not.

        Returns:
            (bool): True (convergence criterion met) or False.
        """

        if len(self.memory) >= self.memory_length:
            self.memory.pop(0)
        self.memory.append(single_value)
        self.cur_length += 1

        if len(self.memory) < 2:
            if ret_dev:
                return False, np.nan
            else:
                return False

        mem_d = np.diff(self.memory, axis=0)
        if comparison_mode == 'sup':
            dev = np.max(np.abs(mem_d))
        elif comparison_mode == 'avg':
            dev = np.mean(np.abs(mem_d))
        elif comparison_mode == 'l2':
            dev = np.sqrt(np.mean(mem_d ** 2))
        elif comparison_mode == 'sem':
            dev = np.mean(np.std(self.memory, axis=0)) / np.sqrt(len(self.memory))
        else:
            raise TypeError('Invalid option for comparison_mode')

        if self.tolerance_mode == 'relative':
            converge_bool = dev < self.tolerance * np.mean(np.abs(self.memory))
            if verbose:
                print(self.tolerance * np.mean(np.abs(self.memory)) - dev)
        elif self.tolerance_mode == 'absolute':
            converge_bool = dev < self.tolerance
            if verbose:
                print(self.tolerance - dev)
        else:
            raise TypeError('Invalid option for tolerance_mode')

        if self.cur_length < self.min_length:
            converge_bool = False

        if ret_dev:
            return converge_bool, dev
        else:
            return converge_bool

    def check_increasing(self, single_value, verbose=False):
        """
        Check if sequence is nonincreasing on average. Tolerance is amount that it can increase and still be reported as
        nonincreasing (use a negative tolerance of you want to force it to decrease by at least a certain amount to
        be reported as nonincreasing).

        single_value (float or array of floats): Most recent value of sequence.
        verbose (bool): Verbose mode.

        Returns:
            (bool): True (nonincreasing criterion met) or False.

        """
        if len(self.memory) >= self.memory_length:
            self.memory.pop(0)
        self.memory.append(single_value)
        self.cur_length += 1

        if len(self.memory) < 2:
            return False

        mem_d = np.diff(self.memory, axis=0)
        change = np.sum(mem_d)

        if self.tolerance_mode == 'relative':
            increase_bool = change >= self.tolerance * np.mean(np.abs(self.memory))
        else:
            increase_bool = change >= self.tolerance

        if self.cur_length < self.min_length:
            increase_bool = False

        return increase_bool

    def clear_memory(self):
        self.memory.clear()
