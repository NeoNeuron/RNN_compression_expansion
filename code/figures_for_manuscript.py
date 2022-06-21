# %% Script for generating the figures used in the manuscript
# "Robust representations learned in RNNs through implicit balance of
# expansion and compression"

import itertools
import time
from multiprocessing import Process, Lock
from pathlib import Path
import plots
import seaborn as sns

# %% Setting up the options and parameters

# FAST_RUN = False  # Takes about 4.5 hours
FAST_RUN = True  # Takes about 45 minutes (change "seeds" below to speed up)

CLEAR_PREVIOUS_RUNS = False
# CLEAR_PREVIOUS_RUNS = True # Delete saved weights from previous runs

plots.LEGEND = True
# plots.LEGEND = False

if FAST_RUN:
    seeds = list(range(5))
else:
    seeds = list(range(30))

if CLEAR_PREVIOUS_RUNS:
    import shutil
    shutil.rmtree('../data/output')

# Number parallel threads to use
n_cores = 1 
# n_cores = 4
# n_cores = 10

# See initialize_and_train.initialize_and_train for a description of these
# parameters.
high_d_input_edge_of_chaos_params = dict(N=200,
                                         num_epochs=12,
                                         num_train_samples_per_epoch=800,
                                         num_test_samples_per_epoch=100,
                                         X_clusters=60,
                                         X_dim=200,
                                         num_classes=2,
                                         n_lag=10,
                                         g_radius=20,
                                         wout_scale=1.0,
                                         clust_sig=.02,
                                         input_scale=1.0,
                                         n_hold=1,
                                         n_out=1,
                                         loss='cce',
                                         optimizer='rmsprop',
                                         dt=.01,
                                         momentum=0,
                                         learning_rate=1e-3,
                                         batch_size=10,
                                         freeze_input=False,
                                         network='vanilla_rnn',
                                         scheduler='plateau',
                                         learning_patience=5,
                                         Win='orthog',
                                         patience_before_stopping=None,
                                         hid_nonlin='tanh',
                                         saves_per_epoch=1.0,
                                         model_seed=1,
                                         rerun=False)

high_d_input_strongly_chaotic_params = high_d_input_edge_of_chaos_params.copy()
high_d_input_strongly_chaotic_params['g_radius'] = 250
low_d_input_edge_of_chaos_params = high_d_input_edge_of_chaos_params.copy()
low_d_input_edge_of_chaos_params['X_dim'] = 2
low_d_input_edge_of_chaos_params['num_epochs'] = 120
low_d_input_strongly_chaotic_params = high_d_input_edge_of_chaos_params.copy()
low_d_input_strongly_chaotic_params['g_radius'] = 250
low_d_input_strongly_chaotic_params['X_dim'] = 2
low_d_input_strongly_chaotic_params['num_epochs'] = 120

lyap_epochs_edge_of_chaos = [0, 1, 5, 10, 150]
lyap_epochs_strongly_chaotic = [0, 1, 5, 10, 150]

chaos_color_0 = [0, 160/255, 160/255]
chaos_color_1 = [233/255, 38/255, 41/255]
chaos_colors = [chaos_color_0, chaos_color_1]
chaos_palette = sns.color_palette(chaos_colors)

subdir_preface = Path('../results/figs/')

# %% Helper function
def make_chaos_params_list(params):
    params = params.copy()
    plist = []
    for g in [20, 250]:
        ps = params.copy()
        ps['g_radius'] = g
        plist.append(ps)
    return plist

# %% Calls to plotting functions

# % Figure 2
def fig2(multiprocess_lock=None):
    subdir = subdir_preface/"fig2"
    # % Figure 2b (top)
    g = high_d_input_edge_of_chaos_params['g_radius']
    plots.lyaps([0], high_d_input_edge_of_chaos_params,
                lyap_epochs_edge_of_chaos,
                figname="fig_2b_top_g_{}_lyaps".format(g), subdir=subdir)

    # % Figure 2b (bottom)
    g = high_d_input_strongly_chaotic_params['g_radius']
    plots.lyaps([0], high_d_input_strongly_chaotic_params,
                lyap_epochs_strongly_chaotic,
                figname="fig_2b_bottom_g_{}_lyaps".format(g),
                subdir=subdir)

    # % Figure 2c (top)
    g = high_d_input_edge_of_chaos_params['g_radius']
    plots.snapshots_through_time(high_d_input_edge_of_chaos_params,
                                 subdir=f'{subdir}/fig_2c_top_snaps_g_{g}')
    
    # % Figure 2c (bottom)
    g = high_d_input_strongly_chaotic_params['g_radius']
    plots.snapshots_through_time(high_d_input_strongly_chaotic_params,
                                 subdir=f'{subdir}/fig_2c_bottom_snaps_g_{g}')

    # % Figure 2d
    acc_params_eoc = high_d_input_edge_of_chaos_params.copy()
    acc_params_eoc['num_epochs'] = 10
    acc_params_eoc['num_train_samples_per_epoch'] = 400
    acc_params_sc = acc_params_eoc.copy()
    acc_params_sc['g_radius'] = 250
    epochs = list(range(acc_params_eoc['num_epochs'] + 1))
    figname = 'fig_2d_acc_over_training'
    plots.acc_over_training([acc_params_eoc, acc_params_sc], seeds,
                            hue_key='g_radius', figname=figname, subdir=subdir,
                            palette=chaos_palette)

    # % Figure 2e
    fig2e_params = high_d_input_edge_of_chaos_params.copy()
    fig2e_params_0 = fig2e_params.copy()
    fig2e_params_0['num_epochs'] = 0
    plist = make_chaos_params_list(fig2e_params) + \
                    make_chaos_params_list(fig2e_params_0)
    plots.dim_over_layers(plist, seeds,
                          hue_key='g_radius', style_key='num_epochs',
                          figname="fig_2e_dim_over_time",
                          subdir=subdir,
                          style_order=[fig2e_params['num_epochs'], 0],
                          palette=chaos_palette,
                          multiprocess_lock=multiprocess_lock,
                         )

    # % Figures 2f and 2g
    figname="fig_2f_2g_clust_holdout_over_time"
    plots.clust_holdout_over_layers(plist, seeds,
                                    hue_key='g_radius', style_key='num_epochs',
                                    figname=figname, subdir=subdir,
                                    style_order=[fig2e_params['num_epochs'], 0],
                                    multiprocess_lock=multiprocess_lock,
                                    palette=chaos_palette)

# %% Figure 3
def fig3(multiprocess_lock=None):
    subdir = subdir_preface/"fig3"

    # % Figure 3b (top)
    g = low_d_input_edge_of_chaos_params['g_radius']
    plots.lyaps([0], low_d_input_edge_of_chaos_params,
                lyap_epochs_edge_of_chaos, multiprocess_lock=multiprocess_lock,
                figname="fig_3b_top_lyaps_g_{}".format(g), subdir=subdir)

    # % Figure 3b (bottom)
    g = low_d_input_strongly_chaotic_params['g_radius']
    plots.lyaps([0], low_d_input_strongly_chaotic_params,
                lyap_epochs_strongly_chaotic,
                multiprocess_lock=multiprocess_lock,
                figname="fig_3b_bottom_lyaps_g_{}".format(g), subdir=subdir)

    # % Figure 3c (top)
    g = low_d_input_edge_of_chaos_params['g_radius']
    plots.snapshots_through_time(low_d_input_edge_of_chaos_params,
                                 multiprocess_lock=multiprocess_lock,
                                 subdir=f'{subdir}/fig_3c_top_snaps_g_{g}')

    # % Figure 3c (bottom)
    g = low_d_input_strongly_chaotic_params['g_radius']
    plots.snapshots_through_time(low_d_input_strongly_chaotic_params,
                                 multiprocess_lock=multiprocess_lock,
                                 subdir=f'{subdir}/fig_3c_bottom_snaps_g_{g}')

    # % Figure 3d
    acc_params_eoc = low_d_input_edge_of_chaos_params.copy()
    acc_params_sc = acc_params_eoc.copy()
    acc_params_sc['g_radius'] = 250
    figname = 'fig_3d_acc_over_training'
    plots.acc_over_training([acc_params_eoc, acc_params_sc], seeds,
                            multiprocess_lock=multiprocess_lock,
                            hue_key='g_radius', figname=figname, subdir=subdir,
                            palette=chaos_palette)

    # % Figure 3e
    fig3e_params = low_d_input_edge_of_chaos_params.copy()
    fig3e_params_0 = fig3e_params.copy()
    fig3e_params_0['num_epochs'] = 0
    plist = make_chaos_params_list(fig3e_params) + \
                    make_chaos_params_list(fig3e_params_0)
    plots.dim_over_layers(plist, seeds, hue_key='g_radius',
                          style_key='num_epochs',
                          multiprocess_lock=multiprocess_lock,
                          figname="fig_3e_dim_over_time", subdir=subdir,
                          style_order=[fig3e_params['num_epochs'], 0],
                          palette=chaos_palette)

    plots.saturation_through_layers(plist, seeds, hue_key='g_radius',
                                    style_key='num_epochs',
                                    figname="saturation_lowd",
                                    subdir=subdir_preface/"saturation",
                                    style_order=[fig3e_params['num_epochs'],
                                                 0],
                                    multiprocess_lock=multiprocess_lock,
                                    palette=chaos_palette)

    # % Figures 3f and 3g
    plots.clust_holdout_over_layers(plist, seeds, hue_key='g_radius',
                                    style_key='num_epochs',
                                    figname="fig_3f_3g_clust_holdout_over_time",
                                    subdir=subdir,
                                    style_order=[fig3e_params['num_epochs'],
                                                 0],
                                    multiprocess_lock=multiprocess_lock,
                                    palette=chaos_palette)

# %% Figure 4
def fig4(multiprocess_lock=None):
    subdir = subdir_preface/"fig4"
    low_d_2n_input_edge_of_chaos_params = low_d_input_edge_of_chaos_params.copy()
    low_d_2n_input_edge_of_chaos_params['Win'] = 'diagonal_first_two'
    low_d_2n_input_strongly_chaotic_params = \
    low_d_input_strongly_chaotic_params.copy()
    low_d_2n_input_strongly_chaotic_params['Win'] = 'diagonal_first_two'
    # % Figure 4b (top)
    g = low_d_2n_input_edge_of_chaos_params['g_radius']
    plots.lyaps([0], low_d_2n_input_edge_of_chaos_params,
                lyap_epochs_edge_of_chaos, multiprocess_lock=multiprocess_lock,
                figname="fig_4b_top_g_{}_lyaps".format(g), subdir=subdir)

    # % Figure 4b (bottom)
    g = low_d_2n_input_strongly_chaotic_params['g_radius']
    plots.lyaps([0], low_d_2n_input_strongly_chaotic_params,
                lyap_epochs_strongly_chaotic,
                multiprocess_lock=multiprocess_lock,
                figname="fig_4b_bottom_g_{}_lyaps".format(g), subdir=subdir)

    # % Figure 4c (top)
    g = low_d_2n_input_edge_of_chaos_params['g_radius']
    plots.snapshots_through_time(low_d_2n_input_edge_of_chaos_params,
                                 multiprocess_lock=multiprocess_lock,
                                 subdir=f'{subdir}/fig_4c_top_snaps_g_{g}')

    # % Figure 4c (bottom)
    g = low_d_2n_input_strongly_chaotic_params['g_radius']
    plots.snapshots_through_time(low_d_2n_input_strongly_chaotic_params,
                                 multiprocess_lock=multiprocess_lock,
                                 subdir=f'{subdir}/fig_4c_bottom_snaps_g_{g}')

    # % Figure 4d
    acc_params_eoc = low_d_2n_input_edge_of_chaos_params.copy()
    acc_params_sc = acc_params_eoc.copy()
    acc_params_sc['g_radius'] = 250
    plots.acc_over_training([acc_params_eoc, acc_params_sc], seeds,
                            hue_key='g_radius',
                            figname='fig_4d_acc_over_training',
                            multiprocess_lock=multiprocess_lock, subdir=subdir,
                            palette=chaos_palette)

    # % Figure 4e
    fig4e_params = low_d_2n_input_edge_of_chaos_params.copy()
    fig4e_params_0 = fig4e_params.copy()
    fig4e_params_0['num_epochs'] = 0
    plist = make_chaos_params_list(fig4e_params) + \
                    make_chaos_params_list(fig4e_params_0)
    plots.dim_over_layers(plist, seeds=seeds, hue_key='g_radius',
                          style_key='num_epochs',
                          figname="fig_4e_dim_over_time", subdir=subdir,
                          style_order=[fig4e_params['num_epochs'], 0],
                          multiprocess_lock=multiprocess_lock,
                          palette=chaos_palette)

    # % Figures 4f and 4g
    figname = "fig_4f_4g_clust_holdout_over_time"
    plots.clust_holdout_over_layers(plist, seeds=seeds, hue_key='g_radius',
                                    style_key='num_epochs',
                                    figname="fig_4f_4g_clust_holdout_over_time",
                                    subdir=subdir,
                                    style_order=[fig4e_params['num_epochs'], 0],
                                    multiprocess_lock=multiprocess_lock,
                                    palette=chaos_palette)

# %% Figure 5a
def fig5a(multiprocess_lock=None):
    lra_left = 5e-3
    lra_right = 1e-3
    subdir = subdir_preface/"fig5"
    n_epochs = high_d_input_edge_of_chaos_params['num_epochs']
    fig5a_left_params = high_d_input_edge_of_chaos_params.copy()
    fig5a_left_params['loss'] = 'mse'
    fig5a_left_params['learning_rate'] = lra_left
    plist = make_chaos_params_list(fig5a_left_params)
    fig5a_left_params['num_epochs'] = 0
    plist += make_chaos_params_list(fig5a_left_params)
    plots.dim_over_layers(plist, seeds=seeds,
                          hue_key='g_radius', style_key='num_epochs',
                          figname=f"fig5a_left_lr_{lra_left}", subdir=subdir,
                          multiprocess_lock=multiprocess_lock,
                          style_order=[n_epochs, 0], palette=chaos_palette)

    # % Figure 5a_right
    n_epochs = low_d_input_edge_of_chaos_params['num_epochs']
    fig5a_right_params = low_d_input_edge_of_chaos_params.copy()
    fig5a_right_params['loss'] = 'mse'
    fig5a_right_params['learning_rate'] = lra_right
    plist = make_chaos_params_list(fig5a_right_params)
    fig5a_right_params['num_epochs'] = 0
    plist += make_chaos_params_list(fig5a_right_params)
    plots.dim_over_layers(
        plist, seeds=seeds, hue_key='g_radius',
        style_key='num_epochs', figname=f"fig5a_right_lr_{lra_right}",
        multiprocess_lock=multiprocess_lock,
        subdir=subdir, style_order=[n_epochs, 0], palette=chaos_palette
    )


def fig5b_6(param_set, train_params, key_list, keys_abbrev,
             subdir, multiprocess_lock=None):
    subdir = subdir_preface/subdir
    param_set = list(param_set)
    train_params = train_params.copy()

    for i0, key in enumerate(key_list):
        train_params[key] = param_set[i0]

    n_lag = 10
    # n_lag = 2
    train_params['n_lag'] = n_lag
    figname = ''.join(
        key + '_' + str(val) + '_' for key, val in zip(keys_abbrev, param_set))
    figname = figname[:-1]
    num_epochs_bs1 = 40
    bs1_learning_patience = 5
    max_epochs = 80000
    saves = 400
    tps = []
    for bs in [1, 10, 50, 800]:
        tp = train_params.copy()
        tp['batch_size'] = bs
        num_epochs_bs = bs * num_epochs_bs1
        if tp['scheduler'] is not None and tp['scheduler'][:10] == 'onecyclelr':
            bs_learning_patience = bs * bs1_learning_patience
            tp['learning_patience'] = bs_learning_patience
            tp['scheduler_factor'] = 10
            tp['patience_before_stopping'] = num_epochs_bs
        saves_per_epoch = round(saves / num_epochs_bs, 15)
        tp['saves_per_epoch'] = saves_per_epoch
        tp['num_epochs'] = num_epochs_bs
        tps.append(tp)

    n_bins = 10
    if 'use_biases' in train_params and train_params['use_biases']:
        bias_str = 'bias'
    else:
        bias_str = 'nobias'
    # seeds = range(10)
    seeds = range(3)
    plot_ps = (tps, seeds, 'batch_size', None, figname)
    plots.USE_ERRORBARS = False
    plots.dim_over_training(*plot_ps, x_axis='num_updates',
                            y_lim=45, subdir=subdir,
                            multiprocess_lock=multiprocess_lock)
    plots.USE_ERRORBARS = True



def fig5c(multiprocess_lock=None):
    subdir = subdir_preface/'fig5c'
    fig_params = high_d_input_edge_of_chaos_params.copy()
    plist = []
    fig_params['network'] = 'vanilla_rnn'
    net = fig_params['network']
    fig_params['loss'] = 'mse_scalar'
    fig_params['hid_nonlin'] = 'linear'
    fig_params['num_epochs'] = 2*25600
    fig_params['optimizer'] = 'sgd'
    fig_params['learning_rate'] = 2e-4
    fig_params['saves_per_epoch'] = 1/(4*320)
    opt = fig_params['optimizer']
    nl = fig_params['hid_nonlin']
    fig_params['n_lag'] = 2
    fig_params['learning_patience'] = 100000
    large_batch_params = fig_params.copy()
    large_batch_params['batch_size'] = 800
    large_batch_params['grad_noise'] = 0
    large_batch_noise_params_0 = large_batch_params.copy()
    large_batch_noise_params_0['grad_noise'] = 0
    plist.append(large_batch_noise_params_0)
    large_batch_noise_params_1 = large_batch_params.copy()
    large_batch_noise_params_1['grad_noise'] = .5
    large_batch_noise_params_1['grad_noise_type'] = None
    plist.append(large_batch_noise_params_1)
    large_batch_noise_params_2 = large_batch_noise_params_1.copy()
    large_batch_noise_params_2['grad_noise'] = 1
    plist.append(large_batch_noise_params_2)
    gtype = large_batch_noise_params_1['grad_noise_type']
    figname=f"fig_5_dim_mse_{net}_batch_size_gn_{opt}_{nl}_{gtype}"
    seeds = range(1)
    plots.dim_over_training(plist, seeds, hue_key='grad_noise', figname=figname,
                            x_axis='num_updates', subdir=subdir,
                            multiprocess_lock=multiprocess_lock)
    figname=f"fig_5_loss_mse_{net}_batch_size_gn_{opt}_{nl}_{gtype}"
    # plots.loss_over_training(plist, seeds,
                             # hue_key='grad_noise', figname=figname,
                             # x_axis='num_updates', y_scale='log',
                             # subdir=subdir,
                             # multiprocess_lock=multiprocess_lock)



def fig5d(param_set, train_params, key_list, keys_abbrev,
            multiprocess_lock=None):
    subdir = subdir_preface/'fig5d'
    param_set = list(param_set)
    train_params = train_params.copy()

    for i0, key in enumerate(key_list):
        train_params[key] = param_set[i0]
    if train_params['network'] == 'feedforward':
        depths = [2, 3, 4]
    else:
        depths = [2, 3, 4]
    n_lag = train_params['n_lag']
    figname = ''.join(
        key + '_' + str(val) + '_' for key, val in zip(keys_abbrev, param_set))
    figname = figname[:-1]
    num_epochs_bs1 = 80
    bs1_learning_patience = 5
    saves = 800
    tps = []
    bs = 10
    for freeze in [True, False]:
        for depth in depths:
            tp = train_params.copy()
            tp['n_lag'] = depth
            tp['batch_size'] = bs
            tp['train_output_weights'] = not freeze
            num_epochs_bs = bs * num_epochs_bs1
            saves_per_epoch = round(saves / num_epochs_bs, 15)
            tp['saves_per_epoch'] = saves_per_epoch
            tp['num_epochs'] = num_epochs_bs
            tps.append(tp)

    seeds = range(3)
    plot_ps = (tps, seeds, 'n_lag', 'train_output_weights', figname)
    plots.USE_ERRORBARS = False
    plots.dim_over_training(*plot_ps, x_axis='num_updates',
                               subdir=subdir,
                               multiprocess_lock=multiprocess_lock)
    plots.USE_ERRORBARS = True


def ED_figs_1(param_set, train_params, key_list, keys_abbrev, subdir,
            multiprocess_lock=None):
    subdir = subdir_preface/subdir
    param_set = list(param_set)
    train_params = train_params.copy()

    for i0, key in enumerate(key_list):
        train_params[key] = param_set[i0]
    train_params_0 = train_params.copy()
    train_params_0['num_epochs'] = 0
    plist = make_chaos_params_list(train_params) + \
                    make_chaos_params_list(train_params_0)
    figname = ''.join(
        key + '_' + str(val) + '_' for key, val in zip(keys_abbrev, param_set))
    figname = figname[:-1]

    plots.dim_over_layers(plist, seeds,
                          hue_key='g_radius',
                          style_key='num_epochs',
                          style_order=[train_params['num_epochs'], 0],
                          figname='dim_' + figname,
                          subdir=subdir,
                          palette=chaos_palette,
                          multiprocess_lock=multiprocess_lock,
                         )
    plots.clust_holdout_over_layers(plist, seeds,
                          hue_key='g_radius',
                          style_key='num_epochs',
                          style_order=[train_params['num_epochs'], 0],
                          figname='clust_hold_' + figname,
                          subdir=subdir,
                          palette=chaos_palette,
                          multiprocess_lock=multiprocess_lock,
                         )

def run_lowd_chaos(param_set, train_params, key_list, keys_abbrev, subdir,
                   multiprocess_lock=None):
    plots.USE_ERRORBARS = True
    subdir = subdir_preface/subdir
    param_set = list(param_set)
    # time.sleep(i0)
    tps = train_params.copy()
    subdir_prefix = Path('Win_orth_lowd_chaos/')
    for i0, key in enumerate(key_list):
        tps[key] = param_set[i0]

    tp_list_0 = []
    tp_list_1 = []
    for g in range(20, 261, 40):
        tp = tps.copy()
        tp['g_radius'] = g
        tp_list_1.append(tp)
        tp = tp.copy()
        tp['num_epochs'] = 0
        tp_list_0.append(tp)

    figname = ''.join(key + '_' + str(val) + '_' for key, val in
                      zip(keys_abbrev, param_set))
    figname = figname[:-1]
    # seeds = list(range(5))
    plots.dim_over_layers(tp_list_0, seeds=seeds,
                          hue_key='g_radius', style_key='num_epochs',
                          figname=figname + '_before',
                          subdir=subdir, use_error_bars=True,
                          multiprocess_lock=multiprocess_lock,
                          palette='viridis')
    plots.dim_over_layers(tp_list_1, seeds=seeds,
                          hue_key='g_radius', style_key='num_epochs',
                          figname=figname,
                          use_error_bars=True,
                          subdir=subdir, multiprocess_lock=multiprocess_lock,
                          palette='viridis')
    plots.USE_ERRORBARS = False


base_params = high_d_input_edge_of_chaos_params.copy()
base_params['scheduler'] = None
base_params['learning_patience'] = None

# %% Figure 5b parameters
params_fig5b = {
    'network': ['vanilla_rnn'],
    # 'learning_rate': [1e-4, 1e-3],
    'learning_rate': [1e-4],
    'optimizer': ['rmsprop'],
    'loss': ['mse_scalar'],
    'hid_nonlin': ['linear'],
    'l2_regularization': [0],
    'g_radius': [20],
    }
keys_fig5b = list(params_fig5b.keys())
keys_fig5b_abbrev = ['network', 'lr', 'opt', 'loss', 'nonlin', 'l2', 'g']
ps_list_fig5b = list(itertools.product(*params_fig5b.values()))

# %% Figure 5d parameters
params_fig5d = {
    'network': ['vanilla_rnn'],
    'learning_rate': [1e-4],
    'loss': ['mse_scalar'],
    'hid_nonlin': ['linear', 'tanh'],
    'optimizer': ['rmsprop'],
    'g_radius': [20],
    }
keys_fig5d = list(params_fig5d.keys())
keys_abbrev_fig5d = ['network', 'lr', 'loss', 'nonlin', 'opt', 'g']
ps_list_fig5d = list(itertools.product(*params_fig5d.values()))

# %% Figure 6b parameters
params_fig6b_1 = {
    'network': ['vanilla_rnn'],
    'learning_rate': [1e-4],
    'loss': ['mse_scalar'],
    'hid_nonlin': ['linear'],
    'optimizer': ['rmsprop'],
    'dropout_p': [0, 0.05],
    'unit_injected_noise': [0],
    }
params_fig6b_2 = {
    'network': ['vanilla_rnn'],
    'learning_rate': [1e-4],
    'loss': ['mse_scalar'],
    'hid_nonlin': ['linear'],
    'optimizer': ['rmsprop'],
    'dropout_p': [0],
    'unit_injected_noise': [0.05],
    }
keys_fig6b = list(params_fig6b_1.keys())
keys_abbrev_fig6b = ['network', 'lr', 'loss', 'nonlin', 'opt', 'dropout',
                    'noise']
ps_list_fig6b = list(itertools.product(*params_fig6b_1.values())) \
                + list(itertools.product(*params_fig6b_2.values()))

# %% Figure 6c parameters
params_fig6c = {
    'network': ['vanilla_rnn'],
    'learning_rate': [1e-4],
    'loss': ['mse_scalar'],
    'hid_nonlin': ['linear'],
    'optimizer': ['rmsprop'],
    'l2_regularization': [0.1, 1e-4, 1e-5],
    'g_radius': [20],
    }
keys_fig6c = list(params_fig6c.keys())
keys_abbrev_fig6c = ['network', 'lr', 'loss', 'nonlin', 'opt', 'l2', 'g']
ps_list_fig6c = list(itertools.product(*params_fig6c.values()))

params_ED_fig_1_1 = {
    'learning_rate': [1e-3, 1e-4],
    'num_epochs': [12],
    'loss': ['cce'],
    'n_lag': [6, 10, 14],
    'N': [200]
    }
params_ED_fig_1_2 = {
    'learning_rate': [1e-3, 1e-4],
    'num_epochs': [120],
    'loss': ['mse'],
    'n_lag': [6, 10, 14],
    'N': [200]
    }
params_ED_fig_1_3 = {
    'learning_rate': [1e-3, 1e-4],
    'num_epochs': [12],
    'loss': ['cce'],
    'n_lag': [14],
    'N': [300]
    }
params_ED_fig_1_4 = {
    'learning_rate': [1e-3, 1e-4],
    'num_epochs': [120],
    'loss': ['mse'],
    'n_lag': [14],
    'N': [300]
    }
keys_ED_fig_1 = list(params_ED_fig_1_1.keys())
keys_abbrev_ED_fig_1 = keys_ED_fig_1.copy()
ps_list_ED_fig_1 = list(itertools.product(*params_ED_fig_1_1.values())) + \
        list(itertools.product(*params_ED_fig_1_2.values())) + \
        list(itertools.product(*params_ED_fig_1_3.values())) + \
        list(itertools.product(*params_ED_fig_1_4.values()))

params_ED_fig_2_1 = {
    'loss': ['cce'],
    'n_lag': [6, 10, 14],
    'X_dim': [2, 4, 10],
    'learning_rate': [1e-4, 1e-3],
    'X_clusters': [60, 120],
    'N': [200],
    }
params_ED_fig_2_2 = {
    'loss': ['cce'],
    'n_lag': [14],
    'X_dim': [2, 4, 10],
    'learning_rate': [1e-4, 1e-3],
    'X_clusters': [60, 120],
    'N': [300]
    }
keys_ED_fig_2 = list(params_ED_fig_2_1.keys())
ps_list_ED_fig_2 = list(itertools.product(*params_ED_fig_2_1.values())) + \
        list(itertools.product(*params_ED_fig_2_2.values()))
keys_abbrev_ED_fig_2 = keys_ED_fig_2.copy()
params_ED_fig_6 = {
    'learning_rate': [1e-3, 1e-4],
    }
keys_ED_fig_6 = list(params_ED_fig_6.keys())
keys_abbrev_ED_fig_6 = keys_ED_fig_6.copy()
ps_list_ED_fig_6 = list(itertools.product(*params_ED_fig_6.values()))
ps_list_ED_fig_6 = list(itertools.product(*params_ED_fig_6.values()))

params_ED_fig_10 = {
        'learning_rate': [1e-3, 1e-4],
        'X_dim': [2, 200],
        }
keys_ED_fig_10 = list(params_ED_fig_10.keys())
keys_abbrev_ED_fig_10 = keys_ED_fig_10.copy()
ps_list_ED_fig_10 = list(itertools.product(*params_ED_fig_10.values()))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':

    lock = Lock()
    processes = []

    # %% Run Figures 2 through 5a
    processes += [Process(target=fig2, args=(lock,))]
    processes += [Process(target=fig3, args=(lock,))]
    processes += [Process(target=fig4, args=(lock,))]
    processes += [Process(target=fig5a, args=(lock,))]

    # %% Run Figures 5b through 6. These simulations take much longer
    # (possibly days).
    # processes += [Process(target=fig5b_6,
                          # args=(p, base_params, keys_fig5b, keys_fig5b_abbrev,
                                # 'fig5b', lock))
                  # for p in ps_list_fig5b]
    # processes += [Process(target=fig5c, args=(lock,))]
    # processes += [Process(target=fig5d,
                          # args=(p, base_params, keys_fig5d, keys_abbrev_fig5d,
                                # lock))
                  # for p in ps_list_fig5d]
    # processes += [Process(target=fig5b_6, args=(p, base_params,
                                              # keys_fig6b, keys_abbrev_fig6b,
                                              # 'fig6b', lock))
                 # for p in ps_list_fig6b]
    # processes += [Process(target=fig5b_6, args=(p, base_params,
                                                 # keys_fig6c, keys_abbrev_fig6c,
                                                # 'fig6c', lock))
                 # for p in ps_list_fig6c]
    
    # %% Run Extended Data Figures
    # processes += [Process(target=ED_figs_1,
                          # args=(p, high_d_input_edge_of_chaos_params,
                                # keys_ED_fig_1, keys_abbrev_ED_fig_1,
                                # 'ED_fig_1', lock))
                 # for p in ps_list_ED_fig_1]

    # processes += [Process(target=ED_figs_1,
                          # args=(p, low_d_input_strongly_chaotic_params,
                                # keys_ED_fig_2, keys_abbrev_ED_fig_2,
                                # 'ED_fig_2_3_4', lock))
                 # for p in ps_list_ED_fig_2]

    # processes += [Process(target=ED_figs_1,
                          # args=(p, high_d_input_edge_of_chaos_params,
                                # keys_ED_fig_6, keys_abbrev_ED_fig_6,
                                # 'ED_fig_6_Xdim_200', lock))
                 # for p in ps_list_ED_fig_6]
    # processes += [Process(target=ED_figs_1,
                          # args=(p, low_d_input_edge_of_chaos_params,
                                # keys_ED_fig_6, keys_abbrev_ED_fig_6,
                                # 'ED_fig_6_Xdim_2', lock))
                 # for p in ps_list_ED_fig_6]

    # processes += [Process(target=run_lowd_chaos,
                          # args=(p, low_d_input_strongly_chaotic_params,
                                # keys_ED_fig_10, keys_abbrev_ED_fig_10,
                                # 'ED_fig_10', lock))
                 # for p in ps_list_ED_fig_10]

    if n_cores == 1:
        for proc in processes:
            proc.run()
    else:
        n_processes_simul = max(int(n_cores/2), 1)
        chunked_processes = list(chunks(processes, n_processes_simul))
        for i0, process_chunk in enumerate(chunked_processes):
            print(f"Starting batch {i0+1} of {len(chunked_processes)} batches")
            for process in process_chunk:
                time.sleep(.5)
                process.start()
            [process.join() for process in process_chunk]
