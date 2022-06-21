import copy
import inspect
import time
from typing import *
import math
import torch
from torch.utils.data import Dataset
import numpy as np
import classification_task
import model_output_manager as mom
import models
import model_trainer
from multiprocessing.synchronize import Lock
import os

default_out_dir = './output'

def get_max_eigval(A):
    # ew, __ = torch.eig(A)
    ew, __ = torch.linalg.eig(A)
    # mags = torch.sqrt(ew[:, 0]**2 + ew[:, 1]**2)
    mags = ew.abs()
    idx_sort = torch.flip(torch.argsort(mags), dims=[0])
    ew = ew[idx_sort]
    return ew

def nested_tuple_to_str(inp_tuple):
    temp = ''
    for x in inp_tuple:
        for y in x:
            temp = temp + str(y) + '&'
        temp = temp[:-1]
        temp = temp + '_'
    inp_tuple = temp[:-1]
    return inp_tuple

## This is the main method for this module
def initialize_and_train(N, X_clusters, n_lag, n_hold, n_out, X_dim,
                         num_classes, clust_sig=0.1, model_seed=2,
                         hid_nonlin='tanh', num_epochs=20, learning_rate=0.005,
                         patience_before_stopping=None, batch_size=10, loss='cce',
                         optimizer='rmsprop', momentum=0, grad_noise=0,
                         grad_noise_type=None,
                         l2_regularization=0,
                         dropout_p=0, unit_injected_noise=0,
                         scheduler='plateau', learning_patience=5,
                         scheduler_factor=0.5, network='vanilla_rnn',
                         Win='orthog', use_biases=True, Wrec_rand_proportion=1,
                         input_scale=1., wout_scale=1, g_radius=1., dt=0.01,
                         num_train_samples_per_epoch=None,
                         num_test_samples_per_epoch=None, freeze_input=False,
                         train_input_weights=False,
                         train_output_weights=True, input_style='hypercube',
                         saves_per_epoch=1, rerun=False, out_dir=None,
                         multiprocess_lock=None):
    """
    Parameters
    ----------
    N : int
        Number of units in the "hidden" layer, i.e. number of neurons making
        up the recurrent layer.
    X_clusters : int
        Number of clusters.
    n_lag : int
        Number of timesteps from stimulus onset to end of loss evaluation.
    n_hold : int
        Number of timesteps for which the input is presented.
    n_out : int
        Number of timesteps for which loss is evaluated.
    X_dim : int
        Dimension of the ambient space in which clusters are generated.
    num_classes : int
        Number of class labels.
    clust_sig : float
        Standard deviation of each cluster.
    model_seed : int
        Seed for generating input and model weights.
    hid_nonlin : str
        Activation function for the hidden units, or if using a sompolinsky
        style recurrent network the nonlinear
        transfer function.
    num_epochs : int
        The number of epochs to train for.
    learning_rate : float
        Learning rate for optimizer.
    patience_before_stopping : Optional[int]
        Number of consecutive epochs to wait for which there is no
        improvement to the (cumulative average) validation
        loss before ending training.
    batch_size : int    
        Size of each training data minibatch.
    loss : str
        The loss function to use. Options are "mse" for mean squared error
        and "cce" for categorical cross entropy.
    optimizer : str
        The optimizer to use. Options are "sgd" for standard stochastic
        gradient descent and "rmsprop" for RMSProp.
    momentum : float
        Momentum value to give to optimizer. If optimizer is 'adam' then
        momentum is set to 0.
    l2_regularization : float
        Weighting factor for l2 regularization of parameters. Default: 0.
    dropout_p : float
        Probability value for dropout applied to the hidden units of the
        feedforward network or recurrent units at each recurrent timestep.
        Default: 0. If 0, a dropout layer isn't added.
    unit_injected_noise : float
        Magnitude of i.i.d Gaussian noise to inject in each unit of each hidden
        layer or on each recurrent timestep. Default: 0.
    scheduler : Optional[str]
        The strategy used to adjust the learning rate through training.
        Options are None for constant learning rate
        through training, "plateau" for reducing the learning rate by a
        multiplicative factor after a plateau of a
        certain number of epochs, and "steplr" for reducing the learning rate
        by a multiplicative factor. In both
        cases, the number of epochs is specified by scheduler_patience and
        the multiplicative factor by scheduler_factor.
    learning_patience : Optional[int]
        If using plateau scheduler, this is the number of epochs over which
        to measure that a plateau has been reached. If using steplr scheduler,
        this is the number of epochs after which to reduce the learning rate.
        If None, this is set to num_epochs.
    scheduler_factor : Optional[float]
        The multiplicative factor by which to reduce the learning rate.
    network : str
        The type of network architecture to use. Options are "vanilla_rnn"
        for a vanilla RNN, "sompolinsky" for a
        Sompolinsky style RNN, and "feedforward" for a feedforward network.
    Win : str
        Type of input weights to use. Can be 'diagonal_first_two' for feeding
        inputs to only the first two neurons
         in the network or 'orthogonal' for a (truncated) orthogonal matrix.
    Wrec_rand_proportion : float
        The proportion of Wrec that should initially be random. Only applies
        if network is sompolinsky style (
        network='sompolinsky'). Wrec will be initialized as a convex
        combination of a random matrix and an orthogonal
        matrix, weighted by Wrec_rand_proportion.
    input_scale : float
        Global scaling of the inputs.
    wout_scale : float
        Scaling of output weights.
    g_radius : float
        Magnitude of the largest eigenvalue of the random part of the
        recurrent weight matrix. This holds exactly
        (i.e. the random matrix is rescaled so that this is satisfied
        exactly), not just on average.
    dt : float
        Size of the timestep to use for the discretization of the dynamics if
        'network' is an RNN ('vanilla_rnn' or
        'sompolinksy'). If network='vanilla_rnn', the recurrent weight matrix
        will be (1-dt)*I + dt*J where I is the
        identitiy matrix and J is a random matrix. The entries of J are
        i.i.d. normally distributed, and scaled so that
        the largest eigenvalue of J has magnitude equal to g_radius.
    num_train_samples_per_epoch : int
        Number of training samples to use per epoch.
    num_test_samples_per_epoch : int
        Number of testing samples to use per epoch.
    input_style: str
        Input style. Currently 'hypercube' is the only valid option.
    freeze_input: bool
        Whether or not to present the same input every epoch. If False,
        new input samples are drawn every epoch
    saves_per_epoch: Union[int,float,Iterable[int]]
        The number of times model parameters are saved to disk, per epoch.
        If this is a fraction, then multiple epochs will be completed per
        save: the equation is
        saves_per_epoch = round(1/epochs_per_save).
        If this is an iterable (such as a list), then it must have length
        num_epochs. Each entry in the list specifies
        how many saves should be in that epoch. For example, if num_epochs =
        3, then setting saves_per_epoch = [2,0,1]
        will cause the model to be saved twice during epoch 1, not saved
        during epoch 2, and saved once (at the end of)
        epoch 3. The first save (check_0.pt) always corresponds with the
        initial network, the next save is called
        check_1.pt, and so on
    rerun: bool
        Whether or not to run the simulation again even if a matching run is
        found on disk. True means run the
        simulation again. This parameter is not written to the output table.
    out_dir: str
        Directory for output to be stored.
    multiprocess_lock: Optional[Lock]
        A multiprocessing Lock for ensuring that writing to the output table
        in a parallel way doesn't cause conflicts.
        This parameter is not written to the output table

    Returns
    -------
        torch.nn.Module
            The trained network model.
        dict
            A collection of all the (meta) parameters used to specify the
            run. This is basically a dictionary of the
            input arguments to this function.
        str
            The directory where the model parameters over training are stored.
    """
    if (unit_injected_noise or dropout_p) and network != 'vanilla_rnn':
        raise NotImplementedError(
            "Noise injection is only implemented in vanilla_rnn")
    if scheduler is None:
        learning_patience = None
        scheduler_factor = 1
    if learning_patience == None and scheduler is not None:
        learning_patience = num_epochs
    if out_dir is None:
        out_dir = default_out_dir
    if num_test_samples_per_epoch in (None, 'None', 'NA', 'na'):
        num_test_samples_per_epoch = round(.15*num_train_samples_per_epoch)
    if hasattr(saves_per_epoch, '__len__'):
        saves_per_epoch_copy = copy.copy(saves_per_epoch)
        saves_per_epoch = str(
            saves_per_epoch)  # Make a string copy to save to arg_dict below
    network = network.lower()
    loss = loss.lower()
    if scheduler is not None:
        scheduler = scheduler.lower()
    optimizer = optimizer.lower()
    learning_patience_copy = copy.copy(learning_patience)
    if hasattr(learning_patience, '__len__'):
        learning_patience = '_'.join([str(x) for x in learning_patience])
    if optimizer == 'adam':
        momentum = 0

    ## Record the input parameters in a dictionary
    loc = locals()
    args = inspect.getfullargspec(initialize_and_train)[0]
    arg_dict = {arg: loc[arg] for arg in args}
    del arg_dict['out_dir']
    del arg_dict['rerun']
    del arg_dict['multiprocess_lock']

    learning_patience = learning_patience_copy

    ## Redefine parameter options for consistency
    for key, value in arg_dict.items():
        if value in (None, 'None', 'NA'):
            arg_dict[key] = 'na'

    learning_patience = learning_patience_copy
    if isinstance(saves_per_epoch, str):
        saves_per_epoch = saves_per_epoch_copy

    # import pandas as pd
    # print(pd.Series(arg_dict))
    # print(arg_dict)
    ## Initialize Data.
    print('==> Preparing data..')
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    ## Training datasets
    if network == 'feedforward':
        out = classification_task.delayed_mixed_gaussian(
            num_train_samples_per_epoch, num_test_samples_per_epoch, X_dim,
            num_classes, X_clusters, 0, 0, clust_sig,
            cluster_seed=2*model_seed + 1,
            assignment_and_noise_seed=3*model_seed + 13, avg_magn=1,
            freeze_input=freeze_input)
    else:
        out = classification_task.delayed_mixed_gaussian(
            num_train_samples_per_epoch, num_test_samples_per_epoch, X_dim,
            num_classes, X_clusters, n_hold, n_lag, clust_sig,
            cluster_seed=2*model_seed + 1,
            assignment_and_noise_seed=3*model_seed + 13, avg_magn=1,
            freeze_input=freeze_input)

    datasets, centers, cluster_class_label = out
    trainset = datasets['train']
    testset = datasets['val']

    if num_train_samples_per_epoch != 'na':
        subset_indices = range(num_train_samples_per_epoch)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=False, num_workers=1)
    else:
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  num_workers=1, shuffle=False)

    if num_test_samples_per_epoch != 'na':
        subset_indices = range(num_test_samples_per_epoch)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                     subset_indices),
                                                 num_workers=1)
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 num_workers=1, shuffle=False)

    datasets = {'train': trainset, 'val': testset}
    dataloaders = {'train': trainloader, 'val': testloader}

    ## Convenience functions and variable definitions
    def ident(x):
        return x

    def zero_fun():
        return 0

    if hid_nonlin == 'linear':
        nonlin = ident
    elif hid_nonlin == 'tanh':
        nonlin = torch.tanh
    elif hid_nonlin == 'relu':
        nonlin = torch.relu
    else:
        raise ValueError('Unrecognized option for hid_nonlin')

    ## Find requested network model and put model on appropriate device
    if Win in ('identity', 'diagonal_first_two'):
        Win_instance = input_scale*torch.eye(X_dim, N)
    elif Win in ('orth', 'orthogonal', 'orthog'):
        temp = torch.empty(N, X_dim)
        temp = torch.nn.init.orthogonal_(temp)
        temp = temp/torch.mean(torch.abs(temp))
        temp = input_scale*temp/math.sqrt(X_dim)
        Win_instance = temp.T.clone()
    else:
        raise AttributeError("Win option not recognized.")

    if loss == "mse_scalar":
        Wout_instance = wout_scale*torch.randn(N, 1)*(.3/np.sqrt(N))
        bout = torch.zeros(1)
    else:
        Wout_instance = wout_scale*torch.randn(N, num_classes)*(.3/np.sqrt(N))
        bout = torch.zeros(num_classes)

    brec = torch.zeros(N)

    J = torch.randn(N, N)/math.sqrt(N)
    top_ew = get_max_eigval(J)[0]
    top_ew_mag = top_ew.abs()
    J_scaled = g_radius*(J/top_ew_mag)
    # breakpoint()
    # J_scaled = g_radius*J
    if network in ('somp', 'sompolinsky', 'sompolinskyrnn'):
        Q = torch.nn.init.orthogonal_(torch.empty(N, N))
        Q_scaled = g_radius*Q.T.clone()
        Wrec = Wrec_rand_proportion*J_scaled + (1 - Wrec_rand_proportion)*Q
        model = models.SompolinskyRNN(Win_instance, Wrec, Wout_instance, brec,
                                      bout, nonlin, dt=dt,
                                      train_recurrent_bias=use_biases,
                                      train_output_bias=use_biases,
                                      output_over_recurrent_time=True)

    elif network == 'vanilla_rnn':
        Wrec = (1 - dt)*torch.eye(N, N) + dt*J_scaled
        model = models.RNN(Win_instance, Wrec, Wout_instance, brec, bout,
                           nonlin, output_over_recurrent_time=True,
                           train_output=train_output_weights,
                           train_recurrent_bias=use_biases,
                           train_output_bias=use_biases,
                           dropout_p=dropout_p,
                           unit_injected_noise=unit_injected_noise)

    elif network == 'feedforward':
        Wrec = (1 - dt)*torch.eye(N, N) + dt*g_radius*(J/top_ew_mag)
        layer_weights = [Win_instance.clone()]
        biases = [torch.zeros(N)]
        nonlinearities = [nonlin]
        for i0 in range(n_lag + n_hold - 2):
            layer_weights.append(Wrec.clone())
            biases.append(torch.zeros(N))
            nonlinearities.append(nonlin)
        layer_weights.append(Wout_instance.clone())
        if loss == 'mse_scalar':
            biases.append(torch.zeros(1))
        else:
            biases.append(torch.zeros(num_classes))
        nonlinearities.append(ident)
        layer_train = [True]*len(layer_weights)
        bias_train = [use_biases]*len(biases)
        if not train_input_weights:
            layer_train[0] = False
            bias_train[0] = False
        bias_train[-1] = False  # Todo: change this back
        if not train_output_weights:
            layer_train[-1] = False
            bias_train[-1] = False

        model = models.FeedForward(layer_weights, biases, nonlinearities,
                                   layer_train, bias_train)
    else:
        raise AttributeError('Option for net_architecture not recognized.')
    if torch.cuda.device_count() == 2:
        device = [torch.device("cuda:0"), torch.device("cuda:1")]
    elif torch.cuda.device_count() == 1:
        device = [torch.device("cuda:0")]
    else:
        device = [torch.device("cpu")]
    # device = [torch.device("cpu")]
    # device = [torch.device("cuda")]

    print("Using {}".format(device[0]))
    model = model.to(device[0])
    loss_points = torch.arange(n_lag - n_out, n_lag + n_hold - 1)

    ## Initializing loss functions
    if loss in ('categorical_crossentropy', 'cce'):
        loss = torch.nn.CrossEntropyLoss()
        if network == 'feedforward':
            def loss_function(output, label):
                return loss(output, label)
        else:
            def loss_function(output, label):
                return loss(
                    output[:, loss_points].transpose(1, 2),
                    label[:, loss_points])
    elif loss in ('mean_square_error', 'mse'):
        criterion_mse = torch.nn.MSELoss()

        def criterion_single_timepoint(output,
                                       label):  # The output does not have a
            # time dimension
            label_onehot = torch.zeros(label.shape[0], num_classes,
                                       device=output.device)
            for i0 in range(num_classes):
                label_onehot[label == i0, i0] = 1
            return criterion_mse(output, .7*label_onehot)

        if network == 'feedforward':
            def loss_function(output, label):
                return criterion_single_timepoint(output, label)
        else:
            def loss_function(output, label):
                cum_loss = 0
                for i0 in loss_points:
                    cum_loss += criterion_single_timepoint(output[:, i0],
                                                           label[:, i0])
                cum_loss = cum_loss/len(loss_points)
                return cum_loss
    elif loss == 'mse_scalar':
        criterion_mse = torch.nn.MSELoss()
        if network == 'feedforward':
            def loss_function(output, label):
                label_t = .7*(2.*label - 1.)
                return criterion_mse(output.flatten(), label_t.flatten())
        else:
            def loss_function(output, label):
                label = .7*(2.*label - 1.)
                # raise AttributeError("scalar loss with recurrent network
                # needs to be checked.")
                crit = criterion_mse(
                    output[:, loss_points].transpose(1, 2).flatten(),
                    label[:, loss_points].flatten())
                return crit
    elif loss == 'zero':
        def loss_function(output, label):
            return 0
    else:
        raise AttributeError("loss option not recognized.")
    
    if network == 'feedforward':
        if grad_noise_type == 'output':
            noise_p_idx = [n_lag + n_hold - 1]
        else:
            noise_p_idx = list(range(n_lag + n_hold))
    else:
        if grad_noise_type == 'output':
            noise_p_idx = [2]
        else:
            noise_p_idx = list(range(3))
    if grad_noise_type is not None and grad_noise_type[-6:] == 'scaled':
        scale = True
    else:
        scale = False

    ## Initialize optimizer and learning scheduler
    if optimizer == 'sgd':
        if grad_noise > 0:
            optimizer_instance = models.NoisySGD(
                # filter(lambda p: p.requires_grad, model.parameters()),
                model.parameters(),
                lr=learning_rate, momentum=momentum,
                weight_decay=l2_regularization, grad_noise=grad_noise,
                noise_p_idx=noise_p_idx, scale_noise=scale)
        else:
            optimizer_instance = torch.optim.SGD(
                # filter(lambda p: p.requires_grad, model.parameters()),
                model.parameters(),
                lr=learning_rate, momentum=momentum,
                weight_decay=l2_regularization)
    elif optimizer == 'rmsprop':
        # noinspection PyUnresolvedReferences
        if grad_noise > 0:
            optimizer_instance = models.NoisyRMSprop(
                # filter(lambda p: p.requires_grad, model.parameters()),
                model.parameters(),
                lr=learning_rate, alpha=0.9, momentum=momentum,
                weight_decay=l2_regularization, grad_noise=grad_noise,
                noise_p_idx=noise_p_idx, scale_noise=scale)
        else:
            # optimizer_instance = torch.optim.RMSprop(
                # filter(lambda p: p.requires_grad, model.parameters()),
                # # model.parameters(),
                # lr=learning_rate, alpha=0.9, momentum=momentum,
                # weight_decay=l2_regularization)
            optimizer_instance = models.RMSprop(
                filter(lambda p: p.requires_grad, model.parameters()),
                # model.parameters(),
                lr=learning_rate, alpha=0.9, momentum=momentum,
                weight_decay=l2_regularization)
        
    elif optimizer[:17] == 'sgd_rmsprop_comb_': # Todo: integrate grad noise with this
        factor = float(optimizer[17:])
        optimizer_instance = models.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            # model.parameters(),
            lr=learning_rate, alpha=0.9, momentum=momentum,
            weight_decay=l2_regularization, prop=factor)
    elif optimizer == 'adam':
        # noinspection PyUnresolvedReferences
        if grad_noise > 0.0 or noise_p_idx is not None:
            raise NotImplementedError
        optimizer_instance = torch.optim.Adam(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=learning_rate, weight_decay=l2_regularization)
    else:
        raise AttributeError('optimizer option not recognized.')
    if scheduler is None:
        learning_scheduler_instance = None
    elif scheduler == 'plateau':
        learning_scheduler_instance = model_trainer.ReduceLROnPlateau(
            optimizer_instance, factor=scheduler_factor,
            patience=learning_patience, threshold=1e-7, threshold_mode='abs',
            min_lr=0, verbose=False)
    elif scheduler == 'steplr':
        learning_scheduler_instance = model_trainer.StepLR(optimizer_instance,
                                                           step_size=learning_patience,
                                                           gamma=scheduler_factor)
    elif scheduler == 'multisteplr':
        learning_scheduler_instance = model_trainer.MultiStepLR(
            optimizer_instance, learning_patience, scheduler_factor)
    elif scheduler == 'cyclic':
        learning_scheduler_instance = model_trainer.CyclicLR(optimizer_instance,
                                                             max_lr=learning_rate,
                                                             base_lr=scheduler_factor*learning_rate,
                                                             step_size_up=learning_patience,
                                                             cycle_momentum=False)
    elif scheduler == 'cyclic_halving':
        learning_scheduler_instance = model_trainer.CyclicLR(optimizer_instance,
                                                             mode='triangular2',
                                                             max_lr=learning_rate,
                                                             base_lr=scheduler_factor*learning_rate,
                                                             step_size_up=learning_patience,
                                                             cycle_momentum=False)
    elif 'onecyclelr' in scheduler:
        if '_' in scheduler:
            final_div_factor = float(scheduler.split('_')[-1])
        else:
            final_div_factor = 1e4
        pct_start = learning_patience/patience_before_stopping
        learning_scheduler_instance = model_trainer.OneCycleLR(
            optimizer_instance, total_steps=patience_before_stopping,
            pct_start=pct_start, max_lr=learning_rate,
            div_factor=scheduler_factor, final_div_factor=final_div_factor,
            cycle_momentum=False)
    else:
        raise AttributeError('scheduler option not recognized.')

    ## Determine if the training needs to be run again or if it can be loaded
    # from disk
    verbose = True
    if isinstance(multiprocess_lock, Lock):
        multiprocess_lock.acquire()
        print("Locking output table.")
        # verbose = False
        verbose = True
    else:
        print("No locking")
    arg_dict_mom = arg_dict.copy()
    del arg_dict_mom['num_epochs']
    run_id = mom.get_run_entry(arg_dict_mom, out_dir)
    run_dir = f"{out_dir}/run_{run_id}/"
    # if mom.run_exists(arg_dict_mom, table_path):
        # run_id = mom.get_run_entry(arg_dict_mom, table_path)
        # run_dir = Path(f"{table_path}/run_{run_id}/")
    # else:
        # run_id = 

    # dirs, ids, output_exists = mom.get_dirs_and_ids_for_run(arg_dict,
                                                            # table_path,
                                                            # ['num_epochs'],
                                                            # maximize='num_epochs')
    # breakpoint()
    # if len(dirs) == 0:
        # run_id, run_dir = mom.make_dir_for_run(arg_dict, table_path)
    # else:
        # run_id = ids[0]
        # run_dir = dirs[0]
    if isinstance(multiprocess_lock, Lock):
        time.sleep(.5)
        print("Releasing output table.")
        multiprocess_lock.release()
    ## Now train the model (if necessary)
    saves_per_epoch_is_number = not hasattr(saves_per_epoch, '__len__')
    batches_per_epoch = len(trainloader)
    if saves_per_epoch_is_number and saves_per_epoch > 1:
        mod_factor = int(math.ceil((batches_per_epoch - 1)/saves_per_epoch))
        print(mod_factor)

        def save_model_criterion(stat_dict):
            return stat_dict['batch']%mod_factor == 0
    elif saves_per_epoch_is_number:
        epochs_per_save = round(1/saves_per_epoch)

        def save_model_criterion(stat_dict):
            save_epoch = stat_dict['epoch']%epochs_per_save == 0
            return stat_dict['epoch_end'] and save_epoch
    else:
        def save_model_criterion(stat_dict):
            saves_this_epoch = saves_per_epoch[stat_dict['epoch']]
            if saves_this_epoch == 1 and stat_dict['epoch_end']:
                return True
            elif saves_this_epoch > 1:
                mod_factor = int(
                    math.ceil((batches_per_epoch - 1)/saves_this_epoch))
                return stat_dict['batch']%mod_factor == 0
            else:
                return False

    print('\n==> Training/loading network')
    load_prev = not rerun
    model_backup = copy.deepcopy(model)
    # This modifies model by reference
    try: # To get around a parallelization bug. This is probably problematic in general.
        model_trainer.train_model(model, dataloaders, device[0], loss_function,
                                  optimizer_instance, num_epochs, run_dir,
                                  load_prev,
                                  learning_scheduler=learning_scheduler_instance,
                                  save_model_criterion=save_model_criterion,
                                  verbose=verbose)
    except RuntimeError:
        print("============")
        print("============")
        print("============")
        print("Warning: error encountered when loading/running model.")
        print("Attempting to retrain model from scratch.")
        print("============")
        print("============")
        print("============")
        load_prev = False
        model = model_backup
        model_trainer.train_model(model, dataloaders, device[0], loss_function,
                                  optimizer_instance, num_epochs, run_dir,
                                  load_prev,
                                  learning_scheduler=learning_scheduler_instance,
                                  save_model_criterion=save_model_criterion,
                                  verbose=verbose)
    params = dict(dataloaders=dataloaders, datasets=datasets)
    loss_function_info = dict(loss_function=loss_function)
    params.update(arg_dict)
    params.update(loss_function_info)
    return model, params, run_dir

if __name__ == '__main__':
    # out=initialize_and_train(
        # N=100, X_clusters=60, n_lag=1, n_hold=1, n_out=1, X_dim=20,
        # num_train_samples_per_epoch=800, num_epochs=2,
        # num_classes=2, network='vanilla_rnn', grad_noise=0.1,
        # grad_noise_type='output')
    out=initialize_and_train(
        N=100, X_clusters=60, n_lag=1, n_hold=1, n_out=1, X_dim=20,
        num_train_samples_per_epoch=800, num_epochs=2,
        num_classes=2, network='vanilla_rnn')
    out=initialize_and_train(
        N=100, X_clusters=60, n_lag=1, n_hold=1, n_out=1, X_dim=20,
        num_train_samples_per_epoch=800, num_epochs=4,
        num_classes=2, network='vanilla_rnn')
    print("Done.")

