import sys
import time
from typing import Callable, Union, Dict, Optional
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
# from pdb import set_trace as stp
import model_loader_utils
import models
import utils

DISABLE_CHECKPOINTS = False

def save_checkpoint(state, is_best=True, filename: Union[str, Path] = 'output/checkpoint.pth.tar'):
    """
    Save checkpoint if a new best is achieved. This will always create the
    checkpoint directory as if it were going to create the file, but only
    actually creates and saves the file if is_best is true.

    This also prints some feedback to the standard output stream about
    if the model was saved or not.

    Parameters
    ----------
    state : dict
        the dictionary to serialize and save
    is_best : bool
        true to save the dictionary to file, false just to create the directory
    filename : str
        a path to where the checkpoint file should be saved
    """
    if DISABLE_CHECKPOINTS:
        return
    filename = Path(filename)
    filedir = filename.parents[0]
    Path.mkdir(filedir, parents=True, exist_ok=True)
    if is_best:
        print("=> Saving model to {}".format(filename))
        torch.save(state, str(filename), pickle_protocol=4)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")


class DefaultStatsTracker:
    def __init__(self, accuracy: bool = True, verbose: bool = True):
        self.batch_losses = {'train': [], 'val': []}
        self.epoch_losses = {'train': [], 'val': []}
        self.batch_accs = {'train': [], 'val': []}
        self.epoch_accs = {'train': [], 'val': []}
        self.accuracy = accuracy
        self.verbose = verbose

    def __call__(self, stat_dict):
        phase = stat_dict['current_phase']
        loss = stat_dict['loss']
        self.batch_losses[phase].append(loss)
        if self.accuracy:
            if len(stat_dict['outputs'].shape) > 2:
                out_class = torch.argmax(stat_dict['outputs'][:, -1].detach(), dim=1)
                accuracy = torch.mean((out_class == stat_dict['targets'][:, -1]).double()).item()
            else:
                if stat_dict['outputs'].shape[-1] == 1:
                    out_class = stat_dict['outputs'] > 0
                    accuracy = torch.mean((out_class == stat_dict['targets']).double()).item()
                else:
                    out_class = torch.argmax(stat_dict['outputs'].detach(), dim=1)
                    accuracy = torch.mean((out_class == stat_dict['targets']).double()).item()
            self.batch_accs[phase].append(accuracy)

        if stat_dict['epoch_end'] and stat_dict['epoch']>0:  # We've reached the end of an epoch
            epoch_loss = torch.mean(torch.tensor(self.batch_losses[phase])).item()
            self.epoch_losses[phase].append(epoch_loss)
            epoch_acc = torch.mean(torch.tensor(self.batch_accs[phase])).item()
            self.epoch_accs[phase].append(epoch_acc)
            # print()
            if self.verbose:
                print(f"Average {phase} loss over this epoch: {epoch_loss}")
            if self.accuracy and self.verbose:
                print(f"Average {phase} accuracy over this epoch: {epoch_acc}")
            self.batch_losses[phase] = []
            self.batch_accs[phase] = []

    def export_stats(self):
        out_stats = dict(epoch_losses=self.epoch_losses, epoch_accuracies=self.epoch_accs)
        return out_stats


# %% Learning rate schedulers.
class LRSchedulerTemplate:
    def __init__(self, optimizer, **kwargs):
        pass

    def __call__(self, stats_dict):
        pass

    def state_dict(self):
        pass


class StepLR(LRSchedulerTemplate):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.torch_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)

    def __call__(self, stats_dict):
        if phase == 'val' and stats_dict['epoch_end']:
            self.torch_lr_scheduler.step()

    def state_dict(self):
        return self.torch_lr_scheduler.state_dict()


class CyclicLR(LRSchedulerTemplate):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.torch_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, **kwargs)

    def __call__(self, stats_dict):
        if stats_dict['current_phase'] == 'val':
            if stats_dict['batch'] == stats_dict['num_batches']-1:
                self.torch_lr_scheduler.step()

    def state_dict(self):
        return self.torch_lr_scheduler.state_dict()


class OneCycleLR(LRSchedulerTemplate):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.torch_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, **kwargs)

    def __call__(self, stats_dict):
        if stats_dict['current_phase'] == 'val':
            if stats_dict['batch'] == stats_dict['num_batches']-1:
                self.torch_lr_scheduler.step()

    def state_dict(self):
        return self.torch_lr_scheduler.state_dict()

class ReduceLROnPlateau(LRSchedulerTemplate):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.batch_losses: list = []
        self.torch_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    def __call__(self, stats_dict):
        if stats_dict['current_phase'] == 'val':
            self.batch_losses.append(stats_dict['loss'])
            if stats_dict['batch'] == stats_dict['num_batches']-1:
                avg_loss = torch.mean(torch.tensor(self.batch_losses)).item()
                self.torch_lr_scheduler.step(avg_loss)
                # print(self.torch_lr_scheduler.num_bad_epochs)
                self.batch_losses = []


    def state_dict(self):
        return self.torch_lr_scheduler.state_dict()


def default_save_model_criterion(stat_dict):
    return stat_dict['epoch_end']


def default_stopping_criterion(stat_dict):
    return False


def train_model(model, dataloaders, device, loss_function, optimizer, stopping_epoch=5, out_dir=None, load_prev=True,
                learning_scheduler=None, save_model_criterion=None, stopping_criterion=None, stats_tracker=None,
                verbose=True):
    """

    Parameters
    ----------
    model : nn.Module
        The model to be trained. NOTE: This is modified by reference!
    dataloaders : Dict[str, DataLoader]
        A dictionary with keys "train" and "val". The Dataset underlying the DataLoaders should return a tuple with
        first entry being the input batch and second entry being the output labels.
    device : Union[str, torch.device]
        Device to use for running the network, for instance 'cpu' or 'cuda'
    loss_function : Callable[[torch.Tensor, torch.Tensor], float]
        A function that takes the model output to an input batch drawn from "dataloaders" as the first parameter
        and the corresponding output labels as the second. This looks like loss = loss_function(outputs, targets)
    optimizer : Optimizer
        An instantiation of a class that inherets from a torch Optimizer object. Used to train the model. Examples
        are instantiations of torch.optim.SGD and of torch.optim.RMSprop
    stopping_epoch : int
        The stopping epoch. The number of training samples used in an epoch is defined by the length of the training
        dataset as contained in dataloaders, which is len(dataloaders['train'])
    out_dir : Optional[str, Path]
        The output directory in which to save the model parameters through training.
    load_prev : Union[bool, int]
        If True, check for a model that's already trained in out_dir, and load the most recent epoch. If an int, load
        epoch load_prev (epoch 0 means before training starts). If False, retrain model from epoch 0. If there are
        multiple saves per epoch, this only loads the save the corresponds with the end of an epoch (for instance,
        epoch_1_save_0.pt is the save at the end of epoch 0, so beginning of epoch 1).
    stats_tracker : Union[None, Callable, str]
        Object for tracking the statistics of the model over training.
    learning_scheduler : object
        An obect that takes in a dictionary as first argument and phase as second argument. It can, for instance,
        call an instantiation of a torch scheduler object, like those found in torch.optim.lr_scheduler, based on
        the values of the items in the dictionary.
    save_model_criterion : Optional[Callable[[Dict[int, float]], bool]] = None
        An Optional Callable that takes in the statistics of the run as defined by a dictionary and returns True if the
        model should be saved. The input dictionary has keys 'training_loss', 'validation_loss', 'training_accuracy',
        'validation_accuracy', 'training_loss_batch', 'validation_loss_batch', 'training_accuracy_batch',
        'validation_accuracy_batch', 'batch', and 'epoch'. If None, the model is saved after every epoch.
    stopping_criterion : Optional[Callable[[Dict[int, float]], bool]]
        NOT CURRENTLY IMPLEMENTED. A Callable controlling early stopping. Takes in the statistics of the run as defined
        by a dictionary and returns True if training should stop. The input dictionary has keys 'training_loss',
        'validation_loss', 'training_accuracy', 'validation_accuracy', 'training_loss_batch', 'validation_loss_batch',
        'training_accuracy_batch', 'validation_accuracy_batch', 'batch', and 'epoch'.
    verbose : bool
        Whether or not to print simulation progress.

    Returns
    -------
    Dict
        A dictionary that holds the statistics of the run through training, as generated by stats_tracker. At the end of
        training, the return value of stats_tracker.export_stats() is returned.

    """
    out_dir = Path(out_dir)
    def vprint(print_str=None):
        if verbose:
            if print_str is None:
                print()
            else:
                print(print_str)

    since = time.time()

    if isinstance(load_prev, bool):
        if load_prev:
            # vprint("Loading previous model.")
            most_recent_epoch = model_loader_utils.get_max_epoch(out_dir)
            if most_recent_epoch is not False:
                starting_epoch = min(most_recent_epoch, stopping_epoch)
                model_loader_utils.load_model_from_epoch_and_dir(model, out_dir, starting_epoch)
            else:
                starting_epoch = 0
        else:
            starting_epoch = 0
    elif isinstance(load_prev, int):
        vprint("Loading previous model.")
        check = model_loader_utils.load_model_from_epoch_and_dir(model, out_dir, load_prev)
        if check == -1:
            starting_epoch = 0
        else:
            starting_epoch = load_prev
    else:
        starting_epoch = 0

    model.eval()

    if learning_scheduler is None:
        def learning_scheduler(stat_dict):
            pass
        learning_scheduler.state_dict = lambda : None

    if save_model_criterion is None:
        save_model_criterion = default_save_model_criterion
    if stopping_criterion is None:
        stopping_criterion = default_stopping_criterion

    if stats_tracker is None:
        stats_tracker = DefaultStatsTracker(True, verbose)

    stat_keys = ['loss', 'batch', 'epoch', 'epoch_end', 'outputs', 'labels', 'current_phase']

    stat_dict = {x: None for x in stat_keys}
    stat_dict['final_epoch'] = False
    if out_dir is not None:
        Path.mkdir(out_dir, exist_ok=True)

    def train(epoch):
        save_ctr = 0
        stat_dict['epoch'] = epoch
        phase = 'train'
        stat_dict['current_phase'] = phase
        model.train()
        for batch_num, (inputs, targets) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()  # zero the parameter gradients
            model.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            if torch.isnan(loss).item() or torch.isinf(loss).item():
                print("NoN or inf encountered during training. Aborting.")
                # return stats_tracker.export_stats()
                sys.exit()
            try:
                loss_val = loss.item()
            except AttributeError:
                loss_val = loss

            stat_dict['loss'] = loss_val
            stat_dict['batch'] = batch_num
            stat_dict['num_batches'] = len(dataloaders[phase])
            stat_dict['outputs'] = outputs
            stat_dict['targets'] = targets
            is_checkpoint = save_model_criterion(stat_dict)
            stat_dict['checkpoint'] = is_checkpoint

            if is_checkpoint and out_dir is not None:
                filename = out_dir / f'epoch_{epoch}_save_{save_ctr}.pt'
                ls_sd = learning_scheduler.state_dict()
                save_checkpoint({'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 'learning_scheduler_state_dict': ls_sd},
                                filename=filename)
                save_ctr += 1

            stats_tracker(stat_dict)

            if loss_val > 0:
                loss.backward()
                optimizer.step()

            learning_scheduler(stat_dict)

            stat_dict['epoch_end'] = False

    def validate(epoch):
        vprint()
        vprint('Validation')
        vprint('-' * 10)
        stat_dict['epoch'] = epoch
        phase = 'val'
        stat_dict['current_phase'] = phase
        model.eval()
        with torch.no_grad():
            for batch_num, (inputs, targets) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # forward
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss_val = loss.item()
                stat_dict['loss'] = loss_val
                stat_dict['batch'] = batch_num
                stat_dict['num_batches'] = len(dataloaders[phase])
                stat_dict['outputs'] = outputs
                stat_dict['targets'] = targets
                is_checkpoint = save_model_criterion(stat_dict)
                stat_dict['checkpoint'] = is_checkpoint

                learning_scheduler(stat_dict)
                stats_tracker(stat_dict)
                stat_dict['epoch_end'] = False

    # print(optimizer.param_groups[0]['lr'])
    for epoch in range(starting_epoch, stopping_epoch):
        tic = time.time()
        vprint()
        vprint(f'Epoch {epoch+1}/{stopping_epoch}')
        vprint('-' * 10)
        stat_dict['epoch_end'] = True
        train(epoch)
        stat_dict['epoch_end'] = True
        validate(epoch)
        toc = time.time()
        vprint(f'Elapsed time this epoch: {round(toc - tic, 1)} seconds')
        print(optimizer.param_groups[0]['lr'])

    if starting_epoch < stopping_epoch:
        filename = out_dir / f'epoch_{stopping_epoch}_save_{0}.pt'  # End of the last epoch
        ls_sd = learning_scheduler.state_dict()
        save_checkpoint({'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'learning_scheduler_state_dict': ls_sd},
                        filename=filename)

        #
        time_elapsed = time.time() - since
        vprint('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    elif stopping_epoch == 0:
        filename = out_dir / f'epoch_0_save_{0}.pt'  # End of the last epoch
        ls_sd = learning_scheduler.state_dict()
        save_checkpoint({'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'learning_scheduler_state_dict': ls_sd},
                        filename=filename)

        #
        time_elapsed = time.time() - since
        vprint('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    else:
        vprint(f'Training previously complete -- loading save from {out_dir}')
    return stats_tracker.export_stats()
