import torch
from pathlib import Path
import model_output_manager as mom

import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

def get_epoch_and_save_from_savestr(savestr):
    savestr = str(savestr)
    parts = savestr.split('_')
    epoch = int(parts[1])
    save = int(parts[3].split('.')[0])
    return epoch, save

def load_model(model, filename, optimizer=None, learning_scheduler=None):
    """
    Load the given torch model from the given file. Loads the model by reference, so the passed in model is modified.
    An optimizer and learning_scheduler object can also be loaded if they were saved along with the model.
    This can, for instance, allow training to resume with the same optimizer and learning_scheduler state.

    Parameters
    ----------
        model : torch.nn.Module
            The pytorch module which should have its state loaded. WARNING: The parameters of this model will be
            modified.
        filename: Union[str, Path]
            The file that contains the state of the model
        optimizer: torch.optim.Optimizer
            The pytorch optimizer object which should have its state loaded. WARNING: The state of this object will be
            modified.
        optimizer: Union[torch.optim._LRScheduler, object]
            The pytorch learning rate scheduler object which should have its state loaded. WARNING: The state of this
            object will be
            modified.

    Returns
    ----------
        Optional[int]
        Model is loaded by reference, so nothing is returned if loading is successful. If the file to load isn't found,
        returns -1.
    """
    try:
        model_state_info = torch.load(filename)
    except FileNotFoundError:
        return -1
    model.load_state_dict(model_state_info['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(model_state_info['optimizer_state_dict'])
    if learning_scheduler is not None:
        learning_scheduler.load_state_dict(model_state_info['learning_scheduler_state_dict'])

def get_epochs_and_saves(out_dir):
    """
    Get a list of all the epochs and saves in the current directory.

    Parameters
    ----------
    out_dir: string
        the path to the directory

    Returns
    -------
    list:
        list of epochs
    list[list]:
        list of saves for each epoch
    """
    out_dir = Path(out_dir)
    ps = Path(out_dir).glob('epoch_*_save_*.pt')
    epochs = sorted(list({int(str(p.parts[-1]).split('_')[1]) for p in ps}))
    saves = []
    for epoch in epochs:
        ps = Path(out_dir).glob(f'epoch_{epoch}_save_*.pt')
        saves.append(sorted([int(str(p.parts[-1]).split('_')[-1].split('.')[0]) for p in ps]))
    return epochs, saves

def get_max_epoch_and_save(out_dir):
    """
    Get the maximum epoch and save in save directory

    Parameters
    ----------
    out_dir: string
        the path to the directory

    Returns
    -------
    int
        the maximum checknum that is saved in the specified directory.
    """
    out_dir = Path(out_dir)
    ps = list(Path(out_dir).glob('epoch_*_save_*.pt'))
    # import ipdb; ipdb.set_trace()
    epochs = [get_epoch_and_save_from_savestr(str(p.parts[-1]))[0] for p in ps]
    saves = [get_epoch_and_save_from_savestr(str(p.parts[-1]))[1] for p in ps]
    max_epoch_id = torch.argmax(torch.tensor(epochs)).item()
    max_epoch = epochs[max_epoch_id]
    saves_for_max_epoch = [saves[k] for k in range(len(saves)) if epochs[k] == max_epoch]
    max_save = max(saves_for_max_epoch)
    return max_epoch, max_save

def get_max_epoch(out_dir, require_save_zero=True):
    """
    Get the maximum epoch in save directory

    Parameters
    ----------
    out_dir: string
        the path to the directory

    Returns
    -------
    int
        the maximum checknum that is saved in the specified directory.
    """
    out_dir = Path(out_dir)
    if require_save_zero:
        ps = Path(out_dir).glob('epoch_*_save_0.pt')
    else:
        ps = Path(out_dir).glob('epoch_*_save_*.pt')
    epochs = [get_epoch_and_save_from_savestr(str(p.parts[-1]))[0] for p in ps]
    if len(epochs) > 0:
        return max(epochs)
    else:
        return False

def load_model_from_epoch_and_dir(model, out_dir, epoch_num, save_num=0, optimizer=None, learning_scheduler=None):
    """
    Loads the model as it was on a specific epoch. Loads the model by reference, so the passed in model is modified.

    Parameters
    ----------
    model: torch.nn.Module
        Instantiation of the model which should have its state loaded
    out_dir: Union[str,Path]
        The path to the directory the model's states over training were saved in
    epoch_num: int
        the epoch to load, or -1 to load the max epoch.

    Returns
    -------
    Optional[int]
        Model is loaded by reference, so nothing is returned if loading is successful. If the file to load isn't found,
        returns -1.

    """
    out_dir = Path(out_dir)
    if epoch_num == -1:
        epoch_num = get_max_epoch(out_dir)
    filename = out_dir/f'epoch_{epoch_num}_save_{save_num}.pt'
    return load_model(model, filename, optimizer, learning_scheduler)
    # return load_model(model, filename, optimizer, learning_scheduler)

def load_model_mom(model, epoch, arg_dict, table_path, compare_exclude=[], optimizer=None, learning_scheduler=None):
    """
    Load a specific model using the model output manager paradigm. This uses
    the model output manager to locate/create the folder containing the run,
    and then acts equivalently to load_model_from_epoch_and_dir.

    Args:
        model: (torch.nn.Module) the neural network model which should have its state loaded
        epoch: (number) the epoch to load or -1 for the latest epoch
        arg_dict:
            (dict) the row which should be saved to file. The file will be generated if
            this is the first row.
        table_path:
            (string) the path to where the file containing the table that the row should
            be added to
        run_name: (string) the name of the run

    Returns
    -------
    None
    """
    run_dir = mom.get_dirs_and_ids_for_run(arg_dict, table_path, compare_exclude)[0]
    if epoch == -1:
        epoch = get_max_epoch(run_dir)
    # Todo: use more sophisticated way of choosing the best directory
    return load_model_from_epoch_and_dir(model, run_dir[-1], epoch, 0, optimizer, learning_scheduler)

class model_loader:
    def __init__(self, model_template, run_dir):
        # self.run_dir = format_dir(run_dir)
        self.num_epochs = get_max_epoch(self.run_dir)
        self.model_template = model_template

    def __len__(self):
        return self.num_epochs + 1

    def __getitem__(self, idx):
        n_idx = torch.arange(self.num_epochs + 1)[idx]
        if not hasattr(n_idx, '__len__'):
            n_idx = [n_idx]
        models = []
        for el0 in n_idx:
            models.append(load_model_from_epoch_and_dir(self.model_template, self.run_dir, el0)[0])

        return models