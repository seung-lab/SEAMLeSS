"""Provides abstractions for saving and restoring training states

A `ModelArchive` is an abstaction for an underlying filesystem structure
that saves all the relevant information about a trained model.

It records and archives
    - the model weights
    - the model architecture
    - the state of the optimizer
    - the model's training history
    - the training parameters used
    - the state of the pseudorandom generators used
    - the loss curves
    - debugging outputs

The archive is intended to be explicit enough that
    (a) the training results can be reproduced if desired
    (b) if training is stopped or interrupted at any point, the
        state can be read back from it and training can continue
        almost as if it was never interrupted

Usage:
    Create a new model archive and save the current training state:
    >>> mymodel = ModelArchive('mymodel_v01', readonly=False)
    (some training code ...)
    >>> mymodel.log(loss)
    >>> mymodel.update()  # save the updated state to disk

    Load an existing trained model and run it on data:
    >>> existing_archive = ModelArchive('existing_archive', readonly=True)
    >>> net = existing_archive.model
    >>> output = net(data)

    Create a new model archive from an existing one:
    >>> old_model = ModelArchive('old_model')
    >>> new_model = old_model.start_new('new_model_v01')
    (some training code ...)
    >>> new_model.update()  # save the updated state of the new model to disk
"""
import sys
import warnings
import subprocess
import random
import torch
import numpy as np
import json
import datetime
from pathlib import Path

from helpers import copy

git_root = Path(subprocess.check_output('git rev-parse --show-toplevel'
                                        .split()).strip().decode("utf-8"))
models_location = git_root / Path('models/')


class ModelArchive(object):
    """
    Abstraction for the maintainence of trained model archives
    """

    @classmethod
    def model_exists(cls, name):
        """
        Returns whether a trained model of this name exists
        """
        cls._check_name(name)
        return (models_location / name).is_dir()

    @classmethod
    def _check_name(cls, name):
        """
        Checks a proposed name for formatting irregularities.
        Checking this prevents accidentally writing to arbitrary
        file locations.
        """
        if not len(name):
            raise ValueError('Model name must have non-zero length.')
        if not name.replace('_', '').isalnum():
            raise ValueError('Malformated name: {}\n'
                             'Model name can only contain alphanumeric '
                             'characters and underscores _.'.format(name))

    def __init__(self, name, readonly=True, *args, **kwargs):
        self._check_name(name)  # check name formatting
        self.name = name
        self.readonly = readonly
        self.directory = models_location / self.name
        self.intermediate_models = self.directory / 'intermediate_models/'
        self.debug_outputs = self.directory / 'debug_outputs/'
        self.paths = {
            # the model's trained weights
            'weights': self.directory / 'weights.pt',
            # the state of the optimizer
            'optimizer': self.directory / 'optimizer.pt',
            # the state of the pseudorandom number gerenrators
            'prand': self.directory / 'prand.pt',
            # other paths are added below
        }
        for filename in [
            'loss.csv',
            'command.txt',
            'plan.txt',
            'history.log',  # nets it was fine_tuned from
            'progress.log',
            'seed.txt',
            'architecture.py',
            'commit.txt',
            'state_vars.json',
        ]:
            key = filename.split('.')[0]
            self.paths[key] = self.directory / filename
        self._model = None
        self._optimizer = None

        if ModelArchive.model_exists(name):
            self._load(*args, **kwargs)
        elif not self.readonly:
            self._create(*args, **kwargs)
        else:
            raise ValueError('Could not find a trained model named "{}".\n'
                             'If the intention was to create one, use '
                             '`ModelArchive("{}", readonly=False)`.'
                             .format(name, name))

    def _load(self, *args, **kwargs):
        if not self.readonly:
            print('Writing to exisiting model archive: {}'.format(self.name))
        else:
            print('Reading from exisiting model archive: {}'.format(self.name))
        assert self.directory.is_dir() and self.paths['commit'].exists()

        # check for matching commits
        # this can prevent errors arising from working on the wrong git branch
        saved_commit = self.commit
        current_commit = subprocess.check_output('git rev-parse HEAD'
                                                 .split()).strip()
        if int(saved_commit, 16) != int(current_commit, 16):
            print('Warning: The repository has changed since this '
                  'net was last trained.')
            if not self.readonly:
                print('Continuing may overwrite the archive by '
                      'running the new code. If this was the intent, '
                      'then it might not be a problem.'
                      '\nIf not, exit the process and return to the '
                      'old commit by running `git checkout {}`'
                      '\nDo you wish to proceed?  [y/N]'
                      .format(saved_commit))
                if input().lower() not in {'yes', 'y'}:
                    print('Exiting')
                    sys.exit()
                print('OK, proceeding...')

        # load the model, optimizer, and pseudorandom number generator
        self._load_model(*args, **kwargs)
        self._load_optimizer(*args, **kwargs)
        self._load_prand(self, *args, **kwargs)

    def _create(self, no_optimizer=False, *args, **kwargs):
        print('Creating a new model archive: {}'.format(self.name))

        # create directories
        self.directory.mkdir()
        self.intermediate_models.mkdir()
        self.debug_outputs.mkdir()

        # create archive files
        for filename in [
            'loss.csv',
            'command.txt',
            'plan.txt',
            'history.log',  # nets it was fine_tuned from
            'progress.log',
            'seed.txt',
            'commit.txt',
            'state_vars.json',
        ]:
            key = filename.split('.')[0]
            self.paths[key].touch(exist_ok=False)

        # copy the architecture definition into the archive
        copy(git_root/'training'/'architecture.py', self.paths['architecture'])

        # record the status of the git repository
        with self.paths['commit'].open(mode='wb') as f:
            hash = subprocess.check_output('git rev-parse HEAD'.split())
            f.write(hash)
            branch = subprocess.check_output('git rev-parse --abbrev-ref HEAD'
                                             .split())
            f.write(branch)
            diff = subprocess.check_output('git diff'.split())
            f.write(diff)

        # write out the command used
        with self.paths['command'].open(mode='w') as f:
            f.writelines(' '.join(sys.argv))

        # create a history entry
        with self.paths['history'].open(mode='w') as f:
            f.writelines('Model: {}'.format(self.name))
            f.writelines('Time: {}'.format(datetime.datetime.now()))
            f.writelines('Commit: {}'.format(self.commit))
            f.writelines(' '.join(sys.argv))
            f.writelines('')

        # initialize the model, optimizer, and pseudorandom number generator
        self._load_model(*args, **kwargs)
        if not no_optimizer:
            self._load_optimizer(*args, **kwargs)
        self._load_prand(self, *args, **kwargs)

        self.update()

    def start_new(self, name, *args, **kwargs):
        """
        Creates and returns a new model archive initialized with the
        weights of this model.

        The new model's training history is copied from the old model
        and appended to.
        """
        if self.model_exists(name):
            raise ValueError('The model "{}" already exists.'.format(name))
        new_archive = type(self)(name, readonly=False, *args, **kwargs)
        copy(self.paths['weights'], new_archive.paths['weights'])

        # Copy the old history into the new archive
        tempfile = new_archive.directory / 'history.log.temp'
        copy(new_archive.paths['history'], tempfile)
        copy(self.paths['history'], new_archive.paths['history'])
        with new_archive.paths['history'].open(mode='a') as f:
            f.writelines(tempfile.read_text())
        tempfile.unlink()  # delete the temporary file

        return new_archive

    @property
    def commit(self):
        """
        The git hash for the commit on which the model was first trained
        """
        saved_commit = ''
        if not self.paths['commit'].exists():
            return None
        with self.paths['commit'].open(mode='r') as f:
            saved_commit = f.readline()
        return saved_commit.strip()

    @property
    def state_vars(self):
        """
        Returns a dict of the state variables stored in `state_vars.json`
        """
        dict = {}
        if not self.paths['state_vars'].exists():
            return dict
        with self.paths['state_vars'].open(mode='r') as f:
            dict = json.load(f)
        return dict

    @property
    def model(self):
        """
        A live version of the model's architecture.
        """
        return self._model

    @property
    def optimizer(self):
        """
        The model's optimizer
        """
        return self._optimizer

    def _load_model(self, *args, **kwargs):
        """
        Loads a working version of the model's architecture,
        initialized with its pretrained weights.

        If the model is untrained, loads a newly initialized model.
        """
        sys.path.insert(0, str(self.directory))
        import architecture
        sys.path.remove(str(self.directory))
        if self.paths['weights'].is_file:
            weights = torch.load(self.paths['weights'])
            self._model = architecture.Model.load(*args, weights=weights,
                                                  **kwargs)
        else:
            self._model = architecture.Model(*args, **kwargs)

        # set model to eval or train mode
        if self.readonly:
            for p in self._model.parameters():
                p.requires_grad = False
            self._model.eval().cuda()
        else:
            for p in self._model.parameters():
                p.requires_grad = True
            self._model.train().cuda()
            self._model = torch.nn.DataParallel(self._model)

        return self._model

    def _load_optimizer(self, *args, **kwargs):
        """
        Loads the saved state of the optimizer.

        If the model is untrained, loads a newly initialized optimizer.
        """
        assert self.model is not None, 'The model has not yet been loaded.'
        self._optimizer = torch.optim.Adam(self.model.parameters())
        if self.paths['optimizer'].is_file:
            opt_state_dict = torch.load(self.paths['optimizer'])
            self._optimizer.load_state_dict(opt_state_dict)
        return self._optimizer

    def _load_prand(self, seed=None, *args, **kwargs):
        """
        Loads the saved state of the pseudorandom number generators.
        """
        assert self.optimizer is not None, 'Should not seed before init.'
        if self.paths['prand'].is_file:
            prand_state = torch.load(self.paths['prand'])
            set_random_generator_state(prand_state)
        else:
            with self.paths['seed'].open('w') as f:
                f.write(str(seed))
            set_seed(seed)

    def update(self, **kwargs):
        """
        Updates the saved training state.
        Any optional arguments will be written out to the file `state_vars.json`
        """
        if self.readonly:
            raise ReadOnlyError(self.name)
        if self._model:
            torch.save(self._model.state_dict(), self.paths['weights'])
            # also write to a json for debugging
            with self.paths['weights'].with_suffix('.json').open('w') as f:
                f.write(json.dumps(self._model.state_dict()))
        if self._optimizer:
            torch.save(self._optimizer.state_dict(), self.paths['optimizer'])
            # also write to a json for debugging
            with self.paths['optimizer'].with_suffix('.json').open('w') as f:
                f.write(json.dumps(self._optimizer.state_dict()))
        torch.save(get_random_generator_state(), self.paths['prand'])
        with self.paths['state_vars'].open('w') as f:
            f.write(json.dumps(kwargs))

    def create_checkpoint(self, epoch, time):
        """
        Save a checkpoint in the training.
        This saves an snapshot of the model's current weights.

        Note: This does not update the model, but merely creates a backup
        copy. To update the weights, use `update()`.
        """
        if self.readonly:
            raise ReadOnlyError(self.name)
        checkpt_name = 'e{}_t{}.pt'.format(epoch, time)
        copy(self.paths['weights'], self.intermediate_models / checkpt_name)

    def log(self, values, printout=True):
        """
        Add a new log entry to `loss.csv`.

        A new row is added to the spreadsheet and populated with the
        contents of `values`. If `values` is a list, each element is
        written in its own column.
        Warning: If the string verion of any value contains a comma,
        this will separate that value over two columns.
        """
        if self.readonly:
            raise ReadOnlyError(self.name)
        if not isinstance(values, list):
            values = [values]
        line = ', '.join(str(v) for v in values)
        with self.paths['loss.csv'].open(mode='a') as f:
            f.writelines(line + '\n')
        if printout:
            print('log: {}'.format(line))


def set_seed(seed):
    """
    Seeds all the random number genertators used.
    If `seed` is not None, the seeding is deterministic and reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def get_random_generator_state():
    """
    Returns a tuple of states of the random generators used in training
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    return python_state, numpy_state, torch_state


def set_random_generator_state(state):
    """
    Resets the random generators to the given state.
    Useful when resuming training.

    The state should be a state generated by calling
    `get_random_generator_state()`
    """
    python_state, numpy_state, torch_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if not torch.backends.cudnn.deterministic:
        warnings.warn('Resetting random state might not seed GPU correctly.')
        torch.backends.cudnn.deterministic = True


class ReadOnlyError(AttributeError):
    def __init__(self, name):
        message = ('Cannot modify the archive since it was opened as '
                   'read-only. If modifying is necessary, open it with '
                   '`ModelArchive("{}", readonly=False)`.'.format(name))
        super().__init__(message)


def warn_change(param_name, before, now):
    warnings.warn('The {} has been changed since '
                  'this model was last saved.\n'
                  'Before: {}\n'
                  'Now: {}\n'
                  'If this is not intentional, then something may have gone '
                  'wrong. Proceeding may overwrite the appropriate value '
                  'in the archive.\n'
                  'Would you like to proceed?  [y/N]'
                  .format(param_name, before, now))
    if input().lower() not in {'yes', 'y'}:
        print('Exiting')
        sys.exit()
    print('OK, proceeding...')
