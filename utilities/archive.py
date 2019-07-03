import sys
import warnings
import subprocess
import random
import torch
import numpy as np
import yaml
import datetime
from pathlib import Path
import filecmp
import importlib
import pandas as pd

from utilities.helpers import cp, dotdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: 402


class ModelArchive(object):
    """
    Abstraction for the maintainence of trained model archives

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
            state can be read back from disk and training can continue
            almost as if it was never interrupted

    Usage:
        Create a new model archive and save the current training state:
        >>> mymodel = ModelArchive('mymodel_v01', readonly=False)
        (some training code ...)
        >>> mymodel.log(loss)
        >>> mymodel.save()  # save the updated state to disk

        Load an existing trained model and run it on data:
        >>> existing_archive = ModelArchive('existing_archive', readonly=True)
        >>> net = existing_archive.model
        >>> output = net(data)

        Create a new model archive from an existing one:
        >>> old_model = ModelArchive('old_model')
        >>> new_model = old_model.start_new('new_model_v01')
        (some training code ...)
        >>> new_model.save()  # save the updated state of the new model to disk
    """

    def __init__(self, name, readonly=True, *args, **kwargs):
        name, directory = self._resolve_model(name)
        self._name = name
        self.directory = directory
        self.readonly = readonly
        self.intermediate_models = self.directory / 'intermediate_models/'
        self.debug_outputs = self.directory / 'debug_outputs/'
        self.last_training_record = self.directory / '.last_training_record'
        self.paths = {
            # the model's trained weights
            'weights': self.directory / 'weights.pt',
            # the state of the optimizer
            'optimizer': self.directory / 'optimizer.pt',
            # the state of the pseudorandom number gerenrators
            'prand': self.directory / 'prand.pt',
            # other paths
            'loss': self.directory / 'loss.csv',
            'command': self.directory / 'command.txt',
            'plan': self.directory / 'plan.txt',
            'history': self.directory / 'history.txt',
            'progress': self.directory / 'progress.log',
            'architecture': self.directory / 'architecture.py',
            'objective': self.directory / 'objective.py',
            'preprocessor': self.directory / 'preprocessor.py',
            'commit': self.directory / 'commit.diff',
            'state_vars': self.directory / 'state_vars.yaml',
            'plot': self.directory / 'plot.png',
        }
        self._architecture = None
        self._model = None
        self._optimizer = None
        self._state_vars = None
        self._objective = None
        self._loss = None
        self._val_loss = None
        self._preprocessor = None
        self._current_debug_directory = None
        self._seed = None

        if self._exists():
            self._load(*args, **kwargs)
        elif not self.readonly:
            self._create(*args, **kwargs)
        else:
            raise ValueError('Could not find a trained model named "{}".\n'
                             'If the intention was to create one, use '
                             '`ModelArchive("{}", readonly=False)`.'
                             .format(name, name))

        if not self.readonly:
            self.out = FileLog(sys.stdout, self.paths['progress'])
            self.err = FileLog(sys.stderr, self.paths['progress'])
        else:
            self.out = sys.stdout
            self.err = sys.stderr

    def _load(self, *args, **kwargs):
        if not self.readonly:
            print('Writing to exisiting model archive: {}'.format(self._name))
        else:
            print('Reading from exisiting model archive: {}'.format(self._name))
        assert self.directory.is_dir()

        if not self.readonly:
            # ensure directories exist
            self.intermediate_models.mkdir(exist_ok=True)
            self.debug_outputs.mkdir(exist_ok=True)
            self.last_training_record.mkdir(exist_ok=True)

        # check for matching commits
        # this can prevent errors arising from working on the wrong git branch
        saved_commit = self.commit
        if saved_commit is not None:
            try:
                current_commit = subprocess.check_output('git rev-parse HEAD'
                                                         .split()).strip()
            except subprocess.CalledProcessError:
                current_commit = '0'
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

        # load the model, optimizer, and state variables
        self._load_state_vars(*args, **kwargs)
        kwargs.update(self._state_vars)
        self._load_model(*args, **kwargs)
        self._load_objective(*args, **kwargs)
        self._load_preprocessor(*args, **kwargs)
        self._load_optimizer(*args, **kwargs)
        # load the pseudorandom number generator last
        self._load_prand(*args, **kwargs)

    def _create(self, *args, **kwargs):
        print('Creating a new model archive: {}'.format(self._name))

        # create directories
        self.directory.mkdir()
        self.intermediate_models.mkdir()
        self.debug_outputs.mkdir()
        self.last_training_record.mkdir()

        # create archive files
        for filename in [
            'loss.csv',
            'command.txt',
            'plan.txt',
            'history.txt',
            'progress.log',
            'commit.diff',
            'plot.png',
        ]:
            key = filename.split('.')[0]
            self.paths[key].touch(exist_ok=False)

        # copy the architecture and objective definitions into the archive
        cp(git_root()/'training'/'architecture.py', self.paths['architecture'])
        cp(git_root()/'training'/'objective.py', self.paths['objective'])
        cp(git_root()/'training'/'preprocessor.py', self.paths['preprocessor'])

        # record the status of the git repository
        with self.paths['commit'].open(mode='wb') as f:
            hash = subprocess.check_output('git rev-parse HEAD'.split())
            f.write(hash)
            branch = subprocess.check_output('git rev-parse --abbrev-ref HEAD'
                                             .split())
            f.write(branch)
            diff = subprocess.check_output('git diff HEAD'.split())
            f.write(diff)

        # write out the command used
        with self.paths['command'].open(mode='w') as f:
            f.writelines(' '.join(sys.argv))

        # create a history entry
        with self.paths['history'].open(mode='w') as f:
            f.writelines('Model: {}\n'.format(self._name))
            f.writelines('Time: {}\n'.format(datetime.datetime.now()))
            f.writelines('Commit: {}\n'.format(self.commit))
            f.writelines(' '.join(sys.argv) + '\n')
            f.writelines('\n')

        # when creating an archive, init pseudorandom number generator first
        self._load_prand(*args, **kwargs)
        # initialize the model, optimizer, and state variables
        self._load_state_vars(*args, **kwargs)
        self._load_model(*args, **kwargs)
        self._load_objective(*args, **kwargs)
        self._load_preprocessor(*args, **kwargs)
        self._load_optimizer(*args, **kwargs)

        self.save()

    def start_new(self, name, *args, **kwargs):
        """
        Creates and returns a new model archive initialized with the
        weights of this model.

        The new model's training history is copied from the old model
        and appended to.
        """
        if self._exists():
            raise ValueError('The model "{}" already exists.'.format(name))
        new_archive = type(self)(name, readonly=False,
                                 weights_file=self.paths['weights'],
                                 *args, **kwargs)
        cp(self.paths['weights'], new_archive.paths['weights'])
        cp(self.paths['loss'], new_archive.paths['loss'])
        cp(self.paths['progress'], new_archive.paths['progress'])

        # Copy the old history into the new archive
        tempfile = new_archive.directory / 'history.txt.temp'
        cp(new_archive.paths['history'], tempfile)
        cp(self.paths['history'], new_archive.paths['history'])
        with new_archive.paths['history'].open(mode='a') as f:
            f.writelines(tempfile.read_text())
        tempfile.unlink()  # delete the temporary file

        return new_archive

    @classmethod
    def _resolve_model(cls, model_path):
        """
        Find the model referenced by `model_path`.
        If `model_path` is the path to an existing directory, this
        directory is used.
        Otherwise, find a model by that name in the repository's `models`
        directory.
        """
        model_path = Path(model_path).absolute()
        name = model_path.stem
        if model_path.is_dir():
            directory = model_path
        else:
            root = git_root()
            if root is None:
                root = Path()
            directory = root / 'models' / name
        return name, directory

    @classmethod
    def model_exists(cls, model_path):
        """
        Returns whether a trained model of this name or path exists.
        If `model_path` is the path to an existing directory, this
        directory is checked.
        Otherwise, a model by that name is searched for in the repository's
        `models` directory.
        """
        _, directory = cls._resolve_model(model_path)
        return directory.is_dir()

    def _exists(self):
        """
        Returns whether this trained model already exists
        """
        return self.directory.is_dir()

    @property
    def name(self):
        """
        The name of the model
        """
        return self._name

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
    def architecture(self):
        """
        The python module code used to build the model.
        Useful for calling any non-class functions defined there.
        """
        return self._architecture

    @property
    def model(self):
        """
        A live version of the model.
        """
        return self._model

    @property
    def optimizer(self):
        """
        The model's optimizer
        """
        return self._optimizer

    @property
    def loss(self):
        """
        The loss function of the model, optimized during training
        """
        return self._loss

    @property
    def val_loss(self):
        """
        The validation loss function of the model
        """
        return self._val_loss

    @property
    def preprocessor(self):
        """
        The archive's image preprocessor
        """
        return self._preprocessor

    @property
    def state_vars(self):
        """
        A dict of various training state variables

        This dict should be used to store any additional training information
        that is needed to restore the model's training state when resuming
        training.

        For convenience, elements of this dict can be accessed using
        either notation:
            >>> state_vars['item']
            or
            >>> state_vars.item
        """
        return self._state_vars

    def _load_model(self, *args, **kwargs):
        """
        Loads a working version of the model's architecture,
        initialized with its pretrained weights.

        If the model is untrained, loads a newly initialized model.
        """
        self._architecture = importlib.import_module(
            '{}.{}.architecture'.format(self.directory.parent.stem,
                                        self._name))
        self._model = self._architecture.Model(*args, **kwargs)
        if self.paths['weights'].exists():
            self._model.load(self.paths['weights'])

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

    def _load_objective(self, *args, **kwargs):
        """
        Loads the objective functions stored in the archive
        """
        if self.paths['objective'].exists():
            self._objective = importlib.import_module(
                '{}.{}.objective'.format(self.directory.parent.stem,
                                         self._name))
        else:
            return None
        self._loss = self._objective.Objective(*args, **kwargs)
        self._val_loss = self._objective.ValidationObjective(*args, **kwargs)
        if not self.readonly:
            self._loss = torch.nn.DataParallel(self._loss.cuda())
            self._val_loss = torch.nn.DataParallel(self._val_loss.cuda())
        return self._objective

    def _load_preprocessor(self, *args, **kwargs):
        """
        Loads the archive's image preprocessor
        """
        if self.paths['preprocessor'].exists():
            preprocessor = importlib.import_module(
                '{}.{}.preprocessor'.format(self.directory.parent.stem,
                                            self._name))
        else:
            return None
        self._preprocessor = preprocessor.Preprocessor(*args, **kwargs)
        return self._preprocessor

    def _load_optimizer(self, *args, **kwargs):
        """
        Loads the saved state of the optimizer.

        If the model is untrained, loads a newly initialized optimizer.
        """
        assert self.model is not None, 'The model has not yet been loaded.'
        if not self.readonly:
            self._optimizer = torch.optim.Adam(self.model.parameters())
            if self.paths['optimizer'].is_file():
                with self.paths['optimizer'].open('rb') as f:
                    opt_state_dict = torch.load(f)
                self._optimizer.load_state_dict(opt_state_dict)
        return self._optimizer

    def _load_prand(self, seed=None, *args, **kwargs):
        """
        Loads the saved state of the pseudorandom number generators.
        """
        if self.readonly:
            return  # do not seed for a readonly archive
        if self.paths['prand'].is_file():
            with self.paths['prand'].open('rb') as f:
                prand_state = torch.load(f)
            set_random_generator_state(prand_state)
        else:
            self._seed = seed
            print('Initializing seed to {}'.format(seed))
            set_seed(seed)

    def _load_state_vars(self, *args, **kwargs):
        """
        Loads the dict of state variables stored in `state_vars.yaml`
        """
        self._state_vars = dotdict({'name': self._name})  # default state_vars
        if self.paths['state_vars'].exists():
            with self.paths['state_vars'].open(mode='r') as f:
                self._state_vars = dotdict(yaml.load(f))
        if self._seed is not None:
            self._state_vars.seed = self._seed
        return self._state_vars

    def save(self):
        """
        Saves the live model archive to disk.
        More specifically, this updates the saved archive to match the live
        training state of the model
        """
        if self.readonly:
            raise ReadOnlyError(self._name)
        if self._model:
            self._model.module.save(self.paths['weights'])
        if self._optimizer:
            with self.paths['optimizer'].open('wb') as f:
                torch.save(self._optimizer.state_dict(), f)
        with self.paths['prand'].open('wb') as f:
            torch.save(get_random_generator_state(), f)
        if self._state_vars:
            s = yaml.dump(dict(self._state_vars))
            with self.paths['state_vars'].open('w') as f:
                f.write(s)

    def create_checkpoint(self, epoch, iteration, save=True):
        """
        Save a checkpoint in the training.
        This saves an snapshot of the model's current saved weights.

        Note: To just update the saved weights without creating a checkpoint,
        use `save()`.
        """
        if self.readonly:
            raise ReadOnlyError(self._name)
        if save:
            self.save()  # ensure the saved weights are up to date
        if epoch is None:
            checkpt_name = 'init'
        elif iteration is None:
            checkpt_name = 'e{}'.format(epoch)
        else:
            checkpt_name = 'e{}_t{}'.format(epoch, iteration)
        check_dir = self.intermediate_models / checkpt_name
        if check_dir.exists():
            print('Checkpoint {} already exists. Overwrite? [y/N]'
                  .format(check_dir))
            if input().lower() not in {'yes', 'y'}:
                check_dir = check_dir / 'new_checkpoint'
            else:
                print('OK, overwriting...')
        check_dir.mkdir(exist_ok=True)
        cp(self.paths['weights'], check_dir)
        cp(self.paths['optimizer'], check_dir)
        cp(self.paths['prand'], check_dir)
        cp(self.paths['state_vars'], check_dir)
        cp(self.paths['plot'], check_dir)

    def record_training_session(self):
        """
        Records a new training session with the updated parameters.
        """
        if self.readonly:
            raise ReadOnlyError(self._name)
        tracked = [
            'architecture.py',
            'objective.py',
            'preprocessor.py',
            'state_vars.yaml'
        ]
        _, changed, error = filecmp.cmpfiles(
            str(self.last_training_record), str(self.directory), tracked)
        if len(changed + error) == 0:
            return
        with self.paths['plan'].open(mode='a') as f:
            f.writelines('\nAt epoch {}, iteration {}:\n'.format(
                self.state_vars.epoch, self.state_vars.iteration))
            f.writelines('Time: {}\n'.format(datetime.datetime.now()))
            f.writelines('Commit: {}\n'.format(self.commit))
            f.writelines(' '.join(sys.argv) + '\n')
        with self.paths['plan'].open(mode='ab') as f:
            for filename in changed + error:
                if filename in changed:
                    subprocess.call(
                        'diff -u'
                        ' -I \\s*\"*epoch\"*:\\s'
                        ' -I \\s*\"*iteration\"*:\\s'
                        ' -I \\s*\"*initialized_list\"*:\\s'
                        ' -I \\s*\"*levels\"*:\\s'.split()
                        + [
                            str(self.last_training_record.expanduser()),
                            str(self.paths[filename.split('.')[0]]
                                .expanduser())
                        ], stdout=f)
        for filename in tracked:
            if (self.last_training_record / filename).is_file():
                (self.last_training_record / filename).unlink()
            cp(self.paths[filename.split('.')[0]],
               self.last_training_record / filename)

    def new_debug_directory(self, exist_ok=False):
        """
        Creates a new subdirectory for debugging outputs.

        The new subdirectory will be placed in the `debug_outputs` directory
        of the archive.
        """
        if self.readonly:
            raise ReadOnlyError(self._name)
        if self._state_vars.iteration is not None:
            dirname = 'e{}_t{}'.format(self._state_vars.epoch,
                                        self._state_vars.iteration)
        else:
            dirname = 'e{}_val'.format(self._state_vars.epoch)
        debug_directory = self.debug_outputs / dirname
        if not exist_ok and debug_directory.is_dir():
            raise FileExistsError('The debug directory {} already exists.'
                                  .format(debug_directory))
        debug_directory.mkdir(exist_ok=exist_ok)
        self._current_debug_directory = debug_directory
        return self._current_debug_directory

    def set_log_titles(self, log_titles):
        """
        Set the column titles for `loss.csv`.

        If `log_titles` is a list, each element is written in its own column.
        """
        self.log(log_titles, printout=False)

    def log(self, values, printout=False):
        """
        Add a new log entry to `loss.csv`.

        A new row is added to the spreadsheet and populated with the
        contents of `values`. If `values` is a list, each element is
        written in its own column.

        Note that this is unbuffered, so the values will be written
        immediately, and even without a call to `save()`.

        Warning: If the string verion of any value contains a comma,
        this will separate that value over two columns.
        """
        if self.readonly:
            raise ReadOnlyError(self._name)
        if not isinstance(values, list):
            values = [values]
        line = ', '.join(str(v) for v in values)
        with self.paths['loss'].open(mode='a') as f:
            f.writelines(line + '\n')
        if printout:
            print('log: {}'.format(line))

    def set_optimizer_params(self, learning_rate, weight_decay):
        if self.readonly:
            raise ReadOnlyError(self._name)
        self.state_vars['lr'] = learning_rate
        self.state_vars['wd'] = weight_decay
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = learning_rate
            param_group['weight_decay'] = weight_decay

    def adjust_learning_rate(self):
        """
        Sets the learning rate to the initial learning rate decayed by
        `gamma` every `gamma_step` epochs.

        `gamma`, `gamma_step`, and the current epoch are pulled from the
        archive's `state_vars` dictionary.
        """  # TODO: reformulate as params
        if self.readonly:
            raise ReadOnlyError(self._name)
        epoch = self._state_vars.epoch
        gamma = self._state_vars.gamma
        gamma_step = self._state_vars.gamma_step
        if gamma == 1:
            return
        self._state_vars.lr = (self._state_vars.start_lr
                               * (gamma ** (epoch // gamma_step)))
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self._state_vars.lr

    def visualize_loss(self, *columns, average_over=100):
        """
        Save a plot of the learning curves
        """
        data = pd.read_csv(self.paths['loss'], sep='\\s*,\\s*',
                           encoding='ascii', engine='python',
                           comment='#')[list(columns)]
        # ensure averaging window is reasonable
        if average_over > len(data.index) // 10 + 1:
            average_over = len(data.index) // 10 + 1
        if average_over < 1:
            average_over = 1
        data = data.dropna(axis=1, how='all').interpolate()
        if self._state_vars.plot_from is not None:
            data = data[self._state_vars.plot_from:]
        if data.empty:
            return
        data = data.rolling(window=average_over).mean()
        data.plot(title='Training loss for {}'.format(self._name))
        with self.paths['plot'].open('wb') as f:
            plt.savefig(f)


def git_root():
    """
    Return the root directory of the current git repository, if available
    """
    try:
        return Path(subprocess.check_output('git rev-parse --show-toplevel'
                                            .split()).strip().decode("utf-8"))
    except subprocess.CalledProcessError:
        return None


def set_seed(seed):
    """
    Seeds all the random number genertators used.
    If `seed` is not None, the seeding is deterministic and reproducible.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which may slow down training.')


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


class FileLog:
    """
    A file-like object that writes both to the terminal and to
    a specified file.

    `terminal_out` should be either `sys.stdout` or `sys.stderr`
    """

    def __init__(self, terminal_out, file):
        self.terminal_out = terminal_out
        self.file = file.open('a')

    def write(self, message):
        self.terminal_out.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal_out.flush()
        self.file.flush()


class ReadOnlyError(AttributeError):
    def __init__(self, name):
        message = ('Cannot modify the archive since it was opened as '
                   'read-only. If modifying is necessary, open it with '
                   '`ModelArchive("{}", readonly=False)`.'.format(name))
        super().__init__(message)


def warn_change(param_name, before, now):
    """
    Warns the user of a discrepancy in the stored archive, and asks
    for affirmation to continue.
    """
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
