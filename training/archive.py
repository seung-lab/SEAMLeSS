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
    >>> mymodel.update(model, optimizer, random_generator)

    Load an existing trained model and run it on data:
    >>> existing_model = ModelArchive('existing_model', readonly=True)
    >>> net = existing_model.model
    >>> output = net(data)

    Create a new model archive from an existing one:
    >>> old_model = ModelArchive('old_model')
    >>> new_model = old_model.start_new('new_model_v01')
    (some training code ...)
    >>> new_model.update(model, optimizer, random_generator)
"""
import torch
import shutil
import subprocess
import warnings
import sys
from pathlib import Path

from helpers import copy

git_root = Path(subprocess.check_output('git rev-parse --show-toplevel'
                                        .split()).strip().decode("utf-8"))
models_location = git_root / Path('models/')


class ModelArchive(object):
    """
    Abstraction for the maintainence of trained model archives
    """

    def model_exists(name):
        """
        Returns whether a trained model of this name exists
        """
        if len(name):
            return (models_location / name).is_dir()
        else:
            raise ValueError('"name" must have non-zero length')

    def __init__(self, name, readonly=True):
        if not name.replace('_','').isalnum():
            raise ValueError('Malformated name: {}'.format(name))
        self.name = name
        self.readonly = readonly
        self.directory = models_location / self.name
        self.intermediate_models = self.directory / 'intermediate_models/'
        self.debug_outputs = self.directory / 'debug_outputs/'
        self.paths = {
            # the model's trained weights
            "model": self.directory / 'model.pt',
            # the state of the optimizer
            "optimizer": self.directory / 'optimizer.pt',
            # the state of the pseudorandom number gerenrators
            "prng": self.directory / 'prng.pt',
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
        ]:
            key = filename.split('.')[0]
            self.paths[key] = self.directory / filename

        if ModelArchive.model_exists(name):
            self._load()
        elif not self.readonly:
            self._create()
        else:
            raise ValueError('Could not find a trained model named "{}".\n'
                             'If the intention was to create one, use '
                             '`ModelArchive("{}", readonly=False)`.'
                             .format(name, name))

    def _load(self):
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
            message = ('The repository has changed since this '
                       'net was last trained.')
            print('Warning: ' + message)
            if not self.readonly:
                print('Continuing may overwrite the archive by '
                      'running the new code. If this was the intent, '
                      'then it might not be a problem.'
                      '\nIf not, exit the process and return to the '
                      'old commit by running `git checkout {}`'
                      '\nDo you wish to proceed?  [y/N]'
                      .format(saved_commit.strip()))
                if input().lower() not in {'yes', 'y'}:
                    print('Exiting')
                    sys.exit()

    def _create(self):
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
            'architecture.py',
            'commit.txt',
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

        # create a history entry
        with self.paths['history'].open(mode='w'):
            pass  # TODO

    def create_checkpoint(self, epoch, time):
        """
        Save a checkpoint in the training.
        This saves an snapshot of the model's current weights.

        Note: This does not update the model, but merely creates a backup
        copy. To update the weights, use `update()`.
        """
        if self.readonly:
            raise ReadOnlyError(self.name)
        check_name = 'e{}_t{}.pt'
        copy(self.paths['model'], self.intermediate_models / check_name)

    def log(self, values, printout=True):
        """
        Add a new log entry to `loss.csv`.

        A new row is added to the spreadsheet and populated with the
        contents of `values`, which must be an iterable.
        """
        if self.readonly:
            raise ReadOnlyError(self.name)
        with self.paths['loss.csv'].open(mode='a') as f:
            line = ', '.join(str(v) for v in values)
            f.writelines(line + '\n')
            if printout:
                print('log: {}'.format(line))

    def update(self, model, optimizer, prng):
        """
        Updates the saved training state
        """
        if self.readonly:
            raise ReadOnlyError(self.name)
        if model:
            torch.save(model.state_dict(), self.paths['model'])
        if optimizer:
            torch.save(optimizer.state_dict(), self.paths['optimizer'])
        if prng:
            torch.save(prng, self.paths['prng'])

    def start_new(self, name):
        """
        Creates and returns a new model archive initialized with the
        weights of this model.

        The new model's training history is copied from the old model
        and appended to.
        """
        if ModelArchive.model_exists(name):
            raise ValueError('The model "{}" already exists.'.format(name))
        new_archive = type(self)(name, readonly=False)
        copy(self.paths['model'], new_archive.paths['model'])

        tempfile = new_archive.directory / 'history.log.temp'
        copy(new_archive.paths['history'], tempfile)
        copy(self.paths['history'], new_archive.paths['history'])
        with new_archive.paths['history'].open(mode='a') as f:
            f.writelines(tempfile.read_text())
        tempfile.unlink()  # delete the temporary file
        return new_archive

    @property
    def architecture(self):
        sys.path.insert(str(self.directory))
        import architecture
        return architecture.Model()

    @property
    def model(self):
        return torch.load(self.paths['model'])

    @property
    def optimizer(self):
        return torch.load(self.paths['optimizer'])

    @property
    def prng(self):
        return torch.load(self.paths['prng'])

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



def copy_aligner(mip_from, mip_to):
    pass


def copy_encoder(mip_from, mip_to):
    pass


class ReadOnlyError(AttributeError):
    def __init__(self, name):
        message = ('Cannot modify the archive since it was opened as '
                   'read-only. If modifying is necessary, open it with '
                   '`ModelArchive("{}", readonly=False)`.'.format(name))
        super().__init__(message)
