import torch.utils.data
import torchvision.transforms as transforms
from data import dataset

class DataLoader(object):
    """Template for training & validation data loaders below
    """
    def __init__(self, args):
        self.transform = None
        self.dataset = None
        self.sampler = None
        self.loader = None

class TrainingDataLoader(DataLoader):
    """Object to manage the DataLoader for training samples:
    """
    def __init__(self, state_vars, archive):
        self.transform = transforms.Compose([
                dataset.ToFloatTensor(),
                dataset.Preprocess(archive.preprocessor),
                dataset.OnlyIf(dataset.RandomRotateAndScale(),
                                     not state_vars.skip_aug),
                dataset.OnlyIf(dataset.RandomFlip(),
                                     not state_vars.skip_aug),
                dataset.Split(),
                dataset.OnlyIf(dataset.RandomTranslation(20),
                                     not state_vars.skip_aug),
                dataset.OnlyIf(dataset.RandomField(),
                                     state_vars.supervised),
                dataset.OnlyIf(dataset.RandomAugmentation(),
                                     not state_vars.skip_aug),
                dataset.ToDevice('cpu'),
        ])
        self.dataset = dataset.compile_dataset(
                            state_vars.training_set_path,
                            transform=self.transform,
                            num_samples=state_vars.num_samples,
                            repeats=state_vars.repeats)
        self.sampler = torch.utils.data.RandomSampler(self.dataset)
        self.loader = torch.utils.data.DataLoader(
                        self.dataset,
                        batch_size=state_vars.batch_size,
                        shuffle=(self.sampler is None),
                        num_workers=state_vars.num_workers,
                        pin_memory=(state_vars.validation_set_path is None),
                        sampler=self.sampler)

class ValidationDataLoader(DataLoader):

    def __init__(self, state_vars, archive):
        if state_vars.validation_set_path:
            self.transform = transforms.Compose([
                                dataset.ToFloatTensor(),
                                dataset.Preprocess(archive.preprocessor),
                                dataset.Split(),
            ])
            self.dataset = dataset.compile_dataset(
                                state_vars.validation_set_path,
                                transform=self.transform)
            self.loader = torch.utils.data.DataLoader(
                            self.dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False)
