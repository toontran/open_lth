import argparse
import sys

from dataclasses import dataclass, fields, MISSING, replace

import typing
import warnings
import os
import torch
import copy
from functools import partial
import json
import numpy as np
import abc
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import pathlib


try:
    import apex
    NO_APEX = False
except ImportError:
    NO_APEX = True

import abc
import argparse
import copy
from typing import Tuple, Union

import bisect

import time
import hashlib

import torch

from models import bn_initializers, initializers
from dataclasses import dataclass, field, make_dataclass, fields
import inspect


def maybe_get_arg(arg_name, positional=False, position=0, boolean_arg=False):
    parser = argparse.ArgumentParser(add_help=False)
    prefix = '' if positional else '--'
    if positional:
        for i in range(position):
            parser.add_argument(f'arg{i}')
    if boolean_arg:
        parser.add_argument(prefix + arg_name, action='store_true')
    else:
        parser.add_argument(prefix + arg_name, type=str, default=None)
    try:
        args = parser.parse_known_args()[0]
        return getattr(args, arg_name) if arg_name in args else None
    except:
        return None

class Step:
    """Represents a particular step of training.

    A step can be represented as either an iteration or a pair of an epoch and an iteration within that epoch.
    This class encapsulates a step of training such that it can be freely converted between the two representations.
    """

    def __init__(self, iteration: int, iterations_per_epoch: int) -> 'Step':
        if iteration < 0: raise ValueError('iteration must >= 0.')
        if iterations_per_epoch <= 0: raise ValueError('iterations_per_epoch must be > 0.')
        self._iteration = iteration
        self._iterations_per_epoch = iterations_per_epoch

    @staticmethod
    def str_is_zero(s: str):
        return s in ['0ep', '0it', '0ep0it']

    @staticmethod
    def from_iteration(iteration: int, iterations_per_epoch: int) -> 'Step':
        return Step(iteration, iterations_per_epoch)

    @staticmethod
    def from_epoch(epoch: int, iteration: int, iterations_per_epoch: int) -> 'Step':
        return Step(epoch * iterations_per_epoch + iteration, iterations_per_epoch)

    @staticmethod
    def from_str(s: str, iterations_per_epoch: int) -> 'Step':
        """Creates a step from a string that describes the number of epochs, iterations, or both.

        Epochs: '120ep'
        Iterations: '2000it'
        Both: '120ep50it'"""

        if 'ep' in s and 'it' in s:
            ep = int(s.split('ep')[0])
            it = int(s.split('ep')[1].split('it')[0])
            if s != '{}ep{}it'.format(ep, it): raise ValueError('Malformed string step: {}'.format(s))
            return Step.from_epoch(ep, it, iterations_per_epoch)
        elif 'ep' in s:
            ep = int(s.split('ep')[0])
            if s != '{}ep'.format(ep): raise ValueError('Malformed string step: {}'.format(s))
            return Step.from_epoch(ep, 0, iterations_per_epoch)
        elif 'it' in s:
            it = int(s.split('it')[0])
            if s != '{}it'.format(it): raise ValueError('Malformed string step: {}'.format(s))
            return Step.from_iteration(it, iterations_per_epoch)
        else:
            raise ValueError('Malformed string step: {}'.format(s))

    @staticmethod
    def zero(iterations_per_epoch: int) -> 'Step':
        return Step(0, iterations_per_epoch)

    @property
    def iteration(self):
        """The overall number of steps of training completed so far."""
        return self._iteration

    @property
    def ep(self):
        """The current epoch of training."""
        return self._iteration // self._iterations_per_epoch

    @property
    def it(self):
        """The iteration within the current epoch of training."""
        return self._iteration % self._iterations_per_epoch

    def _check(self, other):
        if not isinstance(other, Step):
            raise ValueError('Invalid type for other: {}.'.format(type(other)))
        if self._iterations_per_epoch != other._iterations_per_epoch:
            raise ValueError('Cannot compare steps when epochs are of different lengths.')

    def __lt__(self, other):
        self._check(other)
        return self._iteration < other._iteration

    def __le__(self, other):
        self._check(other)
        return self._iteration <= other._iteration

    def __eq__(self, other):
        self._check(other)
        return self._iteration == other._iteration

    def __ne__(self, other):
        self._check(other)
        return self._iteration != other._iteration

    def __gt__(self, other):
        self._check(other)
        return self._iteration > other._iteration

    def __ge__(self, other):
        self._check(other)
        return self._iteration >= other._iteration

    def __str__(self):
        return '(Iteration {}; Iterations per Epoch: {})'.format(self._iteration, self._iterations_per_epoch)

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# from foundations.hparams import TrainingHparams
# from foundations.step import Step


@dataclass
class Hparams(abc.ABC):
    """A collection of hyperparameters.

    Add desired hyperparameters with their types as fields. Provide default values where desired.
    You must provide default values for _name and _description. Help text for field `f` is
    optionally provided in the field `_f`,
    """

    def __post_init__(self):
        if not hasattr(self, '_name'): raise ValueError('Must have field _name with string value.')
        if not hasattr(self, '_description'): raise ValueError('Must have field _name with string value.')

    @classmethod
    def add_args(cls, parser, defaults: 'Hparams' = None, prefix: str = None,
                 name: str = None, description: str = None, create_group: bool = True):
        if defaults and not isinstance(defaults, cls):
            raise ValueError(f'defaults must also be type {cls}.')

        if create_group:
            parser = parser.add_argument_group(name or cls._name, description or cls._description)

        for field in fields(cls):
            if field.name.startswith('_'): continue
            arg_name = f'--{field.name}' if prefix is None else f'--{prefix}_{field.name}'
            helptext = getattr(cls, f'_{field.name}') if hasattr(cls, f'_{field.name}') else ''

            if defaults: default = copy.deepcopy(getattr(defaults, field.name, None))
            elif field.default != MISSING: default = copy.deepcopy(field.default)
            else: default = None

            if field.type == bool:
                if (defaults and getattr(defaults, field.name) is not False) or field.default is not False:
                    raise ValueError(f'Boolean hyperparameters must default to False: {field.name}.')
                parser.add_argument(arg_name, action='store_true', help='(optional) ' + helptext)

            elif field.type in [str, float, int]:
                required = field.default is MISSING and (not defaults or not getattr(defaults, field.name))
                if required:  helptext = '(required: %(type)s) ' + helptext
                elif default: helptext = f'(default: {default}) ' + helptext
                else:         helptext = '(optional: %(type)s) ' + helptext
                parser.add_argument(arg_name, type=field.type, default=default, required=required, help=helptext)

            # If it is a nested hparams, use the field name as the prefix and add all arguments.
            elif isinstance(field.type, type) and issubclass(field.type, Hparams):
                subprefix = f'{prefix}_{field.name}' if prefix else field.name
                field.type.add_args(parser, defaults=default, prefix=subprefix, create_group=False)

            else: raise ValueError(f'Invalid field type {field.type} for hparams.')

    @classmethod
    def create_from_args(cls, args: argparse.Namespace, prefix: str = None) -> 'Hparams':
        d = {}
        for field in fields(cls):
            if field.name.startswith('_'): continue

            # Base types.
            if field.type in [bool, str, float, int]:
                arg_name = f'{field.name}' if prefix is None else f'{prefix}_{field.name}'
                if not hasattr(args, arg_name): raise ValueError(f'Missing argument: {arg_name}.')
                d[field.name] = getattr(args, arg_name)

            # Nested hparams.
            elif isinstance(field.type, type) and issubclass(field.type, Hparams):
                subprefix = f'{prefix}_{field.name}' if prefix else field.name
                d[field.name] = field.type.create_from_args(args, subprefix)

            else: raise ValueError(f'Invalid field type {field.type} for hparams.')

        return cls(**d)

    @property
    def display(self):
        nondefault_fields = [f for f in fields(self)
                             if not f.name.startswith('_') and ((f.default is MISSING) or getattr(self, f.name))]
        s = self._name + '\n'
        return s + '\n'.join(f'    * {f.name} => {getattr(self, f.name)}' for f in nondefault_fields)

    def __str__(self):
        fs = {}
        for f in fields(self):
            if f.name.startswith('_'): continue
            if f.default is MISSING or (getattr(self, f.name) != f.default):
                value = getattr(self, f.name)
                if isinstance(value, str): value = "'" + value + "'"
                if isinstance(value, Hparams): value = str(value)
                if isinstance(value, Tuple): value = 'Tuple(' + ','.join(str(h) for h in value) + ')'
                fs[f.name] = value
        elements = [f'{name}={fs[name]}' for name in sorted(fs.keys())]
        return 'Hparams(' + ', '.join(elements) + ')'


@dataclass
class DatasetHparams(Hparams):
    dataset_name: str
    batch_size: int
    do_not_augment: bool = False
    transformation_seed: int = None
    subsample_fraction: float = None
    random_labels_fraction: float = None
    unsupervised_labels: str = None
    blur_factor: int = None

    _name: str = 'Dataset Hyperparameters'
    _description: str = 'Hyperparameters that select the dataset, data augmentation, and other data transformations.'
    _dataset_name: str = 'The name of the dataset. Examples: mnist, cifar10'
    _batch_size: str = 'The size of the mini-batches on which to train. Example: 64'
    _do_not_augment: str = 'If True, data augmentation is disabled. It is enabled by default.'
    _transformation_seed: str = 'The random seed that controls dataset transformations like ' \
                                'random labels, subsampling, and unsupervised labels.'
    _subsample_fraction: str = 'Subsample the training set, retaining the specified fraction: float in (0, 1]'
    _random_labels_fraction: str = 'Apply random labels to a fraction of the training set: float in (0, 1]'
    _unsupervised_labels: str = 'Replace the standard labels with alternative, unsupervised labels. Example: rotation'
    _blur_factor: str = 'Blur the training set by downsampling and then upsampling by this multiple.'


@dataclass
class ModelHparams(Hparams):
    model_name: str
    model_init: str
    batchnorm_init: str
    batchnorm_frozen: bool = False
    output_frozen: bool = False
    others_frozen: bool = False
    others_frozen_exceptions: str = None

    _name: str = 'Model Hyperparameters'
    _description: str = 'Hyperparameters that select the model, initialization, and weight freezing.'
    _model_name: str = 'The name of the model. Examples: mnist_lenet, cifar_resnet_20, cifar_vgg_16'
    _model_init: str = 'The model initializer. Examples: kaiming_normal, kaiming_uniform, binary, orthogonal'
    _batchnorm_init: str = 'The batchnorm initializer. Examples: uniform, fixed'
    _batchnorm_frozen: str = 'If True, all batch normalization parameters are frozen at initialization.'
    _output_frozen: str = 'If True, all outputt layer parameters are frozen at initialization.'
    _others_frozen: str = 'If true, all other (non-output, non-batchnorm) parameters are frozen at initialization.'
    _others_frozen_exceptions: str = 'A comma-separated list of any tensors that should not be frozen.'


@dataclass
class TrainingHparams(Hparams):
    optimizer_name: str
    lr: float
    training_steps: str
    data_order_seed: int = None
    momentum: float = 0.0
    nesterov_momentum: float = 0.0
    milestone_steps: str = None
    gamma: float = None
    warmup_steps: str = None
    weight_decay: float = None
    apex_fp16: bool = False

    _name: str = 'Training Hyperparameters'
    _description: str = 'Hyperparameters that determine how the model is trained.'
    _optimizer_name: str = 'The opimizer with which to train the network. Examples: sgd, adam'
    _lr: str = 'The learning rate'
    _training_steps: str = 'The number of steps to train as epochs (\'160ep\') or iterations (\'50000it\').'
    _momentum: str = 'The momentum to use with the SGD optimizer.'
    _nesterov: bool = 'The nesterov momentum to use with the SGD optimizer. Cannot set both momentum and nesterov.'
    _milestone_steps: str = 'Steps when the learning rate drops by a factor of gamma. Written as comma-separated ' \
                            'steps (80ep,160ep,240ep) where steps are epochs (\'160ep\') or iterations (\'50000it\').'
    _gamma: str = 'The factor at which to drop the learning rate at each milestone.'
    _data_order_seed: str = 'The random seed for the data order. If not set, the data order is random and unrepeatable.'
    _warmup_steps: str = "Steps of linear lr warmup at the start of training as epochs ('20ep') or iterations ('800it')"
    _weight_decay: str = 'The L2 penalty to apply to the weights.'
    _apex_fp16: bool = 'Whether to train the model in float16 using the NVIDIA Apex library.'


@dataclass
class PruningHparams(Hparams):
    pruning_strategy: str

    _name: str = 'Pruning Hyperparameters'
    _description: str = 'Hyperparameters that determine how the model is pruned. ' \
                        'More hyperparameters will appear once the pruning strategy is selected.'
    _pruning_strategy: str = 'The pruning strategy to use.'

@dataclass
class JobArgs(Hparams):
    """Arguments shared across lottery ticket jobs."""

    replicate: int = 1
    default_hparams: str = None
    quiet: bool = False
    evaluate_only_at_end: bool = False

    _name: str = 'High-Level Arguments'
    _description: str = 'Arguments that determine how the job is run and where it is stored.'
    _replicate: str = 'The index of this particular replicate. ' \
                      'Use a different replicate number to run another copy of the same experiment.'
    _default_hparams: str = 'Populate all arguments with the default hyperparameters for this model.'
    _quiet: str = 'Suppress output logging about the training status.'
    _evaluate_only_at_end: str = 'Run the test set only before and after training. Otherwise, will run every epoch.'


def maybe_get_default_hparams():
    default_hparams = arg_utils.maybe_get_arg('default_hparams')
    return models.registry.get_default_hparams(default_hparams) if default_hparams else None


class Runner(abc.ABC):
    """An instance of a training run of some kind."""

    @staticmethod
    @abc.abstractmethod
    def description() -> str:
        """A description of this runner."""

        pass

    @staticmethod
    @abc.abstractmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add all command line flags necessary for this runner."""

        pass

    @staticmethod
    @abc.abstractmethod
    def create_from_args(args: argparse.Namespace) -> 'Runner':
        """Create a runner from command line arguments."""

        pass

    @abc.abstractmethod
    def display_output_location(self) -> None:
        """Print the output location for the job."""

        pass

    @abc.abstractmethod
    def run(self) -> None:
        """Run the job."""

        pass




def paths_checkpoint(root): return os.path.join(root, 'checkpoint.pth')


def paths_logger(root): return os.path.join(root, 'logger')


def paths_mask(root): return os.path.join(root, 'mask.pth')


def paths_sparsity_report(root): return os.path.join(root, 'sparsity_report.json')


def paths_model(root, step): return os.path.join(root, 'model_ep{}_it{}.pth'.format(step.ep, step.it))


def paths_hparams(root): return os.path.join(root, 'hparams.log')



class Dataset(abc.ABC, torch.utils.data.Dataset):
    """The base class for all datasets in this framework."""

    @staticmethod
    @abc.abstractmethod
    def num_test_examples() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def num_train_examples() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def num_classes() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_train_set(use_augmentation: bool) -> 'Dataset':
        pass

    @staticmethod
    @abc.abstractmethod
    def get_test_set() -> 'Dataset':
        pass

    def __init__(self, examples: np.ndarray, labels):
        """Create a dataset object.

        examples is a numpy array of the examples (or the information necessary to get them).
        Only the first dimension matters for use in this abstract class.

        labels is a numpy array of the labels. Each entry is a zero-indexed integer encoding
        of the label.
        """

        if examples.shape[0] != labels.shape[0]:
            raise ValueError('Different number of examples ({}) and labels ({}).'.format(
                             examples.shape[0], examples.shape[0]))
        self._examples = examples
        self._labels = labels if isinstance(labels, np.ndarray) else labels.numpy()
        self._subsampled = False

    def randomize_labels(self, seed: int, fraction: float) -> None:
        """Randomize the labels of the specified fraction of the dataset."""

        num_to_randomize = np.ceil(len(self._labels) * fraction).astype(int)
        randomized_labels = np.random.RandomState(seed=seed).randint(self.num_classes(), size=num_to_randomize)
        examples_to_randomize = np.random.RandomState(seed=seed+1).permutation(len(self._labels))[:num_to_randomize]
        self._labels[examples_to_randomize] = randomized_labels

    def subsample(self, seed: int, fraction: float) -> None:
        """Subsample the dataset."""

        if self._subsampled:
            raise ValueError('Cannot subsample more than once.')
        self._subsampled = True

        examples_to_retain = np.ceil(len(self._labels) * fraction).astype(int)
        examples_to_retain = np.random.RandomState(seed=seed+1).permutation(len(self._labels))[:examples_to_retain]
        self._examples = self._examples[examples_to_retain]
        self._labels = self._labels[examples_to_retain]

    def __len__(self):
        return self._labels.size

    def __getitem__(self, index):
        """If there is custom logic for example loading, this method should be overridden."""

        return self._examples[index], self._labels[index]


class ImageDataset(Dataset):
    @abc.abstractmethod
    def example_to_image(self, example: np.ndarray) -> Image: pass

    def __init__(self, examples, labels, image_transforms=None, tensor_transforms=None,
                 joint_image_transforms=None, joint_tensor_transforms=None):
        super(ImageDataset, self).__init__(examples, labels)
        self._image_transforms = image_transforms or []
        self._tensor_transforms = tensor_transforms or []
        self._joint_image_transforms = joint_image_transforms or []
        self._joint_tensor_transforms = joint_tensor_transforms or []

        self._composed = None

    def __getitem__(self, index):
        if not self._composed:
            self._composed = torchvision.transforms.Compose(
                self._image_transforms + [torchvision.transforms.ToTensor()] + self._tensor_transforms)

        example, label = self._examples[index], self._labels[index]
        example = self.example_to_image(example)
        for t in self._joint_image_transforms: example, label = t(example, label)
        example = self._composed(example)
        for t in self._joint_tensor_transforms: example, label = t(example, label)
        return example, label

    def blur(self, blur_factor: float) -> None:
        """Add a transformation that blurs the image by downsampling by blur_factor."""

        def blur_transform(image):
            size = list(image.size)
            image = torchvision.transforms.Resize([int(s / blur_factor) for s in size])(image)
            image = torchvision.transforms.Resize(size)(image)
            return image
        self._image_transforms.append(blur_transform)

    def unsupervised_rotation(self, seed: int):
        """Switch the task to unsupervised rotation."""

        self._labels = np.random.RandomState(seed=seed).randint(4, size=self._labels.size)

        def rotate_transform(image, label):
            return torchvision.transforms.RandomRotation(label*90)(image), label
        self._joint_image_transforms.append(rotate_transform)


class ShuffleSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_examples):
        self._num_examples = num_examples
        self._seed = -1

    def __iter__(self):
        if self._seed == -1:
            indices = list(range(self._num_examples))
        elif self._seed is None:
            indices = torch.randperm(self._num_examples).tolist()
        else:
            g = torch.Generator()
            if self._seed is not None: g.manual_seed(self._seed)
            indices = torch.randperm(self._num_examples, generator=g).tolist()

        return iter(indices)

    def __len__(self):
        return self._num_examples

    def shuffle_dataorder(self, seed: int):
        self._seed = seed


class DistributedShuffleSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset):
        super(DistributedShuffleSampler, self).__init__(
            dataset, num_replicas=get_platform().world_size, rank=get_platform().rank)
        self._seed = -1

    def __iter__(self):
        indices = torch.arange(len(self.dataset))

        if self._seed != -1:
            g = torch.Generator()
            g.manual_seed(self._seed or np.random.randint(10e6))
            perm = torch.randperm(len(indices), generator=g)
            indices = indices[perm]

        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices.tolist())

    def shuffle_dataorder(self, seed: int):
        self._seed = seed


class DataLoader(torch.utils.data.DataLoader):
    """A wrapper that makes it possible to access the custom shuffling logic."""

    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int, pin_memory: bool = True):
        if False:
            self._sampler = DistributedShuffleSampler(dataset)
        else:
            self._sampler = ShuffleSampler(len(dataset))

        self._iterations_per_epoch = np.ceil(len(dataset) / batch_size).astype(int)

        if False:
            batch_size //= get_platform().world_size
            num_workers //= get_platform().world_size

        super(DataLoader, self).__init__(
            dataset, batch_size, sampler=self._sampler, num_workers=num_workers,
            pin_memory=pin_memory and (torch.cuda.is_available() and torch.cuda.device_count() > 0))

    def shuffle(self, seed: int):
        self._sampler.shuffle_dataorder(seed)

    @property
    def iterations_per_epoch(self):
        return self._iterations_per_epoch




class DataParallel(torch.nn.DataParallel):
    def __init__(self, module):
        super(DataParallel, self).__init__(module=module)

    @property
    def prunable_layer_names(self): return self.module.prunable_layer_names

    @property
    def output_layer_names(self): return self.module.output_layer_names

    @property
    def loss_criterion(self): return self.module.loss_criterion

    @staticmethod
    def get_model_from_name(model_name, outputs, initializer): raise NotImplementedError

    @staticmethod
    def is_valid_model_name(model_name): raise NotImplementedError

    @staticmethod
    def default_hparams(): raise NotImplementedError

    def save(self, save_location: str, save_step: Step):
        self.module.save(save_location, save_step)


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __init__(self, module, device_ids):
        super(DistributedDataParallel, self).__init__(module=module, device_ids=device_ids)

    @property
    def prunable_layer_names(self): return self.module.prunable_layer_names

    @property
    def output_layer_names(self): return self.module.output_layer_names

    @property
    def loss_criterion(self): return self.module.loss_criterion

    @staticmethod
    def get_model_from_name(model_name, outputs, initializer): raise NotImplementedError

    @staticmethod
    def is_valid_model_name(model_name): raise NotImplementedError

    @staticmethod
    def default_hparams(): raise NotImplementedError

    def save(self, save_location: str, save_step: Step):
        self.module.save(save_location, save_step)


# Standard callbacks.
# Callback frequencies. Each takes a callback as an argument and returns a new callback
# that runs only at the specified frequency.
def run_every_epoch(callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step.it != 0:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback


def run_every_step(callback):
    return callback


def run_at_step(step1, callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step != step1:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback


# The standard set of callbacks that should be used for a normal training run.
def standard_callbacks(training_hparams: TrainingHparams, train_set_loader: DataLoader,
                       test_set_loader: DataLoader, eval_on_train: bool = False, verbose: bool = True,
                       start_step: Step = None, evaluate_every_epoch: bool = True):
    start = start_step or Step.zero(train_set_loader.iterations_per_epoch)
    end = Step.from_str(training_hparams.training_steps, train_set_loader.iterations_per_epoch)
    test_eval_callback = create_eval_callback('test', test_set_loader, verbose=verbose)
    train_eval_callback = create_eval_callback('train', train_set_loader, verbose=verbose)

    # Basic checkpointing and state saving at the beginning and end.
    result = [
        run_at_step(start, save_model),
        run_at_step(end, save_model),
        run_at_step(end, save_logger),
        run_every_epoch(save_checkpoint_callback),
    ]

    # Test every epoch if requested.
    if evaluate_every_epoch: result = [run_every_epoch(test_eval_callback)] + result
    elif verbose: result.append(run_every_epoch(create_timekeeper_callback()))

    # Ensure that testing occurs at least at the beginning and end of training.
    if start.it != 0 or not evaluate_every_epoch: result = [run_at_step(start, test_eval_callback)] + result
    if end.it != 0 or not evaluate_every_epoch: result = [run_at_step(end, test_eval_callback)] + result

    # Do the same for the train set if requested.
    if eval_on_train:
        if evaluate_every_epoch: result = [run_every_epoch(train_eval_callback)] + result
        if start.it != 0 or not evaluate_every_epoch: result = [run_at_step(start, train_eval_callback)] + result
        if end.it != 0 or not evaluate_every_epoch: result = [run_at_step(end, train_eval_callback)] + result

    return result

def save_model(output_location, step, model, optimizer, logger):
    model.save(output_location, step)


def save_logger(output_location, step, model, optimizer, logger):
    logger.save(output_location)


def create_timekeeper_callback():
    time_of_last_call = None

    def callback(output_location, step, model, optimizer, logger):
        nonlocal time_of_last_call
        t = 0.0 if time_of_last_call is None else time.time() - time_of_last_call
        print(f'Ep {step.ep}\tIt {step.it}\tTime Elapsed {t:.2f}')
        time_of_last_call = time.time()

    return callback


def create_eval_callback(eval_name: str, loader: DataLoader, verbose=False):
    """This function returns a callback."""

    time_of_last_call = None
    def device_str():
        # GPU device.
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_ids = ','.join([str(x) for x in range(torch.cuda.device_count())])
            return f'cuda:{device_ids}'
        # CPU device.
        else:
            return 'cpu'
    torch_device = torch.device(device_str())

    def eval_callback(output_location, step, model, optimizer, logger):
        example_count = torch.tensor(0.0).to(torch_device)
        total_loss = torch.tensor(0.0).to(torch_device)
        total_correct = torch.tensor(0.0).to(torch_device)

        def correct(labels, outputs):
            return torch.sum(torch.eq(labels, output.argmax(dim=1)))

        model.eval()

        with torch.no_grad():
            for examples, labels in loader:
                examples = examples.to(torch_device)
                labels = labels.squeeze().to(torch_device)
                output = model(examples)

                labels_size = torch.tensor(len(labels), device=torch_device)
                example_count += labels_size
                total_loss += model.loss_criterion(output, labels) * labels_size
                total_correct += correct(labels, output)

        # Share the information if distributed.
        if False:
            torch.distributed.reduce(total_loss, 0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(total_correct, 0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(example_count, 0, op=torch.distributed.ReduceOp.SUM)

        total_loss = total_loss.cpu().item()
        total_correct = total_correct.cpu().item()
        example_count = example_count.cpu().item()

        logger.add('{}_loss'.format(eval_name), step, total_loss / example_count)
        logger.add('{}_accuracy'.format(eval_name), step, total_correct / example_count)
        logger.add('{}_examples'.format(eval_name), step, example_count)

        if verbose:
            nonlocal time_of_last_call
            elapsed = 0 if time_of_last_call is None else time.time() - time_of_last_call
            print('{}\tep {:03d}\tit {:03d}\tloss {:.3f}\tacc {:.2f}%\tex {:d}\ttime {:.2f}s'.format(
                eval_name, step.ep, step.it, total_loss/example_count, 100 * total_correct/example_count,
                int(example_count), elapsed))
            time_of_last_call = time.time()

    return eval_callback


def optimizers_get_optimizer(training_hparams: TrainingHparams, model) -> torch.optim.Optimizer:
    if training_hparams.optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=training_hparams.lr,
            momentum=training_hparams.momentum or training_hparams.nesterov_momentum or 0,
            weight_decay=training_hparams.weight_decay or 0,
            nesterov=training_hparams.nesterov_momentum is not None and training_hparams.nesterov_momentum > 0
        )
    elif training_hparams.optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=training_hparams.lr,
            weight_decay=training_hparams.weight_decay or 0
        )

    raise ValueError('No such optimizer: {}'.format(training_hparams.optimizer_name))


def optimizers_get_lr_schedule(training_hparams: TrainingHparams, optimizer: torch.optim.Optimizer, iterations_per_epoch: int):
    lambdas = [lambda it: 1.0]

    # Drop the learning rate according to gamma at the specified milestones.
    if bool(training_hparams.gamma) != bool(training_hparams.milestone_steps):
        raise ValueError('milestones and gamma hyperparameters must both be set or not at all.')
    if training_hparams.milestone_steps:
        milestones = [Step.from_str(x, iterations_per_epoch).iteration
                      for x in training_hparams.milestone_steps.split(',')]
        lambdas.append(lambda it: training_hparams.gamma ** bisect.bisect(milestones, it))

    # Add linear learning rate warmup if specified.
    if training_hparams.warmup_steps:
        warmup_iters = Step.from_str(training_hparams.warmup_steps, iterations_per_epoch).iteration
        lambdas.append(lambda it: min(1.0, it / warmup_iters))

    # Combine the lambdas.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: np.product([l(it) for l in lambdas]))



class MetricLogger:
    def __init__(self):
        self.log = {}

    def add(self, name: str, step: Step, value: float):
        self.log[(name, step.iteration)] = value

    def __str__(self):
        return '\n'.join(['{},{},{}'.format(k[0], k[1], v) for k, v in self.log.items()])

    @staticmethod
    def create_from_string(as_str):
        logger = MetricLogger()
        if len(as_str.strip()) == 0:
            return logger

        rows = [row.split(',') for row in as_str.strip().split('\n')]
        logger.log = {(name, int(iteration)): float(value) for name, iteration, value in rows}
        return logger

    @staticmethod
    def create_from_file(filename):
        with open(paths_logger(filename)) as fp:
            as_str = fp.read()
        return MetricLogger.create_from_string(as_str)

    def save(self, location):
        if not os.path.exists(location):
            os.makedirs(location)
        with open(paths_logger(location), 'w') as fp:
            fp.write(str(self))

    def get_data(self, desired_name):
        d = {k[1]: v for k, v in self.log.items() if k[0] == desired_name}
        return [(k, d[k]) for k in sorted(d.keys())]



def save_checkpoint_callback(output_location, step, model, optimizer, logger):
    torch.save({
        'ep': step.ep,
        'it': step.it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'logger': str(logger),
    }, paths_checkpoint(output_location))


def restore_checkpoint(output_location, model, optimizer, iterations_per_epoch):
    checkpoint_location = paths_checkpoint(output_location)
    if not os.path.exists(checkpoint_location):
        return None, None
    checkpoint = torch.load(checkpoint_location, map_location=torch.device('cpu'))

    # Handle DataParallel.
    module_in_name = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if module_in_name and not all(k.startswith('module.') for k in checkpoint['model_state_dict']):
        checkpoint['model_state_dict'] = {'module.' + k: v for k, v in checkpoint['model_state_dict'].items()}
    elif all(k.startswith('module.') for k in checkpoint['model_state_dict']) and not module_in_name:
        checkpoint['model_state_dict'] = {k[len('module.'):]: v for k, v in checkpoint['model_state_dict'].items()}

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = Step.from_epoch(checkpoint['ep'], checkpoint['it'], iterations_per_epoch)
    logger = MetricLogger.create_from_string(checkpoint['logger'])

    return step, logger



class Mask(dict):
    def __init__(self, other_dict=None):
        super(Mask, self).__init__()
        if other_dict is not None:
            for k, v in other_dict.items(): self[k] = v

    def __setitem__(self, key, value):
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError('Invalid tensor name: {}'.format(key))
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError('value for key {} must be torch Tensor or numpy ndarray.'.format(key))
        if ((value != 0) & (value != 1)).any(): raise ValueError('All entries must be 0 or 1.')

        super(Mask, self).__setitem__(key, value)

    @staticmethod
    def ones_like(model) -> 'Mask':
        mask = Mask()
        for name in model.prunable_layer_names:
            mask[name] = torch.ones(list(model.state_dict()[name].shape))
        return mask

    def save(self, output_location):
        if not os.path.exists(output_location): os.makedirs(output_location)
        torch.save({k: v.cpu().int() for k, v in self.items()}, paths_mask(output_location))

        # Create a sparsity report.
        total_weights = np.sum([v.size for v in self.numpy().values()]).item()
        total_unpruned = np.sum([np.sum(v) for v in self.numpy().values()]).item()
        with open(paths_sparsity_report(output_location), 'w') as fp:
            fp.write(json.dumps({'total': float(total_weights), 'unpruned': float(total_unpruned)}, indent=4))

    @staticmethod
    def load(output_location):
        if not Mask.exists(output_location):
            raise ValueError('Mask not found at {}'.format(output_location))
        return Mask(torch.load(paths_mask(output_location)))

    @staticmethod
    def exists(output_location):
        return os.path.exists(paths_mask(output_location))

    def numpy(self):
        return {k: v.cpu().numpy() for k, v in self.items()}

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""

        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in self.values()]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in self.values()]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity

class PrunedModel(torch.nn.Module):
    @staticmethod
    def to_mask_name(name):
        return 'mask_' + name.replace('.', '___')

    def __init__(self, model, mask: Mask):
        if isinstance(model, PrunedModel): raise ValueError('Cannot nest pruned models.')
        super(PrunedModel, self).__init__()
        self.model = model

        for k in self.model.prunable_layer_names:
            if k not in mask: raise ValueError('Missing mask value {}.'.format(k))
            if not np.array_equal(mask[k].shape, np.array(self.model.state_dict()[k].shape)):
                raise ValueError('Incorrect mask shape {} for tensor {}.'.format(mask[k].shape, k))

        for k in mask:
            if k not in self.model.prunable_layer_names:
                raise ValueError('Key {} found in mask but is not a valid model tensor.'.format(k))

        for k, v in mask.items(): self.register_buffer(PrunedModel.to_mask_name(k), v.float())
        self._apply_mask()

    def _apply_mask(self):
        for name, param in self.model.named_parameters():
            if hasattr(self, PrunedModel.to_mask_name(name)):
                param.data *= getattr(self, PrunedModel.to_mask_name(name))

    def forward(self, x):
        self._apply_mask()
        return self.model.forward(x)

    @property
    def prunable_layer_names(self):
        return self.model.prunable_layer_names

    @property
    def output_layer_names(self):
        return self.model.output_layer_names

    @property
    def loss_criterion(self):
        return self.model.loss_criterion

    def save(self, save_location, save_step):
        self.model.save(save_location, save_step)

    @staticmethod
    def default_hparams(): raise NotImplementedError()
    @staticmethod
    def is_valid_model_name(model_name): raise NotImplementedError()
    @staticmethod
    def get_model_from_name(model_name, outputs, initializer): raise NotImplementedError()


class BaseStrategy(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def get_pruning_hparams() -> type:
        pass

    @staticmethod
    @abc.abstractmethod
    def prune(pruning_hparams: PruningHparams, trained_model, current_mask: Mask = None) -> Mask:
        pass

@dataclass
class PruningHparams(PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None

    _name = 'Hyperparameters for Sparse Global Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'

class Strategy(BaseStrategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model, current_mask: Mask = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_tensors}

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

        new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                         for k, v in weights.items()})
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask

registered_strategies = {'sparse_global': Strategy}


def pruning_registry_get(pruning_hparams: PruningHparams):
    """Get the pruning function."""

    return partial(registered_strategies[pruning_hparams.pruning_strategy].prune,
                   copy.deepcopy(pruning_hparams))


def pruning_registry_get_pruning_hparams(pruning_strategy: str) -> type:
    """Get a complete lottery schema as specialized for a particular pruning strategy."""

    if pruning_strategy not in registered_strategies:
        raise ValueError('Pruning strategy {} not found.'.format(pruning_strategy))

    return registered_strategies[pruning_strategy].get_pruning_hparams()
