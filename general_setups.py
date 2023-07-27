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
from utils.general_utils import *
from models.models import *
from datasets import *
# from callbacks import *
from dataclasses import dataclass, field, make_dataclass, fields
from utils.tensor_utils import vectorize, unvectorize, shuffle_tensor, shuffle_state_dict
import inspect



def datasets_registry_get(dataset_hparams: DatasetHparams, train: bool = True):
    """Get the train or test set corresponding to the hyperparameters."""

    seed = dataset_hparams.transformation_seed or 0
    # Get the dataset itself.
    if dataset_hparams.dataset_name in registered_datasets:
        use_augmentation = train and not dataset_hparams.do_not_augment
        if train:
            dataset = registered_datasets[dataset_hparams.dataset_name].Dataset.get_train_set(use_augmentation)
        else:
            dataset = registered_datasets[dataset_hparams.dataset_name].Dataset.get_test_set()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))
    # Transform the dataset.
    if train and dataset_hparams.random_labels_fraction is not None:
        dataset.randomize_labels(seed=seed, fraction=dataset_hparams.random_labels_fraction)

    if train and dataset_hparams.subsample_fraction is not None:
        dataset.subsample(seed=seed, fraction=dataset_hparams.subsample_fraction)

    if train and dataset_hparams.blur_factor is not None:
        if not isinstance(dataset, ImageDataset):
            raise ValueError('Can blur images.')
        else:
            dataset.blur(seed=seed, blur_factor=dataset_hparams.blur_factor)
    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError('Unknown unsupervised labels: {}'.format(dataset_hparams.unsupervised_labels))
        elif not isinstance(dataset, ImageDataset):
            raise ValueError('Can only do unsupervised rotation to images.')
        else:
            dataset.unsupervised_rotation(seed=seed)
    # Create the loader.
    return registered_datasets[dataset_hparams.dataset_name].DataLoader(
        dataset, batch_size=dataset_hparams.batch_size, num_workers=0)


def datasets_registry_iterations_per_epoch(dataset_hparams: DatasetHparams):
    """Get the number of iterations per training epoch."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_train_examples = registered_datasets[dataset_hparams.dataset_name].Dataset.num_train_examples()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    if dataset_hparams.subsample_fraction is not None:
        num_train_examples *= dataset_hparams.subsample_fraction

    return np.ceil(num_train_examples / dataset_hparams.batch_size).astype(int)


def datasets_registry_num_classes(dataset_hparams: DatasetHparams):
    """Get the number of classes."""

    if dataset_hparams.dataset_name in registered_datasets:
        num_classes = registered_datasets[dataset_hparams.dataset_name].Dataset.num_classes()
    else:
        raise ValueError('No such dataset: {}'.format(dataset_hparams.dataset_name))

    if dataset_hparams.unsupervised_labels is not None:
        if dataset_hparams.unsupervised_labels != 'rotation':
            raise ValueError('Unknown unsupervised labels: {}'.format(dataset_hparams.unsupervised_labels))
        else:
            return 4

    return num_classes

def model_registry_get(model_hparams: ModelHparams, outputs=None):
    """Get the model for the corresponding hyperparameters."""

    # Select the initializer.
    if hasattr(initializers, model_hparams.model_init):
        initializer = getattr(initializers, model_hparams.model_init)
    else:
        raise ValueError('No initializer: {}'.format(model_hparams.model_init))

    # Select the BatchNorm initializer.
    if hasattr(bn_initializers, model_hparams.batchnorm_init):
        bn_initializer = getattr(bn_initializers, model_hparams.batchnorm_init)
    else:
        raise ValueError('No batchnorm initializer: {}'.format(model_hparams.batchnorm_init))

    # Create the overall initializer function.
    def init_fn(w):
        initializer(w)
        bn_initializer(w)

    # Select the model.
    model = None
    for registered_model in registered_models:
        if registered_model.is_valid_model_name(model_hparams.model_name):
            model = registered_model.get_model_from_name(model_hparams.model_name, init_fn, outputs)
            break

    if model is None:
        raise ValueError('No such model: {}'.format(model_hparams.model_name))

    # Freeze various subsets of the network.
    bn_names = []
    for k, v in model.named_modules():
        if isinstance(v, torch.nn.BatchNorm2d):
            bn_names += [k + '.weight', k + '.bias']

    if model_hparams.others_frozen_exceptions:
        others_exception_names = model_hparams.others_frozen_exceptions.split(',')
        for name in others_exception_names:
            if name not in model.state_dict():
                raise ValueError(f'Invalid name to except: {name}')
    else:
        others_exception_names = []

    for k, v in model.named_parameters():
        if k in bn_names and model_hparams.batchnorm_frozen:
            v.requires_grad = False
        elif k in model.output_layer_names and model_hparams.output_frozen:
            v.requires_grad = False
        elif k not in bn_names and k not in model.output_layer_names and model_hparams.others_frozen:
            if k in others_exception_names: continue
            v.requires_grad = False

    return model


def model_registry_load(save_location: str, save_step: Step, model_hparams: ModelHparams, outputs=None):
    state_dict = torch.load(paths_model(save_location, save_step))
    model = model_registry_get(model_hparams, outputs)
    model.load_state_dict(state_dict)
    return model


def model_registry_exists(save_location, save_step):
    return os.path.exists(paths_model(save_location, save_step))


def model_registry_get_default_hparams(model_name):
    """Get the default hyperparameters for a particular model."""

    for registered_model in registered_models:
        if registered_model.is_valid_model_name(model_name):
            params = registered_model.default_hparams()
            params.model_hparams.model_name = model_name
            return params

    raise ValueError('No such model: {}'.format(model_name))



def train(
    training_hparams: TrainingHparams,
    model,
    train_loader: DataLoader,
    output_location: str,
    callbacks: typing.List[typing.Callable] = [],
    start_step: Step = None,
    end_step: Step = None
):

    """The main training loop for this framework.

    Args:
      * training_hparams: The training hyperparameters whose schema is specified in hparams.py.
      * model: The model to train. Must be a models.base.Model
      * train_loader: The training data. Must be a datasets.base.DataLoader
      * output_location: The string path where all outputs should be stored.
      * callbacks: A list of functions that are called before each training step and once more
        after the last training step. Each function takes five arguments: the current step,
        the output location, the model, the optimizer, and the logger.
        Callbacks are used for running the test set, saving the logger, saving the state of the
        model, etc. The provide hooks into the training loop for customization so that the
        training loop itself can remain simple.
      * start_step: The step at which the training data and learning rate schedule should begin.
        Defaults to step 0.
      * end_step: The step at which training should cease. Otherwise, training will go for the
        full `training_hparams.training_steps` steps.
    """

    # Create the output location if it doesn't already exist.
    if not os.path.exists(output_location):
        os.makedirs(output_location)

    # Get the optimizer and learning rate schedule.
    def device_str():
        # GPU device.
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_ids = ','.join([str(x) for x in range(torch.cuda.device_count())])
            return f'cuda:{device_ids}'
        # CPU device.
        else:
            return 'cpu'
    torch_device = torch.device(device_str())
    model.to(torch_device)
    optimizer = optimizers_get_optimizer(training_hparams, model)
    step_optimizer = optimizer
    lr_schedule = optimizers_get_lr_schedule(training_hparams, optimizer, train_loader.iterations_per_epoch)

    # Adapt for FP16.
    if training_hparams.apex_fp16:
        if NO_APEX: raise ImportError('Must install nvidia apex to use this model.')
        model, step_optimizer = apex.amp.initialize(model, optimizer, loss_scale='dynamic', verbosity=0)

    # Handle parallelism if applicable.
    if False:
        model = DistributedDataParallel(model, device_ids=[get_platform().rank])
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = DataParallel(model)

    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Restore the model from a saved checkpoint if the checkpoint exists.
    cp_step, cp_logger = restore_checkpoint(output_location, model, optimizer, train_loader.iterations_per_epoch)
    start_step = cp_step or start_step or Step.zero(train_loader.iterations_per_epoch)
    logger = cp_logger or MetricLogger()
    with warnings.catch_warnings():  # Filter unnecessary warning.
        warnings.filterwarnings("ignore", category=UserWarning)
        for _ in range(start_step.iteration): lr_schedule.step()

    # Determine when to end training.
    end_step = end_step or Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)
    if end_step <= start_step: return

    # The training loop.
    for ep in range(start_step.ep, end_step.ep + 1):

        # Ensure the data order is different for each epoch.
        train_loader.shuffle(None if data_order_seed is None else (data_order_seed + ep))

        for it, (examples, labels) in enumerate(train_loader):

            # Advance the data loader until the start epoch and iteration.
            if ep == start_step.ep and it < start_step.it: continue

            # Run the callbacks.
            step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, model, optimizer, logger)

            # Exit at the end step.
            if ep == end_step.ep and it == end_step.it: return

            # Otherwise, train.
            examples = examples.to(device=torch_device)
            labels = labels.to(device=torch_device)

            step_optimizer.zero_grad()
            model.train()
            loss = model.loss_criterion(model(examples), labels)
            if training_hparams.apex_fp16:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Step forward. Ignore extraneous warnings that the lr_schedule generates.
            step_optimizer.step()
            with warnings.catch_warnings():  # Filter unnecessary warning.
                warnings.filterwarnings("ignore", category=UserWarning)
                lr_schedule.step()

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


@dataclass
class Desc(abc.ABC):
    """The bundle of hyperparameters necessary for a particular kind of job. Contains many hparams objects.

    Each hparams object should be a field of this dataclass.
    """

    @staticmethod
    @abc.abstractmethod
    def name_prefix() -> str:
        """The name to prefix saved runs with."""

        pass

    @property
    def hashname(self) -> str:
        """The name under which experiments with these hyperparameters will be stored."""

        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        hparams_strs = [str(fields_dict[k]) for k in sorted(fields_dict) if isinstance(fields_dict[k], Hparams)]
        hash_str = hashlib.md5(';'.join(hparams_strs).encode('utf-8')).hexdigest()
        return f'{self.name_prefix()}_{hash_str}'

    @staticmethod
    @abc.abstractmethod
    def add_args(parser: argparse.ArgumentParser, defaults: 'Desc' = None) -> None:
        """Add the necessary command-line arguments."""

        pass

    @staticmethod
    @abc.abstractmethod
    def create_from_args(args: argparse.Namespace) -> 'Desc':
        """Create from command line arguments."""

        pass

    def save(self, output_location):
        if not os.path.exists(output_location): os.makedirs(output_location)

        fields_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        hparams_strs = [fields_dict[k].display for k in sorted(fields_dict) if isinstance(fields_dict[k], Hparams)]
        with open(paths_hparams(output_location), 'w') as fp:
            fp.write('\n'.join(hparams_strs))




@dataclass
class LotteryDesc(Desc):
    """The hyperparameters necessary to describe a lottery ticket training backbone."""

    model_hparams: ModelHparams
    dataset_hparams: DatasetHparams
    training_hparams: TrainingHparams
    pruning_hparams: PruningHparams
    pretrain_dataset_hparams: DatasetHparams = None
    pretrain_training_hparams: TrainingHparams = None

    @staticmethod
    def name_prefix(): return 'lottery'

    @staticmethod
    def _add_pretrain_argument(parser):
        help_text = \
            'Perform a pre-training phase prior to running the main lottery ticket process. Setting this argument '\
            'will enable arguments to control how the dataset and training during this pre-training phase. Rewinding '\
            'is a specific case of of pre-training where pre-training uses the same dataset and training procedure '\
            'as the main training run.'
        parser.add_argument('--pretrain', action='store_true', help=help_text)

    @staticmethod
    def _add_rewinding_argument(parser):
        help_text = \
            'The number of steps for which to train the network before the lottery ticket process begins. This is ' \
            'the \'rewinding\' step as described in recent lottery ticket research. Can be expressed as a number of ' \
            'epochs (\'160ep\') or a number  of iterations (\'50000it\'). If this flag is present, no other '\
            'pretraining arguments  may be set. Pretraining will be conducted using the same dataset and training '\
            'hyperparameters as for the main training run. For the full range of pre-training options, use --pretrain.'
        parser.add_argument('--rewinding_steps', type=str, help=help_text)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: 'LotteryDesc' = None):
        # Add the rewinding/pretraining arguments.
        rewinding_steps = arg_utils.maybe_get_arg('rewinding_steps')
        pretrain = arg_utils.maybe_get_arg('pretrain', boolean_arg=True)

        if rewinding_steps is not None and pretrain: raise ValueError('Cannot set --rewinding_steps and --pretrain')
        pretraining_parser = parser.add_argument_group(
            'Rewinding/Pretraining Arguments', 'Arguments that control how the network is pre-trained')
        LotteryDesc._add_rewinding_argument(pretraining_parser)
        LotteryDesc._add_pretrain_argument(pretraining_parser)

        # Get the proper pruning hparams.
        pruning_strategy = arg_utils.maybe_get_arg('pruning_strategy')
        if defaults and not pruning_strategy: pruning_strategy = defaults.pruning_hparams.pruning_strategy
        if pruning_strategy:
            pruning_hparams = pruning_registry_get_pruning_hparams(pruning_strategy)
            if defaults and defaults.pruning_hparams.pruning_strategy == pruning_strategy:
                def_ph = defaults.pruning_hparams
        else:
            pruning_hparams = PruningHparams
            def_ph = None

        # Add the main arguments.
        DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None)
        ModelHparams.add_args(parser, defaults=defaults.model_hparams if defaults else None)
        TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None)
        pruning_hparams.add_args(parser, defaults=def_ph if defaults else None)

        # Handle pretraining.
        if pretrain:
            if defaults: def_th = replace(defaults.training_hparams, training_steps='0ep')
            TrainingHparams.add_args(parser, defaults=def_th if defaults else None,
                                             name='Training Hyperparameters for Pretraining', prefix='pretrain')
            DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None,
                                            name='Dataset Hyperparameters for Pretraining', prefix='pretrain')

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> 'LotteryDesc':
        # Get the main arguments.
        dataset_hparams = DatasetHparams.create_from_args(args)
        model_hparams = ModelHparams.create_from_args(args)
        training_hparams = TrainingHparams.create_from_args(args)
        pruning_hparams = pruning_registry_get_pruning_hparams(args.pruning_strategy).create_from_args(args)

        # Create the desc.
        desc = cls(model_hparams, dataset_hparams, training_hparams, pruning_hparams)

        # Handle experiment folder path
        if args.experiment_folder_path:
            desc.root_folder = args.experiment_folder_path
        else:
            raise Exception("Need to set argument \"experiment_folder_path\"") 

        # Handle pretraining.
        if args.pretrain and not Step.str_is_zero(args.pretrain_training_steps):
            desc.pretrain_dataset_hparams = DatasetHparams.create_from_args(args, prefix='pretrain')
            desc.pretrain_dataset_hparams._name = 'Pretraining ' + desc.pretrain_dataset_hparams._name
            desc.pretrain_training_hparams = TrainingHparams.create_from_args(args, prefix='pretrain')
            desc.pretrain_training_hparams._name = 'Pretraining ' + desc.pretrain_training_hparams._name
        elif 'rewinding_steps' in args and args.rewinding_steps and not Step.str_is_zero(args.rewinding_steps):
            desc.pretrain_dataset_hparams = copy.deepcopy(dataset_hparams)
            desc.pretrain_dataset_hparams._name = 'Pretraining ' + desc.pretrain_dataset_hparams._name
            desc.pretrain_training_hparams = copy.deepcopy(training_hparams)
            desc.pretrain_training_hparams._name = 'Pretraining ' + desc.pretrain_training_hparams._name
            desc.pretrain_training_hparams.training_steps = args.rewinding_steps

        return desc

    def str_to_step(self, s: str, pretrain: bool = False) -> Step:
        dataset_hparams = self.pretrain_dataset_hparams if pretrain else self.dataset_hparams
        iterations_per_epoch = datasets_registry_iterations_per_epoch(dataset_hparams)
        return Step.from_str(s, iterations_per_epoch)

    @property
    def pretrain_end_step(self):
        return self.str_to_step(self.pretrain_training_hparams.training_steps, True)

    @property
    def train_start_step(self):
        if self.pretrain_training_hparams: return self.str_to_step(self.pretrain_training_hparams.training_steps)
        else: return self.str_to_step('0it')

    @property
    def train_end_step(self):
        return self.str_to_step(self.training_hparams.training_steps)

    @property
    def pretrain_outputs(self):
        return datasets_registry_num_classes(self.pretrain_dataset_hparams)

    @property
    def train_outputs(self):
        return datasets_registry_num_classes(self.dataset_hparams)

    def run_path(self, replicate: int, pruning_level: Union[str, int], experiment: str = 'main'):
        """The location where any run is stored."""

        if not isinstance(replicate, int) or replicate <= 0:
            raise ValueError('Bad replicate: {}'.format(replicate))

        # return os.path.join(get_platform().root, self.hashname,
        #                     f'replicate_{replicate}', f'level_{pruning_level}', experiment)
        # root = "/root/open_lth_data/"
        # root = "/content/drive/My Drive/Experiments"
        #root = "/home/dragon/xin/projects/toon/Experiments"
        return os.path.join(self.root_folder, self.hashname,
                            f'replicate_{replicate}', f'level_{pruning_level}', experiment)

    @property
    def display(self):
        ls = [self.dataset_hparams.display, self.model_hparams.display,
              self.training_hparams.display, self.pruning_hparams.display]
        if self.pretrain_training_hparams:
            ls += [self.pretrain_dataset_hparams.display, self.pretrain_training_hparams.display]
        return '\n'.join(ls)



def standard_train(
  model,
  output_location: str,
  dataset_hparams: DatasetHparams,
  training_hparams: TrainingHparams,
  start_step: Step = None,
  verbose: bool = True,
  evaluate_every_epoch: bool = True
):
    """Train using the standard callbacks according to the provided hparams."""
    # If the model file for the end of training already exists in this location, do not train.
    iterations_per_epoch = datasets_registry_iterations_per_epoch(dataset_hparams)
    train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)
    if (model_registry_exists(output_location, train_end_step) and
        os.path.exists(paths_logger(output_location))): return
    train_loader = datasets_registry_get(dataset_hparams, train=True)
    test_loader = datasets_registry_get(dataset_hparams, train=False)
    callbacks = standard_callbacks(
        training_hparams, train_loader, test_loader, start_step=start_step,
        verbose=verbose, evaluate_every_epoch=evaluate_every_epoch)
    train(training_hparams, model, train_loader, output_location, callbacks, start_step=start_step)





@dataclass
class LotteryRunner(Runner):
    replicate: int
    levels: int
    desc: LotteryDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return 'Run a lottery ticket hypothesis experiment.'

    @staticmethod
    def _add_levels_argument(parser):
        help_text = \
            'The number of levels of iterative pruning to perform. At each level, the network is trained to ' \
            'completion, pruned, and rewound, preparing it for the next lottery ticket iteration. The full network ' \
            'is trained at level 0, and level 1 is the first level at which pruning occurs. Set this argument to 0 ' \
            'to just train the full network or to N to prune the network N times.'
        parser.add_argument('--levels', required=True, type=int, help=help_text)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # Get preliminary information.
        defaults = shared_args.maybe_get_default_hparams()

        # Add the job arguments.
        shared_args.JobArgs.add_args(parser)
        lottery_parser = parser.add_argument_group(
            'Lottery Ticket Hyperparameters', 'Hyperparameters that control the lottery ticket process.')
        LotteryRunner._add_levels_argument(lottery_parser)
        LotteryDesc.add_args(parser, defaults)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'LotteryRunner':
        return LotteryRunner(args.replicate, args.levels, LotteryDesc.create_from_args(args),
                             not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.replicate, 0))

    def return_output_location(self):
        return self.desc.run_path(self.replicate, 0)

    def run(self) -> None:
        print('='*82 + f'\nLottery Ticket Experiment (Replicate {self.replicate})\n' + '-'*82)
        print(self.desc.display)
        print(self.desc.train_outputs)
        print(f'Output Location: {self.desc.run_path(self.replicate, 0)}' + '\n' + '='*82 + '\n')

        self.desc.save(self.desc.run_path(self.replicate, 0))
        if self.desc.pretrain_training_hparams: self._pretrain()
        self._establish_initial_weights()

        for level in range(self.levels+1):
            self._prune_level(level)
            self._train_level(level)

    # Helper methods for running the lottery.
    def _pretrain(self):
        location = self.desc.run_path(self.replicate, 'pretrain')
        if model_registry_exists(location, self.desc.pretrain_end_step): return

        if self.verbose: print('-'*82 + '\nPretraining\n' + '-'*82)
        model = model_registry_get(self.desc.model_hparams, outputs=self.desc.pretrain_outputs)
        standard_train(model, location, self.desc.pretrain_dataset_hparams, self.desc.pretrain_training_hparams,
                             verbose=self.verbose, evaluate_every_epoch=self.evaluate_every_epoch)

    def _establish_initial_weights(self):
        location = self.desc.run_path(self.replicate, 0)
        if model_registry_exists(location, self.desc.train_start_step): return

        new_model = model_registry_get(self.desc.model_hparams, outputs=self.desc.train_outputs)

        # If there was a pretrained model, retrieve its final weights and adapt them for training.
        if self.desc.pretrain_training_hparams is not None:
            pretrain_loc = self.desc.run_path(self.replicate, 'pretrain')
            old = model_registry_load(pretrain_loc, self.desc.pretrain_end_step,
                                       self.desc.model_hparams, self.desc.pretrain_outputs)
            state_dict = {k: v for k, v in old.state_dict().items()}

            # Select a new output layer if number of classes differs.
            if self.desc.train_outputs != self.desc.pretrain_outputs:
                state_dict.update({k: new_model.state_dict()[k] for k in new_model.output_layer_names})

            new_model.load_state_dict(state_dict)

        new_model.save(location, self.desc.train_start_step)

    def _train_level(self, level: int):
        location = self.desc.run_path(self.replicate, level)
        if model_registry_exists(location, self.desc.train_end_step): return

        model = model_registry_load(self.desc.run_path(self.replicate, 0), self.desc.train_start_step,
                                     self.desc.model_hparams, self.desc.train_outputs)
        pruned_model = PrunedModel(model, Mask.load(location))
        pruned_model.save(location, self.desc.train_start_step)
        if self.verbose:
            print('-'*82 + '\nPruning Level {}\n'.format(level) + '-'*82)
        standard_train(pruned_model, location, self.desc.dataset_hparams, self.desc.training_hparams,
                             start_step=self.desc.train_start_step, verbose=self.verbose,
                             evaluate_every_epoch=self.evaluate_every_epoch)

    def _prune_level(self, level: int):
        new_location = self.desc.run_path(self.replicate, level)
        if Mask.exists(new_location): return

        if level == 0:
            Mask.ones_like(model_registry_get(self.desc.model_hparams,
                                              outputs=self.desc.train_outputs)).save(new_location)
        else:
            old_location = self.desc.run_path(self.replicate, level-1)
            model = model_registry_load(old_location, self.desc.train_end_step,
                                         self.desc.model_hparams, self.desc.train_outputs)
            pruning_registry_get(self.desc.pruning_hparams)(model, Mask.load(old_location)).save(new_location)

from dataclasses import dataclass, field, make_dataclass, fields
import inspect

def make_BranchDesc(BranchHparams: type, name: str):
    @dataclass
    class BranchDesc(Desc):
        lottery_desc: LotteryDesc
        branch_hparams: BranchHparams

        @staticmethod
        def name_prefix(): return 'lottery_branch_' + name

        @staticmethod
        def add_args(parser: argparse.ArgumentParser, defaults: LotteryDesc = None):
            LotteryDesc.add_args(parser, defaults)
            BranchHparams.add_args(parser)

        @classmethod
        def create_from_args(cls, args: argparse.Namespace):
            return BranchDesc(LotteryDesc.create_from_args(args), BranchHparams.create_from_args(args))

    return BranchDesc

@dataclass
class Branch(Runner):
    """A lottery branch. Implement `branch_function`, add a name and description, and add to the registry."""

    replicate: int
    levels: str
    desc: Desc
    verbose: bool = False
    level: int = None

    # Interface that needs to be overriden for each branch.
    @staticmethod
    @abc.abstractmethod
    def description() -> str:
        """A description of this branch. Override this."""
        pass

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """The name of this branch. Override this."""
        pass

    @abc.abstractmethod
    def branch_function(self) -> None:
        """The method that is called to execute the branch.

        Override this method with any additional arguments that the branch will need.
        These arguments will be converted into command-line arguments for the branch.
        Each argument MUST have a type annotation. The first argument must still be self.
        """
        pass

    # Interface that is useful for writing branches.
    @property
    def lottery_desc(self) -> LotteryDesc:
        """The lottery description of this experiment."""

        return self.desc.lottery_desc

    @property
    def experiment_name(self) -> str:
        """The name of this experiment."""

        return self.desc.hashname

    @property
    def branch_root(self) -> str:
        """The root for where branch results will be stored for a specific invocation of run()."""

        return self.lottery_desc.run_path(self.replicate, self.level, self.experiment_name)

    @property
    def level_root(self) -> str:
        """The root of the main experiment on which this branch is based."""

        return self.lottery_desc.run_path(self.replicate, self.level)

    # Interface that deals with command line arguments.
    @dataclass
    class ArgHparams(Hparams):
        levels: str
        pretrain_training_steps: str = None

        _name: str = 'Lottery Ticket Hyperparameters'
        _description: str = 'Hyperparameters that control the lottery ticket process.'
        _levels: str = \
            'The pruning levels on which to run this branch. Can include a comma-separate list of levels or ranges, '\
            'e.g., 1,2-4,9'
        _pretrain_training_steps: str = 'The number of steps to train the network prior to the lottery ticket process.'

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        defaults = shared_args.maybe_get_default_hparams()
        shared_args.JobArgs.add_args(parser)
        Branch.ArgHparams.add_args(parser)
        cls.BranchDesc.add_args(parser, defaults)

    @staticmethod
    def level_str_to_int_list(levels: str):
        level_list = []
        elements = levels.split(',')
        for element in elements:
            if element.isdigit():
                level_list.append(int(element))
            elif len(element.split('-')) == 2:
                level_list += list(range(int(element.split('-')[0]), int(element.split('-')[1]) + 1))
            else:
                raise ValueError(f'Invalid level: {element}')
        return sorted(list(set(level_list)))

    @classmethod
    def create_from_args(cls, args: argparse.Namespace):
        levels = Branch.level_str_to_int_list(args.levels)
        return cls(args.replicate, levels, cls.BranchDesc.create_from_args(args), not args.quiet)

    @classmethod
    def create_from_hparams(cls, replicate, levels, desc: LotteryDesc, hparams: Hparams, verbose=False):
        return cls(replicate, levels, cls.BranchDesc(desc, hparams), verbose)

    def display_output_location(self):
        print(self.branch_root)

    def return_output_location(self):
        return self.branch_root

    def run(self):
        for self.level in self.levels:
            if self.verbose:
                print('='*82)
                print(f'Branch {self.name()} (Replicate {self.replicate}, Level {self.level})\n' + '-'*82)
            args = {f.name: getattr(self.desc.branch_hparams, f.name)
                    for f in fields(self.BranchHparams) if not f.name.startswith('_')}
            self.branch_function(**args)

    # Initialize instances and subclasses (metaprogramming).
    def __init_subclass__(cls):
        """Metaprogramming: modify the attributes of the subclass based on information in run().

        The goal is to make it possible for users to simply write a single run() method and have
        as much functionality as possible occur automatically. Specifically, this function converts
        the annotations and defaults in run() into a `BranchHparams` property.
        """

        fields = []
        for arg_name, parameter in list(inspect.signature(cls.branch_function).parameters.items())[1:]:
            t = parameter.annotation
            if t == inspect._empty: raise ValueError(f'Argument {arg_name} needs a type annotation.')
            elif t in [str, float, int, bool] or (isinstance(t, type) and issubclass(t, Hparams)):
                if parameter.default != inspect._empty: fields.append((arg_name, t, field(default=parameter.default)))
                else: fields.append((arg_name, t))
            else:
                raise ValueError('Invalid branch type: {}'.format(parameter.annotation))

        fields += [('_name', str, 'Branch Arguments'), ('_description', str, 'Arguments specific to the branch.')]
        setattr(cls, 'BranchHparams', make_dataclass('BranchHparams', fields, bases=(Hparams,)))
        setattr(cls, 'BranchDesc', make_BranchDesc(cls.BranchHparams, cls.name()))


