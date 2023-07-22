# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from general_setups import *
from branches import *




# args = argparse.Namespace(apex_fp16=False,
#             batch_size=128,
#             batchnorm_frozen=False,
#             batchnorm_init='uniform',
#             blur_factor=None,
#             data_order_seed=None,
#             dataset_name='cifar10',
#             default_hparams='cifar_resnet_20',
#             display_output_location=False,
#             do_not_augment=False,
#             evaluate_only_at_end=False,
#             gamma=0.1,
#             levels=20,
#             lr=0.1,
#             milestone_steps='80ep,120ep',
#             model_init='kaiming_normal',
#             model_name='cifar_resnet_20',
#             momentum=0.9,
#             nesterov_momentum=0.0,
#             num_workers=0,
#             optimizer_name='sgd',
#             others_frozen=False,
#             others_frozen_exceptions=None,
#             output_frozen=False,
#             platform='local',
#             pretrain=False,
#             pruning_fraction=0.2,
#             pruning_layers_to_ignore=None,
#             pruning_strategy='sparse_global',
#             quiet=False,
#             random_labels_fraction=None,
#             replicate=5,
#             rewinding_steps=None,
#             subcommand='lottery',
#             subsample_fraction=None,
#             training_steps='160ep',
#             transformation_seed=None,
#             unsupervised_labels=None,
#             warmup_steps=None,
#             weight_decay=0.0001)

# LotteryRunner.create_from_args(args).run()

branch_args = argparse.Namespace(apex_fp16=False,
          batch_size=128,
          batchnorm_frozen=False,
          batchnorm_init='uniform',
          blur_factor=None,
          branch_name='randomly_prune',
          data_order_seed=None,
          dataset_name='cifar10',
          default_hparams='cifar_resnet_20',
          display_output_location=False,
          do_not_augment=False,
          evaluate_only_at_end=False,
          gamma=0.1,
          layers_to_ignore='',
          levels='2',
          lr=0.1,
          milestone_steps='80ep,120ep',
          model_init='kaiming_normal',
          model_name='cifar_resnet_20',
          momentum=0.9,
          nesterov_momentum=0.0,
          num_workers=0,
          optimizer_name='sgd',
          others_frozen=False,
          others_frozen_exceptions=None,
          output_frozen=False,
          platform='local',
          pretrain=False,
          pretrain_training_steps=None,
          pruning_fraction=0.2,
          pruning_layers_to_ignore=None,
          pruning_strategy='sparse_global',
          quiet=False,
          random_labels_fraction=None,
          replicate=1,
          rewinding_steps=None,
          seed=7,
          start_at='rewind',
          strategy="3dfilterpos & 2dfilterwise",#'2dfilterwise', #"3dfilterpos"
          start_at_step_zero=False,
          subcommand='lottery_branch',
          subsample_fraction=None,
          training_steps='160ep',
          transformation_seed=None,
          unsupervised_labels=None,
          warmup_steps=None,
          weight_decay=0.0001)
RandomPruneBranch.create_from_args(branch_args).run()





