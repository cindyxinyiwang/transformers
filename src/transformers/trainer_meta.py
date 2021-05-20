# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
import collections
import gc
import inspect
import math
import os
import re
import shutil
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


# Integrations must be imported before ML frameworks:
from .integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    init_deepspeed,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_distributed_available,
    is_torch_tpu_available,
)
from .modeling_utils import PreTrainedModel
from .models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from .optimization import Adafactor, AdamW, get_scheduler
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedTensorGatherer,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
    speed_metrics,
)
from .training_args import ParallelMode, TrainingArguments
from .utils import logging
from .aug_utils import switchout, mlm_switch_tokens
from .trainer import Trainer

_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_distributed_available():
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
else:
    import torch.distributed as dist

if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


def _model_unwrap(model: nn.Module) -> nn.Module:
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return _model_unwrap(model.module)
    else:
        return model


class TrainerMeta(Trainer):
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.PreTrainedModel` or :obj:`torch.nn.Module`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.

            .. note::

                :class:`~transformers.Trainer` is optimized to work with the :class:`~transformers.PreTrainedModel`
                provided by the library. You can still use your own models defined as :obj:`torch.nn.Module` as long as
                they work the same way as the 🤗 Transformers models.
        args (:class:`~transformers.TrainingArguments`, `optional`):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~transformers.TrainingArguments` with the ``output_dir`` set to a directory named `tmp_trainer` in
            the current directory if not provided.
        data_collator (:obj:`DataCollator`, `optional`):
            The function to use to form a batch from a list of elements of :obj:`train_dataset` or :obj:`eval_dataset`.
            Will default to :func:`~transformers.default_data_collator` if no ``tokenizer`` is provided, an instance of
            :func:`~transformers.DataCollatorWithPadding` otherwise.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
            The dataset to use for training. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation. If it is an :obj:`datasets.Dataset`, columns not accepted by the
             ``model.forward()`` method are automatically removed.
        tokenizer (:class:`PreTrainedTokenizerBase`, `optional`):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (:obj:`Callable[[], PreTrainedModel]`, `optional`):
            A function that instantiates the model to be used. If provided, each call to
            :meth:`~transformers.Trainer.train` will start from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune trial object, to be
            able to choose different architectures according to hyper parameters (such as layer count, sizes of inner
            layers, dropout probabilities etc).
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        callbacks (List of :obj:`~transformers.TrainerCallback`, `optional`):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in :doc:`here <callback>`.

            If you want to remove one of the default callbacks used, use the :meth:`Trainer.remove_callback` method.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a
          :class:`~transformers.PreTrainedModel` subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under ``DeepSpeed``,
          the inner model is wrapped in ``DeepSpeed`` and then again in ``torch.nn.DistributedDataParallel``. If the
          inner model hasn't been wrapped, then ``self.model_wrapped`` is the same as ``self.model``.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to :obj:`False` if model parallel or deepspeed is used, or if the default
          ``TrainingArguments.place_model_on_device`` is overridden to return :obj:`False` .
        - **is_in_train** -- Whether or not a model is currently running ``train`` (e.g. when ``evaluate`` is called
          while in ``train``)

    """

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        meta_train_dataset: Dataset = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        augment_data_collator: Optional[DataCollator] = None,
    ):
        self.meta_train_dataset = meta_train_dataset
        Trainer.__init__(self, model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, augment_data_collator)

    def _get_meta_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.meta_train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.meta_train_dataset, collections.abc.Sized
        ):
            return None

        # Gather the number of processes and this process index.
        if self.args.parallel_mode == ParallelMode.TPU:
            num_processes = xm.xrt_world_size()
            process_index = xm.get_ordinal()
        elif (
            self.args.parallel_mode == ParallelMode.DISTRIBUTED
            or self.args.parallel_mode == ParallelMode.SAGEMAKER_DISTRIBUTED
        ):
            num_processes = dist.get_world_size()
            process_index = dist.get_rank()
        else:
            num_processes = 1
            process_index = 0

        # Build the sampler.
        if self.args.group_by_length:
            if num_processes <= 1:
                return LengthGroupedSampler(self.meta_train_dataset, self.args.train_batch_size)
            else:
                return DistributedLengthGroupedSampler(
                    self.meta_train_dataset, self.args.train_batch_size, num_replicas=num_processes, rank=process_index
                )

        else:
            if num_processes <= 1:
                return RandomSampler(self.meta_train_dataset)
            else:
                return DistributedSampler(self.meta_train_dataset, num_replicas=num_processes, rank=process_index)

    def get_meta_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.meta_train_dataset is None:
            raise ValueError("Trainer: training requires a meta_train_dataset.")
        meta_train_sampler = self._get_meta_train_sampler()

        return DataLoader(
            self.meta_train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=meta_train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None

        # Gather the number of processes and this process index.
        if self.args.parallel_mode == ParallelMode.TPU:
            num_processes = xm.xrt_world_size()
            process_index = xm.get_ordinal()
        elif (
            self.args.parallel_mode == ParallelMode.DISTRIBUTED
            or self.args.parallel_mode == ParallelMode.SAGEMAKER_DISTRIBUTED
        ):
            num_processes = dist.get_world_size()
            process_index = dist.get_rank()
        else:
            num_processes = 1
            process_index = 0

        # Build the sampler.
        if self.args.group_by_length:
            if num_processes <= 1:
                return LengthGroupedSampler(self.train_dataset, self.args.train_batch_size)
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset, self.args.train_batch_size, num_replicas=num_processes, rank=process_index
                )

        else:
            if num_processes <= 1:
                return RandomSampler(self.train_dataset)
            else:
                return DistributedSampler(self.train_dataset, num_replicas=num_processes, rank=process_index)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.augment_data_collator if self.augment_data_collator is not None else self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        if is_torch_tpu_available():
            return SequentialDistributedSampler(eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        elif self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        elif is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            self._remove_unused_columns(eval_dataset, description="evaluation")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")
        elif is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            self._remove_unused_columns(test_dataset, description="test")
        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str`, `optional`):
                Local path to a saved checkpoint as saved by a previous instance of :class:`~transformers.Trainer`. If
                present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        # memory metrics - must set up as early as possible
        #self._memory_tracker.start()

        self.is_in_train = True

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            logger.info(f"Loading model from {resume_from_checkpoint}).")
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(resume_from_checkpoint)
                model_reloaded = True
            else:
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self.model = self.model.to(self.args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        meta_train_dataloader = self.get_meta_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        if self.args.deepspeed:
            model, optimizer, lr_scheduler = init_deepspeed(self, num_training_steps=max_steps)
            self.model = model.module
            self.model_wrapped = model  # will get further wrapped in DDP
            self.deepspeed = model  # DeepSpeedEngine object
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        else:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        if is_torch_tpu_available():
            world_size = xm.xrt_world_size()
        elif self.args.local_rank != -1:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * world_size
        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
                if self.meta_train_dataset is not None:
                    meta_train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if train_dataset_is_sized
                else self.args.max_steps * self.args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                if self.meta_train_dataset is not None:
                    try:
                        meta_inputs = next(meta_epoch_iterator)
                    except:
                        if is_torch_tpu_available():
                            meta_parallel_loader = pl.ParallelLoader(meta_train_dataloader, [self.args.device]).per_device_loader(
                                self.args.device
                            )
                            meta_epoch_iterator = iter(meta_parallel_loader)
                        else:
                            meta_epoch_iterator = iter(meta_train_dataloader)
                        meta_inputs = next(meta_epoch_iterator)
                else:
                    meta_inputs = None

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if (
                    ((step + 1) % self.args.gradient_accumulation_steps != 0)
                    and self.args.local_rank != -1
                    and not self.args.deepspeed
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs, meta_inputs)
                else:
                    tr_loss += self.training_step(model, inputs, meta_inputs)
                self._total_flos += float(self.floating_point_ops(inputs))

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                            )

                    # Optimizer step
                    if self.deepspeed:
                        self.deepspeed.step()
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint)
                if self.place_model_on_device:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        if self.deepspeed:
            # free up any memory that might be useful for eval
            self.deepspeed = None
            self.optimizer = None
            self.lr_scheduler = None
            self.model_wrapped = self.model
            gc.collect()  # force memory release
            # to restore normal behavior outside of train replay the place_model_on_device logic w/o deepspeed
            self.place_model_on_device = self.args.place_model_on_device
            if self.is_model_parallel:
                self.place_model_on_device = False

        self.is_in_train = False

        #self._memory_tracker.stop_and_update_metrics(metrics)

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], meta_inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()

        #print(inputs)
        # input_ids, labels, attention_mask
        #if self.args.tau > 0:
        #    #inputs["input_ids"] = switchout(inputs["input_ids"], inputs["attention_mask"], tau=self.args.tau, tokenizer=self.tokenizer)
        #    inputs["input_ids"] = mlm_switch_tokens(inputs["input_ids"], inputs["attention_mask"], p=self.args.tau, tokenizer=self.tokenizer, pretrained_model=model)
        inputs = self._prepare_inputs(inputs)
        meta_inputs = self._prepare_inputs(meta_inputs)
        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
                meta_loss = self.compute_loss(model, meta_inputs)
                loss = loss+meta_loss
        else:
            loss = self.compute_loss(model, inputs)
            meta_loss = self.compute_loss(model, meta_inputs)
            loss = loss+meta_loss

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class TrainerMetaGradMask(TrainerMeta):
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.PreTrainedModel` or :obj:`torch.nn.Module`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.

            .. note::

                :class:`~transformers.Trainer` is optimized to work with the :class:`~transformers.PreTrainedModel`
                provided by the library. You can still use your own models defined as :obj:`torch.nn.Module` as long as
                they work the same way as the 🤗 Transformers models.
        args (:class:`~transformers.TrainingArguments`, `optional`):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~transformers.TrainingArguments` with the ``output_dir`` set to a directory named `tmp_trainer` in
            the current directory if not provided.
        data_collator (:obj:`DataCollator`, `optional`):
            The function to use to form a batch from a list of elements of :obj:`train_dataset` or :obj:`eval_dataset`.
            Will default to :func:`~transformers.default_data_collator` if no ``tokenizer`` is provided, an instance of
            :func:`~transformers.DataCollatorWithPadding` otherwise.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
            The dataset to use for training. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation. If it is an :obj:`datasets.Dataset`, columns not accepted by the
             ``model.forward()`` method are automatically removed.
        tokenizer (:class:`PreTrainedTokenizerBase`, `optional`):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (:obj:`Callable[[], PreTrainedModel]`, `optional`):
            A function that instantiates the model to be used. If provided, each call to
            :meth:`~transformers.Trainer.train` will start from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune trial object, to be
            able to choose different architectures according to hyper parameters (such as layer count, sizes of inner
            layers, dropout probabilities etc).
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        callbacks (List of :obj:`~transformers.TrainerCallback`, `optional`):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in :doc:`here <callback>`.

            If you want to remove one of the default callbacks used, use the :meth:`Trainer.remove_callback` method.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a
          :class:`~transformers.PreTrainedModel` subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under ``DeepSpeed``,
          the inner model is wrapped in ``DeepSpeed`` and then again in ``torch.nn.DistributedDataParallel``. If the
          inner model hasn't been wrapped, then ``self.model_wrapped`` is the same as ``self.model``.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to :obj:`False` if model parallel or deepspeed is used, or if the default
          ``TrainingArguments.place_model_on_device`` is overridden to return :obj:`False` .
        - **is_in_train** -- Whether or not a model is currently running ``train`` (e.g. when ``evaluate`` is called
          while in ``train``)

    """

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        meta_train_dataset: Dataset = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        augment_data_collator: Optional[DataCollator] = None,
    ):
        self.meta_train_dataset = meta_train_dataset
        TrainerMeta.__init__(self, model, args, data_collator, train_dataset, meta_train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, augment_data_collator)
        self.trainable_params = [p for (n, p) in model.named_parameters() if "layer.5" in n]
        #self.trainable_params = [p for (n, p) in model.named_parameters() if p.requires_grad]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], meta_inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()

        #print(inputs)
        # input_ids, labels, attention_mask
        #if self.args.tau > 0:
        #    #inputs["input_ids"] = switchout(inputs["input_ids"], inputs["attention_mask"], tau=self.args.tau, tokenizer=self.tokenizer)
        #    inputs["input_ids"] = mlm_switch_tokens(inputs["input_ids"], inputs["attention_mask"], p=self.args.tau, tokenizer=self.tokenizer, pretrained_model=model)
        inputs = self._prepare_inputs(inputs)
        meta_inputs = self._prepare_inputs(meta_inputs)
        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
                meta_loss = self.compute_loss(model, meta_inputs)
                loss = loss+meta_loss
        else:
            loss = self.compute_loss(model, inputs, reduction="none")
            meta_loss = self.compute_loss(model, meta_inputs, reduction="mean")
            
            z = torch.ones(loss.shape).to(loss.device)
            z.requires_grad = True
            loss_grad = torch.autograd.grad(loss, self.trainable_params, grad_outputs=z, create_graph=True)
            meta_loss_grad = torch.autograd.grad(meta_loss, self.trainable_params, retain_graph=True)
            dot_prod = torch.autograd.grad(loss_grad, z, grad_outputs=meta_loss_grad)[0]

            gradient_mask = (dot_prod > 0)
            #print(gradient_mask.long().sum())
            #print((~gradient_mask).long().sum())
            labels = inputs["labels"]
            labels_mask = (labels.view(-1) != -100)
            mask = (labels_mask & gradient_mask).float()
            loss = (loss*mask).sum() / (mask.sum()+1e-8)

            loss = loss+meta_loss

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, reduction="mean"):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        inputs["reduction"] = reduction
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


