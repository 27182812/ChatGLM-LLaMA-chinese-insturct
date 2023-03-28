# Copyright 2020 The HuggingFace Team. All rights reserved.
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
Integrations with other Python libraries.
"""
import functools
import importlib.util
import json
import numbers
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from . import __version__ as version
from .utils import flatten_dict, is_datasets_available, is_torch_available, logging


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

# comet_ml requires to be imported before any ML frameworks
_has_comet = importlib.util.find_spec("comet_ml") is not None and os.getenv("COMET_MODE", "").upper() != "DISABLED"
if _has_comet:
    try:
        import comet_ml  # noqa: F401

        if hasattr(comet_ml, "config") and comet_ml.config.get_config("comet.api_key"):
            _has_comet = True
        else:
            if os.getenv("COMET_MODE", "").upper() != "DISABLED":
                logger.warning("comet_ml is installed but `COMET_API_KEY` is not set.")
            _has_comet = False
    except (ImportError, ValueError):
        _has_comet = False

_has_neptune = importlib.util.find_spec("neptune") is not None
if TYPE_CHECKING and _has_neptune:
    from neptune.new.metadata_containers.run import Run

from .trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from .trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from .training_args import ParallelMode  # noqa: E402
from .utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402


# Integration functions:
def is_wandb_available():
    # any value of WANDB_DISABLED disables wandb
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        logger.warning(
            "Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        return False
    return importlib.util.find_spec("wandb") is not None


def is_clearml_available():
    return importlib.util.find_spec("clearml") is not None


def is_comet_available():
    return _has_comet


def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


def is_optuna_available():
    return importlib.util.find_spec("optuna") is not None


def is_ray_available():
    return importlib.util.find_spec("ray") is not None


def is_ray_tune_available():
    if not is_ray_available():
        return False
    return importlib.util.find_spec("ray.tune") is not None


def is_sigopt_available():
    return importlib.util.find_spec("sigopt") is not None


def is_azureml_available():
    if importlib.util.find_spec("azureml") is None:
        return False
    if importlib.util.find_spec("azureml.core") is None:
        return False
    return importlib.util.find_spec("azureml.core.run") is not None


def is_mlflow_available():
    if os.getenv("DISABLE_MLFLOW_INTEGRATION", "FALSE").upper() == "TRUE":
        return False
    return importlib.util.find_spec("mlflow") is not None


def is_dagshub_available():
    return None not in [importlib.util.find_spec("dagshub"), importlib.util.find_spec("mlflow")]


def is_fairscale_available():
    return importlib.util.find_spec("fairscale") is not None


def is_neptune_available():
    return _has_neptune


def is_codecarbon_available():
    return importlib.util.find_spec("codecarbon") is not None


def hp_params(trial):
    if is_optuna_available():
        import optuna

        if isinstance(trial, optuna.Trial):
            return trial.params
    if is_ray_tune_available():
        if isinstance(trial, dict):
            return trial

    if is_sigopt_available():
        if isinstance(trial, dict):
            return trial

    if is_wandb_available():
        if isinstance(trial, dict):
            return trial

    raise RuntimeError(f"Unknown type for trial {trial.__class__}")


def default_hp_search_backend():
    if is_optuna_available():
        return "optuna"
    elif is_ray_tune_available():
        return "ray"
    elif is_sigopt_available():
        return "sigopt"


def run_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import optuna

    if trainer.args.process_index == 0:

        def _objective(trial, checkpoint_dir=None):
            checkpoint = None
            if checkpoint_dir:
                for subdir in os.listdir(checkpoint_dir):
                    if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                        checkpoint = os.path.join(checkpoint_dir, subdir)
            trainer.objective = None
            if trainer.args.world_size > 1:
                if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                    raise RuntimeError("only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently.")
                trainer._hp_search_setup(trial)
                torch.distributed.broadcast_object_list(pickle.dumps(trainer.args), src=0)
                trainer.train(resume_from_checkpoint=checkpoint)
            else:
                trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
            # If there hasn't been any evaluation during the training loop.
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
            return trainer.objective

        timeout = kwargs.pop("timeout", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        study = optuna.create_study(direction=direction, **kwargs)
        study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
        best_trial = study.best_trial
        return BestRun(str(best_trial.number), best_trial.value, best_trial.params)
    else:
        for i in range(n_trials):
            trainer.objective = None
            args_main_rank = list(pickle.dumps(trainer.args))
            if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise RuntimeError("only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently.")
            torch.distributed.broadcast_object_list(args_main_rank, src=0)
            args = pickle.loads(bytes(args_main_rank))
            for key, value in asdict(args).items():
                if key != "local_rank":
                    setattr(trainer.args, key, value)
            trainer.train(resume_from_checkpoint=None)
            # If there hasn't been any evaluation during the training loop.
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
        return None


def run_hp_search_ray(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import ray

    def _objective(trial, local_trainer, checkpoint_dir=None):
        try:
            from transformers.utils.notebook import NotebookProgressCallback

            if local_trainer.pop_callback(NotebookProgressCallback):
                local_trainer.add_callback(ProgressCallback)
        except ModuleNotFoundError:
            pass

        checkpoint = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    checkpoint = os.path.join(checkpoint_dir, subdir)
        local_trainer.objective = None
        local_trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(local_trainer, "objective", None) is None:
            metrics = local_trainer.evaluate()
            local_trainer.objective = local_trainer.compute_objective(metrics)
            local_trainer._tune_save_checkpoint()
            ray.tune.report(objective=local_trainer.objective, **metrics, done=True)

    if not trainer._memory_tracker.skip_memory_metrics:
        from .trainer_utils import TrainerMemoryTracker

        logger.warning(
            "Memory tracking for your Trainer is currently "
            "enabled. Automatically disabling the memory tracker "
            "since the memory tracker is not serializable."
        )
        trainer._memory_tracker = TrainerMemoryTracker(skip_memory_metrics=True)

    # The model and TensorBoard writer do not pickle so we have to remove them (if they exists)
    # while doing the ray hp search.
    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    trainer.model = None

    # Setup default `resources_per_trial`.
    if "resources_per_trial" not in kwargs:
        # Default to 1 CPU and 1 GPU (if applicable) per trial.
        kwargs["resources_per_trial"] = {"cpu": 1}
        if trainer.args.n_gpu > 0:
            kwargs["resources_per_trial"]["gpu"] = 1
        resource_msg = "1 CPU" + (" and 1 GPU" if trainer.args.n_gpu > 0 else "")
        logger.info(
            "No `resources_per_trial` arg was passed into "
            "`hyperparameter_search`. Setting it to a default value "
            f"of {resource_msg} for each trial."
        )
    # Make sure each trainer only uses GPUs that were allocated per trial.
    gpus_per_trial = kwargs["resources_per_trial"].get("gpu", 0)
    trainer.args._n_gpu = gpus_per_trial

    # Setup default `progress_reporter`.
    if "progress_reporter" not in kwargs:
        from ray.tune import CLIReporter

        kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])
    if "keep_checkpoints_num" in kwargs and kwargs["keep_checkpoints_num"] > 0:
        # `keep_checkpoints_num=0` would disabled checkpointing
        trainer.use_tune_checkpoints = True
        if kwargs["keep_checkpoints_num"] > 1:
            logger.warning(
                f"Currently keeping {kwargs['keep_checkpoints_num']} checkpoints for each trial. "
                "Checkpoints are usually huge, "
                "consider setting `keep_checkpoints_num=1`."
            )
    if "scheduler" in kwargs:
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule, PopulationBasedTraining

        # Check if checkpointing is enabled for PopulationBasedTraining
        if isinstance(kwargs["scheduler"], PopulationBasedTraining):
            if not trainer.use_tune_checkpoints:
                logger.warning(
                    "You are using PopulationBasedTraining but you haven't enabled checkpointing. "
                    "This means your trials will train from scratch everytime they are exploiting "
                    "new configurations. Consider enabling checkpointing by passing "
                    "`keep_checkpoints_num=1` as an additional argument to `Trainer.hyperparameter_search`."
                )

        # Check for `do_eval` and `eval_during_training` for schedulers that require intermediate reporting.
        if isinstance(
            kwargs["scheduler"], (ASHAScheduler, MedianStoppingRule, HyperBandForBOHB, PopulationBasedTraining)
        ) and (not trainer.args.do_eval or trainer.args.evaluation_strategy == IntervalStrategy.NO):
            raise RuntimeError(
                "You are using {cls} as a scheduler but you haven't enabled evaluation during training. "
                "This means your trials will not report intermediate results to Ray Tune, and "
                "can thus not be stopped early or used to exploit other trials parameters. "
                "If this is what you want, do not use {cls}. If you would like to use {cls}, "
                "make sure you pass `do_eval=True` and `evaluation_strategy='steps'` in the "
                "Trainer `args`.".format(cls=type(kwargs["scheduler"]).__name__)
            )

    trainable = ray.tune.with_parameters(_objective, local_trainer=trainer)

    @functools.wraps(trainable)
    def dynamic_modules_import_trainable(*args, **kwargs):
        """
        Wrapper around `tune.with_parameters` to ensure datasets_modules are loaded on each Actor.

        Without this, an ImportError will be thrown. See https://github.com/huggingface/transformers/issues/11565.

        Assumes that `_objective`, defined above, is a function.
        """
        if is_datasets_available():
            import datasets.load

            dynamic_modules_path = os.path.join(datasets.load.init_dynamic_modules(), "__init__.py")
            # load dynamic_modules from path
            spec = importlib.util.spec_from_file_location("datasets_modules", dynamic_modules_path)
            datasets_modules = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = datasets_modules
            spec.loader.exec_module(datasets_modules)
        return trainable(*args, **kwargs)

    # special attr set by tune.with_parameters
    if hasattr(trainable, "__mixins__"):
        dynamic_modules_import_trainable.__mixins__ = trainable.__mixins__

    analysis = ray.tune.run(
        dynamic_modules_import_trainable,
        config=trainer.hp_space(None),
        num_samples=n_trials,
        **kwargs,
    )
    best_trial = analysis.get_best_trial(metric="objective", mode=direction[:3], scope=trainer.args.ray_scope)
    best_run = BestRun(best_trial.trial_id, best_trial.last_result["objective"], best_trial.config)
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)
    return best_run


def run_hp_search_sigopt(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import sigopt

    from transformers.utils.versions import importlib_metadata

    if trainer.args.process_index == 0:
        if importlib_metadata.version("sigopt") >= "8.0.0":
            sigopt.set_project("huggingface")

            experiment = sigopt.create_experiment(
                name="huggingface-tune",
                type="offline",
                parameters=trainer.hp_space(None),
                metrics=[{"name": "objective", "objective": direction, "strategy": "optimize"}],
                parallel_bandwidth=1,
                budget=n_trials,
            )

            logger.info(f"created experiment: https://app.sigopt.com/experiment/{experiment.id}")

            for run in experiment.loop():
                with run:
                    trainer.objective = None
                    if trainer.args.world_size > 1:
                        if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                            raise RuntimeError("only support DDP Sigopt HPO for ParallelMode.DISTRIBUTED currently.")
                        trainer._hp_search_setup(run.run)
                        torch.distributed.broadcast_object_list(pickle.dumps(trainer.args), src=0)
                        trainer.train(resume_from_checkpoint=None)
                    else:
                        trainer.train(resume_from_checkpoint=None, trial=run.run)
                    # If there hasn't been any evaluation during the training loop.
                    if getattr(trainer, "objective", None) is None:
                        metrics = trainer.evaluate()
                        trainer.objective = trainer.compute_objective(metrics)
                    run.log_metric("objective", trainer.objective)

            best = list(experiment.get_best_runs())[0]
            best_run = BestRun(best.id, best.values["objective"].value, best.assignments)
        else:
            from sigopt import Connection

            conn = Connection()
            proxies = kwargs.pop("proxies", None)
            if proxies is not None:
                conn.set_proxies(proxies)

            experiment = conn.experiments().create(
                name="huggingface-tune",
                parameters=trainer.hp_space(None),
                metrics=[{"name": "objective", "objective": direction, "strategy": "optimize"}],
                parallel_bandwidth=1,
                observation_budget=n_trials,
                project="huggingface",
            )
            logger.info(f"created experiment: https://app.sigopt.com/experiment/{experiment.id}")

            while experiment.progress.observation_count < experiment.observation_budget:
                suggestion = conn.experiments(experiment.id).suggestions().create()
                trainer.objective = None
                if trainer.args.world_size > 1:
                    if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                        raise RuntimeError("only support DDP Sigopt HPO for ParallelMode.DISTRIBUTED currently.")
                    trainer._hp_search_setup(suggestion)
                    torch.distributed.broadcast_object_list(pickle.dumps(trainer.args), src=0)
                    trainer.train(resume_from_checkpoint=None)
                else:
                    trainer.train(resume_from_checkpoint=None, trial=suggestion)
                # If there hasn't been any evaluation during the training loop.
                if getattr(trainer, "objective", None) is None:
                    metrics = trainer.evaluate()
                    trainer.objective = trainer.compute_objective(metrics)

                values = [{"name": "objective", "value": trainer.objective}]
                obs = conn.experiments(experiment.id).observations().create(suggestion=suggestion.id, values=values)
                logger.info(f"[suggestion_id, observation_id]: [{suggestion.id}, {obs.id}]")
                experiment = conn.experiments(experiment.id).fetch()

            best = list(conn.experiments(experiment.id).best_assignments().fetch().iterate_pages())[0]
            best_run = BestRun(best.id, best.value, best.assignments)
        return best_run
    else:
        for i in range(n_trials):
            trainer.objective = None
            args_main_rank = list(pickle.dumps(trainer.args))
            if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise RuntimeError("only support DDP Sigopt HPO for ParallelMode.DISTRIBUTED currently.")
            torch.distributed.broadcast_object_list(args_main_rank, src=0)
            args = pickle.loads(bytes(args_main_rank))
            for key, value in asdict(args).items():
                if key != "local_rank":
                    setattr(trainer.args, key, value)
            trainer.train(resume_from_checkpoint=None)
            # If there hasn't been any evaluation during the training loop.
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
        return None


def run_hp_search_wandb(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    from .integrations import is_wandb_available

    if not is_wandb_available():
        raise ImportError("This function needs wandb installed: `pip install wandb`")
    import wandb

    # add WandbCallback if not already added in trainer callbacks
    reporting_to_wandb = False
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, WandbCallback):
            reporting_to_wandb = True
            break
    if not reporting_to_wandb:
        trainer.add_callback(WandbCallback())
    trainer.args.report_to = "wandb"
    best_trial = {"run_id": None, "objective": None, "hyperparameters": None}
    sweep_id = kwargs.pop("sweep_id", None)
    project = kwargs.pop("project", None)
    name = kwargs.pop("name", None)
    entity = kwargs.pop("entity", None)
    metric = kwargs.pop("metric", "eval/loss")

    sweep_config = trainer.hp_space(None)
    sweep_config["metric"]["goal"] = direction
    sweep_config["metric"]["name"] = metric
    if name:
        sweep_config["name"] = name

    def _objective():
        run = wandb.run if wandb.run else wandb.init()
        trainer.state.trial_name = run.name
        run.config.update({"assignments": {}, "metric": metric})
        config = wandb.config

        trainer.objective = None

        trainer.train(resume_from_checkpoint=None, trial=vars(config)["_items"])
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
            format_metrics = rewrite_logs(metrics)
            if metric not in format_metrics:
                logger.warning(
                    f"Provided metric {metric} not found. This might result in unexpected sweeps charts. The available"
                    f" metrics are {format_metrics.keys()}"
                )
        best_score = False
        if best_trial["run_id"] is not None:
            if direction == "minimize":
                best_score = trainer.objective < best_trial["objective"]
            elif direction == "maximize":
                best_score = trainer.objective > best_trial["objective"]

        if best_score or best_trial["run_id"] is None:
            best_trial["run_id"] = run.id
            best_trial["objective"] = trainer.objective
            best_trial["hyperparameters"] = dict(config)

        return trainer.objective

    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity) if not sweep_id else sweep_id
    logger.info(f"wandb sweep id - {sweep_id}")
    wandb.agent(sweep_id, function=_objective, count=n_trials)

    return BestRun(best_trial["run_id"], best_trial["objective"], best_trial["hyperparameters"])


def get_available_reporting_integrations():
    integrations = []
    if is_azureml_available() and not is_mlflow_available():
        integrations.append("azure_ml")
    if is_comet_available():
        integrations.append("comet_ml")
    if is_dagshub_available():
        integrations.append("dagshub")
    if is_mlflow_available():
        integrations.append("mlflow")
    if is_neptune_available():
        integrations.append("neptune")
    if is_tensorboard_available():
        integrations.append("tensorboard")
    if is_wandb_available():
        integrations.append("wandb")
    if is_codecarbon_available():
        integrations.append("codecarbon")
    if is_clearml_available():
        integrations.append("clearml")
    return integrations


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class TensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writer=None):
        has_tensorboard = is_tensorboard_available()
        if not has_tensorboard:
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401

                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter

                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writer = tb_writer

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writer = self._SummaryWriter(log_dir=log_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(self.tb_writer, "add_hparams"):
                self.tb_writer.add_hparams(args.to_sanitized_dict(), metric_dict={})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None


class WandbCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that logs metrics, media, model checkpoints to [Weight and Biases](https://www.wandb.com/).
    """

    def __init__(self):
        has_wandb = is_wandb_available()
        if not has_wandb:
            raise RuntimeError("WandbCallback requires wandb to be installed. Run `pip install wandb`.")
        if has_wandb:
            import wandb

            self._wandb = wandb
        self._initialized = False
        # log model
        if os.getenv("WANDB_LOG_MODEL", "FALSE").upper() in ENV_VARS_TRUE_VALUES.union({"TRUE"}):
            DeprecationWarning(
                f"Setting `WANDB_LOG_MODEL` as {os.getenv('WANDB_LOG_MODEL')} is deprecated and will be removed in "
                "version 5 of transformers. Use one of `'end'` or `'checkpoint'` instead."
            )
            logger.info(f"Setting `WANDB_LOG_MODEL` from {os.getenv('WANDB_LOG_MODEL')} to `end` instead")
            self._log_model = "end"
        else:
            self._log_model = os.getenv("WANDB_LOG_MODEL", "false").lower()

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.

        One can subclass and override this method to customize the setup if needed. Find more information
        [here](https://docs.wandb.ai/guides/integrations/huggingface). You can also override the following environment
        variables:

        Environment:
        - **WANDB_LOG_MODEL** (`str`, *optional*, defaults to `"false"`):
            Whether to log model and checkpoints during training. Can be `"end"`, `"checkpoint"` or `"false"`. If set
            to `"end"`, the model will be uploaded at the end of training. If set to `"checkpoint"`, the checkpoint
            will be uploaded every `args.save_steps` . If set to `"false"`, the model will not be uploaded. Use along
            with [`~transformers.TrainingArguments.load_best_model_at_end`] to upload best model.

            <Deprecated version="5.0">

            Setting `WANDB_LOG_MODEL` as `bool` will be deprecated in version 5 of 🤗 Transformers.

            </Deprecated>
        - **WANDB_WATCH** (`str`, *optional* defaults to `"false"`):
            Can be `"gradients"`, `"all"`, `"parameters"`, or `"false"`. Set to `"all"` to log gradients and
            parameters.
        - **WANDB_PROJECT** (`str`, *optional*, defaults to `"huggingface"`):
            Set this to a custom string to store results in a different project.
        - **WANDB_DISABLED** (`bool`, *optional*, defaults to `False`):
            Whether to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                init_args["name"] = trial_name
                init_args["group"] = args.run_name
            else:
                if not (args.run_name is None or args.run_name == args.output_dir):
                    init_args["name"] = args.run_name

            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    **init_args,
                )
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            _watch_model = os.getenv("WANDB_WATCH", "false")
            if not is_torch_tpu_available() and _watch_model in ("all", "parameters", "gradients"):
                self._wandb.watch(model, log=_watch_model, log_freq=max(100, args.logging_steps))

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._wandb is None:
            return
        hp_search = state.is_hyper_param_search
        if hp_search:
            self._wandb.finish()
            self._initialized = False
            args.run_name = None
        if not self._initialized:
            self.setup(args, state, model, **kwargs)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model in ("end", "checkpoint") and self._initialized and state.is_world_process_zero:
            from .trainer import Trainer

            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                    }
                )
                logger.info("Logging model artifacts. ...")
                model_name = (
                    f"model-{self._wandb.run.id}"
                    if (args.run_name is None or args.run_name == args.output_dir)
                    else f"model-{self._wandb.run.name}"
                )
                artifact = self._wandb.Artifact(name=model_name, type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})

    def on_save(self, args, state, control, **kwargs):
        if self._log_model == "checkpoint" and self._initialized and state.is_world_process_zero:
            checkpoint_metadata = {
                k: v
                for k, v in dict(self._wandb.summary).items()
                if isinstance(v, numbers.Number) and not k.startswith("_")
            }

            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. ...")
            checkpoint_name = (
                f"checkpoint-{self._wandb.run.id}"
                if (args.run_name is None or args.run_name == args.output_dir)
                else f"checkpoint-{self._wandb.run.name}"
            )
            artifact = self._wandb.Artifact(name=checkpoint_name, type="model", metadata=checkpoint_metadata)
            artifact.add_dir(artifact_path)
            self._wandb.log_artifact(artifact, aliases=[f"checkpoint-{state.global_step}"])


class CometCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [Comet ML](https://www.comet.ml/site/).
    """

    def __init__(self):
        if not _has_comet:
            raise RuntimeError("CometCallback requires comet-ml to be installed. Run `pip install comet-ml`.")
        self._initialized = False
        self._log_assets = False

    def setup(self, args, state, model):
        """
        Setup the optional Comet.ml integration.

        Environment:
        - **COMET_MODE** (`str`, *optional*, defaults to `ONLINE`):
            Whether to create an online, offline experiment or disable Comet logging. Can be `OFFLINE`, `ONLINE`, or
            `DISABLED`.
        - **COMET_PROJECT_NAME** (`str`, *optional*):
            Comet project name for experiments.
        - **COMET_OFFLINE_DIRECTORY** (`str`, *optional*):
            Folder to use for saving offline experiments when `COMET_MODE` is `OFFLINE`.
        - **COMET_LOG_ASSETS** (`str`, *optional*, defaults to `TRUE`):
            Whether or not to log training assets (tf event logs, checkpoints, etc), to Comet. Can be `TRUE`, or
            `FALSE`.

        For a number of configurable items in the environment, see
        [here](https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables).
        """
        self._initialized = True
        log_assets = os.getenv("COMET_LOG_ASSETS", "FALSE").upper()
        if log_assets in {"TRUE", "1"}:
            self._log_assets = True
        if state.is_world_process_zero:
            comet_mode = os.getenv("COMET_MODE", "ONLINE").upper()
            experiment = None
            experiment_kwargs = {"project_name": os.getenv("COMET_PROJECT_NAME", "huggingface")}
            if comet_mode == "ONLINE":
                experiment = comet_ml.Experiment(**experiment_kwargs)
                experiment.log_other("Created from", "transformers")
                logger.info("Automatic Comet.ml online logging enabled")
            elif comet_mode == "OFFLINE":
                experiment_kwargs["offline_directory"] = os.getenv("COMET_OFFLINE_DIRECTORY", "./")
                experiment = comet_ml.OfflineExperiment(**experiment_kwargs)
                experiment.log_other("Created from", "transformers")
                logger.info("Automatic Comet.ml offline logging enabled; use `comet upload` when finished")
            if experiment is not None:
                experiment._set_model_graph(model, framework="transformers")
                experiment._log_parameters(args, prefix="args/", framework="transformers")
                if hasattr(model, "config"):
                    experiment._log_parameters(model.config, prefix="config/", framework="transformers")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            experiment = comet_ml.config.get_global_experiment()
            if experiment is not None:
                experiment._log_metrics(logs, step=state.global_step, epoch=state.epoch, framework="transformers")

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            experiment = comet_ml.config.get_global_experiment()
            if experiment is not None:
                if self._log_assets is True:
                    logger.info("Logging checkpoints. This may take time.")
                    experiment.log_asset_folder(
                        args.output_dir, recursive=True, log_file_name=True, step=state.global_step
                    )
                experiment.end()


class AzureMLCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [AzureML](https://pypi.org/project/azureml-sdk/).
    """

    def __init__(self, azureml_run=None):
        if not is_azureml_available():
            raise RuntimeError("AzureMLCallback requires azureml to be installed. Run `pip install azureml-sdk`.")
        self.azureml_run = azureml_run

    def on_init_end(self, args, state, control, **kwargs):
        from azureml.core.run import Run

        if self.azureml_run is None and state.is_world_process_zero:
            self.azureml_run = Run.get_context()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.azureml_run and state.is_world_process_zero:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.azureml_run.log(k, v, description=k)


class MLflowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [MLflow](https://www.mlflow.org/). Can be disabled by setting
    environment variable `DISABLE_MLFLOW_INTEGRATION = TRUE`.
    """

    def __init__(self):
        if not is_mlflow_available():
            raise RuntimeError("MLflowCallback requires mlflow to be installed. Run `pip install mlflow`.")
        import mlflow

        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH

        self._initialized = False
        self._auto_end_run = False
        self._log_artifacts = False
        self._ml_flow = mlflow

    def setup(self, args, state, model):
        """
        Setup the optional MLflow integration.

        Environment:
        - **HF_MLFLOW_LOG_ARTIFACTS** (`str`, *optional*):
            Whether to use MLflow `.log_artifact()` facility to log artifacts. This only makes sense if logging to a
            remote server, e.g. s3 or GCS. If set to `True` or *1*, will copy each saved checkpoint on each save in
            [`TrainingArguments`]'s `output_dir` to the local or remote artifact storage. Using it without a remote
            storage will just copy the files to your artifact location.
        - **MLFLOW_EXPERIMENT_NAME** (`str`, *optional*, defaults to `None`):
            Whether to use an MLflow experiment_name under which to launch the run. Default to `None` which will point
            to the `Default` experiment in MLflow. Otherwise, it is a case sensitive name of the experiment to be
            activated. If an experiment with this name does not exist, a new experiment with this name is created.
        - **MLFLOW_TAGS** (`str`, *optional*):
            A string dump of a dictionary of key/value pair to be added to the MLflow run as tags. Example:
            `os.environ['MLFLOW_TAGS']='{"release.candidate": "RC1", "release.version": "2.2.0"}'`.
        - **MLFLOW_NESTED_RUN** (`str`, *optional*):
            Whether to use MLflow nested runs. If set to `True` or *1*, will create a nested run inside the current
            run.
        - **MLFLOW_RUN_ID** (`str`, *optional*):
            Allow to reattach to an existing run which can be usefull when resuming training from a checkpoint. When
            `MLFLOW_RUN_ID` environment variable is set, `start_run` attempts to resume a run with the specified run ID
            and other parameters are ignored.
        - **MLFLOW_FLATTEN_PARAMS** (`str`, *optional*, defaults to `False`):
            Whether to flatten the parameters dictionary before logging.
        """
        self._log_artifacts = os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._nested_run = os.getenv("MLFLOW_NESTED_RUN", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", None)
        self._flatten_params = os.getenv("MLFLOW_FLATTEN_PARAMS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._run_id = os.getenv("MLFLOW_RUN_ID", None)
        logger.debug(
            f"MLflow experiment_name={self._experiment_name}, run_name={args.run_name}, nested={self._nested_run},"
            f" tags={self._nested_run}"
        )
        if state.is_world_process_zero:
            if self._ml_flow.active_run() is None or self._nested_run or self._run_id:
                if self._experiment_name:
                    # Use of set_experiment() ensure that Experiment is created if not exists
                    self._ml_flow.set_experiment(self._experiment_name)
                self._ml_flow.start_run(run_name=args.run_name, nested=self._nested_run)
                logger.debug(f"MLflow run started with run_id={self._ml_flow.active_run().info.run_id}")
                self._auto_end_run = True
            combined_dict = args.to_dict()
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            combined_dict = flatten_dict(combined_dict) if self._flatten_params else combined_dict
            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f'Trainer is attempting to log a value of "{value}" for key "{name}" as a parameter. MLflow\'s'
                        " log_param() only accepts values no longer than 250 characters so we dropped this attribute."
                        " You can use `MLFLOW_FLATTEN_PARAMS` environment variable to flatten the parameters and"
                        " avoid this message."
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                self._ml_flow.log_params(dict(combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH]))
            mlflow_tags = os.getenv("MLFLOW_TAGS", None)
            if mlflow_tags:
                mlflow_tags = json.loads(mlflow_tags)
                self._ml_flow.set_tags(mlflow_tags)
        self._initialized = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            metrics = {}
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
                else:
                    logger.warning(
                        f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
                        "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
                    )
            self._ml_flow.log_metrics(metrics=metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            if self._auto_end_run and self._ml_flow.active_run():
                self._ml_flow.end_run()

    def on_save(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero and self._log_artifacts:
            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. This may take time.")
            self._ml_flow.pyfunc.log_model(
                ckpt_dir,
                artifacts={"model_path": artifact_path},
                python_model=self._ml_flow.pyfunc.PythonModel(),
            )

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if (
            self._auto_end_run
            and callable(getattr(self._ml_flow, "active_run", None))
            and self._ml_flow.active_run() is not None
        ):
            self._ml_flow.end_run()


class DagsHubCallback(MLflowCallback):
    """
    A [`TrainerCallback`] that logs to [DagsHub](https://dagshub.com/). Extends [`MLflowCallback`]
    """

    def __init__(self):
        super().__init__()
        if not is_dagshub_available():
            raise ImportError("DagsHubCallback requires dagshub to be installed. Run `pip install dagshub`.")

        from dagshub.upload import Repo

        self.Repo = Repo

    def setup(self, *args, **kwargs):
        """
        Setup the DagsHub's Logging integration.

        Environment:
        - **HF_DAGSHUB_LOG_ARTIFACTS** (`str`, *optional*):
                Whether to save the data and model artifacts for the experiment. Default to `False`.
        """

        self.log_artifacts = os.getenv("HF_DAGSHUB_LOG_ARTIFACTS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self.name = os.getenv("HF_DAGSHUB_MODEL_NAME") or "main"
        self.remote = os.getenv("MLFLOW_TRACKING_URI")
        self.repo = self.Repo(
            owner=self.remote.split(os.sep)[-2],
            name=self.remote.split(os.sep)[-1].split(".")[0],
            branch=os.getenv("BRANCH") or "main",
        )
        self.path = Path("artifacts")

        if self.remote is None:
            raise RuntimeError(
                "DagsHubCallback requires the `MLFLOW_TRACKING_URI` environment variable to be set. Did you run"
                " `dagshub.init()`?"
            )

        super().setup(*args, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        if self.log_artifacts:
            if getattr(self, "train_dataloader", None):
                torch.save(self.train_dataloader.dataset, os.path.join(args.output_dir, "dataset.pt"))

            self.repo.directory(str(self.path)).add_dir(args.output_dir)


class NeptuneMissingConfiguration(Exception):
    def __init__(self):
        super().__init__(
            """
        ------ Unsupported ---- We were not able to create new runs. You provided a custom Neptune run to
        `NeptuneCallback` with the `run` argument. For the integration to work fully, provide your `api_token` and
        `project` by saving them as environment variables or passing them to the callback.
        """
        )


class NeptuneCallback(TrainerCallback):
    """TrainerCallback that sends the logs to [Neptune](https://neptune.ai).

    Args:
        api_token (`str`, optional):
            Neptune API token obtained upon registration. You can leave this argument out if you have saved your token
            to the `NEPTUNE_API_TOKEN` environment variable (strongly recommended). See full setup instructions in the
            [docs](https://docs.neptune.ai/getting-started/installation).
        project (`str`, optional):
            Name of an existing Neptune project, in the form: "workspace-name/project-name". You can find and copy the
            name from the project Settings -> Properties in Neptune. If None (default), the value of the
            `NEPTUNE_PROJECT` environment variable will be used.
        name (`str`, optional): Custom name for the run.
        base_namespace (`str`, optional, defaults to "finetuning"): In the Neptune run, the root namespace
            that will contain all of the logged metadata.
        log_parameters (`bool`, optional, defaults to True):
            If True, logs all Trainer arguments and model parameters provided by the Trainer.
        log_checkpoints (`str`, optional, defaults to None):
            If "same", uploads checkpoints whenever they are saved by the Trainer. If "last", uploads only the most
            recently saved checkpoint. If "best", uploads the best checkpoint (among the ones saved by the Trainer). If
            None, does not upload checkpoints.
        run (`Run`, optional):
            Pass a Neptune run object if you want to continue logging to an existing run. Read more about resuming runs
            in the [docs](https://docs.neptune.ai/how-to-guides/neptune-api/resume-run).
        **neptune_run_kwargs (optional):
            Additional keyword arguments to be passed directly to the
            [neptune.init_run()](https://docs.neptune.ai/api-reference/neptune#.init_run) function when a new run is
            created.
    """

    integration_version_key = "source_code/integrations/transformers"
    model_parameters_key = "model_parameters"
    trial_name_key = "trial"
    trial_params_key = "trial_params"
    trainer_parameters_key = "trainer_parameters"
    flat_metrics = {"train/epoch"}

    def __init__(
        self,
        *,
        api_token: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        base_namespace: str = "finetuning",
        run: Optional["Run"] = None,
        log_parameters: bool = True,
        log_checkpoints: Optional[str] = None,
        **neptune_run_kwargs,
    ):
        if not is_neptune_available():
            raise ValueError(
                "NeptuneCallback requires the Neptune client library to be installed. "
                "To install the library, run `pip install neptune-client`."
            )

        from neptune.new.metadata_containers.run import Run

        try:
            from neptune.new.integrations.utils import verify_type
        except ImportError:
            from neptune.new.internal.utils import verify_type

        verify_type("api_token", api_token, (str, type(None)))
        verify_type("project", project, (str, type(None)))
        verify_type("name", name, (str, type(None)))
        verify_type("base_namespace", base_namespace, str)
        verify_type("run", run, (Run, type(None)))
        verify_type("log_parameters", log_parameters, bool)
        verify_type("log_checkpoints", log_checkpoints, (str, type(None)))

        self._base_namespace_path = base_namespace
        self._log_parameters = log_parameters
        self._log_checkpoints = log_checkpoints
        self._initial_run: Optional[Run] = run

        self._run = None
        self._is_monitoring_run = False
        self._run_id = None
        self._force_reset_monitoring_run = False
        self._init_run_kwargs = {"api_token": api_token, "project": project, "name": name, **neptune_run_kwargs}

        self._volatile_checkpoints_dir = None
        self._should_upload_checkpoint = self._log_checkpoints is not None
        self._recent_checkpoint_path = None

        if self._log_checkpoints in {"last", "best"}:
            self._target_checkpoints_namespace = f"checkpoints/{self._log_checkpoints}"
            self._should_clean_recently_uploaded_checkpoint = True
        else:
            self._target_checkpoints_namespace = "checkpoints"
            self._should_clean_recently_uploaded_checkpoint = False

    def _stop_run_if_exists(self):
        if self._run:
            self._run.stop()
            del self._run
            self._run = None

    def _initialize_run(self, **additional_neptune_kwargs):
        from neptune.new import init_run
        from neptune.new.exceptions import NeptuneMissingApiTokenException, NeptuneMissingProjectNameException

        self._stop_run_if_exists()

        try:
            self._run = init_run(**self._init_run_kwargs, **additional_neptune_kwargs)
            self._run_id = self._run["sys/id"].fetch()
        except (NeptuneMissingProjectNameException, NeptuneMissingApiTokenException) as e:
            raise NeptuneMissingConfiguration() from e

    def _use_initial_run(self):
        self._run = self._initial_run
        self._is_monitoring_run = True
        self._run_id = self._run["sys/id"].fetch()
        self._initial_run = None

    def _ensure_run_with_monitoring(self):
        if self._initial_run is not None:
            self._use_initial_run()
        else:
            if not self._force_reset_monitoring_run and self._is_monitoring_run:
                return

            if self._run and not self._is_monitoring_run and not self._force_reset_monitoring_run:
                self._initialize_run(run=self._run_id)
                self._is_monitoring_run = True
            else:
                self._initialize_run()
                self._force_reset_monitoring_run = False

    def _ensure_at_least_run_without_monitoring(self):
        if self._initial_run is not None:
            self._use_initial_run()
        else:
            if not self._run:
                self._initialize_run(
                    run=self._run_id,
                    capture_stdout=False,
                    capture_stderr=False,
                    capture_hardware_metrics=False,
                    capture_traceback=False,
                )
                self._is_monitoring_run = False

    @property
    def run(self):
        if self._run is None:
            self._ensure_at_least_run_without_monitoring()
        return self._run

    @property
    def _metadata_namespace(self):
        return self.run[self._base_namespace_path]

    def _log_integration_version(self):
        self.run[NeptuneCallback.integration_version_key] = version

    def _log_trainer_parameters(self, args):
        self._metadata_namespace[NeptuneCallback.trainer_parameters_key] = args.to_sanitized_dict()

    def _log_model_parameters(self, model):
        if model and hasattr(model, "config") and model.config is not None:
            self._metadata_namespace[NeptuneCallback.model_parameters_key] = model.config.to_dict()

    def _log_hyper_param_search_parameters(self, state):
        if state and hasattr(state, "trial_name"):
            self._metadata_namespace[NeptuneCallback.trial_name_key] = state.trial_name

        if state and hasattr(state, "trial_params") and state.trial_params is not None:
            self._metadata_namespace[NeptuneCallback.trial_params_key] = state.trial_params

    def _log_model_checkpoint(self, source_directory: str, checkpoint: str):
        target_path = relative_path = os.path.join(source_directory, checkpoint)

        if self._volatile_checkpoints_dir is not None:
            consistent_checkpoint_path = os.path.join(self._volatile_checkpoints_dir, checkpoint)
            try:
                shutil.copytree(relative_path, os.path.join(consistent_checkpoint_path, relative_path))
                target_path = consistent_checkpoint_path
            except IOError as e:
                logger.warning(
                    "NeptuneCallback was unable to made a copy of checkpoint due to I/O exception: '{}'."
                    "Could fail trying to upload.".format(e)
                )

        self._metadata_namespace[self._target_checkpoints_namespace].upload_files(target_path)

        if self._should_clean_recently_uploaded_checkpoint and self._recent_checkpoint_path is not None:
            self._metadata_namespace[self._target_checkpoints_namespace].delete_files(self._recent_checkpoint_path)

        self._recent_checkpoint_path = relative_path

    def on_init_end(self, args, state, control, **kwargs):
        self._volatile_checkpoints_dir = None
        if self._log_checkpoints and (args.overwrite_output_dir or args.save_total_limit is not None):
            self._volatile_checkpoints_dir = tempfile.TemporaryDirectory().name

        if self._log_checkpoints == "best" and not args.load_best_model_at_end:
            raise ValueError("To save the best model checkpoint, the load_best_model_at_end argument must be enabled.")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return

        self._ensure_run_with_monitoring()
        self._force_reset_monitoring_run = True

        self._log_integration_version()
        if self._log_parameters:
            self._log_trainer_parameters(args)
            self._log_model_parameters(model)

        if state.is_hyper_param_search:
            self._log_hyper_param_search_parameters(state)

    def on_train_end(self, args, state, control, **kwargs):
        self._stop_run_if_exists()

    def __del__(self):
        if self._volatile_checkpoints_dir is not None:
            shutil.rmtree(self._volatile_checkpoints_dir, ignore_errors=True)

        self._stop_run_if_exists()

    def on_save(self, args, state, control, **kwargs):
        if self._should_upload_checkpoint:
            self._log_model_checkpoint(args.output_dir, f"checkpoint-{state.global_step}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._log_checkpoints == "best":
            best_metric_name = args.metric_for_best_model
            if not best_metric_name.startswith("eval_"):
                best_metric_name = f"eval_{best_metric_name}"

            metric_value = metrics.get(best_metric_name)

            operator = np.greater if args.greater_is_better else np.less

            self._should_upload_checkpoint = state.best_metric is None or operator(metric_value, state.best_metric)

    @classmethod
    def get_run(cls, trainer):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, cls):
                return callback.run

        raise Exception("The trainer doesn't have a NeptuneCallback configured.")

    def on_log(self, args, state, control, logs: Optional[Dict[str, float]] = None, **kwargs):
        if not state.is_world_process_zero:
            return

        if logs is not None:
            for name, value in rewrite_logs(logs).items():
                if isinstance(value, (int, float)):
                    if name in NeptuneCallback.flat_metrics:
                        self._metadata_namespace[name] = value
                    else:
                        self._metadata_namespace[name].log(value, step=state.global_step)


class CodeCarbonCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that tracks the CO2 emission of training.
    """

    def __init__(self):
        if not is_codecarbon_available():
            raise RuntimeError(
                "CodeCarbonCallback requires `codecarbon` to be installed. Run `pip install codecarbon`."
            )
        import codecarbon

        self._codecarbon = codecarbon
        self.tracker = None

    def on_init_end(self, args, state, control, **kwargs):
        if self.tracker is None and state.is_local_process_zero:
            # CodeCarbon will automatically handle environment variables for configuration
            self.tracker = self._codecarbon.EmissionsTracker(output_dir=args.output_dir)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.tracker and state.is_local_process_zero:
            self.tracker.start()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tracker and state.is_local_process_zero:
            self.tracker.stop()


class ClearMLCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [ClearML](https://clear.ml/).

    Environment:
    - **CLEARML_PROJECT** (`str`, *optional*, defaults to `HuggingFace Transformers`):
        ClearML project name.
    - **CLEARML_TASK** (`str`, *optional*, defaults to `Trainer`):
        ClearML task name.
    - **CLEARML_LOG_MODEL** (`bool`, *optional*, defaults to `False`):
        Whether to log models as artifacts during training.
    """

    def __init__(self):
        if is_clearml_available():
            import clearml

            self._clearml = clearml
        else:
            raise RuntimeError("ClearMLCallback requires 'clearml' to be installed. Run `pip install clearml`.")

        self._initialized = False
        self._clearml_task = None

        self._log_model = os.getenv("CLEARML_LOG_MODEL", "FALSE").upper() in ENV_VARS_TRUE_VALUES.union({"TRUE"})

    def setup(self, args, state, model, tokenizer, **kwargs):
        if self._clearml is None:
            return
        if self._initialized:
            return
        if state.is_world_process_zero:
            logger.info("Automatic ClearML logging enabled.")
            if self._clearml_task is None:
                # This might happen when running inside of a pipeline, where the task is already initialized
                # from outside of Hugging Face
                if self._clearml.Task.current_task():
                    self._clearml_task = self._clearml.Task.current_task()
                    self._initialized = True
                    logger.info("External ClearML Task has been connected.")
                else:
                    self._clearml_task = self._clearml.Task.init(
                        project_name=os.getenv("CLEARML_PROJECT", "HuggingFace Transformers"),
                        task_name=os.getenv("CLEARML_TASK", "Trainer"),
                        auto_connect_frameworks={"tensorboard": False, "pytorch": False},
                        output_uri=True,
                    )
                    self._initialized = True
                    logger.info("ClearML Task has been initialized.")

            self._clearml_task.connect(args, "Args")
            if hasattr(model, "config") and model.config is not None:
                self._clearml_task.connect(model.config, "Model Configuration")

    def on_train_begin(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._clearml is None:
            return
        if state.is_hyper_param_search:
            self._initialized = False
        if not self._initialized:
            self.setup(args, state, model, tokenizer, **kwargs)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, metrics=None, logs=None, **kwargs):
        if self._clearml is None:
            return
        if self._clearml_task and state.is_world_process_zero:
            # Close ClearML Task at the end end of training
            self._clearml_task.close()

    def on_log(self, args, state, control, model=None, tokenizer=None, logs=None, **kwargs):
        if self._clearml is None:
            return
        if not self._initialized:
            self.setup(args, state, model, tokenizer, **kwargs)
        if state.is_world_process_zero:
            eval_prefix = "eval_"
            eval_prefix_len = len(eval_prefix)
            test_prefix = "test_"
            test_prefix_len = len(test_prefix)
            single_value_scalars = [
                "train_runtime",
                "train_samples_per_second",
                "train_steps_per_second",
                "train_loss",
                "total_flos",
                "epoch",
            ]
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    if k in single_value_scalars:
                        self._clearml_task.get_logger().report_single_value(name=k, value=v)
                    elif k.startswith(eval_prefix):
                        self._clearml_task.get_logger().report_scalar(
                            title=k[eval_prefix_len:], series="eval", value=v, iteration=state.global_step
                        )
                    elif k.startswith(test_prefix):
                        self._clearml_task.get_logger().report_scalar(
                            title=k[test_prefix_len:], series="test", value=v, iteration=state.global_step
                        )
                    else:
                        self._clearml_task.get_logger().report_scalar(
                            title=k, series="train", value=v, iteration=state.global_step
                        )
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of ClearML logger's  report_scalar() "
                        "is incorrect so we dropped this attribute."
                    )

    def on_save(self, args, state, control, **kwargs):
        if self._log_model and self._clearml_task and state.is_world_process_zero:
            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. This may take time.")
            self._clearml_task.update_output_model(artifact_path, iteration=state.global_step, auto_delete_file=False)


INTEGRATION_TO_CALLBACK = {
    "azure_ml": AzureMLCallback,
    "comet_ml": CometCallback,
    "mlflow": MLflowCallback,
    "neptune": NeptuneCallback,
    "tensorboard": TensorBoardCallback,
    "wandb": WandbCallback,
    "codecarbon": CodeCarbonCallback,
    "clearml": ClearMLCallback,
    "dagshub": DagsHubCallback,
}


def get_reporting_integration_callbacks(report_to):
    for integration in report_to:
        if integration not in INTEGRATION_TO_CALLBACK:
            raise ValueError(
                f"{integration} is not supported, only {', '.join(INTEGRATION_TO_CALLBACK.keys())} are supported."
            )

    return [INTEGRATION_TO_CALLBACK[integration] for integration in report_to]
