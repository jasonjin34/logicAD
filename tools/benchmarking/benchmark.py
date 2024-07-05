"""Benchmark all the algorithms in the repo."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import os
import io
import logging
import math
import multiprocessing
import sys
import time
import warnings
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
import distutils

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from utils import (
    upload_to_comet,
    upload_to_wandb,
    write_metrics,
    write_summary,
)

from anomalib.config import get_configurable_parameters, update_input_size_config
from anomalib.data import get_datamodule
from anomalib.deploy import export
from anomalib.deploy.export import ExportMode
from anomalib.models import get_model
from anomalib.utils.loggers import configure_logger
from anomalib.utils.sweep import (
    get_openvino_throughput,
    get_run_config,
    get_sweep_callbacks,
    get_torch_throughput,
    set_in_nested_config,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
configure_logger()
pl_logger = logging.getLogger(__file__)
for logger_name in ["pytorch_lightning", "torchmetrics", "os"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def hide_output(func):
    """Decorator to hide output of the function.

    Args:
        func (function): Hides output of this function.

    Raises:
        Exception: Incase the execution of function fails, it raises an exception.

    Returns:
        object of the called function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        std_out = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            value = func(*args, **kwargs)
        except Exception as exp:
            raise Exception(buf.getvalue()) from exp
        sys.stdout = std_out
        return value

    return wrapper


#@hide_output
def get_single_model_metrics(model_config: DictConfig | ListConfig, openvino_metrics: bool = False) -> dict:
    """Collects metrics for `model_name` and returns a dict of results.

    Args:
        model_config (DictConfig, ListConfig): Configuration for run
        openvino_metrics (bool): If True, converts the model to OpenVINO format and gathers inference metrics.

    Returns:
        dict: Collection of all the metrics such as time taken, throughput and performance scores.
    """
    with TemporaryDirectory() as project_path:
        model_config.project.path = project_path
        datamodule = get_datamodule(model_config)
        model = get_model(model_config)
        callbacks = get_sweep_callbacks(model_config)

        trainer = Trainer(**model_config.trainer, logger=None, callbacks=callbacks)

        start_time = time.time()

        trainer.fit(model=model, datamodule=datamodule)

        # get start time
        training_time = time.time() - start_time

        # Creating new variable is faster according to https://stackoverflow.com/a/4330829
        start_time = time.time()
        # get test results
        test_results = trainer.test(model=model, datamodule=datamodule)

        # get testing time
        testing_time = time.time() - start_time

        # Create dirs for torch export (as default only lighting model is produced)
        export(
            task=model_config.dataset.task,
            transform=trainer.datamodule.test_data.transform.to_dict(),
            input_size=model_config.model.input_size,
            model=model,
            export_mode=ExportMode.TORCH,
            export_root=project_path,
        )

        throughput = get_torch_throughput(
            model_path=project_path,
            test_dataset=datamodule.test_dataloader().dataset,
            device=model_config.trainer.accelerator,
        )

        # Get OpenVINO metrics
        openvino_throughput = float("nan")
        if openvino_metrics:
            # Create dirs for openvino model export
            export(
                task=model_config.dataset.task,
                transform=trainer.datamodule.test_data.transform.to_dict(),
                input_size=model_config.model.input_size,
                model=model,
                export_mode=ExportMode.OPENVINO,
                export_root=project_path,
            )
            openvino_throughput = get_openvino_throughput(model_path=project_path, test_dataset=datamodule.test_data)

        # arrange the data
        data = {
            "Training Time (s)": training_time,
            "Testing Time (s)": testing_time,
            f"Inference Throughput {model_config.trainer.accelerator} (fps)": throughput,
            "OpenVINO Inference Throughput (fps)": openvino_throughput,
        }
        for key, val in test_results[0].items():
            data[key] = float(val)

    return data


def compute_on_cpu(sweep_config: DictConfig | ListConfig, folder: str | None = None, configs = None):
    """Compute all run configurations over a sigle CPU."""
    for run_config in get_run_config(sweep_config.grid_search):
        model_metrics = sweep(run_config, 0, sweep_config.seed, False, configs=configs)
        write_metrics(model_metrics, sweep_config.writer, folder)


def compute_on_gpu(
        run_configs: list[DictConfig],
        device: int,
        seed: int,
        writers: list[str],
        folder: str | None = None,
        compute_openvino: bool = False,
        configs = None,
):
    """Go over each run config and collect the result.

    Args:
        run_configs (DictConfig | ListConfig): List of run configurations.
        device (int): The GPU id used for running the sweep.
        seed (int): Fix a seed.
        writers (list[str]): Destinations to write to.
        folder (optional, str): Sub-directory to which runs are written to. Defaults to None. If none writes to root.
        compute_openvino (bool, optional): Compute OpenVINO throughput. Defaults to False.
    """
    for run_config in run_configs:
        if isinstance(run_config, (DictConfig, ListConfig)):
            model_metrics = sweep(run_config, device, seed, compute_openvino, configs=configs)
            write_metrics(model_metrics, writers, folder)
        else:
            raise ValueError(
                f"Expecting `run_config` of type DictConfig or ListConfig. Got {type(run_config)} instead."
            )


def distribute_over_gpus(sweep_config: DictConfig | ListConfig, folder: str | None = None):
    """Distribute metric collection over all available GPUs. This is done by splitting the list of configurations."""
    if torch.cuda.device_count() == 1:
        print("Only one GPU detected. Running benchmarking on single GPU, no need to launch multiple processes pool executor")
        run_configs = list(get_run_config(sweep_config.grid_search))
        compute_on_gpu(
            run_configs, 
            1, 
            sweep_config.seed, 
            sweep_config.writer, 
            folder, 
            sweep_config.compute_openvino,
            configs=sweep_config)
    else:
        with ProcessPoolExecutor(
                max_workers=torch.cuda.device_count(), mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            run_configs = list(get_run_config(sweep_config.grid_search))
            jobs = []
            for device_id, run_split in enumerate(
                    range(0, len(run_configs), math.ceil(len(run_configs) / torch.cuda.device_count()))
            ):
                jobs.append(
                    executor.submit(
                        compute_on_gpu,
                        run_configs[run_split: run_split + math.ceil(len(run_configs) / torch.cuda.device_count())],
                        device_id + 1,
                        sweep_config.seed,
                        sweep_config.writer,
                        folder,
                        sweep_config.compute_openvino,
                    )
                )
            for job in jobs:
                try:
                    job.result()
                except Exception as exc:
                    raise Exception(f"Error occurred while computing benchmark on GPU {job}") from exc


def distribute(config: DictConfig | ListConfig):
    """Run all cpu experiments on a single process. Distribute gpu experiments over all available gpus.

    Args:
        config: (DictConfig | ListConfig): Sweep configuration.
    """
    runs_folder = datetime.strftime(datetime.now(), "%Y_%m_%d-%H_%M_%S")
    if config.task_name:
        runs_folder = f"{config.task_name}-{runs_folder}"
    devices = config.hardware
    if not torch.cuda.is_available() and "gpu" in devices:
        pl_logger.warning("Config requested GPU benchmarking but torch could not detect any cuda enabled devices")
    elif {"cpu", "gpu"}.issubset(devices):
        # Create process for gpu and cpu
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as executor:
            jobs = [
                executor.submit(compute_on_cpu, config, runs_folder),
                executor.submit(distribute_over_gpus, config, runs_folder),
            ]
            for job in as_completed(jobs):
                try:
                    job.result()
                except Exception as exception:
                    raise Exception(f"Error occurred while computing benchmark on device {job}") from exception
    elif "cpu" in devices:
        compute_on_cpu(config, folder=runs_folder)
    elif "gpu" in devices:
        distribute_over_gpus(config, folder=runs_folder)
    if "wandb" in config.writer:
        upload_to_wandb(team="anomalib", folder=runs_folder)
    if "comet" in config.writer:
        upload_to_comet(folder=runs_folder)


def sweep(
        run_config: DictConfig | ListConfig, 
        device: int = 0, 
        seed: int = 42, 
        convert_openvino: bool = False,
        configs = None,
) -> dict[str, str | float]:
    """Go over all the values mentioned in `grid_search` parameter of the benchmarking config.

    Args:
        run_config: (DictConfig | ListConfig, optional): Configuration for current run.
        device (int, optional): Name of the device on which the model is trained. Defaults to 0 "cpu".
        convert_openvino (bool, optional): Whether to convert the model to openvino format. Defaults to False.

    Returns:
        dict[str, str | float]: Dictionary containing the metrics gathered from the sweep.
    """
    seed_everything(seed, workers=True)
    # This assumes that `model_name` is always present in the sweep config.

    if run_config["dataset.name"] == "visa":
        config_filename = "config_visa"
    elif run_config["dataset.name"] == "mvtec":
        config_filename = "config_mvtec"
    elif run_config["dataset.name"] == "mini_mvtec":
        config_filename = "config_mini_mvtec"
    else:
        raise ValueError(f"dataset name {run_config['dataset.name']} is not supported")
    
    model_config = get_configurable_parameters(
        model_name=run_config.model_name, config_filename=config_filename)
    model_config.project.seed = seed

    if run_config.model_name in ["winclip", "sclipad", "gemad"]:
        # update the run_config by model category for text-prompt / clip based model
        run_config['model.category'] = run_config['dataset.category']
        if run_config.model_name == "sclipad":
            # update the run config image size
            model_config.model.k_shot = configs.few_shot_k
            model_config.model.zero_shot = configs.zero_shot
            model_config.model.apply_key_smoothing = configs.key_smoothing
            model_config.model.isCSA = configs.csa
            model_config.model.clsCSA = configs.cls_csa
            model_config.model.cls_type_windows = configs.cls_type
            model_config.model.pretrained = configs.pretrained
            model_config.model.backbone = configs.backbone
            model_config.model.ckpt = configs.ckpt
            model_config.model.text_adapter = configs.text_adapter
            model_config.model.image_adapter = configs.image_adapter
            model_config.model.attn_logit_scale = configs.attn_logit_scale
            model_config.model.img_size = configs.img_size
            model_config.dataset.image_size = [configs.img_size, configs.img_size]
            model_config.dataset.center_crop = [configs.crop_size, configs.crop_size]
            run_config["dataset.image_size"] = configs.img_size

    model_config = cast(DictConfig, model_config)  # placate mypy
    for param in run_config.keys():
        # grid search keys are always assumed to be strings
        param = cast(str, param)  # placate mypy
        set_in_nested_config(model_config, param.split("."), run_config[param])  # type: ignore

    # convert image size to tuple in case it was updated by run config
    model_config = update_input_size_config(model_config)

    # Set device in config. 0 - cpu, [0], [1].. - gpu id
    if device != 0:
        model_config.trainer.devices = [device - 1]
        model_config.trainer.accelerator = "gpu"
    else:
        model_config.trainer.accelerator = "cpu"

    # Remove legacy flags
    for legacy_device in ["num_processes", "gpus", "ipus", "tpu_cores"]:
        if legacy_device in model_config.trainer:
            model_config.trainer[legacy_device] = None

    if run_config.model_name in ["patchcore", "cflow"]:
        convert_openvino = False  # `torch.cdist` is not supported by onnx version 11
        # TODO Remove this line when issue #40 is fixed https://github.com/openvinotoolkit/anomalib/issues/40
        if model_config.model.input_size != (224, 224):
            return {}  # go to next run

    def path_update(model_config, path: str):
        """Update the config with latest save path"""
        # get patch components
        pc = path.split(os.sep)
        # the second last is category component, need to update
        pc[-1] = model_config.dataset.category
        model_config.project.path = os.path.join(*pc)

    path_update(model_config, model_config.project.path)
    path_update(model_config, model_config.trainer.default_root_dir)

    print(model_config.model)

    # Run benchmarking for current config
    model_metrics = get_single_model_metrics(model_config=model_config, openvino_metrics=convert_openvino)
    output = f"One sweep run complete for model {model_config.model.name}"
    output += f" On category {model_config.dataset.category}" if model_config.dataset.category is not None else ""
    output += str(model_metrics)
    logger.info(output)

    # Append configuration of current run to the collected metrics
    for key, value in run_config.items():
        # Skip adding model name to the dataframe
        if key != "model_name" and key != "dataset.name":
            model_metrics[key] = value

    # Add device name to list
    model_metrics["device"] = "gpu" if device > 0 else "cpu"
    model_metrics["model_name"] = run_config.model_name

    return model_metrics


if __name__ == "__main__":
    # Benchmarking entry point.
    # Spawn multiple processes one for cpu and rest for the number of gpus available in the system.
    # The idea is to distribute metrics collection over all the available devices.
    parser = ArgumentParser()
    default_benchmark_path = os.path.join(".", "tools", "benchmarking", "benchmark_params.yaml")
    parser.add_argument("--config", type=Path, default=default_benchmark_path, help="Path to sweep configuration")
    parser.add_argument("--crop_size", type=int, default=224, help="center crop size to use for benchmarking")
    parser.add_argument("--img_size", type=int, default=224, help="image size to use for benchmarking")
    parser.add_argument("--zero_shot", type=lambda x:bool(distutils.util.strtobool(x)), default=True, help="zero-shot or few-shot settings")
    parser.add_argument("--few_shot_k", type=int, default=4, help="few-shot k value")
    parser.add_argument("--csa", type=lambda x:bool(distutils.util.strtobool(x)), default=True, help="using csa for the last layer of visual transblocks")
    parser.add_argument("--gem", type=lambda x:bool(distutils.util.strtobool(x)), default=False, help="using csa for the last layer of visual transblocks")
    parser.add_argument("--cls_csa", type=lambda x:bool(distutils.util.strtobool(x)), default=True, help="using csa for the cls token for classification")
    parser.add_argument("--key_smoothing", type=lambda x:bool(distutils.util.strtobool(x)), default=False, help="center crop size to use for benchmarking")
    parser.add_argument("--cls_type", type=str, default="max", help="cls type to get classification score")
    parser.add_argument("--task_name", type=str, default="test", help="experiment name for benchmarking")
    parser.add_argument("--ckpt", type=str, default=None, help="ckpt to be loaded from alignment training")
    parser.add_argument("--backbone", type=str, default="ViT-B-16", help="")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="")
    parser.add_argument("--text_adapter", type=float, default=0.0, help="text adapter quotient, if ckpt trained with one")
    parser.add_argument("--image_adapter", type=float, default=0.0, help="image adapter quotient, if ckpt trained with one")
    parser.add_argument("--attn_logit_scale", type=float, default=-1, help="csa logit scale for the last layer of visual transblocks")

    _args = parser.parse_args()

    print("Benchmarking started 🏃‍♂️. This will take a while ⏲ depending on your configuration.")
    _sweep_config = OmegaConf.load(_args.config)
    _sweep_config.crop_size = _args.crop_size
    _sweep_config.zero_shot = _args.zero_shot
    _sweep_config.few_shot_k = _args.few_shot_k
    _sweep_config.csa = _args.csa
    _sweep_config.cls_csa = _args.cls_csa
    _sweep_config.key_smoothing = _args.key_smoothing
    _sweep_config.cls_type = _args.cls_type
    _sweep_config.img_size = _args.img_size
    _sweep_config.task_name = _args.task_name
    _sweep_config.ckpt = _args.ckpt
    _sweep_config.backbone = _args.backbone
    _sweep_config.pretrained = _args.pretrained
    _sweep_config.text_adapter = _args.text_adapter
    _sweep_config.image_adapter = _args.image_adapter
    _sweep_config.attn_logit_scale = _args.attn_logit_scale
    
    # backbone = ["ViT-B-16", "ViT-B-16", "ViT-L-14", "ViT-B-16-quickgelu", "ViT-B-16-plus-240", "EVA02-B-16", "EVA02-L-14"]
    # pretrained = ["laion400m_e32", "laion2b_s34b_b88k", "datacomp_xl_s13b_b90k", "metaclip_fullcc", "laion400m_e32", "merged2b_s8b_b131k", "merged2b_s4b_b131k"]

    csa = "sclip" if _args.csa else "clip"
    cls_csa = "cls_csa" if _args.cls_csa else "cls_att"
    
    if _args.zero_shot:
        few_shot_or_zero_shot = "zeroshot"
    else:
        few_shot_or_zero_shot = f"few_shot_{str(k)}"
    print(f"Commencing {few_shot_or_zero_shot} benchmark")

    _sweep_config.task_name = f"{_sweep_config.grid_search.dataset.name}-{few_shot_or_zero_shot}" + \
        f"-{_args.backbone}_{_args.pretrained}" + \
        (f"-ckpt_{''.join(_args.ckpt.split('/')[-1].split('.')[:-1])}" if _args.ckpt else "") + \
        f"-image_{_args.img_size}-crop_{_args.crop_size}-{csa}-{cls_csa}_{_args.cls_type}"
    distribute(_sweep_config)
    print("Finished gathering results ⚡")
    print("Generate average metrics statistic")
    write_summary()
