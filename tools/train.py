"""Anomalib Training Script.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
MVETEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood","zipper",
]

VISA = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4"]

import logging
import warnings
from argparse import ArgumentParser, Namespace
import os

from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import TestSplitMode
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger

logger = logging.getLogger("anomalib")


def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")
    return parser


def train(args: Namespace):
    """Train an anomaly model.

    Args:
        args (Namespace): The arguments from the command line.
    """

    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config = get_configurable_parameters(model_name=args.model, config_path=args.config)

    def _run_experiment(config):
        """Run experiment."""
        if config.project.get("seed") is not None:
            seed_everything(config.project.seed)
        datamodule = get_datamodule(config)
        model = get_model(config)
        experiment_logger = get_experiment_logger(config)
        callbacks = get_callbacks(config)
        trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
        logger.info("Training the model.")
        trainer.fit(model=model, datamodule=datamodule)
        logger.info("Loading the best model weights.")
        load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
        trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member
        if config.dataset.test_split_mode == TestSplitMode.NONE:
            logger.info("No test set provided. Skipping test stage.")
        else:
            logger.info("Testing the model.")
            trainer.test(model=model, datamodule=datamodule)

    if config.dataset.name in ["mvtec"] and config.dataset.category == "all":
        categories = MVETEC
        for c in categories:
            config.dataset.category = c
            path = config.project.path
            pathelems = path.split(os.sep)
            pathelems[-2] = c
            config.project.path = os.path.join(*pathelems)
            try:
                config.model.category = c
            except Exception as e:
                continue
            _run_experiment(config) 
    
    else:
        _run_experiment(config)


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
