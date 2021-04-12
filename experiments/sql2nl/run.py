#!/usr/bin/env python

import os
import _jsonnet
import json
import argparse
import attr
import wandb

from experiments.sql2nl import train
from tensor2struct.commands import eval


@attr.s
class TrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class InferConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib()
    debug = attr.ib(default=False)
    method = attr.ib(default="beam_search")
    mode = attr.ib(default="infer")
    limit = attr.ib(default=None)
    output_history = attr.ib(default=False)


@attr.s
class EvalConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    inferred = attr.ib()
    output = attr.ib()
    etype = attr.ib(default="match")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["train"], help="train",
    )
    parser.add_argument("exp_config_file", help="jsonnet file for experiments")
    args = parser.parse_args()

    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = json.dumps(exp_config["model_config_args"])
    else:
        model_config_args = None

    # cluster base dir
    log_base_dir = os.environ.get("LOG_BASE_DIR", None)
    if log_base_dir is None:
        print(f"Using default log base dir {os.getcwd()}")
        logdir = exp_config["logdir"]
    else:
        logdir = os.path.join(log_base_dir, exp_config["logdir"])

    # wandb init
    expname = exp_config["logdir"].split("/")[-1]
    project = exp_config["project"]

    # dist train need to start a wandb session in each process, not a global one
    if args.mode not in ["dist_train"]:
        wandb.init(project=project, group=expname, job_type=args.mode)

    if args.mode == "train":
        train_config = TrainConfig(model_config_file, model_config_args, logdir)
        train.main(train_config)


if __name__ == "__main__":
    main()
