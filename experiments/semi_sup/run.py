#!/usr/bin/env python

import _jsonnet
import json
import argparse
import attr
import os
import wandb

from experiments.semi_sup import semi_train
from tensor2struct.commands.run import load_args


@attr.s
class SemiTrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["semi_train"], help="semi_train")
    parser.add_argument("exp_config_file", help="jsonnet file for experiments")
    parser.add_argument("--config_args", help="additional exp configs")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    exp_config, model_config_args = load_args(args)

    # model jsonnet
    model_config_file = exp_config["model_config"]

    # cluster base dir
    log_base_dir = os.environ.get("LOG_BASE_DIR", None)
    if log_base_dir is None:
        print(f"Using default log base dir {os.getcwd()}")
        logdir = exp_config["logdir"]
    else:
        logdir = os.path.join(log_base_dir, exp_config["logdir"])

    # wandb init
    if args.mode in ["semi_train"]:
        expname = exp_config["logdir"].split("/")[-1]
        project = exp_config["project"]
        wandb.init(project=project, group=expname, job_type=args.mode)

    if args.mode == "semi_train":
        semi_train_config = SemiTrainConfig(
            model_config_file, model_config_args, logdir
        )
        semi_train.main(semi_train_config)


if __name__ == "__main__":
    main()
