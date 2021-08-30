#!/usr/bin/env python
import os
import _jsonnet
import json
import argparse
import attr
import collections
import wandb

import tensor2struct
import experiments

from tensor2struct.commands import infer, batched_infer, eval
from tensor2struct.commands.run import InferConfig, EvalConfig, load_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["eval_cogs", "batched_eval_cogs"],
        help="eval_cogs/batched_eval_cogs",
    )
    parser.add_argument("exp_config_file", help="jsonnet file for experiments")
    parser.add_argument("--config_args", help="additional exp configs")
    args = parser.parse_args()
    return args


def eval_and_report_cogs(args, exp_config, model_config_args, logdir, infer_mod):
    # model jsonnet
    model_config_file = exp_config["model_config"]

    summary = collections.defaultdict(float)
    for step in exp_config["eval_steps"]:
        infer_output_path = "{}/{}-step{}.infer".format(
            exp_config["eval_output"], exp_config["eval_name"], step
        )

        infer_config = InferConfig(
            model_config_file,
            model_config_args,
            logdir,
            exp_config["eval_section"],
            exp_config["eval_beam_size"],
            infer_output_path,
            step,
            debug=exp_config["eval_debug"],
            method=exp_config["eval_method"],
        )

        # try:
        #     infer_mod.main(infer_config)
        # except Exception as e:
        #     print(f"Infer error {str(e)}")
        #     continue

        eval_output_path = "{}/{}-step{}.eval".format(
            exp_config["eval_output"], exp_config["eval_name"], step
        )
        eval_config = EvalConfig(
            model_config_file,
            model_config_args,
            logdir,
            exp_config["eval_section"],
            infer_output_path,
            eval_output_path,
        )

        # etype
        if "eval_type" in exp_config:
            eval_config.etype = exp_config["eval_type"]
        else:
            assert eval_config.etype == "match"

        try:
            metrics = eval.main(eval_config)
        except Exception as e:
            raise e
            print(f"Eval error {str(e)}")
            continue

        # update config
        wandb.config.update(
            {
                "eval_type": eval_config.etype,
                "eval_method": exp_config["eval_method"],
                "eval_section": exp_config["eval_section"],
                "beam_size": exp_config["eval_beam_size"],
            }
        )

        if "args" in exp_config:
            wandb.config.update({"exp_args": exp_config["args"]})

        # commit with step
        eval_section = exp_config["eval_section"]
        for category in metrics:
            if category in ("per_item"):
                continue
            lf_accuracy = metrics[category]["lf_accuracy"]

            if category == "total_scores":
                metric_name = f"{eval_section}-lf-accuracy"
            else:
                metric_name = f"{eval_section}-{category}-lf-accuracy"

            wandb.log(
                {metric_name: lf_accuracy}, step=step,
            )

        overall_lf_accuracy = metrics["total_scores"]["lf_accuracy"]
        print(step, metrics["total_scores"])

        if overall_lf_accuracy > summary[f"{eval_section}-best-lf-accuracy"]:
            summary[f"{eval_section}-best-lf-accuracy"] = overall_lf_accuracy
            summary[f"{eval_section}-best_lf_accuracy_step"] = step

    # sync summary to wandb
    print("Summary:", str(summary))
    for item in summary:
        wandb.run.summary[item] = summary[item]


def main():
    args = parse_args()
    exp_config, model_config_args = load_args(args)

    # log dir
    log_base_dir = os.environ.get("LOG_BASE_DIR", None)
    if log_base_dir is None:
        print(f"Using default log base dir {os.getcwd()}")
        logdir = exp_config["logdir"]
    else:
        logdir = os.path.join(log_base_dir, exp_config["logdir"])

    # wandb init
    if args.mode in ["eval_cogs", "batched_eval_cogs"]:
        expname = exp_config["logdir"].split("/")[-1]
        project = exp_config["project"]
        wandb.init(project=project, group=expname, job_type=args.mode)

    # execute commands
    if args.mode == "eval_cogs":
        eval_and_report_cogs(
            args, exp_config, model_config_args, logdir, infer_mod=infer
        )
    elif args.mode == "batched_eval_cogs":
        eval_and_report_cogs(
            args, exp_config, model_config_args, logdir, infer_mod=batched_infer
        )


if __name__ == "__main__":
    main()
