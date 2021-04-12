import argparse
import os
import json
import _jsonnet
import wandb

from tensor2struct.utils import registry


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-args")
    parser.add_argument("--section", required=True)
    parser.add_argument("--inferred", required=True)
    parser.add_argument("--etype", default="match", help="match, exec, all")
    parser.add_argument("--output")
    parser.add_argument("--logdir")
    args = parser.parse_args()
    return args


def compute_metrics(config_path, config_args, section, inferred_path, etype, logdir=None):
    if config_args:
        config = json.loads(
            _jsonnet.evaluate_file(config_path, tla_codes={"args": config_args})
        )
    else:
        config = json.loads(_jsonnet.evaluate_file(config_path))

    # update config to wandb
    wandb.config.update(config)

    if "model_name" in config and logdir:
        logdir = os.path.join(logdir, config["model_name"])
    if logdir:
        inferred_path = inferred_path.replace("__LOGDIR__", logdir)

    inferred = open(inferred_path)
    data = registry.construct("dataset", config["data"][section])
    metrics = data.Metrics(data, etype)

    inferred_lines = list(inferred)
    if len(inferred_lines) < len(data):
        raise Exception(
            "Not enough inferred: {} vs {}".format(len(inferred_lines), len(data))
        )

    for i, line in enumerate(inferred_lines):
        infer_results = json.loads(line)
        if infer_results["beams"]:
            inferred_codes = [beam["inferred_code"] for beam in infer_results["beams"]]
        else:
            inferred_codes = [None]
        assert "index" in infer_results

        if etype in ["execution", "all"]:
            # if eval by execution, then we choose the first executable one from the beams
            metrics.add_beams(data[infer_results["index"]], inferred_codes, data[i].orig["question"])
        else:
            assert etype in ["match", "sacreBLEU", "tokenizedBLEU"]
            metrics.add_one(data[infer_results["index"]], inferred_codes[0])
    return logdir, metrics.finalize()


def main(args):
    real_logdir, metrics = compute_metrics(
        args.config, args.config_args, args.section, args.inferred, args.etype, args.logdir
    )

    if args.output:
        if real_logdir:
            output_path = args.output.replace("__LOGDIR__", real_logdir)
        else:
            output_path = args.output
        with open(output_path, "w") as f:
            json.dump(metrics, f)
        print("Wrote eval results to {}".format(output_path))
    else:
        print(metrics)
    return metrics


if __name__ == "__main__":
    args = add_parser()
    main(args)
