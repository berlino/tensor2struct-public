import argparse
import ast
import attr
import itertools
import json
import os
import sys

import _jsonnet
import asdl
import astor
import torch
import tqdm

from tensor2struct.utils import registry
from tensor2struct.utils import saver as saver_mod

class CheckpointNotFoundError(Exception):
    pass


class Inferer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(1)

        # 0. Construct preprocessors
        self.model_preproc = registry.instantiate(
            registry.lookup("model", config["model"]).Preproc,
            config["model"],
            unused_keys=("name",),
        )
        self.model_preproc.load()

    def load_model(self, logdir, step):
        """Load a model (identified by the config used for construction) and return it"""
        # 1. Construct model
        model = registry.construct(
            "model",
            self.config["model"],
            preproc=self.model_preproc,
            device=self.device,
            unused_keys=("decoder_preproc", "encoder_preproc"),
        )
        model.to(self.device)
        model.eval()
        model.visualize_flag = False

        # 2. Restore its parameters
        saver = saver_mod.Saver({"model": model})
        last_step = saver.restore(
            logdir, step=step, map_location=self.device, item_keys=["model"]
        )
        if last_step == 0:  # which is fine fro pretrained model
            # print("Warning: infer on untrained model")
            raise CheckpointNotFoundError(f"Attempting to infer on untrained model, logdir {logdir}, step {step}")
        return model

    def infer(self, model, output_path, args):
        output = open(output_path, "w")

        infer_func = registry.lookup("infer_method", args.method)
        with torch.no_grad():
            if args.mode == "infer":
                orig_data = registry.construct(
                    "dataset", self.config["data"][args.section]
                )
                preproc_data = self.model_preproc.dataset(args.section)
                if args.limit:
                    sliced_orig_data = itertools.islice(orig_data, args.limit)
                    sliced_preproc_data = itertools.islice(preproc_data, args.limit)
                else:
                    sliced_orig_data = orig_data
                    sliced_preproc_data = preproc_data
                assert len(orig_data) == len(preproc_data)
                self._inner_infer(
                    model,
                    infer_func,
                    args.beam_size,
                    sliced_orig_data,
                    sliced_preproc_data,
                    output,
                    args.debug,
                )

    def _inner_infer(
        self,
        model,
        infer_func,
        beam_size,
        sliced_orig_data,
        sliced_preproc_data,
        output,
        debug=False,
    ):
        for i, (orig_item, preproc_item) in enumerate(
            tqdm.tqdm(
                zip(sliced_orig_data, sliced_preproc_data), total=len(sliced_orig_data)
            )
        ):
            beams = infer_func(
                model, orig_item, preproc_item, beam_size=beam_size, max_steps=1000
            )

            decoded = []
            for beam in beams:
                model_output, inferred_code = beam.inference_state.finalize()
                if inferred_code is None:
                    continue

                decoded.append(
                    {
                        # "orig_question": orig_item["question"],
                        "model_output": model_output,
                        "inferred_code": inferred_code,
                        "score": beam.score,
                        **(
                            {
                                "choice_history": beam.choice_history,
                                "score_history": beam.score_history,
                            }
                            if debug
                            else {}
                        ),
                    }
                )

            if debug:
                output.write(
                    json.dumps(
                        {
                            "index": i,
                            "beams": decoded,
                            "orig_item": attr.asdict(orig_item),
                            "preproc_item": preproc_item[0],
                        }
                    )
                    + "\n"
                )
            else:
                output.write(json.dumps({"index": i, "beams": decoded}) + "\n")
            output.flush()


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-args")

    parser.add_argument("--step", type=int)
    parser.add_argument("--section", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beam-size", required=True, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", default="infer", choices=["infer"])

    parser.add_argument("--method", default="beam_search")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


def setup(args):
    if args.config_args:
        config = json.loads(
            _jsonnet.evaluate_file(args.config, tla_codes={"args": args.config_args})
        )
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if "model_name" in config:
        args.logdir = os.path.join(args.logdir, config["model_name"])

    output_path = args.output.replace("__LOGDIR__", args.logdir)
    dir_name = os.path.dirname(output_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    if os.path.exists(output_path):
        print("WARNING Output file {} already exists".format(output_path))
        # sys.exit(1)
    return config, output_path


def main(args):
    config, output_path = setup(args)
    inferer = Inferer(config)
    model = inferer.load_model(args.logdir, args.step)
    inferer.infer(model, output_path, args)


if __name__ == "__main__":
    args = add_parser()
    main(args)
