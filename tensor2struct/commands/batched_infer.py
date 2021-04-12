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

from tensor2struct.commands import infer
from tensor2struct.utils import registry
from tensor2struct.utils import saver as saver_mod


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class BatchedInferer(infer.Inferer):
    def infer(self, model, output_path, args):
        output = open(output_path, "w")
        chunk_size = 128  # this is manually set, TODO add it to config

        assert args.method.startswith("batched")
        infer_func = registry.lookup("infer_method", args.method)
        with torch.no_grad():
            orig_data = registry.construct("dataset", self.config["data"][args.section])
            preproc_data = self.model_preproc.dataset(args.section)
            assert len(orig_data) == len(preproc_data)
            chunked_orig_data = chunks(orig_data, chunk_size)
            chunked_preproc_data = chunks(preproc_data, chunk_size)
            pbar = tqdm.tqdm(total=len(preproc_data))
            self._inner_infer(
                model,
                infer_func,
                args.beam_size,
                chunked_orig_data,
                chunked_preproc_data,
                output,
                pbar,
            )
            pbar.close()

    def _inner_infer(
        self,
        model,
        infer_func,
        beam_size,
        chunked_orig_data,
        chunked_preproc_data,
        output,
        pbar,
    ):
        i = 0
        for orig_data, preproc_data in zip(chunked_orig_data, chunked_preproc_data):
            code_list = infer_func(
                model, orig_data, preproc_data, beam_size=beam_size, max_steps=256
            )

            for inferred_code in code_list:
                decoded = []
                decoded.append(
                    {
                        # "model_output": model_output,
                        "inferred_code": inferred_code,
                        # "score": score,
                    }
                )

                output.write(json.dumps({"index": i, "beams": decoded}) + "\n")
                output.flush()
                i += 1
                pbar.update(1)


def main(args):
    config, output_path = infer.setup(args)
    inferer = BatchedInferer(config)
    model = inferer.load_model(args.logdir, args.step)
    inferer.infer(model, output_path, args)


if __name__ == "__main__":
    args = infer.add_parser()
    main(args)
