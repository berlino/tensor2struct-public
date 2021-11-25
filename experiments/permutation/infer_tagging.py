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
import torch.nn.functional as F

from tensor2struct.utils import registry
from tensor2struct.utils import saver as saver_mod
from tensor2struct.commands import infer

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class Inferer(infer.Inferer):
    def infer(self, model, output_path, args):
        """
        Manually set args.limit, TODO: pass the argument via run.py
        """
        output = open(output_path, "w")
        args.limit = 256

        infer_func = registry.lookup("infer_method", args.method)
        with torch.no_grad():
            assert args.mode == "infer"
            orig_data = registry.construct(
                "dataset", self.config["data"][args.section]
            )
            preproc_data = self.model_preproc.dataset(args.section)
            assert len(orig_data) == len(preproc_data)
            chunked_orig_data = chunks(orig_data, args.limit)
            chunked_preproc_data = chunks(preproc_data, args.limit)
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
    
    def batched_decode(self, model, preproc_data):
        model.eval()
        enc_batch = [enc_input for enc_input, dec_output in preproc_data]
        enc_state = model.encoder(enc_batch)
        logits = model.decoder.score_f(enc_state.src_memory)

        # greedy decoding
        code_list = []
        _, decode_t = logits.max(dim=-1)
        for batch_idx in range(len(enc_batch)):
            decoded_tokens = []
            for token_idx in range(enc_state.lengths[batch_idx]):
                best_idx = int(decode_t[batch_idx][token_idx])
                decoded_tokens.append(model.decoder.vocab[best_idx])
            code_list.append(" ".join(decoded_tokens))
        return code_list

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
            zipped_preproc_data = zip(*preproc_data)
            code_list = self.batched_decode(model, zipped_preproc_data)

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
    inferer = Inferer(config)
    model = inferer.load_model(args.logdir, args.step)
    inferer.infer(model, output_path, args)


if __name__ == "__main__":
    args = infer.add_parser()
    main(args)
