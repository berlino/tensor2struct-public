import collections
import itertools
import json
import os

import attr
import torch
import numpy as np

from tensor2struct.models import abstract_preproc
from tensor2struct.modules import embedders, lstm, attention, rat
from tensor2struct.utils import serialization, vocab, registry


@attr.s
class EncoderState:
    src_memory = attr.ib()
    src_summary = attr.ib()


class EncPreproc(abstract_preproc.AbstractPreproc):
    def __init__(self, save_path, word_emb=None, min_freq=0, max_count=10000):
        self.save_path = save_path
        self.data_dir = os.path.join(save_path, "enc")
        self.texts = collections.defaultdict(list)
        self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        self.vocab_path = os.path.join(save_path, "enc_vocab")

        # pretrained embeddings, e.g., Glove
        if word_emb is None:
            self.embedder = word_emb
        else:
            self.embedder = registry.construct("word_emb", word_emb)
        self.learnable_words = None

    def _tokenize(self, unsplit, presplit):
        if self.embedder:
            return self.embedder.tokenize(unsplit)
        return presplit

    def validate_item(self, item, section):
        return True, None

    def add_item(self, item, section, validation_info):
        if isinstance(item.src, (list, tuple)):
            unsplit = " ".join(item.src)
            tokens = self._tokenize(unsplit, item.src)
        else:
            tokens = self._tokenize(item.src, item.src.split())

        self.texts[section].append({"tokens": tokens})

        if section == "train":
            for token in tokens:
                self.vocab_builder.add_word(token)

    def clear_items(self):
        self.texts = collections.defaultdict(list)

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.vocab = self.vocab_builder.finish()
        print(f"{len(self.vocab)} words in enc vocab")
        self.vocab.save(self.vocab_path)

        for section, texts in self.texts.items():
            with open(os.path.join(self.data_dir, section + ".jsonl"), "w") as f:
                for text in texts:
                    f.write(json.dumps(text) + "\n")

    def load(self):
        self.vocab = vocab.Vocab.load(self.vocab_path)

    def dataset(self, section):
        return [
            json.loads(line)
            for line in open(os.path.join(self.data_dir, section + ".jsonl"))
        ]


@registry.register("encoder", "vanilla")
class Encoder(torch.nn.Module):
    batched = True
    Preproc = EncPreproc

    def __init__(
        self,
        device,
        preproc,
        dropout=0.1,
        word_emb_size=128,
        recurrent_size=256,
        encoder=("emb", "bilstm"),
    ):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.vocab = preproc.vocab

        self.dropout = dropout
        self.word_emb_size = word_emb_size
        self.recurrent_size = recurrent_size

        # with batching
        self.encoder_modules = encoder
        self.encoder = self._build_modules(encoder)

    def _build_modules(self, module_types):
        module_builder = {
            "emb": lambda: embedders.LookupEmbeddings(
                device=self._device,
                vocab=self.vocab,
                embedder=self.preproc.embedder,
                emb_size=self.word_emb_size,
                learnable_words=self.preproc.learnable_words,
            ),
            "bilstm": lambda: lstm.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                use_native=True,
                summarize=False,
            ),
            "cls_glue": lambda: rat.PadCLS(
                device=self._device, hidden_size=self.recurrent_size, pos_encode=False,
            ),
            "cls_glue_p": lambda: rat.PadCLS(
                device=self._device, hidden_size=self.recurrent_size, pos_encode=True,
            ),
            "transformer": lambda: rat.TransformerEncoder(
                device=self._device,
                num_layers=1,
                num_heads=4,
                hidden_size=self.recurrent_size,
            ),
        }

        modules = []
        for module_type in module_types:
            modules.append(module_builder[module_type]())
        return torch.nn.Sequential(*modules)

    def forward(self, enc_inputs):
        tokens_list = [desc["tokens"] for desc in enc_inputs]
        return self.compute_encoding(tokens_list)

    def compute_encoding(self, tokens_list):
        src_enc = self.encoder(tokens_list)

        ret_list = []
        for i in range(len(tokens_list)):
            if self.encoder_modules[-1] == "bilstm":
                src_memory = src_enc.select(i)
                src_summary = lstm.extract_last_hidden_state(src_memory)

                # batch size 1
                src_memory = src_memory.unsqueeze(0)
                src_summary = src_summary.unsqueeze(0)
            else:
                assert self.encoder_modules[-1] == "transformer"
                src_rep = src_enc[i]
                assert src_rep.size()[1] == len(tokens_list[i]) + 1  # CLS token

                src_summary = src_rep[:, 0, :]  # the first token is used as summary
                src_memory = src_rep[:, 1:, :]
            ret_list.append(
                EncoderState(src_memory=src_memory, src_summary=src_summary,)
            )
        return ret_list
