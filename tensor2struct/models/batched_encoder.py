import collections
import itertools
import json
import os

import attr
import torch
import numpy as np

from tensor2struct.models import abstract_preproc, encoder
from tensor2struct.modules import embedders, lstm, attention, rat, bert
from tensor2struct.utils import serialization, vocab, registry


@attr.s
class EncoderState:
    src_memory = attr.ib()
    lengths = attr.ib()
    src_summary = attr.ib()


@registry.register("encoder", "batched_vanilla")
class Encoder(torch.nn.Module):
    batched = True
    Preproc = encoder.EncPreproc

    def __init__(
        self,
        device,
        preproc,
        dropout=0.1,
        word_emb_size=128,
        recurrent_size=256,
        num_heads=4,
        use_native_lstm=True,
        bert_version="bert-base-uncased",
        encoder=("emb", "bilstm"),
    ):
        """
        Args:
            num_heads: attention heads for transformer module if used
            use_native_lstm: whether using native lstm is lstm is used
            bert_version: which bert to use if bert is a encoder moduel
        """
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.vocab = preproc.vocab

        self.dropout = dropout
        self.word_emb_size = word_emb_size
        self.recurrent_size = recurrent_size
        self.use_native_lstm = use_native_lstm
        self.num_heads = num_heads
        self.bert_version = bert_version

        # with batching
        self.encoder_modules = encoder
        self.last_enc_module = self.encoder_modules[-1]

        # shared modules
        self.shared_modules = {}
        if any(m.startswith("shared") for m in encoder):
            self.shared_modules["shared-bert"] = bert.BERTEncoder(
                device=self._device, bert_version=self.bert_version
            )

        self.encoder = self._build_modules(encoder, self.shared_modules)

    def _build_modules(self, module_types, shared_modules=None):
        module_builder = {
            "bert": lambda: bert.BERTEncoder(
                device=self._device, bert_version=self.bert_version
            ),
            "bert2emb": lambda: bert.BERT2Embed(
                device=self._device,
                bert_version=self.bert_version,
                emb_size=self.word_emb_size,
            ),
            "emb": lambda: embedders.LookupEmbeddings(
                device=self._device,
                vocab=self.vocab,
                embedder=self.preproc.embedder,
                emb_size=self.word_emb_size,
                learnable_words=self.preproc.learnable_words,
            ),
            "unilstm": lambda: lstm.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                use_native=self.use_native_lstm,
                summarize=False,
                bidirectional=False,
            ),
            "bilstm": lambda: lstm.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                use_native=self.use_native_lstm,
                summarize=False,
                bidirectional=True,
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
                num_heads=self.num_heads,
                hidden_size=self.recurrent_size,
                dropout=self.dropout
            ),
        }

        modules = []
        for module_type in module_types:
            if shared_modules and module_type in shared_modules:
                modules.append(shared_modules[module_type])
            else:
                modules.append(module_builder[module_type]())
        return torch.nn.Sequential(*modules)

    def forward(self, enc_inputs):
        tokens_list = [desc["tokens"] for desc in enc_inputs]
        return self.compute_encoding(tokens_list)

    def compute_encoding(self, tokens_list):
        src_enc = self.encoder(tokens_list)
        if self.last_enc_module in ["bilstm", "lstm"]:
            src_enc_memory, lengths = src_enc.pad(batch_first=True)
            bidirectional = self.last_enc_module == "bilstm"
            src_enc_summary = lstm.extract_last_hidden_state_batched(
                src_enc_memory, lengths, bidirectional=bidirectional
            )
        else:
            assert self.last_enc_module == "transformer"
            raw_src_enc_memory, lengths = src_enc

            # unpack CLS representation as the summary, recover original lengths
            src_enc_summary = raw_src_enc_memory[:, 0, :]
            src_enc_memory = raw_src_enc_memory[:, 1:, :]
            for i in range(len(lengths)):
                lengths[i] = lengths[i] - 1

        return EncoderState(src_enc_memory, lengths, src_enc_summary)
