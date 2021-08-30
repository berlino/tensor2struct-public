import attr
import numpy as np
import collections
import itertools
import pyrsistent
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensor2struct.models import (
    abstract_preproc,
    decoder,
    batched_decoder,
)
from tensor2struct.modules import attention, variational_lstm, lstm, embedders
from tensor2struct.utils import serialization, vocab, registry, bpe
from tensor2struct.models.ast_decoder.utils import lstm_init

import logging

logger = logging.getLogger("tensor2struct")


class ScanDecoderPreproc(decoder.DecoderPreproc):
    def add_item(self, item, section, validation_info):
        actions = item.code.split()

        if section == "train":
            for action in actions:
                self.vocab_builder.add_word(action)

        self.items[section].append({"actions": [vocab.BOS] + actions + [vocab.EOS]})

@registry.register("decoder", "scan")
class ScanDecoder(decoder.Decoder):
    batched = True
    Preproc = ScanDecoderPreproc

class ScanDecoderPreprocV3(decoder.DecoderPreproc):
    """
    Some models does not need bos and eos, e.g., aligned decoder or CTC
    """
    def add_item(self, item, section, validation_info):
        actions = item.code.split()

        if section == "train":
            for action in actions:
                self.vocab_builder.add_word(action)

        self.items[section].append({"actions": actions})


@registry.register("decoder", "scan_lstm_batched_vanilla")
class ScanDecoder(batched_decoder.Decoder):
    batched = True
    Preproc = ScanDecoderPreproc

@registry.register("decoder", "scan_transformer_batched_vanilla")
class ScanTransformerDecoder(batched_decoder.TransformerDecoder):
    batched = True
    Preproc = ScanDecoderPreproc
