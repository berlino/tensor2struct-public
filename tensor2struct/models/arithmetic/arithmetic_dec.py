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

from tensor2struct.models import abstract_preproc, decoder, tagging_decoder
from tensor2struct.modules import attention, variational_lstm, lstm, embedders
from tensor2struct.utils import serialization, vocab, registry

import logging

logger = logging.getLogger("tensor2struct")


class ArithmeticDecoderPreproc(decoder.DecoderPreproc):
    def add_item(self, item, section, validation_info):
        actions = item.tgt

        if section == "train":
            for action in actions:
                self.vocab_builder.add_word(action)

        self.items[section].append({"actions": actions})


@registry.register("decoder", "arithmetic_tagging_dec")
class ArithmeticDecoder(tagging_decoder.TaggingDecoder):
    batched = True
    Preproc = ArithmeticDecoderPreproc
