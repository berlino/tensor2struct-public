import collections
import itertools
import json
import os

import attr
import torch
import numpy as np

from tensor2struct.models import (
    abstract_preproc,
    encoder,
    batched_encoder,
)
from tensor2struct.modules import embedders, lstm, attention, rat
from tensor2struct.utils import serialization, vocab, registry

import logging

logger = logging.getLogger("tensor2struct")


class ScanEncPreproc(encoder.EncPreproc):
    def add_item(self, item, section, validation_info):
        tokens = item.text.split()
        self.texts[section].append({"tokens": tokens})

        if section == "train":
            for token in tokens:
                self.vocab_builder.add_word(token)


@registry.register("encoder", "scan")
class ScanEncoder(encoder.Encoder):
    batched = True
    Preproc = ScanEncPreproc


@registry.register("encoder", "scan_batched_vanilla")
class ScanLatPerEncoder(batched_encoder.Encoder):
    batched = True
    Preproc = ScanEncPreproc
