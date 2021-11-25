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

from tensor2struct.models import abstract_preproc, decoder
from tensor2struct.modules import attention, variational_lstm, lstm, embedders
from tensor2struct.utils import serialization, vocab, registry

import logging

logger = logging.getLogger("tensor2struct")


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input2feat = nn.Linear(input_size, hidden_size)
        self.feat2output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.feat2output(F.relu(self.input2feat(x))).squeeze(0)


class TaggingPreproc(decoder.DecoderPreproc):
    def add_item(self, item, section, validation_info):
        actions = item.tgt

        if section == "train":
            for action in actions:
                self.vocab_builder.add_word(action)

        self.items[section].append({"actions": actions})


@registry.register("decoder", "tagging_dec")
class TaggingDecoder(torch.nn.Module):
    """
    Decode re-ordered input word-by-word
    """

    batched = True
    Preproc = TaggingPreproc

    def __init__(self, device, preproc, enc_recurrent_size=256, score_f="linear"):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.vocab = preproc.vocab
        self.enc_recurrent_size = enc_recurrent_size

        if score_f == "linear":
            # use Linear layer instead of MLP results to a convex loss function
            self.score_f = torch.nn.Linear(enc_recurrent_size, len(self.vocab))
        else:
            assert score_f == "mlp"
            self.score_f = MLP(
                enc_recurrent_size, enc_recurrent_size * 2, len(self.vocab)
            )

    def forward(self, dec_batch, enc_state, compute_loss=True, infer=False):
        ret_dic = {}
        if compute_loss:
            ret_dic["loss"] = self.compute_loss(dec_batch, enc_state)
        if infer:
            dec_item = dec_batch  # by default, inference model is with bs 1
            enc_state.src_memory.squeeze_(0)  # enc_state is 1 * seq_len * hidden_size
            traversal, initial_choices = self.begin_inference(dec_item, enc_state)
            ret_dic["initial_state"] = traversal
            ret_dic["initial_choices"] = initial_choices
        return ret_dic

    def compute_loss(self, dec_batch, enc_state):
        bs = len(dec_batch)
        src_memory = enc_state.src_memory
        ignore_index = len(self.vocab) + 1
        gold = self.obtain_gold_action(dec_batch, ignore_index=ignore_index)
        logits = self.score_f(src_memory)

        loss = F.cross_entropy(
            logits.view(-1, len(self.vocab)),
            gold.view(-1),
            reduction="sum",
            ignore_index=ignore_index,
        )
        loss = loss / bs
        return loss

    def obtain_gold_action(self, dec_batch, ignore_index):
        actions_list = [dec_item["actions"] for dec_item in dec_batch]
        max_len = max(len(al) for al in actions_list)

        res_list = []
        for al in actions_list:
            action_ids = [self.vocab.index(a) for a in al] + [ignore_index] * (
                max_len - len(al)
            )
            al_t = torch.LongTensor(action_ids)
            res_list.append(al_t)
        res = torch.stack(res_list, dim=0).to(self._device)
        return res

    def begin_inference(self, dec_item, enc_output):
        """
        In inference mode, inputs are not batched
        For batched inference, use "python experiments/permutation/run.py eval_aligned configfile" for now
        """
        logits = F.log_softmax(self.score_f(enc_output.src_memory), dim=1)
        inferer = Inference(self, logits)
        choices = inferer.step()
        return inferer, choices


class Inference:
    def __init__(self, model, logits):
        if model is None:
            return None

        self.model = model
        self.logits = logits
        self.vocab = model.vocab

        self.decode_idx = 0
        self.actions = pyrsistent.pvector()

    def clone(self):
        other = self.__class__(None, None)
        other.model = self.model
        other.vocab = self.vocab
        other.logits = self.logits

        other.actions = self.actions
        other.decode_idx = self.decode_idx
        return other

    def step(self, action=None):
        if action is not None:
            self.actions = self.actions.append(action)

        # last word
        num_tokens = self.logits.size()[0]
        if self.decode_idx >= num_tokens - 1:
            return None

        scores = self.logits[self.decode_idx]
        self.decode_idx += 1
        return [(self.vocab[i], scores[i]) for i in range(len(self.vocab))]

    def finalize(self):
        res_actions = [a for a in self.actions]
        code = " ".join(res_actions)
        return res_actions, code
