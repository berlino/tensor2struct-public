import ast
import collections
import collections.abc
import enum
import itertools
import json
import os
import operator
import re
import copy
import random

import asdl
import attr
import pyrsistent
import entmax
import torch
import torch.nn.functional as F

from tensor2struct.models.ast_decoder.tree_traversal import TreeTraversal


@attr.s
class ChoiceHistoryEntry:
    rule_left = attr.ib()
    choices = attr.ib()
    probs = attr.ib()
    valid_choices = attr.ib()

@attr.s
class LogitHistoryEntry:
    logits = attr.ib()
    log_prob = attr.ib()

class TrainTreeTraversal(TreeTraversal):
    @attr.s(frozen=True)
    class XentChoicePoint:
        logits = attr.ib() # unnormalized logits

        def compute_loss(self, outer, idx, extra_indices):
            if extra_indices:
                logprobs = torch.nn.functional.log_softmax(self.logits, dim=1)
                valid_logprobs = logprobs[:, [idx] + extra_indices]
                return outer.model.multi_loss_reduction(valid_logprobs)
            else:
                # idx shape: batch (=1)
                idx = outer.model._tensor([idx])
                # loss_piece shape: batch (=1)
                return outer.model.xent_loss(self.logits, idx)
        
        def compute_kd_loss(self, outer, label_logits):
            normalized_logits = torch.log_softmax(self.logits, dim=1) 
            label_logits = torch.Tensor(label_logits).to(outer.model._device)
            normalized_label_logits = torch.log_softmax(label_logits, dim=1)
            q_t = torch.exp(normalized_label_logits)
            loss = (-1 * q_t * normalized_logits).sum()
            return loss

    @attr.s(frozen=True)
    class TokenChoicePoint:
        lstm_output = attr.ib()
        gen_logodds = attr.ib()

        def compute_loss(self, outer, token, extra_tokens):
            return outer.model.gen_token_loss(
                self.lstm_output, self.gen_logodds, token, outer.desc_enc
            )

    def __init__(self, model, desc_enc, record_logits=False, lambda_mixture=0.5, kd_logits=None):
        """
        Support record logits and load logits for knowledge distillation
        """
        super().__init__(model, desc_enc)
        self.choice_point = None
        self.loss = pyrsistent.pvector()

        # self knowledge distillation
        self.record_logits = record_logits
        if record_logits:
            self.logits = pyrsistent.pvector()
        self.lambda_mixture = lambda_mixture
        self.kd_logits = kd_logits

    def clone(self):
        super_clone = super().clone()
        super_clone.choice_point = self.choice_point
        super_clone.loss = self.loss
        return super_clone

    def rule_choice(self, node_type, rule_logits):
        self.choice_point = self.XentChoicePoint(rule_logits)

    def token_choice(self, output, gen_logodds):
        self.choice_point = self.TokenChoicePoint(output, gen_logodds)

    def pointer_choice(self, node_type, logits, attention_logits):
        self.choice_point = self.XentChoicePoint(logits)
        self.attention_choice = self.XentChoicePoint(attention_logits)

    def update_using_last_choice(
        self, last_choice, extra_choice_info, attention_offset
    ):
        super().update_using_last_choice(
            last_choice, extra_choice_info, attention_offset
        )
        if last_choice is None:
            return

        # compute loss
        nll_loss = self.choice_point.compute_loss(self, last_choice, extra_choice_info)
        if self.kd_logits is not None and isinstance(self.choice_point, self.XentChoicePoint):
            # pop one item
            cur_label = self.kd_logits[0]
            self.kd_logits = self.kd_logits[1:]
            # print(f"Current {len(self.kd_logits)} items in kd logits ")

            kd_loss = self.choice_point.compute_kd_loss(self, cur_label)
            mixture_loss = self.lambda_mixture * nll_loss + (1 - self.lambda_mixture) * kd_loss
            self.loss = self.loss.append(mixture_loss)
        else:
            self.loss = self.loss.append(nll_loss)

        if self.record_logits and isinstance(self.choice_point, self.XentChoicePoint):
            logit_entry = LogitHistoryEntry(
                self.choice_point.logits.detach().cpu().numpy(),
                -nll_loss.detach().cpu().numpy()[0],
            )
            self.logits = self.logits.append(logit_entry)

        # check if attention choice was used
        if attention_offset is not None and self.attention_choice is not None:
            self.loss = self.loss.append(
                self.attention_choice.compute_loss(
                    self, attention_offset, extra_indices=None
                )
            )

        # empty the choice point
        self.choice_point = None
        self.attention_choice = None
