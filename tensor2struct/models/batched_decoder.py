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
from tensor2struct.modules import attention, variational_lstm, lstm, embedders, rat
from tensor2struct.utils import serialization, vocab, registry, bpe
from tensor2struct.models.ast_decoder.utils import lstm_init

import logging

logger = logging.getLogger("tensor2struct")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@attr.s
class DecItem:
    actions = attr.ib()


@registry.register("decoder", "batched_lstm_dec")
class Decoder(torch.nn.Module):
    batched = True
    Preproc = decoder.DecoderPreproc

    def __init__(
        self,
        device,
        preproc,
        action_emb_size,
        desc_attn="bahdanau",
        enc_recurrent_size=256,
        recurrent_size=256,
        dropout=0.1,
        input_feed=True,
        tie_weights=False,
        layernorm=False,
        label_smooth=0.0,
    ):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.vocab = preproc.vocab

        self.dropout = dropout
        self.action_emb_size = action_emb_size
        self.enc_recurrent_size = enc_recurrent_size
        self.recurrent_size = recurrent_size
        self.input_feed = input_feed
        self.tie_weights = tie_weights
        self.label_smooth = label_smooth
        self.layernorm = layernorm

        # attention
        self.attn_type = desc_attn
        self.desc_attn = attention.BahdanauAttention(
            query_size=self.recurrent_size,
            value_size=self.enc_recurrent_size,
            proj_size=50,
        )
        self.embedder = embedders.LookupEmbeddings(
            device=self._device,
            vocab=self.vocab,
            embedder=None,
            emb_size=self.action_emb_size,
            learnable_words=None,
        )

        if self.input_feed:
            self.h_input_size = recurrent_size
            self.feat2hidden = nn.Linear(
                enc_recurrent_size + recurrent_size, self.h_input_size, bias=False
            )
            self.hidden2action = nn.Linear(
                self.h_input_size, len(self.vocab), bias=False
            )
            self.lstm = lstm.UniLSTM(
                input_size=self.action_emb_size + self.h_input_size,
                hidden_size=recurrent_size,
                dropout=self.dropout,
                layernorm=self.layernorm,
            )
        else:
            self.h_input_size = recurrent_size
            self.hidden2action = nn.Linear(
                self.h_input_size, len(self.vocab), bias=False
            )
            self.lstm = lstm.UniLSTM(
                input_size=self.action_emb_size,
                hidden_size=recurrent_size,
                dropout=self.dropout,
                layernorm=self.layernorm,
            )

        if tie_weights:
            self.hidden2action.weight = self.embedder.embedding.weight

    def forward(self, dec_input, enc_output, compute_loss=True, infer=False):
        ret_dic = {}
        if compute_loss:
            ret_dic["loss"] = self.compute_loss(dec_input, enc_output)
        if infer:
            traversal, initial_choices_list = self.begin_batched_inference(
                dec_input, enc_output
            )
            ret_dic["initial_state"] = traversal
            ret_dic["initial_choices_list"] = initial_choices_list
        return ret_dic

    def compute_loss(self, dec_input, enc_output):
        if self.input_feed:
            loss = self.compute_loss_with_input_feeding(dec_input, enc_output)
        else:
            loss = self.compute_loss_without_input_feeding(dec_input, enc_output)
        return loss

    def compute_loss_without_input_feeding(self, dec_input, enc_output):
        """
        TODO: haven't test this function yet
        """
        logger.warn("Decoder without input feeding has not been tested")

        input_actions = [item["actions"][:-1] for item in dec_input]
        embed = self.embedder(input_actions)
        init_hidden = enc_output.src_summary
        dec_rep_packed, _ = self.lstm(embed.ps, hidden_state=(init_hidden, init_hidden))
        dec_rep = dec_rep_packed.data

        # attention
        context, _ = self.desc_attn(
            dec_rep, enc_output.src_memory.expand(len(input_actions), -1, -1),
        )
        feat = torch.cat([dec_rep, context], dim=1)
        logits = self.score_action(feat)

        ignore_index = len(self.vocab) + 1
        gold = self.obtain_gold_action(input_actions, ignore_index=1)
        loss = F.cross_entropy(logits, gold, reduction="sum", ignore_index=ignore_index)

        return loss

    def compute_loss_with_input_feeding(self, dec_input, enc_output):
        input_actions = [item["actions"][:-1] for item in dec_input]
        embed_packed = self.embedder(input_actions)
        embed_batched, tgt_lengths = embed_packed.pad(batch_first=True)

        bs = len(dec_input)
        init_hidden = enc_output.src_summary
        assert len(self.lstm.lstm_cells) == 1
        lstm_cell = self.lstm.lstm_cells[0]
        lstm_cell.set_dropout_masks(batch_size=bs)

        h_input = torch.zeros([bs, self.h_input_size]).to(self._device)
        recurrent_state = (init_hidden, init_hidden)
        embed_batch_second = embed_batched.transpose(0, 1)
        logits_list = []
        for i in range(max(tgt_lengths)):
            embed = embed_batch_second[i]
            _input = torch.cat([embed, h_input], dim=1)
            recurrent_state = lstm_cell(_input, recurrent_state)
            h = recurrent_state[0]
            c, _ = self.desc_attn(h, enc_output.src_memory)

            h_input = torch.tanh(self.feat2hidden(torch.cat([c, h], dim=1)))
            logits = torch.log_softmax(self.hidden2action(h_input), dim=1)
            logits_list.append(logits)

        logits = torch.stack(logits_list, dim=1)  # bs * seq_len * vocab_len
        ignore_index = self.vocab.index(vocab.BOS)
        target_actions = [item["actions"][1:] for item in dec_input]
        target_idx = self.actions_list_to_idx(target_actions, ignore_index)

        # TODO: make token_nll and seq_nll more explicit
        if self.label_smooth > 0:
            smooth_loss, nll_loss = label_smoothed_nll_loss(
                logits.view(-1, len(self.vocab)),
                target_idx.view(-1),
                self.label_smooth,
                ignore_index=ignore_index,
            )

            num_tgts = sum(tgt_lengths)
            if self.training:
                loss = smooth_loss / num_tgts
            else:
                loss = nll_loss / num_tgts
        else:
            sum_loss = F.cross_entropy(
                logits.view(-1, len(self.vocab)),
                target_idx.view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )
            loss = sum_loss / bs

        return loss

    def actions_list_to_idx(self, actions_list, ignore_index):
        max_len = max(len(al) for al in actions_list)

        res_list = []
        for al in actions_list:
            true_action_ids = [self.vocab.index(a) for a in al]
            assert ignore_index not in true_action_ids
            action_ids = true_action_ids + [ignore_index] * (
                max_len - len(al)
            )
            al_t = torch.LongTensor(action_ids)
            res_list.append(al_t)
        res = torch.stack(res_list, dim=0).to(self._device)
        return res

    def begin_batched_inference(self, orig_item, enc_state):
        inferer = BatchedInference(self, enc_state)
        choices = inferer.step()
        return inferer, choices

    def begin_inference(self, orig_item, enc_state):
        inferer = decoder.Inference(self, enc_state)
        choices = inferer.step()
        return inferer, choices


class BatchedInference:
    batched = True

    def __init__(self, model, enc_output):
        if model is None:
            return None

        self.model = model
        self.vocab = model.vocab
        self.embedder = model.embedder.embedding
        self.src_memory = enc_output.src_memory
        self._device = model._device

        self.bs = self.src_memory.size()[0]
        self.rnn_cell = model.lstm.lstm_cells[0]
        self.rnn_cell.set_dropout_masks(batch_size=self.bs)

        # init state
        init_hidden = enc_output.src_summary
        self.recurrent_state = (init_hidden, init_hidden)
        self.h_input = torch.zeros([self.bs, self.model.h_input_size]).to(self._device)

        self.actions_list = [pyrsistent.pvector()] * self.bs

    def clone(self):
        other = self.__class__(None, None)
        other.model = self.model
        other.embedder = self.embedder
        other.rnn_cell = self.rnn_cell

        other.vocab = self.vocab
        other._device = self._device

        other.bs = self.bs
        other.src_memory = self.src_memory
        other.recurrent_state = self.recurrent_state
        other.h_input = self.h_input
        other.actions_list = self.actions_list
        return other

    def step(self, action=None):
        if self.model.input_feed:
            return self.step_with_input_feed(action)
        else:
            return self.step_without_input_feed(action)

    def step_without_input_feed(self, action=None):
        raise NotImplementedError

    def step_with_input_feed(self, actions=None):
        if actions is None:
            actions = [vocab.BOS] * self.bs

        for i, ac in enumerate(actions):
            self.actions_list[i] = self.actions_list[i].append(ac)

        if all(vocab.EOS in ac for ac in self.actions_list):
            return None

        action_idx = torch.LongTensor(
            [self.vocab.index(action) for action in actions]
        ).to(self.model._device)
        action_emb = self.embedder(action_idx)
        lstm_input = torch.cat([action_emb, self.h_input], dim=1)

        new_state = self.rnn_cell(lstm_input, self.recurrent_state)
        self.recurrent_state = new_state

        hidden_state = new_state[0]
        context, _ = self.model.desc_attn(hidden_state, self.src_memory)

        h_input = torch.tanh(
            self.model.feat2hidden(torch.cat([context, hidden_state], dim=1))
        )
        scores = F.log_softmax(self.model.hidden2action(h_input), dim=1).squeeze(0)
        self.h_input = h_input

        res = []
        num_k = 60 if len(self.model.vocab) > 60 else len(self.model.vocab)
        topk_values, topk_indices = scores.topk(k=num_k, dim=1)
        for b_idx in range(self.bs):
            candidates = [
                (self.vocab[i.item()], scores[b_idx, i]) for i in topk_indices[b_idx]
            ]
            res.append(candidates)
        return res

    def finalize(self,):
        res = []
        for actions in self.actions_list:
            if vocab.EOS in actions:
                eos_idx = actions.index(vocab.EOS)
            else:
                eos_idx = len(actions) - 1
            code = " ".join(actions[1:eos_idx])
            res.append(code)
        return res


@registry.register("decoder", "batched_transformer_dec")
class TransformerDecoder(torch.nn.Module):
    batched = True
    Preproc = decoder.DecoderPreproc

    def __init__(
        self,
        device,
        preproc,
        action_emb_size,
        num_layers=2,
        num_heads=4,
        enc_recurrent_size=256,
        recurrent_size=256,
        dropout=0.1,
        tie_weights=False,
        label_smooth=0.0,
    ):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.vocab = preproc.vocab

        self.dropout = dropout
        self.action_emb_size = action_emb_size
        self.enc_recurrent_size = enc_recurrent_size
        self.recurrent_size = recurrent_size
        self.tie_weights = tie_weights
        self.label_smooth = label_smooth

        self.embedder = embedders.LookupEmbeddings(
            device=device,
            vocab=self.vocab,
            embedder=None,
            emb_size=self.action_emb_size,
            learnable_words=None,
        )

        self.decoder = rat.TransformerDecoder(
            device=device,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=recurrent_size,
            dropout=dropout
        )

        self.score_fn = torch.nn.Linear(recurrent_size, len(self.vocab))
        if tie_weights:
            self.score_fn.weight = self.embedder.embedding.weight

    def forward(self, dec_input, enc_output, compute_loss=True, infer=False):
        ret_dic = {}
        if compute_loss:
            ret_dic["loss"] = self.compute_loss(dec_input, enc_output)
        if infer:
            traversal, initial_choices_list = self.begin_batched_inference(
                dec_input, enc_output
            )
            ret_dic["initial_state"] = traversal
            ret_dic["initial_choices_list"] = initial_choices_list
        return ret_dic

    def compute_loss(self, dec_input, enc_output):
        bs = len(dec_input)
        src_memory = enc_output.src_memory
        src_mask = rat.get_src_attn_mask(enc_output.lengths).to(self._device)

        input_actions = [item["actions"][:-1] for item in dec_input]
        embed_packed = self.embedder(input_actions)
        tgt_emb, tgt_lengths = embed_packed.pad(batch_first=True)

        ignore_index = len(self.vocab) + 1
        tgt_input_idx = self.actions_list_to_idx(input_actions, ignore_index)
        tgt_mask = rat.make_std_mask(tgt_input_idx, ignore_index).to(self._device)

        output_actions = [item["actions"][1:] for item in dec_input]
        tgt_output_idx = self.actions_list_to_idx(output_actions, ignore_index)

        target_enc = self.decoder(tgt_emb, src_memory, src_mask, tgt_mask)
        logits = torch.log_softmax(self.score_fn(target_enc), dim=-1)

        if self.label_smooth > 0:
            num_tgts = sum(tgt_lengths)
            smooth_loss, nll_loss = label_smoothed_nll_loss(
                logits.view(-1, len(self.vocab)),
                tgt_output_idx.view(-1),
                self.label_smooth,
                ignore_index=ignore_index,
            )

            if self.training:
                loss = smooth_loss / num_tgts
            else:
                loss = nll_loss / num_tgts
        else:
            sum_loss = F.cross_entropy(
                logits.view(-1, len(self.vocab)),
                tgt_output_idx.view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )
            loss = sum_loss / bs

        return loss
 
    def actions_list_to_idx(self, actions_list, ignore_index):
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

    def begin_batched_inference(self, dec_input, enc_state):
        inferer = BatchedTransformerInference(self, enc_state)
        choices = inferer.step()
        return inferer, choices

    def begin_inference(self, orig_item, enc_state):
        inferer = UnBatchedTransformerInference(self, enc_state)
        choices = inferer.step()
        return inferer, choices


class BatchedTransformerInference:
    batched = True

    def __init__(self, model, enc_output):
        if model is None:
            return None

        self.model = model
        self.vocab = model.vocab
        self._device = model._device
        self.src_memory = enc_output.src_memory
        self.src_mask = rat.get_src_attn_mask(enc_output.lengths).to(self._device)

        self.bs = self.src_memory.size()[0]
        self.actions_list = [pyrsistent.pvector() for _ in range(self.bs)]

    def clone(self):
        other = self.__class__(None, None)
        other.model = self.model
        other.vocab = self.vocab
        other._device = self._device

        other.bs = self.bs
        other.src_memory = self.src_memory
        other.src_mask = self.src_mask
        other.actions_list = self.actions_list
        return other

    def step(self, actions=None):
        if actions is None:
            actions = [vocab.BOS] * self.bs

        for i, ac in enumerate(actions):
            self.actions_list[i] = self.actions_list[i].append(ac)

        if all(vocab.EOS in ac for ac in self.actions_list):
            return None

        action_emb_packed = self.model.embedder(self.actions_list)
        action_emb, _ = action_emb_packed.pad(batch_first=True)

        # TODO: use incremental decoding
        tgt_mask = rat.subsequent_mask(len(self.actions_list[0])).to(self._device)
        tgt_enc = self.model.decoder(
            action_emb, self.src_memory, self.src_mask, tgt_mask
        )
        tgt_enc = tgt_enc[:, -1]
        scores = F.log_softmax(self.model.score_fn(tgt_enc), dim=1)

        res = []
        # TODO: this num should be set per task, not fixed
        num_k = 3 if len(self.model.vocab) > 3 else len(self.model.vocab)
        topk_values, topk_indices = scores.topk(k=num_k, dim=1)
        for b_idx in range(self.bs):
            candidates = [
                (self.vocab[i.item()], scores[b_idx, i]) for i in topk_indices[b_idx]
            ]
            res.append(candidates)
        return res

    def finalize(self,):
        res = []
        for actions in self.actions_list:
            if vocab.EOS in actions:
                eos_idx = actions.index(vocab.EOS)
            else:
                eos_idx = len(actions) - 1
            code = " ".join(actions[1:eos_idx])  # exclude bos and eos
            res.append(code)
        return res


class UnBatchedTransformerInference:
    batched = False

    def __init__(self, model, enc_output):
        if model is None:
            return None

        self.model = model
        self.vocab = model.vocab
        self._device = model._device
        self.src_memory = enc_output.src_memory
        self.src_mask = rat.get_src_attn_mask(enc_output.lengths).to(self._device)

        self.actions = pyrsistent.pvector()

    def clone(self):
        other = self.__class__(None, None)
        other.model = self.model
        other.vocab = self.vocab
        other._device = self._device

        other.src_memory = self.src_memory
        other.src_mask = self.src_mask
        other.actions = self.actions
        return other

    def step(self, action=None):
        if action is None:
            action = vocab.BOS

        self.actions = self.actions.append(action)

        if action == vocab.EOS:
            return None

        action_emb_packed = self.model.embedder([self.actions])
        action_emb, _ = action_emb_packed.pad(batch_first=True)

        tgt_mask = rat.subsequent_mask(len(self.actions)).to(self._device)
        tgt_enc = self.model.decoder(
            action_emb, self.src_memory, self.src_mask, tgt_mask
        )
        tgt_enc = tgt_enc[:, -1]
        scores = F.log_softmax(self.model.score_fn(tgt_enc), dim=1)

        num_k = 100 if len(self.model.vocab) > 100 else len(self.model.vocab)
        topk_values, topk_indices = scores.topk(k=num_k, dim=1)
        candidates = [(self.vocab[i.item()], scores[0, i]) for i in topk_indices[0]]
        return candidates

    def finalize(self,):
        actions = [a for a in self.actions]
        code = " ".join(actions[1:-1])  # exclude bos and eos
        return actions, code
