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

from tensor2struct.models import abstract_preproc
from tensor2struct.modules import attention, variational_lstm, lstm, embedders
from tensor2struct.utils import serialization, vocab, registry, bpe
from tensor2struct.models.ast_decoder.utils import lstm_init

import logging

logger = logging.getLogger("tensor2struct")


@attr.s
class DecItem:
    actions = attr.ib()


class DecoderPreproc(abstract_preproc.AbstractPreproc):
    def __init__(self, save_path, min_freq=0, max_count=10000):
        self.data_dir = os.path.join(save_path, "dec")
        self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        self.vocab_path = os.path.join(save_path, "dec_vocab")

        self.items = collections.defaultdict(list)

    def validate_item(self, item, section):
        return True, None

    def add_item(self, item, section, validation_info):
        actions = item.tgt

        if section == "train":
            for action in actions:
                self.vocab_builder.add_word(action)

        self.items[section].append({"actions": [vocab.BOS] + actions + [vocab.EOS]})

    def clear_items(self):
        self.items = collections.defaultdict(list)

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.vocab = self.vocab_builder.finish()
        print(f"{len(self.vocab)} words in dec vocab")
        self.vocab.save(self.vocab_path)

        for section, texts in self.items.items():
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


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input2feat = nn.Linear(input_size, hidden_size)
        self.feat2output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.feat2output(F.relu(self.input2feat(x))).squeeze(0)


@registry.register("decoder", "vanilla")
class Decoder(torch.nn.Module):
    batched = False
    Preproc = DecoderPreproc

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
            )
        else:
            self.score_action = MLP(
                enc_recurrent_size + recurrent_size, recurrent_size, len(self.vocab)
            )
            self.lstm = lstm.UniLSTM(
                input_size=self.action_emb_size,
                hidden_size=recurrent_size,
                dropout=self.dropout,
            )

    def forward(self, dec_input, enc_output, compute_loss=True, infer=False):
        ret_dic = {}
        if compute_loss:
            ret_dic["loss"] = self.compute_loss(dec_input, enc_output)
        if infer:
            traversal, initial_choices = self.begin_inference(dec_input, enc_output)
            ret_dic["initial_state"] = traversal
            ret_dic["initial_choices"] = initial_choices
        return ret_dic

    def compute_loss(self, dec_input, enc_output):
        if self.input_feed:
            loss = self.compute_loss_with_input_feeding(
                dec_input["actions"], enc_output
            )
        else:
            loss = self.compute_loss_without_input_feeding(
                dec_input["actions"], enc_output
            )
        return loss

    def compute_loss_without_input_feeding(self, actions, enc_output):
        # Attention! the following encoding only works for batch size=1
        input_actions = actions[:-1]
        embed = self.embedder([input_actions])
        init_hidden = enc_output.src_summary
        # TODO: this need to be tested
        dec_rep_packed = self.lstm(embed, hidden_state=(init_hidden, init_hidden))
        dec_rep = dec_rep_packed.select(0)

        # attention
        context, _ = self.desc_attn(
            dec_rep, enc_output.src_memory.expand(len(input_actions), -1, -1),
        )
        feat = torch.cat([dec_rep, context], dim=1)
        logits = self.score_action(feat)
        gold = self.obtain_gold_action(actions)
        loss = F.cross_entropy(logits, gold, reduction="sum")

        return loss

    def compute_loss_with_input_feeding(self, actions, enc_output):
        # encode, only works for batch size=1
        target_actions = actions[1:]
        input_actions = actions[:-1]
        all_embed = self.embedder([input_actions])

        init_hidden = enc_output.src_summary
        lstm_cell = self.lstm.lstm_cells[0]
        lstm_cell.set_dropout_masks(batch_size=1)
        losses = []

        h_input = torch.zeros([1, self.h_input_size]).to(self._device)
        recurrent_state = (init_hidden, init_hidden)
        for i, target_action in enumerate(target_actions):
            embed = all_embed.ps.data[i].unsqueeze(0)  # bs 1
            _input = torch.cat([embed, h_input], dim=1)
            recurrent_state = lstm_cell(_input, recurrent_state)
            h = recurrent_state[0]
            c, _ = self.desc_attn(h, enc_output.src_memory)

            h_input = torch.tanh(self.feat2hidden(torch.cat([c, h], dim=1)))
            logits = torch.log_softmax(self.hidden2action(h_input), dim=1)

            target_id = self.vocab.index(target_action)
            losses.append(-logits[0, target_id])

        loss = sum(losses)
        return loss

    def obtain_gold_action(self, actions):
        output_actions = actions[1:]
        action_ids = [self.vocab.index(a) for a in output_actions]
        return torch.LongTensor(action_ids).to(self._device)

    def begin_inference(self, dec_input, enc_output):
        inferer = Inference(self, enc_output)
        choices = inferer.step()
        return inferer, choices

    def record(self, dec_input, enc_output):
        assert not self.training and self.input_feed
        input_actions = dec_input["actions"][:-1]
        all_embed = self.embedder([input_actions])

        init_hidden = enc_output.src_summary
        lstm_cell = self.lstm.lstm_cells[0]
        lstm_cell.set_dropout_masks(batch_size=1)
        h_input_lists = []

        target_actions = dec_input["actions"][1:]
        target_ids = [self.vocab.index(ac) for ac in target_actions]

        h_input = torch.zeros([1, self.h_input_size]).to(self._device)
        recurrent_state = (init_hidden, init_hidden)
        for i, target_id in enumerate(target_ids):
            embed = all_embed.ps.data[i].unsqueeze(0)  # bs 1
            _input = torch.cat([embed, h_input], dim=1)
            recurrent_state = lstm_cell(_input, recurrent_state)
            h = recurrent_state[0]
            c, _ = self.desc_attn(h, enc_output.src_memory)

            h_input = torch.tanh(self.feat2hidden(torch.cat([c, h], dim=1)))
            h_input_lists.append(h_input.squeeze(0))

        return list(zip(h_input_lists, target_ids))


class Inference:
    def __init__(self, model, enc_output):
        if model is None:
            return None

        self.model = model
        self.vocab = model.vocab
        self.embedder = model.embedder
        self.src_memory = enc_output.src_memory
        self._device = model._device

        self.rnn_cell = model.lstm.lstm_cells[0]
        self.rnn_cell.set_dropout_masks(batch_size=1)

        # init state
        init_hidden = enc_output.src_summary
        self.recurrent_state = (init_hidden, init_hidden)
        self.h_input = torch.zeros([1, self.model.h_input_size]).to(self._device)

        self.actions = pyrsistent.pvector()

    def clone(self):
        other = self.__class__(None, None)
        other.model = self.model
        other.embedder = self.embedder
        other.rnn_cell = self.rnn_cell

        other.vocab = self.vocab
        other._device = self._device

        other.src_memory = self.src_memory
        other.recurrent_state = self.recurrent_state
        other.h_input = self.h_input
        other.actions = self.actions
        return other

    def step(self, action=None):
        if self.model.input_feed:
            return self.step_with_input_feed(action)
        else:
            return self.step_without_input_feed(action)

    def step_without_input_feed(self, action=None):
        if action is None:
            action = vocab.BOS

        self.actions = self.actions.append(action)

        if action == vocab.EOS:
            return None

        action_emb = self.embedder._embed_token(action)
        new_state = self.rnn_cell(action_emb, self.recurrent_state)
        self.recurrent_state = new_state

        hidden_state = new_state[0]
        context, _ = self.model.desc_attn(hidden_state, self.src_memory)
        feat = torch.cat([hidden_state, context], dim=1)
        self.h_input = feat  # for searching
        scores = F.log_softmax(self.model.score_action(feat), dim=0)
        return [(self.vocab[i], scores[i]) for i in range(len(self.vocab))]

    def step_with_input_feed(self, action=None):
        if action is None:
            action = vocab.BOS

        self.actions = self.actions.append(action)

        if action == vocab.EOS:
            return None

        action_emb = self.embedder._embed_token(action).unsqueeze(0)
        _input = torch.cat([action_emb, self.h_input], dim=1)

        new_state = self.rnn_cell(_input, self.recurrent_state)
        self.recurrent_state = new_state

        hidden_state = new_state[0]
        context, _ = self.model.desc_attn(hidden_state, self.src_memory)

        h_input = torch.tanh(
            self.model.feat2hidden(torch.cat([context, hidden_state], dim=1))
        )
        scores = F.log_softmax(self.model.hidden2action(h_input), dim=1).squeeze(0)
        self.h_input = h_input

        return [(self.vocab[i], scores[i]) for i in range(len(self.vocab))]

    def finalize(self,):
        actions = [a for a in self.actions]
        code = " ".join(actions[1:-1])  # exclude bos and eos
        return actions, code

