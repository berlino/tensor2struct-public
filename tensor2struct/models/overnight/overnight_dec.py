import attr
import os
import json
import logging
import itertools
import collections
import pyrsistent
from typing import Union, List, Dict, Any, Set

import torch
from torch import nn
import torch.nn.functional as F

from tensor2struct.utils import registry
from tensor2struct.models import abstract_preproc
from tensor2struct.modules import attention

from tensor2struct.modules import lstm
import tensor2struct.languages.dsl.common.errors as lf_errors
import tensor2struct.languages.dsl.common.util as lf_util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# disable for training
# logging.disable(logging.CRITICAL)


class DecoderPreproc(abstract_preproc.AbstractPreproc):
    def __init__(self, grammar, save_path, min_freq=3, max_count=5000):
        self.grammar_config = grammar
        self.data_dir = os.path.join(save_path, "dec")
        self.items = collections.defaultdict(list)

        self.prod_dict = collections.defaultdict(set)
        self.domain_prod_dict = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )
        # self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        # self.vocab_path = os.path.join(save_path, 'dec_vocab.json')
        self.observed_productions_path = os.path.join(
            save_path, "observed_productions.json"
        )
        self.domain_productions_path = os.path.join(
            save_path, "domain_productions.json"
        )
        self.data_dir = os.path.join(save_path, "dec")

    def validate_item(self, item, section):
        return True, None

    def add_item(self, item, section, validation_info):
        preprocessed = self.preprocess_item(item, section, validation_info)
        self.items[section].append(preprocessed)

    def preprocess_item(self, item, section, validation_info):
        grammar = registry.construct("grammar", self.grammar_config, domain=item.domain)
        norm_lf = grammar.normalize_lf(item.lf)
        actions = grammar.logical_form_to_action_sequence(norm_lf)

        # for convenice, should be adapted so that it doesn't depend on data
        d_t_rules_dict = grammar.get_domain_terminal_productions()
        for pt in d_t_rules_dict:
            self.domain_prod_dict[item.domain][pt] = self.domain_prod_dict[item.domain][
                pt
            ].union(d_t_rules_dict[pt])

        # those are the pre-defined rules, does not need to induce from train data
        p_rules_dict = grammar.get_non_terminal_productions()
        t_rules_dict = grammar.get_general_terminal_productions()  # treat as labels
        for pt in p_rules_dict:
            self.prod_dict[pt] = self.prod_dict[pt].union(p_rules_dict[pt])
        for pt in t_rules_dict:
            self.prod_dict[pt] = self.prod_dict[pt].union(t_rules_dict[pt])

        return {"domain": item.domain, "productions": actions}

    def clear_items(self):
        self.items = collections.defaultdict(list)

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        # self.vocab = self.vocab_builder.finish()
        # self.vocab.save(self.vocab_path)

        with open(self.observed_productions_path, "w") as f:
            self.prod_dict = {k: sorted(v) for k, v in self.prod_dict.items()}
            json.dump(self.prod_dict, f)
        with open(self.domain_productions_path, "w") as f:
            for d in self.domain_prod_dict:
                self.domain_prod_dict[d] = {
                    k: sorted(v) for k, v in self.domain_prod_dict[d].items()
                }
            json.dump(self.domain_prod_dict, f)

        for section, items in self.items.items():
            with open(os.path.join(self.data_dir, section + ".jsonl"), "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")

    def load(self):
        # self.vocab = vocab.Vocab.load(self.vocab_path)
        with open(self.observed_productions_path, "r") as f:
            self.prod_dict = json.load(f)
        with open(self.domain_productions_path, "r") as f:
            self.domain_prod_dict = json.load(f)
        self.prod_list = sorted(set(itertools.chain(*self.prod_dict.values())))

    def dataset(self, section):
        # for codalab eval
        if len(self.items[section]) > 0:
            return self.items[section]
        else:
            return [
                json.loads(line)
                for line in open(os.path.join(self.data_dir, section + ".jsonl"))
            ]


@attr.s
class RnnStatelet:
    state = attr.ib()
    memory = attr.ib()
    action_embed = attr.ib()
    state_hat = attr.ib()  # Luong's input feeding


class ConstantModule(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.param = nn.Parameter(nn.init.normal_(torch.empty(1, size)))

    def forward(self, input_m):
        batch_size, _ = input_m.size()
        return self.param.expand(batch_size, self.size)


class InferLF:
    def __init__(self, model, enc_state, example):
        self.model = model
        self.example = example
        self.enc_state = enc_state

        self.domain = self.example.domain
        self.general_prod_dict = self.model.prod_dict
        self.pointer_prod_dict = self.model.domain_dict[self.domain]
        self.pointers = self.model.get_prod_pointers(example.domain, enc_state)

        self.cur_state = None
        self.queue = pyrsistent.pvector([lf_util.START_SYMBOL])
        self.history = pyrsistent.pvector()

    def clone(self):
        other = self.__class__(self.model, self.enc_state, self.example)
        other.queue = self.queue
        other.cur_state = self.cur_state
        other.history = self.history  # note it's pyrisistent
        return other

    def pop(self):
        if self.queue:
            item = self.queue[-1]
            self.queue = self.queue.delete(-1)
            return item
        return None

    def peek(self):
        if self.queue:
            return self.queue[-1]
        return None

    def update(self, action):
        self.history = self.history.append(action)
        right_items = lf_util.get_right_side_parts(action)
        non_terminal = lf_util.get_left_side_part(action)

        # pop previous non-terminal
        queue_item = self.pop()
        assert queue_item == non_terminal

        # update current non-terminals
        for p in reversed(right_items):
            if p in self.general_prod_dict or p in self.pointer_prod_dict:
                self.queue = self.queue.append(p)

        # update action embedding
        if (
            non_terminal in self.pointer_prod_dict
            and action in self.pointer_prod_dict[non_terminal]
        ):
            pointer_type = non_terminal
            target_id = self.pointer_prod_dict[pointer_type].index(action)
            pointer_v = self.pointers[pointer_type][target_id].unsqueeze(0)
            action_embed = self.model.pointer_action_emb_proj[pointer_type](pointer_v)
        else:
            assert (
                non_terminal in self.general_prod_dict
                and action in self.general_prod_dict[non_terminal]
            )
            action_embed = self.model.rule_embedding.weight[
                self.model.prod2id[action]
            ].unsqueeze(0)

        self.cur_state.action_embed = action_embed
        self.cur_state, _ = self.model.update_state(self.cur_state, self.enc_state)

    def pointer_rules(self, non_terminal):
        candidates = []
        if non_terminal in self.pointer_prod_dict:
            pointer_type = non_terminal
            if pointer_type not in self.pointers:
                return []
            pointer_logits = self.model.compute_pointer(
                self.cur_state,
                self.enc_state,
                pointer_type,
                self.pointers[pointer_type],
            )
            pointer_logits = pointer_logits.squeeze(0)
            for cand_action, pointer_logit in zip(
                self.pointer_prod_dict[pointer_type], pointer_logits
            ):
                candidates.append((cand_action, pointer_logit))
        return candidates

    def general_rules(self, non_terminal):
        candidates = []
        if non_terminal in self.general_prod_dict:
            valid_prod_strs = self.general_prod_dict[non_terminal]
            valid_prod_ids = [self.model.prod2id[p] for p in valid_prod_strs]
            valid_prod_id_t = torch.LongTensor(valid_prod_ids).to(self.model._device)
            _prod_scores = self.model.rule_logits(self.cur_state.state_hat).squeeze(0)
            prod_logits = _prod_scores[valid_prod_id_t]

            for prod_str, prod_logit in zip(valid_prod_strs, prod_logits):
                candidates.append((prod_str, prod_logit))
        return candidates

    def step(self, action, debug=False):
        if debug:
            print(action)
            print(self.queue)
            print(self.history)

        if action is None:
            self.cur_state = self.model.inital_state
            self.cur_state, _ = self.model.update_state(self.cur_state, self.enc_state)
            cur_non_terminal = lf_util.START_SYMBOL
        else:
            self.update(action)
            cur_non_terminal = self.peek()

        # output next choices
        if cur_non_terminal is None:
            return None
        # (prod, score/logit)
        g_candidates = self.general_rules(cur_non_terminal)
        t_candidates = self.pointer_rules(cur_non_terminal)
        candidates = g_candidates + t_candidates

        if len(candidates) == 0:
            return None

        # renormalize the score
        normalized_candidates = []
        if len(candidates) > 1:
            _logits = torch.log_softmax(torch.stack([a[1] for a in candidates]), dim=-1)
        else:
            _logits = torch.log_softmax(candidates[0][1], dim=-1).unsqueeze(0)
        for i in range(len(candidates)):
            normalized_candidates.append((candidates[i][0], _logits[i]))
        return normalized_candidates

    def finalize(self):
        grammar_class = registry.lookup("grammar", "overnight")
        grammar = grammar_class(self.domain)
        try:
            lf = grammar.action_seq_to_raw_lf(self.history)
        except lf_errors.ParsingError:
            lf = None
        return list(self.history), lf


@attr.s
class TrainCandidate:
    prod_str = attr.ib()
    logit = attr.ib()
    prod_v = attr.ib()
    prod_type = attr.ib()


@registry.register("decoder", "overnight_decoder")
class Decoder(torch.nn.Module):
    Preproc = DecoderPreproc

    def __init__(
        self,
        preproc,
        device,
        enc_recurrent_size=256,
        rule_emb_size=64,
        recurrent_size=256,
        dropout=0.1,
    ):
        super().__init__()
        self.preproc = preproc
        self._device = device

        self.enc_recurrent_size = enc_recurrent_size
        self.rule_emb_size = rule_emb_size
        self.recurrent_size = recurrent_size
        self.prod2id = {v: k for k, v in enumerate(self.preproc.prod_list)}
        self.prod_dict = preproc.prod_dict
        self.domain_dict = preproc.domain_prod_dict

        self.state_update = lstm.VarLSTMCell(
            input_size=self.rule_emb_size + self.recurrent_size * 2,
            hidden_size=self.recurrent_size,
            dropout=dropout,
        )

        self.desc_attn = attention.MultiHeadedAttention(
            h=8, query_size=self.recurrent_size, value_size=self.enc_recurrent_size
        )

        self.rule_logits = torch.nn.Sequential(
            torch.nn.Linear(self.recurrent_size * 2, self.rule_emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.rule_emb_size, len(self.prod2id)),
        )
        self.rule_embedding = torch.nn.Embedding(
            num_embeddings=len(self.prod2id), embedding_dim=self.rule_emb_size
        )

        pointer_types = set()
        for d in self.domain_dict:
            pointer_types = pointer_types.union(set(self.domain_dict[d].keys()))
        self.pointer_types = sorted(pointer_types)
        self.pointers = torch.nn.ModuleDict()
        self.pointer_action_emb_proj = torch.nn.ModuleDict()
        for pointer_type in self.pointer_types:
            self.pointers[pointer_type] = attention.ScaledDotProductPointer(
                query_size=self.recurrent_size, key_size=self.enc_recurrent_size
            )
            # self.pointer_action_emb_proj[pointer_type] = torch.nn.Linear(
            #    self.enc_recurrent_size, self.rule_emb_size
            # )
            self.pointer_action_emb_proj[pointer_type] = ConstantModule(
                self.rule_emb_size
            )

        self.inital_state = self._get_initial_state()

    def _get_initial_state(self) -> RnnStatelet:
        """
        Initial states are trainable parameters
        """
        self.initial_state = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.recurrent_size)).to(self._device)
        )
        self.initial_memory = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.recurrent_size)).to(self._device)
        )
        self.initial_action_embed = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.rule_emb_size)).to(self._device)
        )
        self.inital_state_hat = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.recurrent_size * 2)).to(self._device)
        )
        initial_rnn_state = RnnStatelet(
            self.initial_state,
            self.initial_memory,
            self.initial_action_embed,
            self.inital_state_hat,
        )
        return initial_rnn_state

    def update_state(self, rnn_state, desc_enc):
        context_v, att_logits = self.compute_attention(rnn_state, desc_enc)
        cur_hidden, cur_memory = rnn_state.state, rnn_state.memory
        new_input = torch.cat([rnn_state.action_embed, rnn_state.state_hat], dim=-1)
        next_state, next_memory = self.state_update(new_input, (cur_hidden, cur_memory))
        next_state_hat = torch.cat([next_state, context_v], dim=-1)
        new_decoder_state = RnnStatelet(next_state, next_memory, None, next_state_hat)
        return new_decoder_state, att_logits

    def compute_attention(self, rnn_state, enc_state):
        query = rnn_state.state
        context_v, att_logits = self.desc_attn(query, enc_state.memory)
        return context_v, att_logits

    def compute_pointer_with_align(
        self, rnn_state, enc_state, pointer_type, pointer_v_dic
    ):
        "TODO: this function needs to be changed"
        memory_pointer_logits = self.pointers[pointer_type](
            rnn_state.state, enc_state.memory
        )
        memory_pointer_probs = torch.nn.functional.softmax(memory_pointer_logits, dim=1)
        pointer_probs = torch.mm(
            memory_pointer_probs, enc_state.pointer_align_mat[pointer_type]
        )
        pointer_probs = pointer_probs.clamp(min=1e-9)
        pointer_logits = torch.log(pointer_probs)
        return pointer_logits

    def compute_pointer(self, rnn_state, enc_state, pointer_type, pointer_memory):
        pointer_memory = pointer_memory.unsqueeze(0)  # 1 * mem_len * dim
        memory_pointer_logits = self.pointers[pointer_type](
            rnn_state.state, pointer_memory
        )
        return memory_pointer_logits

    def forward(self, example, enc_state, compute_loss=True, infer=False):
        ret_dict = {}
        ret_dict["mentioned"] = enc_state.mentioned
        if compute_loss:
            ret_dict["loss"] = self.compute_loss(example, enc_state)
        if infer:
            initial_state, initial_choices = self.begin_inference(example, enc_state)
            ret_dict["initial_state"] = initial_state
            ret_dict["initial_choices"] = initial_choices
        return ret_dict

    def get_prod_pointers(self, domain, enc_state):
        v_pointers = collections.defaultdict(list)
        for tp in self.domain_dict[domain]:
            if tp == "Property":
                p = "property"
            else:
                assert "Value" in tp
                p = "value"
            for prod in self.domain_dict[domain][tp]:
                ref = lf_util.get_right_side_parts(prod)[0]
                if ref in enc_state.pointer_refs[p]:
                    ind = enc_state.pointer_refs[p].index(ref)
                    v_pointers[tp].append(enc_state.pointer_memories[p][:, ind])

        for tp in v_pointers:
            if len(v_pointers[tp]) > 0:
                v_pointers[tp] = torch.cat(v_pointers[tp], 0)
        return v_pointers

    def compute_loss(self, dec_output, enc_state):
        self.state_update.set_dropout_masks(batch_size=1)
        domain = dec_output["domain"]
        actions = dec_output["productions"]
        domain_prod_dict = self.domain_dict[domain]
        pointer_v_dic = self.get_prod_pointers(domain, enc_state)

        losses = []
        rnn_state = self.inital_state
        for i, action in enumerate(actions):
            left_side, right_side = action.split(" -> ")
            new_rnn_state, att_logits = self.update_state(rnn_state, enc_state)

            # aggreate candidates
            candidates = []  # (prod, logit, rep (1*dim), type)
            if left_side in domain_prod_dict:
                pointer_type = left_side
                pointer_logits = self.compute_pointer(
                    new_rnn_state, enc_state, pointer_type, pointer_v_dic[pointer_type]
                ).squeeze(0)

                for i, prod in enumerate(domain_prod_dict[pointer_type]):
                    logit = pointer_logits[i]
                    rep = pointer_v_dic[pointer_type][i].unsqueeze(0)
                    candidate = TrainCandidate(prod, logit, rep, "pointer")
                    candidates.append(candidate)

            if left_side in self.prod_dict:
                valid_prod_strs = self.prod_dict[left_side]
                valid_prod_ids = [self.prod2id[p] for p in valid_prod_strs]
                valid_prod_id_t = torch.LongTensor(valid_prod_ids).to(self._device)
                _target_logits = self.rule_logits(new_rnn_state.state_hat).squeeze(0)
                target_logits = _target_logits[valid_prod_id_t]

                for i, prod in enumerate(self.prod_dict[left_side]):
                    logit = target_logits[i]
                    rep = self.rule_embedding.weight[self.prod2id[prod]].unsqueeze(0)
                    candidate = TrainCandidate(prod, logit, rep, "non-pointer")
                    candidates.append(candidate)

            # choose candidates
            cand_prod_strs = [c.prod_str for c in candidates]
            if len(cand_prod_strs) > 1:
                cand_logits = torch.stack([c.logit for c in candidates], dim=0)
            else:
                cand_logits = candidates[0].logit.unsqueeze(0)
            target_id = cand_prod_strs.index(action)
            if candidates[target_id].prod_type == "pointer":
                new_action_embed = self.pointer_action_emb_proj[left_side](
                    candidates[target_id].prod_v
                )
            else:
                new_action_embed = candidates[target_id].prod_v

            loss = -1 * torch.log_softmax(cand_logits, dim=-1)[target_id]
            losses.append(loss)
            new_rnn_state.action_embed = new_action_embed
            rnn_state = new_rnn_state
        return sum(losses)

    def begin_inference(self, example, enc_state):
        self.state_update.set_dropout_masks(batch_size=1)
        # assert not self.training
        infer_lf = InferLF(self, enc_state, example)
        choices = infer_lf.step(None)
        return infer_lf, choices
