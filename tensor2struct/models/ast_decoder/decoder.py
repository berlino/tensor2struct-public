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

from tensor2struct.languages.ast import ast_util
from tensor2struct.models import abstract_preproc
from tensor2struct.modules import attention, variational_lstm

from tensor2struct.utils import registry, vocab, serialization
from tensor2struct.models.ast_decoder.tree_traversal import TreeTraversal, TreeState
from tensor2struct.models.ast_decoder.train_tree_traversal import TrainTreeTraversal
from tensor2struct.models.ast_decoder.infer_tree_traversal import InferenceTreeTraversal
from tensor2struct.modules import lstm, bert_tokenizer
from tensor2struct.models.ast_decoder.utils import (
    accumulate_logprobs,
    maybe_stack,
    lstm_init,
    get_field_presence_info,
)

import logging

logger = logging.getLogger("tensor2struct")


@attr.s
class NL2CodeDecoderPreprocItem:
    tree = attr.ib()
    orig_code = attr.ib()

    kd_logits = attr.ib(default=None)

class NL2CodeDecoderPreproc(abstract_preproc.AbstractPreproc):
    def __init__(
        self,
        grammar,
        save_path,
        min_freq=3,
        max_count=5000,
        use_seq_elem_rules=False,
        value_tokenizer=None,
    ):
        self.grammar = registry.construct("grammar", grammar)
        self.ast_wrapper = self.grammar.ast_wrapper

        # tokenizer for value prediction, lazy init
        self.value_tokenizer_config = value_tokenizer

        self.vocab_path = os.path.join(save_path, "dec_vocab.json")
        self.observed_productions_path = os.path.join(
            save_path, "observed_productions.json"
        )
        self.grammar_rules_path = os.path.join(save_path, "grammar_rules.json")
        self.data_dir = os.path.join(save_path, "dec")

        self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        self.use_seq_elem_rules = use_seq_elem_rules

        self.items = collections.defaultdict(list)
        self.sum_type_constructors = collections.defaultdict(set)
        self.field_presence_infos = collections.defaultdict(set)
        self.seq_lengths = collections.defaultdict(set)
        self.primitive_types = set()

        self.vocab = None
        self.all_rules = None
        self.rules_mask = None

    @property
    def value_tokenizer(self):
        if not hasattr(self, "_value_tokenizer"):
            if self.value_tokenizer_config is not None:
                self._value_tokenizer = bert_tokenizer.BERTokenizer(
                    self.value_tokenizer_config
                )
            else:
                self._value_tokenizer = None
        return self._value_tokenizer

    def tokenize_field_value(self, value):
        """
        Tokenization comes from two sources:
        1) grammar tokenizer 
        2) decoder tokenizer that should be aligned with encoder tokenizer, since it will be used for copying values from input

        Note that by default, the space are kept in the vocab. So in order to predict multi-word values like "Hello World", it 
        has to generate the space explicitly 
        Since we only need to collect words that cannot be copied, we use tokenize_with_orig to obtain the
        original (sub-)words (possibly cased).
        """
        orig_tok_vals = self.grammar.tokenize_field_value(value)

        if self.grammar.include_literals and self.value_tokenizer is not None:
            tok_vals = []
            for tok in orig_tok_vals:
                if tok in [" ", "True", "False", "None"]:  # keep space
                    tok_vals.append(tok)
                else:
                    tok_vals += self.value_tokenizer.tokenize_with_orig(tok)
        else:
            tok_vals = orig_tok_vals
        return tok_vals

    def validate_item(self, item, section):
        # for syn data, item.code is already tree-structured
        if "_type" in item.code:
            parsed = item.code
        else:
            parsed = self.grammar.parse(item.code)

        if parsed:
            try:
                self.ast_wrapper.verify_ast(parsed)
            except AssertionError:
                logger.warn("One AST cannot be verified!")
            return True, parsed
        return section not in ["train", "syn_train"], None

    def add_item(self, item, section, validation_info):
        root = validation_info
        if section in ["train", "syn_train"]:
            for token in self._all_tokens(root):
                self.vocab_builder.add_word(token)
            self._record_productions(root)

        self.items[section].append(
            NL2CodeDecoderPreprocItem(tree=root, orig_code=item.code)
        )

    def clear_items(self):
        self.items = collections.defaultdict(list)

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.vocab = self.vocab_builder.finish()
        self.vocab.save(self.vocab_path)

        for section, items in self.items.items():
            with open(os.path.join(self.data_dir, section + ".jsonl"), "w") as f:
                for item in items:
                    f.write(json.dumps(attr.asdict(item)) + "\n")

        # observed_productions
        self.sum_type_constructors = serialization.to_dict_with_sorted_values(
            self.sum_type_constructors
        )
        self.field_presence_infos = serialization.to_dict_with_sorted_values(
            self.field_presence_infos, key=str
        )
        self.seq_lengths = serialization.to_dict_with_sorted_values(self.seq_lengths)
        self.primitive_types = sorted(self.primitive_types)
        with open(self.observed_productions_path, "w") as f:
            json.dump(
                {
                    "sum_type_constructors": self.sum_type_constructors,
                    "field_presence_infos": self.field_presence_infos,
                    "seq_lengths": self.seq_lengths,
                    "primitive_types": self.primitive_types,
                },
                f,
                indent=2,
                sort_keys=True,
            )

        # grammar
        self.all_rules, self.rules_mask = self._calculate_rules()
        with open(self.grammar_rules_path, "w") as f:
            json.dump(
                {"all_rules": self.all_rules, "rules_mask": self.rules_mask},
                f,
                indent=2,
                sort_keys=True,
            )

    def load(self):
        self.vocab = vocab.Vocab.load(self.vocab_path)

        # load grammar rules, this still follows origianl api
        observed_productions = json.load(open(self.observed_productions_path))
        self.sum_type_constructors = observed_productions["sum_type_constructors"]
        self.field_presence_infos = observed_productions["field_presence_infos"]
        self.seq_lengths = observed_productions["seq_lengths"]
        self.primitive_types = observed_productions["primitive_types"]

        grammar = json.load(open(self.grammar_rules_path))
        self.all_rules = serialization.tuplify(grammar["all_rules"])
        self.rules_mask = grammar["rules_mask"]

    def dataset(self, section):
        if len(self.items[section]) > 0:
            return self.items[section]
        else:
            return [
                NL2CodeDecoderPreprocItem(**json.loads(line))
                for line in open(os.path.join(self.data_dir, section + ".jsonl"))
            ]

    def _record_productions(self, tree):
        queue = [(tree, False)]
        while queue:
            node, is_seq_elem = queue.pop()
            node_type = node["_type"]

            # Rules of the form:
            # expr -> Attribute | Await | BinOp | BoolOp | ...
            # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
            for type_name in [node_type] + node.get("_extra_types", []):
                if type_name in self.ast_wrapper.constructors:
                    sum_type_name = self.ast_wrapper.constructor_to_sum_type[type_name]
                    if is_seq_elem and self.use_seq_elem_rules:
                        self.sum_type_constructors[sum_type_name + "_seq_elem"].add(
                            type_name
                        )
                    else:
                        self.sum_type_constructors[sum_type_name].add(type_name)

            # Rules of the form:
            # FunctionDef
            # -> identifier name, arguments args
            # |  identifier name, arguments args, stmt* body
            # |  identifier name, arguments args, expr* decorator_list
            # |  identifier name, arguments args, expr? returns
            # ...
            # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
            assert node_type in self.ast_wrapper.singular_types
            field_presence_info = get_field_presence_info(
                self.ast_wrapper,
                node,
                self.ast_wrapper.singular_types[node_type].fields,
            )
            self.field_presence_infos[node_type].add(field_presence_info)

            # TODO: does not need record for fields that are not present
            for field_info in self.ast_wrapper.singular_types[node_type].fields:
                field_value = node.get(field_info.name, [] if field_info.seq else None)
                to_enqueue = []
                if field_info.seq:
                    # Rules of the form:
                    # stmt* -> stmt
                    #        | stmt stmt
                    #        | stmt stmt stmt
                    self.seq_lengths[field_info.type + "*"].add(len(field_value))
                    to_enqueue = field_value
                else:
                    to_enqueue = [field_value]
                for child in to_enqueue:
                    if isinstance(child, collections.abc.Mapping) and "_type" in child:
                        queue.append((child, field_info.seq))
                    else:
                        self.primitive_types.add(type(child).__name__)

    def _calculate_rules(self):
        offset = 0

        all_rules = []
        rules_mask = {}

        # Rules of the form:
        # expr -> Attribute | Await | BinOp | BoolOp | ...
        # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
        for parent, children in sorted(self.sum_type_constructors.items()):
            assert not isinstance(children, set)
            rules_mask[parent] = (offset, offset + len(children))
            offset += len(children)
            all_rules += [(parent, child) for child in children]

        # Rules of the form:
        # FunctionDef
        # -> identifier name, arguments args
        # |  identifier name, arguments args, stmt* body
        # |  identifier name, arguments args, expr* decorator_list
        # |  identifier name, arguments args, expr? returns
        # ...
        # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
        for name, field_presence_infos in sorted(self.field_presence_infos.items()):
            assert not isinstance(field_presence_infos, set)
            rules_mask[name] = (offset, offset + len(field_presence_infos))
            offset += len(field_presence_infos)
            all_rules += [(name, presence) for presence in field_presence_infos]

        # Rules of the form:
        # stmt* -> stmt
        #        | stmt stmt
        #        | stmt stmt stmt
        for seq_type_name, lengths in sorted(self.seq_lengths.items()):
            assert not isinstance(lengths, set)
            rules_mask[seq_type_name] = (offset, offset + len(lengths))
            offset += len(lengths)
            all_rules += [(seq_type_name, i) for i in lengths]

        return tuple(all_rules), rules_mask

    def _all_tokens(self, root):
        queue = [root]
        while queue:
            node = queue.pop()
            type_info = self.ast_wrapper.singular_types[node["_type"]]

            for field_info in reversed(type_info.fields):
                field_value = node.get(field_info.name)
                if field_info.type in self.grammar.pointers:
                    pass
                elif field_info.type in self.ast_wrapper.primitive_types:
                    for token in self.tokenize_field_value(field_value):
                        yield token
                elif isinstance(field_value, (list, tuple)):
                    queue.extend(field_value)
                elif field_value is not None:
                    queue.append(field_value)


@registry.register("decoder", "NL2CodeV2")
class NL2CodeDecoderV2(torch.nn.Module):

    Preproc = NL2CodeDecoderPreproc

    def __init__(
        self,
        device,
        preproc,
        rule_emb_size=128,
        node_embed_size=64,
        enc_recurrent_size=256,
        recurrent_size=256,
        dropout=0.0,
        desc_attn="bahdanau",
        copy_pointer=None,
        multi_loss_type="logsumexp",
        sup_att=None,
        use_align_mat=False,
        use_align_loss=False,
        enumerate_order=False,
        loss_type="softmax",
        alpha_smooth=0.1,
    ):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.ast_wrapper = preproc.ast_wrapper
        self.terminal_vocab = preproc.vocab
        self.rule_emb_size = rule_emb_size
        self.node_emb_size = node_embed_size
        self.enc_recurrent_size = enc_recurrent_size
        self.recurrent_size = recurrent_size

        # state lstm
        self.state_update = lstm.VarLSTMCell(
            input_size=self.rule_emb_size * 2
            + self.enc_recurrent_size
            + self.recurrent_size
            + self.node_emb_size,
            hidden_size=self.recurrent_size,
            dropout=dropout,
        )

        # alignment
        self.use_align_mat = use_align_mat
        self.use_align_loss = use_align_loss
        self.enumerate_order = enumerate_order
        if use_align_loss:
            assert use_align_mat

        # decoder attention
        self.attn_type = desc_attn
        if desc_attn == "bahdanau":
            self.desc_attn = attention.BahdanauAttention(
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size,
                proj_size=50,
            )
        elif desc_attn == "mha":
            self.desc_attn = attention.MultiHeadedAttention(
                h=8, query_size=self.recurrent_size, value_size=self.enc_recurrent_size
            )
        elif desc_attn == "mha-1h":
            self.desc_attn = attention.MultiHeadedAttention(
                h=1, query_size=self.recurrent_size, value_size=self.enc_recurrent_size
            )
        elif desc_attn == "sep":
            self.question_attn = attention.MultiHeadedAttention(
                h=1, query_size=self.recurrent_size, value_size=self.enc_recurrent_size
            )
            self.schema_attn = attention.MultiHeadedAttention(
                h=1, query_size=self.recurrent_size, value_size=self.enc_recurrent_size
            )
        else:
            # TODO: Figure out how to get right sizes (query, value) to module
            self.desc_attn = desc_attn
        self.sup_att = sup_att

        # rule classification and embedding
        self.rules_index = {v: idx for idx, v in enumerate(self.preproc.all_rules)}
        self.rule_logits = torch.nn.Sequential(
            torch.nn.Linear(self.recurrent_size, self.rule_emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.rule_emb_size, len(self.rules_index)),
        )
        self.rule_embedding = torch.nn.Embedding(
            num_embeddings=len(self.rules_index), embedding_dim=self.rule_emb_size
        )

        # token classification and embedding
        self.gen_logodds = torch.nn.Linear(self.recurrent_size, 1)
        self.terminal_logits = torch.nn.Sequential(
            torch.nn.Linear(self.recurrent_size, self.rule_emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.rule_emb_size, len(self.terminal_vocab)),
        )
        self.terminal_embedding = torch.nn.Embedding(
            num_embeddings=len(self.terminal_vocab), embedding_dim=self.rule_emb_size
        )

        # copy pointer
        if copy_pointer is None:
            self.copy_pointer = attention.BahdanauPointer(
                query_size=self.recurrent_size,
                key_size=self.enc_recurrent_size,
                proj_size=50,
            )
        else:
            self.copy_pointer = copy_pointer

        # column/table pointers
        self.pointers = torch.nn.ModuleDict()
        self.pointer_action_emb_proj = torch.nn.ModuleDict()
        for pointer_type in self.preproc.grammar.pointers:
            self.pointers[pointer_type] = attention.ScaledDotProductPointer(
                query_size=self.recurrent_size, key_size=self.enc_recurrent_size
            )
            self.pointer_action_emb_proj[pointer_type] = torch.nn.Linear(
                self.enc_recurrent_size, self.rule_emb_size
            )

        # parent type
        if self.preproc.use_seq_elem_rules:
            self.node_type_vocab = vocab.Vocab(
                sorted(self.preproc.primitive_types)
                + sorted(self.ast_wrapper.custom_primitive_types)
                + sorted(self.preproc.sum_type_constructors.keys())
                + sorted(self.preproc.field_presence_infos.keys())
                + sorted(self.preproc.seq_lengths.keys()),
                special_elems=(),
            )
        else:
            self.node_type_vocab = vocab.Vocab(
                sorted(self.preproc.primitive_types)
                + sorted(self.ast_wrapper.custom_primitive_types)
                + sorted(self.ast_wrapper.sum_types.keys())
                + sorted(self.ast_wrapper.singular_types.keys())
                + sorted(self.preproc.seq_lengths.keys()),
                special_elems=(),
            )
        self.node_type_embedding = torch.nn.Embedding(
            num_embeddings=len(self.node_type_vocab), embedding_dim=self.node_emb_size
        )

        # Other neural facilities
        self.zero_rule_emb = torch.zeros(1, self.rule_emb_size, device=self._device)
        self.zero_recurrent_emb = torch.zeros(
            1, self.recurrent_size, device=self._device
        )

        if multi_loss_type == "logsumexp":
            self.multi_loss_reduction = lambda logprobs: -torch.logsumexp(
                logprobs, dim=1
            )
        elif multi_loss_type == "mean":
            self.multi_loss_reduction = lambda logprobs: -torch.mean(logprobs, dim=1)

        self.losses = {
            # "softmax": torch.nn.CrossEntropyLoss(reduction="none"),
            "softmax": lambda X, target: torch.nn.functional.cross_entropy(X, target, reduction="none"),
            "entmax": entmax.entmax15_loss,
            "sparsemax": entmax.sparsemax_loss,
            "label_smooth": self.label_smooth_loss,
        }
        self.alpha_smooth = alpha_smooth
        self.xent_loss = self.losses[loss_type]

    def label_smooth_loss(self, X, target):
        if self.training:
            logits = torch.log_softmax(X, dim=1)
            size = X.size()[1]
            if size > 1:
                if isinstance(self.alpha_smooth, torch.Tensor):
                    peak_value = torch.sigmoid(self.alpha_smooth)
                    avg_value = (1-peak_value) / (size - 1)
                    softlabel = avg_value.unsqueeze(0).expand(1, size)
                    softlabel = softlabel.scatter(1, target.unsqueeze(0), peak_value)
                else:
                    softlabel = torch.full(X.size(), self.alpha_smooth / (size - 1)).to(
                        X.device
                    )
                    softlabel.scatter_(1, target.unsqueeze(0), 1 - self.alpha_smooth)
            else:
                softlabel = torch.full(X.size(), 1.0).to(X.device)
            # loss = F.kl_div(logits, one_hot, reduction="batchmean")
            loss = (-1 * softlabel * logits).sum(dim=1)
            return loss
        else:
            return torch.nn.functional.cross_entropy(X, target, reduction="none")

    def forward(self, example, desc_enc, compute_loss=True, infer=False):
        """
        This is the only entry point of decoder
        """
        ret_dic = {}
        if compute_loss:
            try:
                ret_dic["loss"] = self.compute_loss(example, desc_enc)
            # except KeyError as e1, AssertionError as e:
            except Exception as e:
                # TODO: synthetic data have unseen rules, not sure why
                logger.info(f"Loss computation error: {str(e)}")
                ret_dic["loss"] = torch.Tensor([0.0]).to(self._device)
        if infer:
            traversal, initial_choices = self.begin_inference(example, desc_enc)
            ret_dic["initial_state"] = traversal
            ret_dic["initial_choices"] = initial_choices
        return ret_dic

    def compute_loss(self, example, desc_enc):
        if not self.enumerate_order or not self.training:
            mle_loss = self.compute_mle_loss(example, desc_enc)
        else:
            mle_loss = self.compute_loss_from_all_ordering(example, desc_enc)

        if self.use_align_loss:
            if desc_enc.relation.cq_relation is not None:
                # if cq_relation is not None, then latent relations are extracted, so alignment
                # loss will be enforced on latent relations
                align_loss = self.compute_align_loss_from_latent_relation(
                    desc_enc, example
                )
            else:
                # otherwise, we put the loss over alignment matrix
                align_loss = self.compute_align_loss_from_alignment_mat(
                    desc_enc, example
                )
            return mle_loss + align_loss
        return mle_loss

    def compute_loss_from_all_ordering(self, example, desc_enc):
        def get_permutations(node):
            def traverse_tree(node):
                nonlocal permutations
                if isinstance(node, (list, tuple)):
                    p = itertools.permutations(range(len(node)))
                    permutations.append(list(p))
                    for child in node:
                        traverse_tree(child)
                elif isinstance(node, dict):
                    for node_name in node:
                        traverse_tree(node[node_name])

            permutations = []
            traverse_tree(node)
            return permutations

        def get_perturbed_tree(node, permutation):
            def traverse_tree(node, parent_type, parent_node):
                if isinstance(node, (list, tuple)):
                    nonlocal permutation
                    p_node = [node[i] for i in permutation[0]]
                    parent_node[parent_type] = p_node
                    permutation = permutation[1:]
                    for child in node:
                        traverse_tree(child, None, None)
                elif isinstance(node, dict):
                    for node_name in node:
                        traverse_tree(node[node_name], node_name, node)

            node = copy.deepcopy(node)
            traverse_tree(node, None, None)
            return node

        orig_tree = example.tree
        permutations = get_permutations(orig_tree)
        products = itertools.product(*permutations)
        loss_list = []
        for product in products:
            tree = get_perturbed_tree(orig_tree, product)
            example.tree = tree
            loss = self.compute_mle_loss(example, desc_enc)
            loss_list.append(loss)
        example.tree = orig_tree
        loss_v = torch.stack(loss_list, 0)
        return torch.logsumexp(loss_v, 0)

    def compute_mle_loss(
        self, example, desc_enc, record_logits=False, kd_logits=None, lambda_mixture=0.5
    ):
        traversal = TrainTreeTraversal(
            self, desc_enc, record_logits=record_logits, lambda_mixture=lambda_mixture, kd_logits=kd_logits
        )
        traversal.step(None)
        queue = [
            TreeState(
                node=example.tree, parent_field_type=self.preproc.grammar.root_type
            )
        ]
        while queue:
            item = queue.pop()
            node = item.node
            parent_field_type = item.parent_field_type

            if isinstance(node, (list, tuple)):
                node_type = parent_field_type + "*"
                rule = (node_type, len(node))
                rule_idx = self.rules_index[rule]
                assert traversal.cur_item.state == TreeTraversal.State.LIST_LENGTH_APPLY
                traversal.step(rule_idx)

                if (
                    self.preproc.use_seq_elem_rules
                    and parent_field_type in self.ast_wrapper.sum_types
                ):
                    parent_field_type += "_seq_elem"

                for i, elem in reversed(list(enumerate(node))):
                    queue.append(
                        TreeState(node=elem, parent_field_type=parent_field_type)
                    )
                continue

            if parent_field_type in self.preproc.grammar.pointers:
                assert isinstance(node, int)
                assert traversal.cur_item.state == TreeTraversal.State.POINTER_APPLY
                pointer_map = desc_enc.pointer_maps.get(parent_field_type)
                if pointer_map:
                    values = pointer_map[node]
                    traversal.step(values[0], values[1:])
                else:
                    traversal.step(node)
                continue

            if parent_field_type in self.ast_wrapper.primitive_types:
                # identifier, int, string, bytes, object, singleton
                # - could be bytes, str, int, float, bool, NoneType
                # - terminal tokens vocabulary is created by turning everything into a string (with `str`)
                # - at decoding time, cast back to str/int/float/bool
                field_type = type(node).__name__
                field_value_split = self.preproc.tokenize_field_value(node) + [
                    vocab.EOS
                ]

                for token in field_value_split:
                    assert traversal.cur_item.state == TreeTraversal.State.GEN_TOKEN
                    traversal.step(token)
                continue

            type_info = self.ast_wrapper.singular_types[node["_type"]]

            if parent_field_type in self.preproc.sum_type_constructors:
                # ApplyRule, like expr -> Call
                rule = (parent_field_type, type_info.name)
                rule_idx = self.rules_index[rule]
                assert traversal.cur_item.state == TreeTraversal.State.SUM_TYPE_APPLY
                extra_rules = [
                    self.rules_index[parent_field_type, extra_type]
                    for extra_type in node.get("_extra_types", [])
                ]
                traversal.step(rule_idx, extra_rules)

            if type_info.fields:
                # ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
                # Figure out which rule needs to be applied
                present = get_field_presence_info(
                    self.ast_wrapper, node, type_info.fields
                )
                rule = (node["_type"], tuple(present))
                rule_idx = self.rules_index[rule]
                assert traversal.cur_item.state == TreeTraversal.State.CHILDREN_APPLY
                traversal.step(rule_idx)

            # reversed so that we perform a DFS in left-to-right order
            for field_info in reversed(type_info.fields):
                if field_info.name not in node:
                    continue

                queue.append(
                    TreeState(
                        node=node[field_info.name], parent_field_type=field_info.type
                    )
                )

        loss = torch.sum(torch.stack(tuple(traversal.loss), dim=0), dim=0)
        if record_logits:
            return loss, traversal.logits
        else:
            return loss

    def begin_inference(self, example, desc_enc):
        traversal = InferenceTreeTraversal(self, desc_enc, example)
        choices = traversal.step(None)
        return traversal, choices

    def _desc_attention(self, prev_state, desc_enc):
        # prev_state shape:
        # - h_n: batch (=1) x emb_size
        # - c_n: batch (=1) x emb_size
        query = prev_state[0]
        if self.attn_type != "sep":
            return self.desc_attn(query, desc_enc.memory, attn_mask=None)
        else:
            question_context, question_attention_logits = self.question_attn(
                query, desc_enc.question_memory
            )
            schema_context, schema_attention_logits = self.schema_attn(
                query, desc_enc.schema_memory
            )
            return question_context + schema_context, schema_attention_logits

    def _tensor(self, data, dtype=None):
        return torch.tensor(data, dtype=dtype, device=self._device)

    def _index(self, vocab, word):
        return self._tensor([vocab.index(word)])

    def _update_state(
        self,
        node_type,
        prev_state,
        prev_action_emb,
        parent_h,
        parent_action_emb,
        desc_enc,
    ):
        # desc_context shape: batch (=1) x emb_size
        desc_context, attention_logits = self._desc_attention(prev_state, desc_enc)

        # node_type_emb shape: batch (=1) x emb_size
        node_type_emb = self.node_type_embedding(
            self._index(self.node_type_vocab, node_type)
        )

        state_input = torch.cat(
            (
                prev_action_emb,  # a_{t-1}: rule_emb_size
                desc_context,  # c_t: enc_recurrent_size
                parent_h,  # s_{p_t}: recurrent_size
                parent_action_emb,  # a_{p_t}: rule_emb_size
                node_type_emb,  # n_{f-t}: node_emb_size
            ),
            dim=-1,
        )
        new_state = self.state_update(
            # state_input shape: batch (=1) x (emb_size * 5)
            state_input,
            prev_state,
        )
        return new_state, attention_logits

    def apply_rule(
        self,
        node_type,
        prev_state,
        prev_action_emb,
        parent_h,
        parent_action_emb,
        desc_enc,
    ):
        new_state, attention_logits = self._update_state(
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc,
        )
        # output shape: batch (=1) x emb_size
        output = new_state[0]
        # rule_logits shape: batch (=1) x num choices
        rule_logits = self.rule_logits(output)

        return output, new_state, rule_logits

    def rule_infer(self, node_type, rule_logits):
        rule_logprobs = torch.nn.functional.log_softmax(rule_logits, dim=-1)
        rules_start, rules_end = self.preproc.rules_mask[node_type]

        # TODO: Mask other probabilities first?
        return list(
            zip(range(rules_start, rules_end), rule_logprobs[0, rules_start:rules_end])
        )

    def gen_token(
        self,
        node_type,
        prev_state,
        prev_action_emb,
        parent_h,
        parent_action_emb,
        desc_enc,
    ):
        new_state, attention_logits = self._update_state(
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc,
        )
        # output shape: batch (=1) x emb_size
        output = new_state[0]

        # gen_logodds shape: batch (=1)
        gen_logodds = self.gen_logodds(output).squeeze(1)

        return new_state, output, gen_logodds

    def gen_token_loss(self, output, gen_logodds, token, desc_enc):
        # token_idx shape: batch (=1), LongTensor
        token_idx = self._index(self.terminal_vocab, token)
        # action_emb shape: batch (=1) x emb_size
        action_emb = self.terminal_embedding(token_idx)

        # +unk, +in desc: copy
        # +unk, -in desc: gen (an unk token)
        # -unk, +in desc: copy, gen
        # -unk, -in desc: gen
        # gen_logodds shape: batch (=1)
        desc_loc = desc_enc.find_word_occurrences(token)
        if desc_loc is not None:
            # copy: if the token appears in the description at least once
            # copy_loc_logits shape: batch (=1) x desc length
            copy_loc_logits = self.copy_pointer(output, desc_enc.question_memory)
            copy_logprob = (
                # log p(copy | output)
                # shape: batch (=1)
                torch.nn.functional.logsigmoid(-gen_logodds)
                -
                # xent_loss: -log p(start_location | output) - log p(end_location | output)
                # shape: batch (=1)
                self.xent_loss(copy_loc_logits, self._tensor([desc_loc]))
            )
        else:
            copy_logprob = None

        # gen: ~(unk & in desc), equivalent to  ~unk | ~in desc
        if token in self.terminal_vocab or copy_logprob is None:
            # if copy_logprob is None:
            token_logits = self.terminal_logits(output)
            # shape:
            gen_logprob = (
                # log p(gen | output)
                # shape: batch (=1)
                torch.nn.functional.logsigmoid(gen_logodds)
                -
                # xent_loss: -log p(token | output)
                # shape: batch (=1)
                self.xent_loss(token_logits, token_idx)
            )

        else:
            gen_logprob = None

        # choose either copying or generating
        # if gen_logprob is None:
        #     loss_piece = -copy_logprob
        # else:
        #     loss_piece = -gen_logprob

        # loss should be -log p(...), so negate
        loss_piece = -torch.logsumexp(
            maybe_stack([copy_logprob, gen_logprob], dim=1), dim=1
        )
        return loss_piece

    def token_infer(self, output, gen_logodds, desc_enc):
        # Copy tokens
        # log p(copy | output)
        # shape: batch (=1)
        copy_logprob = torch.nn.functional.logsigmoid(-gen_logodds)
        copy_loc_logits = self.copy_pointer(output, desc_enc.question_memory)
        # log p(loc_i | copy, output)
        # shape: batch (=1) x seq length
        copy_loc_logprobs = torch.nn.functional.log_softmax(copy_loc_logits, dim=-1)
        # log p(loc_i, copy | output)
        copy_loc_logprobs += copy_logprob

        log_prob_by_word = {}
        # accumulate_logprobs is needed because the same word may appear
        # multiple times in desc_enc.words.
        accumulate_logprobs(
            log_prob_by_word,
            zip(desc_enc.words_for_copying, copy_loc_logprobs.squeeze(0)),
        )

        # Generate tokens
        # log p(~copy | output)
        # shape: batch (=1)
        gen_logprob = torch.nn.functional.logsigmoid(gen_logodds)
        token_logits = self.terminal_logits(output)
        # log p(v | ~copy, output)
        # shape: batch (=1) x vocab size
        _token_logprobs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        # log p(v, ~copy| output)
        # shape: batch (=1) x vocab size
        token_logprobs = _token_logprobs + gen_logprob

        accumulate_logprobs(
            log_prob_by_word,
            (
                (self.terminal_vocab[idx], token_logprobs[0, idx])
                for idx in range(token_logprobs.shape[1])
            ),
        )

        return list(log_prob_by_word.items())

    def compute_pointer(
        self,
        node_type,
        prev_state,
        prev_action_emb,
        parent_h,
        parent_action_emb,
        desc_enc,
    ):
        new_state, attention_logits = self._update_state(
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc,
        )
        # output shape: batch (=1) x emb_size
        output = new_state[0]
        # pointer_logits shape: batch (=1) x num choices
        pointer_logits = self.pointers[node_type](
            output, desc_enc.pointer_memories[node_type]
        )

        return output, new_state, pointer_logits, attention_logits

    def compute_pointer_with_align(
        self,
        node_type,
        prev_state,
        prev_action_emb,
        parent_h,
        parent_action_emb,
        desc_enc,
    ):
        new_state, attention_weights = self._update_state(
            node_type,
            prev_state,
            prev_action_emb,
            parent_h,
            parent_action_emb,
            desc_enc,
        )
        # output shape: batch (=1) x emb_size
        output = new_state[0]
        memory_pointer_logits = self.pointers[node_type](output, desc_enc.memory)
        memory_pointer_probs = torch.nn.functional.softmax(memory_pointer_logits, dim=1)
        # pointer_logits shape: batch (=1) x num choices
        if node_type == "column":
            pointer_probs = torch.mm(memory_pointer_probs, desc_enc.m2c_align_mat)
        else:
            assert node_type == "table"
            pointer_probs = torch.mm(memory_pointer_probs, desc_enc.m2t_align_mat)
        pointer_probs = pointer_probs.clamp(min=1e-9)
        pointer_logits = torch.log(pointer_probs)
        return output, new_state, pointer_logits, attention_weights

    def pointer_infer(self, node_type, logits):
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        return list(
            zip(
                # TODO batching
                range(logits.shape[1]),
                logprobs[0],
            )
        )

    def compute_align_loss_from_alignment_mat(self, desc_enc, example):
        """model: a nl2code decoder"""
        # find relevant columns
        root_node = example.tree
        rel_cols = list(
            reversed(
                [
                    val
                    for val in self.ast_wrapper.find_all_descendants_of_type(
                        root_node, "column"
                    )
                ]
            )
        )
        rel_tabs = list(
            reversed(
                [
                    val
                    for val in self.ast_wrapper.find_all_descendants_of_type(
                        root_node, "table"
                    )
                ]
            )
        )

        rel_cols_t = torch.LongTensor(sorted(list(set(rel_cols)))).to(self._device)
        rel_tabs_t = torch.LongTensor(sorted(list(set(rel_tabs)))).to(self._device)

        mc_att_on_rel_col = desc_enc.m2c_align_mat.index_select(1, rel_cols_t)
        mc_max_rel_att, _ = mc_att_on_rel_col.max(dim=0)
        mc_max_rel_att.clamp_(min=1e-9)

        mt_att_on_rel_tab = desc_enc.m2t_align_mat.index_select(1, rel_tabs_t)
        mt_max_rel_att, _ = mt_att_on_rel_tab.max(dim=0)
        mt_max_rel_att.clamp_(min=1e-9)

        c_num = desc_enc.m2c_align_mat.size()[1]
        ir_rel_cols_t = torch.LongTensor(
            sorted(list(set(range(c_num)) - set(rel_cols)))
        ).to(self._device)
        mc_att_on_unrel_col = desc_enc.m2c_align_mat.index_select(1, ir_rel_cols_t)
        mc_max_ir_rel_att, _ = mc_att_on_unrel_col.max(dim=0)
        mc_max_ir_rel_att.clamp_(min=1e-9)
        mc_margin = (
            torch.log(mc_max_ir_rel_att).mean() - torch.log(mc_max_rel_att).mean()
        )

        t_num = desc_enc.m2t_align_mat.size()[1]
        if t_num > len(set(rel_tabs)):
            ir_rel_tabs_t = torch.LongTensor(
                sorted(list(set(range(t_num)) - set(rel_tabs)))
            ).to(self._device)
            mt_att_on_unrel_tab = desc_enc.m2t_align_mat.index_select(1, ir_rel_tabs_t)
            mt_max_ir_rel_att, _ = mt_att_on_unrel_tab.max(dim=0)
            mt_max_ir_rel_att.clamp_(min=1e-9)
            mt_margin = (
                torch.log(mt_max_ir_rel_att).mean() - torch.log(mt_max_rel_att).mean()
            )
        else:
            mt_margin = torch.tensor(0.0).to(self._device)

        gamma = 1
        # align_loss = - torch.log(mc_max_rel_att).mean()
        align_loss = (
            -torch.log(mc_max_rel_att).mean() - torch.log(mt_max_rel_att).mean()
        )
        # align_loss = mc_margin.clamp(min=-gamma)
        # align_loss = mc_margin.clamp(min=-gamma) + mt_margin.clamp(min=-gamma)
        return align_loss

    def compute_align_loss_from_latent_relation(self, desc_enc, example):
        # find relevant columns
        root_node = example.tree
        rel_cols = list(
            reversed(
                [
                    val
                    for val in self.ast_wrapper.find_all_descendants_of_type(
                        root_node, "column"
                    )
                ]
            )
        )
        rel_tabs = list(
            reversed(
                [
                    val
                    for val in self.ast_wrapper.find_all_descendants_of_type(
                        root_node, "table"
                    )
                ]
            )
        )

        cq_relation = desc_enc.relation.cq_relation
        tq_relation = desc_enc.relation.tq_relation
        c_num = cq_relation.size()[0]
        t_num = tq_relation.size()[0]

        rel_cols_t = torch.LongTensor(sorted(list(set(rel_cols)))).to(self._device)
        rel_tabs_t = torch.LongTensor(sorted(list(set(rel_tabs)))).to(self._device)
        ir_rel_cols_t = torch.LongTensor(
            sorted(list(set(range(c_num)) - set(rel_cols)))
        ).to(self._device)

        # could be greater than 1 bc of merging and inaccurate sinkhorn
        mean_rel_col_prob = cq_relation[rel_cols_t].max(dim=1)[0].mean()
        mean_irrel_col_prob = (1 - cq_relation[ir_rel_cols_t].max(dim=1)[0]).mean()
        align_col_loss = -torch.log(mean_irrel_col_prob) - torch.log(mean_rel_col_prob)

        mean_rel_tab_prob = tq_relation[rel_tabs_t].max(dim=1)[0].mean()
        if len(rel_tabs) < t_num:
            ir_rel_tabs_t = torch.LongTensor(
                sorted(list(set(range(t_num)) - set(rel_tabs)))
            ).to(self._device)
            mean_irrel_tab_prob = (1 - tq_relation[ir_rel_tabs_t].max(dim=1)[0]).mean()
            align_tab_loss = -torch.log(mean_irrel_tab_prob) - torch.log(
                mean_rel_tab_prob
            )
        else:
            align_tab_loss = -torch.log(mean_rel_tab_prob)

        align_loss = align_col_loss + align_tab_loss
        if align_tab_loss.item() < 0 or align_col_loss.item() < 0:
            logger.warn("Negative align loss found!")
        logger.info(
            f"align column loss: {align_col_loss.item()}, align tab loss: {align_tab_loss.item()}"
        )
        return align_loss
