import collections
import itertools
import json
import os

import attr
import torch
import numpy as np

from tensor2struct.models import abstract_preproc
from tensor2struct.utils import serialization, vocab, registry
from tensor2struct.modules import rat, lstm, embedders

import logging

logger = logging.getLogger("tensor2struct")


@attr.s
class SpiderEncoderState:
    state = attr.ib()
    memory = attr.ib()
    question_memory = attr.ib()
    schema_memory = attr.ib()
    words_for_copying = attr.ib()

    pointer_memories = attr.ib()
    pointer_maps = attr.ib()

    relation = attr.ib(default=None)
    m2c_align_mat = attr.ib(default=None)
    m2t_align_mat = attr.ib(default=None)

    def find_word_occurrences(self, token):
        occurrences = [i for i, w in enumerate(self.words_for_copying) if w == token]
        if len(occurrences) > 0:
            return occurrences[0]
        else:
            return None


class SpiderEncoderV3Preproc(abstract_preproc.AbstractPreproc):
    def __init__(
        self,
        save_path,
        context,
        min_freq=3,
        max_count=5000,
        include_table_name_in_column=True,
        word_emb=None,
        count_tokens_in_word_emb_for_vocab=False,
        compute_sc_link=False,
        compute_cv_link=False,
        use_ch_vocab=False,
        ch_word_emb=None,
    ):
        if word_emb is None:
            self.word_emb = None
        else:
            self.word_emb = registry.construct("word_emb", word_emb)

        self.data_dir = os.path.join(save_path, "enc")
        self.include_table_name_in_column = include_table_name_in_column
        self.count_tokens_in_word_emb_for_vocab = count_tokens_in_word_emb_for_vocab
        self.compute_sc_link = compute_sc_link
        self.compute_cv_link = compute_cv_link
        self.context_config = context

        self.texts = collections.defaultdict(list)
        self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        self.vocab_path = os.path.join(save_path, "enc_vocab.json")
        self.vocab_word_freq_path = os.path.join(save_path, "enc_word_freq.json")
        self.vocab = None
        self.use_ch_vocab = use_ch_vocab
        if use_ch_vocab:
            assert ch_word_emb is not None
            self.ch_word_emb = registry.construct("word_emb", ch_word_emb)
            self.ch_vocab_builder = vocab.VocabBuilder(min_freq, max_count)
            self.ch_vocab_path = os.path.join(save_path, "ch_enc_vocab.json")
            self.ch_vocab_word_freq_path = os.path.join(
                save_path, "ch_enc_word_freq.json"
            )
            self.ch_vocab = None
        self.counted_db_ids = set()
        self.relations = set()

        self.context_cache = {}

    def validate_item(self, item, section):
        return True, None

    def add_item(self, item, section, validation_info):
        preprocessed = self.preprocess_item(item, validation_info)
        self.texts[section].append(preprocessed)

        if section == "train":
            for relation_name in itertools.chain(
                preprocessed["schema_relations"].keys(),
                preprocessed["sc_relations"].keys(),
                preprocessed["cv_relations"].keys(),
            ):
                self.relations.add(relation_name)

            q_to_count = preprocessed["question"]
            if item.schema.db_id not in self.counted_db_ids:
                self.counted_db_ids.add(item.schema.db_id)
                to_count = itertools.chain(
                    *preprocessed["columns"], *preprocessed["tables"]
                )
            else:
                to_count = []

            # only question is possibly chinese
            if self.use_ch_vocab:
                for token in q_to_count:
                    count_token = (
                        self.ch_word_emb is None
                        or self.count_tokens_in_word_emb_for_vocab
                        or self.ch_word_emb.lookup(token) is None
                    )
                    if count_token:
                        self.ch_vocab_builder.add_word(token)
            else:
                to_count = itertools.chain(to_count, q_to_count)

            for token in to_count:
                count_token = (
                    self.word_emb is None
                    or self.count_tokens_in_word_emb_for_vocab
                    or self.word_emb.lookup(token) is None
                )
                if count_token:
                    self.vocab_builder.add_word(token)

    def clear_items(self):
        self.texts = collections.defaultdict(list)

    def preprocess_item(self, item, validation_info):
        if self.use_ch_vocab:
            question, question_for_copying = self._ch_tokenize_for_copying(
                item.text, item.orig["question"]
            )
        else:
            question, question_for_copying = self._tokenize_for_copying(
                item.text, item.orig["question"]
            )

        if item.schema.db_id in self.context_cache:
            context = self.context_cache[item.schema.db_id]
        else:
            context = registry.construct(
                "context",
                self.context_config,
                schema=item.schema,
                word_emb=self.word_emb,
            )
            self.context_cache[item.schema.db_id] = context

        preproc_schema = context.preproc_schema
        schema_relations = context.compute_schema_relations()
        sc_relations = (
            context.compute_schema_linking(question) if self.compute_sc_link else {}
        )
        cv_relations = (
            context.compute_schema_linking(question) if self.compute_cv_link else {}
        )

        return {
            "raw_question": item.orig["question"],
            "question": question,
            "question_for_copying": question_for_copying,
            "db_id": item.schema.db_id,
            "schema_relations": schema_relations,
            "sc_relations": sc_relations,
            "cv_relations": cv_relations,
            "columns": preproc_schema.column_names,
            "tables": preproc_schema.table_names,
            "table_bounds": preproc_schema.table_bounds,
            "column_to_table": preproc_schema.column_to_table,
            "table_to_columns": preproc_schema.table_to_columns,
            "foreign_keys": preproc_schema.foreign_keys,
            "foreign_keys_tables": preproc_schema.foreign_keys_tables,
            "primary_keys": preproc_schema.primary_keys,
        }

    def _tokenize(self, presplit, unsplit):
        if self.word_emb:
            return self.word_emb.tokenize(unsplit)
        return presplit

    def _ch_tokenize(self, presplit, unsplit):
        if self.ch_word_emb:
            return self.ch_word_emb.tokenize(unsplit)
        return presplit

    def _tokenize_for_copying(self, presplit, unsplit):
        if self.word_emb:
            return self.word_emb.tokenize_for_copying(unsplit)
        return presplit, presplit

    def _ch_tokenize_for_copying(self, presplit, unsplit):
        if self.ch_word_emb:
            return self.ch_word_emb.tokenize_for_copying(unsplit)
        return presplit, presplit

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.vocab = self.vocab_builder.finish()
        print(f"{len(self.vocab)} words in vocab")
        self.vocab.save(self.vocab_path)
        self.vocab_builder.save(self.vocab_word_freq_path)
        if self.use_ch_vocab:
            self.ch_vocab = self.ch_vocab_builder.finish()
            print(f"{len(self.ch_vocab)} chinese words in vocab")
            self.ch_vocab.save(self.ch_vocab_path)
            self.ch_vocab_builder.save(self.ch_vocab_word_freq_path)

        default_relations = registry.lookup(
            "context", self.context_config["name"]
        ).get_default_relations()
        self.relations = sorted(self.relations.union(default_relations))
        print(f"{len(self.relations)} relations extracted")
        with open(os.path.join(self.data_dir, "relations.json"), "w") as f:
            json.dump(self.relations, f)

        for section, texts in self.texts.items():
            with open(os.path.join(self.data_dir, section + ".jsonl"), "w") as f:
                for text in texts:
                    f.write(json.dumps(text) + "\n")

    def deprecated_load(self):
        self.vocab = vocab.Vocab.load(self.vocab_path)
        self.vocab_builder.load(self.vocab_word_freq_path)
        if self.use_ch_vocab:
            self.ch_vocab = vocab.Vocab.load(self.ch_vocab_path)
            self.ch_vocab_builder.load(self.ch_vocab_word_freq_path)
        with open(os.path.join(self.data_dir, "relation2id.json"), "r") as f:
            self.relations2id = json.load(f)

    def load(self):
        self.vocab = vocab.Vocab.load(self.vocab_path)
        self.vocab_builder.load(self.vocab_word_freq_path)
        if self.use_ch_vocab:
            self.ch_vocab = vocab.Vocab.load(self.ch_vocab_path)
            self.ch_vocab_builder.load(self.ch_vocab_word_freq_path)
        with open(os.path.join(self.data_dir, "relations.json"), "r") as f:
            relations = json.load(f)
            self.relations = sorted(relations)
        self.relations2id = {r: ind for ind, r in enumerate(self.relations)}

    def dataset(self, section):
        # for codalab eval
        if len(self.texts[section]) > 0:
            return self.texts[section]
        else:
            return [
                json.loads(line)
                for line in open(os.path.join(self.data_dir, section + ".jsonl"))
            ]


@registry.register("encoder", "spiderv3")
class SpiderEncoderV3(torch.nn.Module):

    batched = True
    Preproc = SpiderEncoderV3Preproc

    def __init__(
        self,
        device,
        preproc,
        word_emb_size=128,
        recurrent_size=256,
        dropout=0.0,
        question_encoder=("emb", "bilstm"),
        column_encoder=("emb", "bilstm"),
        table_encoder=("emb", "bilstm"),
        linking_config={},
        rat_config={},
        top_k_learnable=0,
        include_in_memory=("question", "column", "table"),
    ):
        super().__init__()
        self._device = device
        self.preproc = preproc

        self.vocab = preproc.vocab
        self.word_emb_size = word_emb_size
        self.recurrent_size = recurrent_size
        assert self.recurrent_size % 2 == 0
        word_freq = self.preproc.vocab_builder.word_freq
        top_k_words = set([_a[0] for _a in word_freq.most_common(top_k_learnable)])
        self.learnable_words = top_k_words
        self.include_in_memory = set(include_in_memory)
        self.dropout = dropout

        shared_modules = {
            "shared-en-emb": embedders.LookupEmbeddings(
                self._device,
                self.vocab,
                self.preproc.word_emb,
                self.word_emb_size,
                self.learnable_words,
            ),
            "shared-bilstm": lstm.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=False,
            ),
        }

        # chinese vocab and module
        if self.preproc.use_ch_vocab:
            self.ch_vocab = preproc.ch_vocab
            ch_word_freq = self.preproc.ch_vocab_builder.word_freq
            ch_top_k_words = set(
                [_a[0] for _a in ch_word_freq.most_common(top_k_learnable)]
            )
            self.ch_learnable_words = ch_top_k_words
            shared_modules["shared-ch-emb"] = embedders.LookupEmbeddings(
                self._device,
                self.ch_vocab,
                self.preproc.ch_word_emb,
                self.preproc.ch_word_emb.dim,
                self.ch_learnable_words,
            )
            shared_modules["ch-bilstm"] = lstm.BiLSTM(
                input_size=self.preproc.ch_word_emb.dim,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                use_native=False,
                summarize=False,
            )
            shared_modules["ch-bilstm-native"] = lstm.BiLSTM(
                input_size=self.preproc.ch_word_emb.dim,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                use_native=True,
                summarize=False,
            )

        self.question_encoder = self._build_modules(
            question_encoder, shared_modules=shared_modules
        )
        self.column_encoder = self._build_modules(
            column_encoder, shared_modules=shared_modules
        )
        self.table_encoder = self._build_modules(
            table_encoder, shared_modules=shared_modules
        )

        # matching
        self.schema_linking = registry.construct(
            "schema_linking",
            linking_config,
            device=device,
            word_emb_size=word_emb_size,
            preproc=preproc,
        )

        # rat
        rat_modules = {"rat": rat.RAT, "none": rat.NoOpUpdate}
        self.rat_update = registry.instantiate(
            rat_modules[rat_config["name"]],
            rat_config,
            unused_keys={"name"},
            device=self._device,
            relations2id=preproc.relations2id,
            hidden_size=recurrent_size,
        )

        # aligner
        self.aligner = rat.AlignmentWithRAT(
            device=device,
            hidden_size=recurrent_size,
            relations2id=preproc.relations2id,
            enable_latent_relations=rat_config["enable_latent_relations"],
            num_latent_relations=rat_config.get("num_latent_relations", None),
            combine_latent_relations=rat_config["combine_latent_relations"],
        )

    def _build_modules(self, module_types, shared_modules=None):
        module_builder = {
            "emb": lambda: embedders.LookupEmbeddings(
                self._device,
                self.vocab,
                self.preproc.word_emb,
                self.word_emb_size,
                self.learnable_words,
            ),
            "bilstm": lambda: lstm.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=False,
            ),
            "bilstm-native": lambda: lstm.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=False,
                use_native=True,
            ),
            "bilstm-summarize": lambda: lstm.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=True,
            ),
            "bilstm-native-summarize": lambda: lstm.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=True,
                use_native=True,
            ),
        }

        modules = []
        for module_type in module_types:
            if module_type in shared_modules:
                modules.append(shared_modules[module_type])
            else:
                modules.append(module_builder[module_type]())
        return torch.nn.Sequential(*modules)

    def forward(self, descs):
        qs = [[desc["question"]] for desc in descs]
        q_enc, _ = self.question_encoder(qs)

        c_enc, c_boundaries = self.column_encoder([desc["columns"] for desc in descs])
        column_pointer_maps = [
            {
                i: list(range(left, right))
                for i, (left, right) in enumerate(
                    zip(c_boundaries_for_item, c_boundaries_for_item[1:])
                )
            }
            for batch_idx, c_boundaries_for_item in enumerate(c_boundaries)
        ]

        t_enc, t_boundaries = self.table_encoder([desc["tables"] for desc in descs])
        table_pointer_maps = [
            {
                i: list(range(left, right))
                for i, (left, right) in enumerate(
                    zip(t_boundaries_for_item, t_boundaries_for_item[1:])
                )
            }
            for batch_idx, (desc, t_boundaries_for_item) in enumerate(
                zip(descs, t_boundaries)
            )
        ]

        result = []
        # TODO: support batching
        for batch_idx, desc in enumerate(descs):
            relation = self.schema_linking(descs[batch_idx])
            q_enc_new_item, c_enc_new_item, t_enc_new_item = self.rat_update(
                desc,
                q_enc.select(batch_idx).unsqueeze(1),
                c_enc.select(batch_idx).unsqueeze(1),
                t_enc.select(batch_idx).unsqueeze(1),
                relation,
            )

            align_mat_item = self.aligner(
                desc, q_enc_new_item, c_enc_new_item, t_enc_new_item, relation
            )

            memory = []
            if "question" in self.include_in_memory:
                memory.append(q_enc_new_item)
            if "column" in self.include_in_memory:
                memory.append(c_enc_new_item)
            if "table" in self.include_in_memory:
                memory.append(t_enc_new_item)
            memory = torch.cat(memory, dim=1)

            result.append(
                SpiderEncoderState(
                    state=None,
                    words_for_copying=desc["question"],
                    memory=memory,
                    question_memory=q_enc_new_item,
                    schema_memory=torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                    pointer_memories={
                        "column": c_enc_new_item,
                        # "table": torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                        "table": t_enc_new_item,
                    },
                    pointer_maps={
                        "column": column_pointer_maps[batch_idx],
                        "table": table_pointer_maps[batch_idx],
                    },
                    relation=relation,
                    m2c_align_mat=align_mat_item[0],
                    m2t_align_mat=align_mat_item[1],
                )
            )
        return result
