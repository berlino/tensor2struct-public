import os
import attr
import logging
import json
import itertools
import collections

import torch
from torch import nn

from tensor2struct.utils import registry, vocab
from tensor2struct.models import abstract_preproc
from tensor2struct.modules import seq2seq, rat, lstm, embedders


@attr.s
class OvernightEncoderState:
    question_memeory = attr.ib()
    property_memory = attr.ib()
    q2p_align_mat = attr.ib()


class OvernightEncPreproc(abstract_preproc.AbstractPreproc):
    def __init__(
        self,
        save_path,
        grammar,
        context,
        word_emb,
        min_freq=3,
        max_count=5000,
        sc_link=True,
        cv_link=True,
    ):
        self.data_dir = os.path.join(save_path, "enc")
        self.compute_sc_link = sc_link
        self.compute_cv_link = cv_link
        self.grammar_config = grammar
        self.context_config = context
        self.texts = collections.defaultdict(list)
        self.word_emb = registry.construct("word_emb", word_emb)

        # vocab
        self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        self.vocab_path = os.path.join(self.data_dir, "vocab.json")
        self.vocab_word_freq_path = os.path.join(self.data_dir, "word_freq.json")

        self.relations = set()
        self.schema_cache = {}

    def validate_item(self, item, section):
        return True, None

    def add_item(self, item, section, validation_info):
        preprocessed = self.preprocess_item(item, validation_info)
        self.texts[section].append(preprocessed)

        if section in ["train", "_train"]:  # _train to build vocab
            for token in preprocessed["question"]:
                self.vocab_builder.add_word(token)
            for value in preprocessed["columns"]:
                for token in value:
                    self.vocab_builder.add_word(token)
            for p in preprocessed["values"]:
                for token in p:
                    self.vocab_builder.add_word(token)

    def clear_items(self):
        self.texts = collections.defaultdict(list)

    def preprocess_item(self, item, validation_info):
        tokens = self.tokenize(item.question)
        grammar = registry.construct("grammar", self.grammar_config, domain=item.domain)
        raw_properties, ref_properties = grammar.get_properties()
        raw_values, ref_values = grammar.get_values()
        schema_raw_relations = grammar.get_schema_relations()

        if item.domain in self.schema_cache:
            context = self.schema_cache[item.domain]
            processed_properties = context.schema["columns"]
            processed_values = context.schema["values"]
            schema_relations = context.compute_schema_relations()
        else:
            processed_properties = [self.tokenize(p) for p in raw_properties]
            processed_values = [self.tokenize(v) for v in raw_values]

            context = registry.construct(
                "context",
                self.context_config,
                schema={
                    "columns": processed_properties,
                    "values": processed_values,
                    "schema_relations": schema_raw_relations,
                },
            )
            self.schema_cache[item.domain] = context
            schema_relations = context.compute_schema_relations()

        sc_relations = (
            context.compute_schema_linking(tokens) if self.compute_sc_link else {}
        )
        cv_relations = (
            context.compute_cell_value_linking(tokens) if self.compute_cv_link else {}
        )
        for relation_name in itertools.chain(
            schema_relations.keys(), sc_relations.keys(), cv_relations.keys()
        ):
            self.relations.add(relation_name)

        return {
            "db_id": item.domain,  # comply with data_scheduler
            "question": tokens,
            "raw_question": item.question,
            "columns": processed_properties,
            "values": processed_values,
            "ref_columns": ref_properties,
            "ref_values": ref_values,
            "schema_relations": schema_relations,
            "sc_relations": sc_relations,
            "cv_relations": cv_relations,
        }

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.vocab = self.vocab_builder.finish()
        print(f"{len(self.vocab)} words in vocab")
        self.vocab.save(self.vocab_path)
        self.vocab_builder.save(self.vocab_word_freq_path)

        default_relations = registry.lookup(
            "context", self.context_config["name"]
        ).get_default_relations()
        self.relations = sorted(self.relations.union(default_relations))
        print(f"{len(self.relations)} relations extracted")
        with open(os.path.join(self.data_dir, "relation.json"), "w") as f:
            json.dump(self.relations, f)

        for section, texts in self.texts.items():
            with open(os.path.join(self.data_dir, section + ".jsonl"), "w") as f:
                for text in texts:
                    f.write(json.dumps(text) + "\n")

    def load(self):
        self.vocab = vocab.Vocab.load(self.vocab_path)
        self.vocab_builder.load(self.vocab_word_freq_path)
        with open(os.path.join(self.data_dir, "relation.json"), "r") as f:
            relations = json.load(f)
            self.relations = sorted(relations)
        self.relations2id = {r: ind for ind, r in enumerate(list(self.relations))}

    def dataset(self, section):
        return [
            json.loads(line)
            for line in open(os.path.join(self.data_dir, section + ".jsonl"))
        ]

    def tokenize(self, text: str):
        assert self.word_emb is not None
        return self.word_emb.tokenize(text)


@attr.s
class OvernightEncoderState:
    memory = attr.ib()
    pointer_memories = attr.ib()
    pointer_refs = attr.ib()
    pointer_align_mat = attr.ib()
    mentioned = attr.ib()


@registry.register("encoder", "overnight")
class OvernightEnc(nn.Module):
    Preproc = OvernightEncPreproc
    batched = True

    def __init__(
        self,
        device,
        preproc,
        word_emb_size=128,
        recurrent_size=256,
        dropout=0.0,
        question_encoder=("emb", "bilstm"),
        column_encoder=("emb", "bilstm"),
        value_encoder=("emb", "bilstm"),
        linking_config={},
        rat_config={},
        top_k_learnable=0,
        include_in_memory=("question",),
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
            )
        }

        self.question_encoder = self._build_modules(
            question_encoder, "question", shared_modules=shared_modules
        )
        self.column_encoder = self._build_modules(
            column_encoder, "column", shared_modules=shared_modules
        )
        self.value_encoder = self._build_modules(
            value_encoder, "value", shared_modules=shared_modules
        )

        update_modules = {"rat": rat.RAT, "none": rat.NoOpUpdate}

        self.schema_linking = registry.construct(
            "schema_linking",
            linking_config,
            device=device,
            word_emb_size=word_emb_size,
            preproc=preproc,
        )

        self.rat_update = registry.instantiate(
            update_modules[rat_config["name"]],
            rat_config,
            unused_keys={"name"},
            device=self._device,
            relations2id=self.preproc.relations2id,
            hidden_size=recurrent_size,
        )

    def _build_modules(self, module_types, prefix, shared_modules=None):
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

    @staticmethod
    def filter_values_with_matching(desc, kind):
        orig_values = desc["values"]
        ref_values = desc["ref_values"]

        filterd_ids = set()
        for r_t in desc["cv_relations"]:
            for r in desc["cv_relations"][r_t]:
                _kind = r_t.split(":")[1]
                if _kind != kind:
                    continue
                if r_t.startswith("q"):
                    q_id, val_id = r
                else:
                    val_id, q_id = r
                filterd_ids.add(val_id)

        filtered_ids = sorted(filterd_ids)

        filtered_values = []
        filtered_ref_values = []
        for ind in filtered_ids:
            filtered_values.append(orig_values[ind])
            filtered_ref_values.append(ref_values[ind])
        return filtered_values, filtered_ref_values

    @staticmethod
    def filter_properties_with_matching(desc, kind):
        orig_properties = desc["columns"]
        ref_properties = desc["ref_columns"]
        filtered_ids = set()
        for r_t in desc["sc_relations"]:
            for r in desc["sc_relations"][r_t]:
                _kind = r_t.split(":")[1]
                if _kind != kind:
                    continue
                if r_t.startswith("q"):
                    q_id, col_id = r
                else:
                    col_id, q_id = r
                filtered_ids.add(col_id)

        filtered_ids = sorted(filtered_ids)

        filtered_properties = []
        filtered_ref_properties = []
        for ind in filtered_ids:
            filtered_properties.append(orig_properties[ind])
            filtered_ref_properties.append(ref_properties[ind])

        return filtered_properties, filtered_ref_properties

    def forward(self, descs):
        qs = [[desc["question"]] for desc in descs]
        q_enc, _ = self.question_encoder(qs)
        col_enc, _ = self.column_encoder([desc["columns"] for desc in descs])
        val_enc, _ = self.value_encoder([desc["values"] for desc in descs])

        result = []
        for batch_idx, desc in enumerate(descs):
            relation = self.schema_linking(descs[batch_idx])
            q_enc_new_item, col_enc_new_item, val_enc_new_item = self.rat_update(
                desc,
                q_enc.select(batch_idx).unsqueeze(1),
                col_enc.select(batch_idx).unsqueeze(1),
                val_enc.select(batch_idx).unsqueeze(1),
                relation,
            )

            memory = []
            if "question" in self.include_in_memory:
                memory.append(q_enc_new_item)
            if "column" in self.include_in_memory:
                memory.append(col_enc_new_item)
            if "value" in self.include_in_memory:
                memory.append(val_enc_new_item)

            if len(memory) > 1:
                memory = torch.cat(memory, dim=1)
            else:
                memory = memory[0]

            _, ex_mentioned_values = self.filter_values_with_matching(desc, "EM")
            _, pa_mentioned_values = self.filter_values_with_matching(desc, "PM")
            _, ex_mentioned_properties = self.filter_properties_with_matching(
                desc, "EM"
            )
            _, pa_mentioned_properties = self.filter_properties_with_matching(
                desc, "PM"
            )

            result.append(
                OvernightEncoderState(
                    memory=memory,
                    pointer_memories={
                        "property": col_enc_new_item,
                        "value": val_enc_new_item,
                    },
                    pointer_refs={
                        "property": desc["ref_columns"],
                        "value": desc["ref_values"],
                    },
                    pointer_align_mat={"property": None, "value": None},
                    mentioned={
                        "exact": {
                            "property": ex_mentioned_properties,
                            "value": ex_mentioned_values,
                        },
                        "partial": {
                            "property": pa_mentioned_properties,
                            "value": pa_mentioned_values,
                        },
                    },
                )
            )
        return result
