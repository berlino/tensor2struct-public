import collections
import itertools
import json
import os

import attr
import nltk.corpus
import torch
import torchtext
import numpy as np

from tensor2struct.models import abstract_preproc
from tensor2struct.utils import serialization, vocab, registry
from tensor2struct.modules import rat, lstm, embedders, bert_tokenizer

from transformers import BertModel, ElectraModel, AutoModel

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

    m2c_align_mat = attr.ib()
    m2t_align_mat = attr.ib()

    # for copying
    tokenizer = attr.ib()

    def find_word_occurrences(self, token):
        occurrences = [i for i, w in enumerate(self.words_for_copying) if w == token]
        if len(occurrences) > 0:
            return occurrences[0]
        else:
            return None

class SpiderEncoderBertPreproc(abstract_preproc.AbstractPreproc):
    def __init__(
        self,
        save_path,
        context,
        bert_version="bert-base-uncased",
        compute_sc_link=True,
        compute_cv_link=True,
    ):

        self.data_dir = os.path.join(save_path, "enc")
        self.texts = collections.defaultdict(list)
        self.compute_sc_link = compute_sc_link
        self.compute_cv_link = compute_cv_link
        self.context_config = context

        self.relations = set()

        # TODO: should get types from the data
        # column_types = ["text", "number", "time", "boolean", "others"]
        # self.tokenizer.add_tokens([f"<type: {t}>" for t in column_types])
        self.tokenizer_config = bert_version  # lazy init

        self.context_cache = {}

    @property
    def tokenizer(self):
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = bert_tokenizer.BERTokenizer(self.tokenizer_config)
        return self._tokenizer

    def validate_item(self, item, section):
        num_words = (
            len(item.text)
            + sum(len(c.name) for c in item.schema.columns)
            + sum(len(t.name) for t in item.schema.tables)
        )
        if "phobert" in self.tokenizer_config and num_words > 256:
            logger.info(f"Found long seq in {item.schema.db_id}")
            return False, None
        if num_words > 512:
            logger.info(f"Found long seq in {item.schema.db_id}")
            return False, None
        else:
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

    def clear_items(self):
        self.texts = collections.defaultdict(list)

    def preprocess_item(self, item, validation_info):
        q_text = " ".join(item.text)

        # use the original words for copying, while they are not necessarily used for encoding
        # question_for_copying = self.tokenizer.tokenize_and_lemmatize(q_text)
        question_for_copying = self.tokenizer.tokenize_with_orig(q_text)

        if item.schema.db_id in self.context_cache:
            context = self.context_cache[item.schema.db_id]
        else:
            context = registry.construct(
                "context",
                self.context_config,
                schema=item.schema,
                tokenizer=self.tokenizer,
            )
            self.context_cache[item.schema.db_id] = context

        preproc_schema = context.preproc_schema
        schema_relations = context.compute_schema_relations()
        sc_relations = (
            context.compute_schema_linking(q_text) if self.compute_sc_link else {}
        )
        cv_relations = (
            context.compute_cell_value_linking(q_text) if self.compute_cv_link else {}
        )

        return {
            "question_text": q_text,
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

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        # self.tokenizer.save_pretrained(self.data_dir)

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

    def load(self):
        # self.tokenizer = BertTokenizer.from_pretrained(self.data_dir)
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


@registry.register("encoder", "spider-bert")
class SpiderEncoderBert(torch.nn.Module):

    Preproc = SpiderEncoderBertPreproc
    batched = True

    def __init__(
        self,
        device,
        preproc,
        bert_token_type=False,
        bert_version="bert-base-uncased",
        summarize_header="avg",
        include_in_memory=("question", "column", "table"),
        rat_config={},
        linking_config={},
    ):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.bert_version = bert_version
        self.bert_token_type = bert_token_type
        self.base_enc_hidden_size = (
            1024 if "large" in bert_version else 768
        )
        self.include_in_memory = include_in_memory

        # ways to summarize header
        assert summarize_header in ["first", "avg"]
        self.summarize_header = summarize_header
        self.enc_hidden_size = self.base_enc_hidden_size

        # matching
        self.schema_linking = registry.construct(
            "schema_linking", linking_config, preproc=preproc, device=device,
        )

        # rat
        rat_modules = {"rat": rat.RAT, "none": rat.NoOpUpdate}
        self.rat_update = registry.instantiate(
            rat_modules[rat_config["name"]],
            rat_config,
            unused_keys={"name"},
            device=self._device,
            relations2id=preproc.relations2id,
            hidden_size=self.enc_hidden_size,
        )

        # aligner
        self.aligner = rat.AlignmentWithRAT(
            device=device,
            hidden_size=self.enc_hidden_size,
            relations2id=preproc.relations2id,
            enable_latent_relations=False,
        )

        if "electra" in bert_version:
            modelclass = ElectraModel
        elif "phobert" in bert_version:
            modelclass = AutoModel
        elif "bert" in bert_version:
            modelclass = BertModel
        else:
            raise NotImplementedError
        self.bert_model = modelclass.from_pretrained(bert_version)
        self.tokenizer = self.preproc.tokenizer
        # self.bert_model.resize_token_embeddings(
        #    len(self.tokenizer)
        # )  # several tokens added

    def forward(self, descs):
        # TODO: abstract the operations of batching for bert
        batch_token_lists = []
        batch_id_to_retrieve_question = []
        batch_id_to_retrieve_column = []
        batch_id_to_retrieve_table = []
        if self.summarize_header == "avg":
            batch_id_to_retrieve_column_2 = []
            batch_id_to_retrieve_table_2 = []
        long_seq_set = set()
        batch_id_map = {}  # some long examples are not included

        # 1) retrieve bert pre-trained embeddings
        for batch_idx, desc in enumerate(descs):
            qs = self.tokenizer.text_to_ids(desc["question_text"], cls=True)
            cols = [self.tokenizer.text_to_ids(c, cls=False) for c in desc["columns"]]
            tabs = [self.tokenizer.text_to_ids(t, cls=False) for t in desc["tables"]]

            token_list = (
                qs + [c for col in cols for c in col] + [t for tab in tabs for t in tab]
            )
            assert self.tokenizer.check_bert_input_seq(token_list)
            if "phobert" in self.bert_version and len(token_list) > 256:
                long_seq_set.add(batch_idx)
                continue

            elif len(token_list) > 512:
                long_seq_set.add(batch_idx)
                continue

            q_b = len(qs)
            col_b = q_b + sum(len(c) for c in cols)
            # leave out [CLS] and [SEP]
            question_indexes = list(range(q_b))[1:-1]
            # use the first/avg representation for column/table
            column_indexes = np.cumsum(
                [q_b] + [len(token_list) for token_list in cols[:-1]]
            ).tolist()
            table_indexes = np.cumsum(
                [col_b] + [len(token_list) for token_list in tabs[:-1]]
            ).tolist()
            if self.summarize_header == "avg":
                column_indexes_2 = np.cumsum(
                    [q_b - 2] + [len(token_list) for token_list in cols]
                ).tolist()[1:]
                table_indexes_2 = np.cumsum(
                    [col_b - 2] + [len(token_list) for token_list in tabs]
                ).tolist()[1:]

            # token_list is already indexed
            indexed_token_list = token_list
            batch_token_lists.append(indexed_token_list)

            # add index for retrieving representations
            question_rep_ids = torch.LongTensor(question_indexes).to(self._device)
            batch_id_to_retrieve_question.append(question_rep_ids)
            column_rep_ids = torch.LongTensor(column_indexes).to(self._device)
            batch_id_to_retrieve_column.append(column_rep_ids)
            table_rep_ids = torch.LongTensor(table_indexes).to(self._device)
            batch_id_to_retrieve_table.append(table_rep_ids)
            if self.summarize_header == "avg":
                assert all(i2 >= i1 for i1, i2 in zip(column_indexes, column_indexes_2))
                column_rep_ids_2 = torch.LongTensor(column_indexes_2).to(self._device)
                batch_id_to_retrieve_column_2.append(column_rep_ids_2)
                assert all(i2 >= i1 for i1, i2 in zip(table_indexes, table_indexes_2))
                table_rep_ids_2 = torch.LongTensor(table_indexes_2).to(self._device)
                batch_id_to_retrieve_table_2.append(table_rep_ids_2)

            batch_id_map[batch_idx] = len(batch_id_map)

        if len(long_seq_set) < len(descs):
            (
                padded_token_lists,
                att_mask_lists,
                tok_type_lists,
            ) = self.tokenizer.pad_sequence_for_bert_batch(batch_token_lists)
            tokens_tensor = torch.LongTensor(padded_token_lists).to(self._device)
            att_masks_tensor = torch.LongTensor(att_mask_lists).to(self._device)

            if self.bert_token_type:
                tok_type_tensor = torch.LongTensor(tok_type_lists).to(self._device)
                bert_output = self.bert_model(
                    tokens_tensor,
                    attention_mask=att_masks_tensor,
                    token_type_ids=tok_type_tensor,
                )[0]
            else:
                bert_output = self.bert_model(
                    tokens_tensor, attention_mask=att_masks_tensor
                )[0]

            enc_output = bert_output

        column_pointer_maps = [
            {i: [i] for i in range(len(desc["columns"]))} for desc in descs
        ]
        table_pointer_maps = [
            {i: [i] for i in range(len(desc["tables"]))} for desc in descs
        ]

        # assert len(long_seq_set) == 0  # remove them for now

        # 2) rat update
        result = []
        for batch_idx, desc in enumerate(descs):
            # retrieve representations
            if batch_idx in long_seq_set:
                q_enc, col_enc, tab_enc = self.encoder_long_seq(desc) 
            else:
                bert_batch_idx = batch_id_map[batch_idx]
                q_enc = enc_output[bert_batch_idx][
                    batch_id_to_retrieve_question[bert_batch_idx]
                ]
                col_enc = enc_output[bert_batch_idx][
                    batch_id_to_retrieve_column[bert_batch_idx]
                ]
                tab_enc = enc_output[bert_batch_idx][
                    batch_id_to_retrieve_table[bert_batch_idx]
                ]

                if self.summarize_header == "avg":
                    col_enc_2 = enc_output[bert_batch_idx][
                        batch_id_to_retrieve_column_2[bert_batch_idx]
                    ]
                    tab_enc_2 = enc_output[bert_batch_idx][
                        batch_id_to_retrieve_table_2[bert_batch_idx]
                    ]

                    col_enc = (col_enc + col_enc_2) / 2.0  # avg of first and last token
                    tab_enc = (tab_enc + tab_enc_2) / 2.0  # avg of first and last token

            words_for_copying = desc["question_for_copying"]
            assert q_enc.size()[0] == len(words_for_copying)
            assert col_enc.size()[0] == len(desc["columns"])
            assert tab_enc.size()[0] == len(desc["tables"])

            # rat update
            # TODO: change this, question is in the protocal of build relations
            desc["question"] = words_for_copying
            relation = self.schema_linking(desc)
            (
                q_enc_new_item,
                c_enc_new_item,
                t_enc_new_item,
            ) = self.rat_update.forward_unbatched(
                desc,
                q_enc.unsqueeze(1),
                col_enc.unsqueeze(1),
                tab_enc.unsqueeze(1),
                relation,
            )

            # attention memory
            memory = []
            if "question" in self.include_in_memory:
                memory.append(q_enc_new_item)
            if "column" in self.include_in_memory:
                memory.append(c_enc_new_item)
            if "table" in self.include_in_memory:
                memory.append(t_enc_new_item)
            memory = torch.cat(memory, dim=1)

            # alignment matrix
            align_mat_item = self.aligner(
                desc, q_enc_new_item, c_enc_new_item, t_enc_new_item, relation
            )

            result.append(
                SpiderEncoderState(
                    state=None,
                    words_for_copying=words_for_copying,
                    tokenizer=self.tokenizer,
                    memory=memory,
                    question_memory=q_enc_new_item,
                    schema_memory=torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                    pointer_memories={
                        "column": c_enc_new_item,
                        "table": t_enc_new_item,
                    },
                    pointer_maps={
                        "column": column_pointer_maps[batch_idx],
                        "table": table_pointer_maps[batch_idx],
                    },
                    m2c_align_mat=align_mat_item[0],
                    m2t_align_mat=align_mat_item[1],
                )
            )
        return result
    
    def _bert_encode(self, ids):
        if not isinstance(ids[0], list):  # encode question words
            tokens_tensor = torch.tensor([ids]).to(self._device)
            outputs = self.bert_model(tokens_tensor)
            return outputs[0][0, 1:-1]  # remove [CLS] and [SEP]
        else:
            max_len = max([len(it) for it in ids])
            tok_ids = []
            for item_ids in ids:
                item_ids = item_ids + [self.tokenizer.pad_token_id] * (max_len - len(item_ids))
                tok_ids.append(item_ids)

            tokens_tensor = torch.tensor(tok_ids).to(self._device)
            outputs = self.bert_model(tokens_tensor)
            return outputs[0][:, 0, :]

    def encoder_long_seq(self, desc):
        """
        Since bert cannot handle sequence longer than 512, each column/table is encoded individually
        The representation of a column/table is the vector of the first token [CLS]
        """
        qs = self.tokenizer.text_to_ids(desc['question_text'], cls=True)
        cols = [self.tokenizer.text_to_ids(c, cls=True) for c in desc['columns']]
        tabs = [self.tokenizer.text_to_ids(t, cls=True) for t in desc['tables']]

        enc_q = self._bert_encode(qs)
        enc_col = self._bert_encode(cols)
        enc_tab = self._bert_encode(tabs)
        return enc_q, enc_col, enc_tab

    