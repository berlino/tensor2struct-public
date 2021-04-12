import os
import attr
import collections
import itertools
import timeout_decorator


from tensor2struct.contexts import abstract_context
from tensor2struct.utils import registry, serialization
from tensor2struct.contexts.spider_context import SpiderContext


@attr.s
class PreprocessedSchema:
    column_names = attr.ib(factory=list)
    table_names = attr.ib(factory=list)
    table_bounds = attr.ib(factory=list)
    column_to_table = attr.ib(factory=dict)
    table_to_columns = attr.ib(factory=dict)
    foreign_keys = attr.ib(factory=dict)
    foreign_keys_tables = attr.ib(factory=lambda: collections.defaultdict(set))
    primary_keys = attr.ib(factory=list)

    normalized_column_names = attr.ib(factory=list)
    normalized_table_names = attr.ib(factory=list)


@registry.register("context", "spider-bert")
class SpiderBertContext(SpiderContext):
    def __init__(self, schema, tokenizer, db_paths) -> None:
        self.tokenizer = tokenizer

        self.schema = schema
        preproc_schema = self.preprocess_schema(self.schema)
        self.preproc_schema = preproc_schema

        self.columns = preproc_schema.normalized_column_names
        self.tables = preproc_schema.normalized_table_names

        self.db_dirs = db_paths

    def _normalize(self, sent_text, ret_meta=False):
        if not ret_meta:
            return self.tokenizer.tokenize_and_lemmatize(sent_text)
        else:
            return self.tokenizer.tokenize_and_spacy(sent_text)
    
    def preprocess_schema(self, schema):
        r = PreprocessedSchema()
        last_table_id = None
        for i, column in enumerate(schema.columns):

            # assert column.type in ["text", "number", "time", "boolean", "others"]
            # type_tok = "<type: {}>".format(column.type)
            # for bert, we take the representation of the first word

            column_text = " ".join(column.name)
            r.column_names.append(column_text)
            r.normalized_column_names.append(self._normalize(column_text))

            table_id = None if column.table is None else column.table.id
            r.column_to_table[str(i)] = table_id
            if table_id is not None:
                columns = r.table_to_columns.setdefault(str(table_id), [])
                columns.append(i)
            if last_table_id != table_id:
                r.table_bounds.append(i)
                last_table_id = table_id

            if column.foreign_key_for is not None:
                r.foreign_keys[str(column.id)] = column.foreign_key_for.id
                r.foreign_keys_tables[str(column.table.id)].add(
                    column.foreign_key_for.table.id
                )

        r.table_bounds.append(len(schema.columns))
        assert len(r.table_bounds) == len(schema.tables) + 1

        for i, table in enumerate(schema.tables):
            table_text = " ".join(table.name)
            r.table_names.append(table_text)
            r.normalized_table_names.append(self._normalize(table_text))

        r.foreign_keys_tables = serialization.to_dict_with_sorted_values(
            r.foreign_keys_tables
        )
        r.primary_keys = [
            column.id for table in schema.tables for column in table.primary_keys
        ]
        return r
    
    def compute_schema_linking(self, q_text):
        question = self._normalize(q_text)
        column, table = self.columns, self.tables
        relations = collections.defaultdict(list)

        col_id2list = dict()
        for col_id, col_item in enumerate(column):
            if col_id == 0:
                continue
            col_id2list[col_id] = col_item

        tab_id2list = dict()
        for tab_id, tab_item in enumerate(table):
            tab_id2list[tab_id] = tab_item

        # 5-gram
        n = 5
        while n > 0:
            for i in range(len(question) - n + 1):
                n_gram_list = question[i : i + n]
                n_gram = " ".join(n_gram_list)
                if len(n_gram.strip()) == 0:
                    continue
                # exact match case
                for col_id in col_id2list:
                    if self.exact_match(n_gram_list, col_id2list[col_id]):
                        for q_id in range(i, i + n):
                            relations["q-col:CEM"].append((q_id, col_id))
                            relations["col-q:CEM"].append((col_id, q_id))
                for tab_id in tab_id2list:
                    if self.exact_match(n_gram_list, tab_id2list[tab_id]):
                        for q_id in range(i, i + n):
                            relations["q-tab:TEM"].append((q_id, tab_id))
                            relations["tab-q:TEM"].append((tab_id, q_id))

                # partial match case
                for col_id in col_id2list:
                    if self.partial_match(n_gram_list, col_id2list[col_id]):
                        for q_id in range(i, i + n):
                            relations[f"q-col:CPM"].append((q_id, col_id))
                            relations[f"col-q:CPM"].append((col_id, q_id))
                for tab_id in tab_id2list:
                    if self.partial_match(n_gram_list, tab_id2list[tab_id]):
                        for q_id in range(i, i + n):
                            relations["q-tab:TPM"].append((q_id, tab_id))
                            relations["tab-q:TPM"].append((tab_id, q_id))
            n -= 1
        return self.remove_duplicates(relations)

    def compute_cell_value_linking(self, q_text):
        """
        Utilize spacy for 1) stop words 2) number
        """
        sp_tokens = self._normalize(q_text, ret_meta=True)

        schema = self.schema
        db_dirs = self.db_dirs

        db_name = schema.db_id
        # find the db path
        for db_dir in db_dirs:
            db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
            if os.path.exists(db_path):
                break
            else:
                continue

        relations = collections.defaultdict(list)
        for q_id, sp_token in enumerate(sp_tokens):
            if sp_token.is_stop or sp_token.is_space:
                continue

            for col_id, column in enumerate(schema.columns):
                if col_id == 0:
                    assert column.orig_name == "*"
                    continue

                if sp_token.like_num:
                    if column.type in ["number", "time"]:  # TODO fine-grained date
                        relations[f"q-col:{column.type.upper()}"].append((q_id, col_id))
                        relations[f"col-q:{column.type.upper()}"].append((col_id, q_id))
                else:
                    word = sp_token.text  # use verbatim for value matching
                    # word = sp_token.lemma_

                    try:
                        ret = self.db_word_match(
                            word, column.orig_name, column.table.orig_name, db_path
                        )
                    except timeout_decorator.TimeoutError as e:
                        ret = False

                    if ret:
                        relations["q-col:CELLMATCH"].append((q_id, col_id))
                        relations["col-q:CELLMATCH"].append((col_id, q_id))

        return self.remove_duplicates(relations)

