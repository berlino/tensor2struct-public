import os
import attr
import collections
import itertools
from tensor2struct.contexts import abstract_context
from tensor2struct.utils import registry, serialization


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


@registry.register("context", "spider")
class SpiderContext(abstract_context.AbstractContext):
    def __init__(self, schema, word_emb, db_path) -> None:
        self.schema = schema
        self.word_emb = word_emb
        preproc_schema = self.preprocess_schema(self.schema)
        self.preproc_schema = preproc_schema
        assert preproc_schema.column_names[0][0].startswith("<type:")
        self.columns = [col[1:] for col in preproc_schema.column_names]
        self.tables = preproc_schema.table_names

        self.db_dir = db_path

    def _tokenize(self, presplit, unsplit):
        if self.word_emb:
            return self.word_emb.tokenize(unsplit)
        return presplit

    def preprocess_schema(self, schema):
        r = PreprocessedSchema()
        last_table_id = None
        for i, column in enumerate(schema.columns):
            col_toks = self._tokenize(column.name, column.unsplit_name)

            # assert column.type in ["text", "number", "time", "boolean", "others"]
            type_tok = "<type: {}>".format(column.type)
            column_name = [type_tok] + col_toks
            r.column_names.append(column_name)

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
            table_toks = self._tokenize(table.name, table.unsplit_name)
            r.table_names.append(table_toks)

        r.foreign_keys_tables = serialization.to_dict_with_sorted_values(
            r.foreign_keys_tables
        )
        r.primary_keys = [
            column.id for table in schema.tables for column in table.primary_keys
        ]

        return r

    def compute_schema_linking(self, question):
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

    def compute_cell_value_linking(self, tokens):
        schema = self.schema
        db_dir = self.db_dir

        db_name = schema.db_id
        db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")

        relations = collections.defaultdict(list)
        for q_id, word in enumerate(tokens):
            if len(word.strip()) == 0:
                continue
            if self.isstopword(word):
                continue

            num_flag, _ = self.isnumber(word)

            for col_id, column in enumerate(schema.columns):
                if col_id == 0:
                    assert column.orig_name == "*"
                    continue

                # word is number
                if num_flag:
                    if column.type in ["number", "time"]:  # TODO fine-grained date
                        relations[f"q-col:{column.type.upper()}"].append((q_id, col_id))
                        relations[f"col-q:{column.type.upper()}"].append((col_id, q_id))
                else:
                    ret = self.db_word_match(
                        word, column.orig_name, column.table.orig_name, db_path
                    )
                    if ret:
                        relations["q-col:CELLMATCH"].append((q_id, col_id))
                        relations["col-q:CELLMATCH"].append((col_id, q_id))

        return self.remove_duplicates(relations)

    def compute_schema_relations(self):
        relations = collections.defaultdict(list)

        for col1, col2 in itertools.product(range(len(self.columns)), repeat=2):
            if col1 == col2:
                relations["col-col:dist0"].append((col1, col2))
            else:
                if self.preproc_schema.foreign_keys.get(str(col1)) == col2:
                    relations["col-col:fkey_forward"].append((col1, col2))
                    relations["col-col:fkey_backward"].append((col2, col1))
                elif (
                    self.preproc_schema.column_to_table[str(col1)]
                    == self.preproc_schema.column_to_table[str(col2)]
                ):
                    relations["col-col:table_match"].append((col1, col2))
                    relations["col-col:table_match"].append((col2, col1))

        def match_foreign_key(col, table):
            foreign_key_for = self.preproc_schema.foreign_keys.get(str(col))
            if foreign_key_for is None:
                return False
            foreign_table = self.preproc_schema.column_to_table[str(foreign_key_for)]
            return self.preproc_schema.column_to_table[str(col)] == foreign_table

        for col, tab in itertools.product(
            range(len(self.columns)), range(len(self.tables))
        ):
            if match_foreign_key(col, tab):
                relations["col-tab:fkey"].append((col, tab))
                relations["tab-col:fkey"].append((tab, col))
            _tab = self.preproc_schema.column_to_table[str(col)]
            if _tab == tab:
                if col in self.preproc_schema.primary_keys:
                    relations["col-tab:pr_key"].append((col, tab))
                    relations["tab-col:pr_key"].append((tab, col))
                else:
                    relations["col-tab:table_match"].append((col, tab))
                    relations["tab-col:table_match"].append((tab, col))
            if _tab is None:
                relations["col-tab:any_table"].append((col, tab))
                relations["tab-col:any_table"].append((tab, col))

        for tab1, tab2 in itertools.product(range(len(self.tables)), repeat=2):
            if tab1 == tab2:
                relations["tab-tab:dist0"].append((tab1, tab2))
            else:
                forward = tab2 in self.preproc_schema.foreign_keys_tables.get(
                    str(tab1), ()
                )
                backward = tab1 in self.preproc_schema.foreign_keys_tables.get(
                    str(tab2), ()
                )
                if forward and backward:
                    relations["tab-tab:fkey_both"].append((tab1, tab2))
                elif forward:
                    relations["tab-tab:fkey_forward"].append((tab1, tab2))
                elif backward:
                    relations["tab-tab:fkey_backward"].append((tab1, tab2))
        return self.remove_duplicates(relations)

    @staticmethod
    def remove_duplicates(relations):
        new_relations = {}
        for relation in relations:
            new_relations[relation] = list(set(relations[relation]))
        return new_relations

    @staticmethod
    def get_default_relations():
        default_rs = set()
        default_rs.add("x-x:default")
        for s1, s2 in itertools.product(("q", "col", "tab"), repeat=2):
            default_rs.add("{}:{}-default".format(s1, s2))
        return default_rs

    def get_column_value_map(self):
        pass

    def get_all_entities(self):
        pass
