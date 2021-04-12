import os
import collections
import itertools

from tensor2struct.contexts import abstract_context
from tensor2struct.utils import registry


@registry.register("context", "overnight")
class OvernightContext(abstract_context.AbstractContext):
    def __init__(self, schema) -> None:
        self.schema = schema
        self._schema_relations = None

    def compute_schema_linking(self, question):
        relations = collections.defaultdict(list)

        col_id2list = dict()
        for col_id, col_item in enumerate(self.schema["columns"]):
            col_id2list[col_id] = col_item

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
                            relations["q-col:EM"].append((q_id, col_id))
                            relations["col-q:EM"].append((col_id, q_id))

                # partial match case
                for col_id in col_id2list:
                    if self.partial_match(n_gram_list, col_id2list[col_id]):
                        for q_id in range(i, i + n):
                            relations[f"q-col:PM"].append((q_id, col_id))
                            relations[f"col-q:PM"].append((col_id, q_id))
            n -= 1
        return self.remove_duplicates(relations)

    def compute_cell_value_linking(self, question):
        relations = collections.defaultdict(list)

        col_id2list = dict()
        for col_id, col_item in enumerate(self.schema["values"]):
            col_id2list[col_id] = col_item

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
                            relations["q-val:EM"].append((q_id, col_id))
                            relations["val-q:EM"].append((col_id, q_id))

                # partial match case
                for col_id in col_id2list:
                    if self.partial_match(n_gram_list, col_id2list[col_id]):
                        for q_id in range(i, i + n):
                            relations[f"q-val:PM"].append((q_id, col_id))
                            relations[f"val-q:PM"].append((col_id, q_id))
            n -= 1
        return self.remove_duplicates(relations)

    def compute_schema_relations(self):
        if self._schema_relations is not None:
            return self._schema_relations
        relations = collections.defaultdict(list)
        schema_relation = self.schema["schema_relations"]
        for relation in schema_relation:
            ent_1_type, ent_1_id, r_name, ent_2_type, ent_2_id = relation
            trans_dic = {"val": "val", "prop": "col"}
            relations[
                f"{trans_dic[ent_1_type]}-{trans_dic[ent_2_type]}:{r_name}"
            ].append((ent_1_id, ent_2_id))
            relations[
                f"{trans_dic[ent_2_type]}-{trans_dic[ent_1_type]}:!{r_name}"
            ].append((ent_2_id, ent_1_id))
        self._schema_relations = relations
        return relations

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
        for s1, s2 in itertools.product(("q", "val", "col"), repeat=2):
            default_rs.add("{}:{}-default".format(s1, s2))
        return default_rs
