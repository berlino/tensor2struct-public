import attr
import json
import torch

from tensor2struct.datasets import spider
from tensor2struct.datasets.spider import Column, Table, SpiderItem, SpiderDataset
from tensor2struct.utils import registry, dataset

from third_party.ssp import process_sql, schema, evaluation

@attr.s
class SSPItem(spider.SpiderItem):
    is_sql_json_valid = attr.ib(kw_only=True)

@registry.register("dataset", "ssp")
class SSPDataset(dataset.Dataset):
    def __init__(self, path, domain, split, tables_path, db_path, data_split="question"):
        self.path = path
        self.split = split
        self.domain = domain
        self.db_path = db_path
        self.data_split = data_split

        assert self.data_split in ("question", "query",)

        # 1. whether check execution
        if domain in ["atis", "advising"]:
            # execution is very inefficient
            self.check_execution = False
        else:
            self.check_execution = True

        # 2. load schema in json format
        self.schemas, self.eval_foreign_key_maps = spider.load_tables([tables_path])

        # 3. load orig schema
        (self.orig_schemas, _, self.orig_tables,) = schema.get_schemas_from_json(
            tables_path
        )

        # 4. load examples
        self.examples = []
        if self.data_split == "question":
            self.load_examples_by_question_split()
        else:
            assert self.data_split == "query"
            self.load_examples_by_query_split()

    def is_valid_sql(self, sql_str, db_name):
        import sqlite3

        db = self.db_path + f"{db_name}/{db_name}.sqlite"
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_str)
        except Exception as e:
            return False
        return True

    def sql_str_to_json(self, sql_str, db_name):
        sql_json = process_sql.get_sql(
            schema.Schema(self.orig_schemas[db_name], self.orig_tables[db_name],),
            sql_str,
        )
        return sql_json

    def load_examples_by_question_split(self):
        with open(self.path, "r") as f:
            orig_examples = json.load(f)

        counter_invalid_ex = 0
        for orig_example in orig_examples:
            anonymized_sql = orig_example["sql"][0]

            sql_constants = {}
            for var_dic in orig_example["variables"]:
                # if var_dic["location"] == "sql-only":
                # "sql-only" is not enough
                sql_constants[var_dic["name"]] = var_dic["example"]

            for example in orig_example["sentences"]:
                if example["question-split"] != self.split:
                    continue

                nl = example["text"]
                sql = anonymized_sql

                ## adapted from https://github.com/alsuhr-c/language/blob/master/language/xsp/data_preprocessing/michigan_preprocessing.py
                ## However, alane's extraction is probably problematic as it does not handle values that not in NL
                for variable_name, value in sorted(
                    example["variables"].items(), key=lambda x: len(x[0]), reverse=True,
                ):
                    if not value:
                        if variable_name not in sql_constants:
                            # TODO: variable name is empty, not sure why
                            continue
                        else:
                            value = sql_constants[variable_name]

                    nl = nl.replace(variable_name, value)
                    sql = sql.replace(variable_name, value)
                ## end adaption

                # remove invalid sql
                if self.check_execution and not self.is_valid_sql(sql, self.domain):
                    if self.split != "train":
                        print(f"Found non-executable programs in {self.split}")
                    else:
                        counter_invalid_ex += 1
                        continue

                sql_str = sql
                try:
                    sql_json = self.sql_str_to_json(sql_str, self.domain)
                    is_sql_json_valid = True
                except Exception as e:
                    # TODO: fix process_sql
                    if self.split == "train":
                        counter_invalid_ex += 1
                        continue
                    else:
                        sql_json = {
                            "except": None,
                            "from": {"conds": [], "table_units": []},
                            "groupBy": [],
                            "having": [],
                            "intersect": None,
                            "limit": None,
                            "orderBy": [],
                            "select": [False, []],
                            "union": None,
                            "where": [],
                        }
                        is_sql_json_valid = False

                item = SSPItem(
                    text=nl.split(" "),
                    code=sql_json,
                    schema=self.schemas[self.domain],
                    orig={"query": sql_str, "question": nl},
                    orig_schema=self.schemas[self.domain].orig,
                    is_sql_json_valid=is_sql_json_valid,
                )
                self.examples.append(item)

        if counter_invalid_ex > 0:
            print(
                f"WARNING: {counter_invalid_ex} out of {counter_invalid_ex + len(self.examples)} invalid examples cannot be parsed in {self.split}"
            )

    def load_examples_by_query_split(self):
        with open(self.path, "r") as f:
            orig_examples = json.load(f)

        counter_invalid_ex = 0
        for orig_example in orig_examples:
            if orig_example["query-split"] != self.split:
                continue

            anonymized_sql = orig_example["sql"][0]

            sql_constants = {}
            for var_dic in orig_example["variables"]:
                # if var_dic["location"] == "sql-only":
                # "sql-only" is not enough
                sql_constants[var_dic["name"]] = var_dic["example"]

            for example in orig_example["sentences"]:
                nl = example["text"]
                sql = anonymized_sql

                ## adapted from https://github.com/alsuhr-c/language/blob/master/language/xsp/data_preprocessing/michigan_preprocessing.py
                ## However, alane's extraction is probably problematic as it does not handle values that not in NL
                for variable_name, value in sorted(
                    example["variables"].items(), key=lambda x: len(x[0]), reverse=True,
                ):
                    if not value:
                        if variable_name not in sql_constants:
                            # TODO: variable name is empty, not sure why
                            continue
                        else:
                            value = sql_constants[variable_name]

                    nl = nl.replace(variable_name, value)
                    sql = sql.replace(variable_name, value)
                ## end adaption

                # remove invalid sql
                if self.check_execution and not self.is_valid_sql(sql, self.domain):
                    if self.split != "train":
                        print(f"Found non-executable programs in {self.split}")
                    else:
                        counter_invalid_ex += 1
                        continue

                sql_str = sql
                try:
                    sql_json = self.sql_str_to_json(sql_str, self.domain)
                    is_sql_json_valid = True
                except Exception as e:
                    # TODO: fix process_sql
                    if self.split == "train":
                        counter_invalid_ex += 1
                        continue
                    else:
                        sql_json = {
                            "except": None,
                            "from": {"conds": [], "table_units": []},
                            "groupBy": [],
                            "having": [],
                            "intersect": None,
                            "limit": None,
                            "orderBy": [],
                            "select": [False, []],
                            "union": None,
                            "where": [],
                        }
                        is_sql_json_valid = False

                item = SSPItem(
                    text=nl.split(" "),
                    code=sql_json,
                    schema=self.schemas[self.domain],
                    orig={"query": sql_str, "question": nl},
                    orig_schema=self.schemas[self.domain].orig,
                    is_sql_json_valid=is_sql_json_valid,
                )
                self.examples.append(item)

        if counter_invalid_ex > 0:
            print(
                f"WARNING: {counter_invalid_ex} out of {counter_invalid_ex + len(self.examples)} invalid examples cannot be parsed in {self.split}"
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    class Metrics:
        """
        Only differ by evaluation module, compared with Spider Metric
        """

        def __init__(self, dataset, etype):
            self.dataset = dataset
            self.etype = etype
            self.foreign_key_maps = {
                db_id: evaluation.build_foreign_key_map(schema.orig)
                for db_id, schema in self.dataset.schemas.items()
            }
            self.evaluator = evaluation.Evaluator(
                self.dataset.db_path, self.foreign_key_maps, etype,
            )
            self.results = []

        def add_one(self, item, inferred_code, orig_question=None):
            ret_dict = self.evaluator.evaluate_one(
                item.schema.db_id, item.orig["query"], inferred_code
            )

            if orig_question:
                ret_dict["orig_question"] = orig_question

            self.results.append(ret_dict)

        def add_beams(self, item, inferred_codes, orig_question=None):
            """
            Find the first executable SQL and run evaluation.
            SSP only support eval by execution for now.
            """
            ret_dict = None
            for i, code in enumerate(inferred_codes):
                if self.evaluator.isValidSQL(code, item.schema.db_id):
                    ret_dict = self.evaluator.evaluate_one_by_exec_only(
                        item.schema.db_id, item.orig["query"], code
                    )
                    break

            # if all failed
            if ret_dict is None:
                ret_dict = self.evaluator.evaluate_one_by_exec_only(
                    item.schema.db_id, item.orig["query"], inferred_codes[0]
                )

            if orig_question:
                ret_dict["orig_question"] = orig_question

            self.results.append(ret_dict)

        def finalize(self):
            self.evaluator.finalize()
            results = {"per_item": self.results, "total_scores": self.evaluator.scores}
            return results
