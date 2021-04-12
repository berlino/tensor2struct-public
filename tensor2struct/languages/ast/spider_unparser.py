import collections
import copy
import itertools
import os

import asdl
import attr
import networkx as nx

from tensor2struct import ast_util
from tensor2struct.utils import registry
from tensor2struct.grammars import spider


def bimap(first, second):
    return {f: s for f, s in zip(first, second)}, {s: f for f, s in zip(first, second)}


def filter_nones(d):
    return {k: v for k, v in d.items() if v is not None and v != []}


def join(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x


def intersperse(delimiter, seq):
    return itertools.islice(
        itertools.chain.from_iterable(zip(itertools.repeat(delimiter), seq)), 1, None
    )


# Another unparser that produce more bert-friendly output
class SpiderUnparser2(spider.SpiderUnparser):

    terminal_placeholder = "#PH#"

    @attr.s
    class SQLToken:
        token = attr.ib()
        col_id = attr.ib(default=None)
        tab_id = attr.ib(default=None)

    def unparse_col_unit(self, col_unit):
        assert "col_id" in col_unit
        col_id = col_unit["col_id"]
        column = self.schema.columns[col_unit["col_id"]]
        column_name = column.unsplit_name

        result = []

        if col_unit["is_distinct"]:
            result.append(self.SQLToken(token="DISTINCT"))

        agg_type = col_unit["agg_id"]["_type"]
        if agg_type != "NoneAggOp":
            result.append(self.SQLToken(token=agg_type))

        result.append(self.SQLToken(token=column_name, col_id=col_id))
        return result

    def unparse_val(self, val):
        if val["_type"] == "Terminal":
            return [self.SQLToken(token=self.terminal_placeholder)]
        if val["_type"] == "String":
            return [self.SQLToken(token=val["s"])]
        if val["_type"] == "ColUnit":
            return self.unparse_col_unit(val["c"])
        if val["_type"] == "Number":
            return [self.SQLToken(token=str(val["f"]))]
        if val["_type"] == "ValSql":
            return [
                self.SQLToken(token="("),
                *self.unparse_sql(val["s"]),
                self.SQLToken(token=")"),
            ]

    def unparse_val_unit(self, val_unit):
        if val_unit["_type"] == "Column":
            return self.unparse_col_unit(val_unit["col_unit1"])
        col1 = self.unparse_col_unit(val_unit["col_unit1"])
        col2 = self.unparse_col_unit(val_unit["col_unit2"])
        return [*col1, self.SQLToken(token=self.UNIT_TYPES_B[val_unit["_type"]]), *col2]

    def unparse_cond(self, cond, negated=False):
        if cond["_type"] == "And":
            assert not negated
            return [
                *self.unparse_cond(cond["left"]),
                self.SQLToken(token="AND"),
                *self.unparse_cond(cond["right"]),
            ]
        elif cond["_type"] == "Or":
            assert not negated
            return [
                *self.unparse_cond(cond["left"]),
                self.SQLToken(token="OR"),
                *self.unparse_cond(cond["right"]),
            ]
        elif cond["_type"] == "Not":
            return self.unparse_cond(cond["c"], negated=True)
        elif cond["_type"] == "Between":
            tokens = [*self.unparse_val_unit(cond["val_unit"])]
            if negated:
                tokens.append(self.SQLToken(token="NOT"))
            tokens += [
                self.SQLToken(token="BETWEEN"),
                *self.unparse_val(cond["val1"]),
                self.SQLToken(token="AND"),
                *self.unparse_val(cond["val2"]),
            ]
            return tokens
        tokens = [*self.unparse_val_unit(cond["val_unit"])]
        if negated:
            tokens.append(self.SQLToken(token="NOT"))
        tokens += [
            self.SQLToken(token=self.COND_TYPES_B[cond["_type"]]),
            *self.unparse_val(cond["val1"]),
        ]
        return tokens

    def unparse_select(self, select):
        tokens = [self.SQLToken(token="SELECT")]
        if select["is_distinct"]:
            tokens.append(self.SQLToken(token="DISTINCT"))
        for agg in select.get("aggs", []):
            tokens += self.unparse_agg(agg)
        return tokens

    def unparse_agg(self, agg):
        unparsed_val_unit = self.unparse_val_unit(agg["val_unit"])
        agg_type = agg["agg_id"]["_type"]
        if agg_type == "NoneAggOp":
            return unparsed_val_unit
        else:
            return [
                self.SQLToken(token=agg_type),
                self.SQLToken(token="("),
                *unparsed_val_unit,
                self.SQLToken(token=")"),
            ]

    def unparse_from(self, from_):
        if "conds" in from_:
            all_conds, keywords = self.linearize_cond(from_["conds"])
        else:
            all_conds, keywords = [], []
        assert all(keyword == "And" for keyword in keywords)

        cond_indices_by_table = collections.defaultdict(set)
        tables_involved_by_cond_idx = collections.defaultdict(set)
        for i, cond in enumerate(all_conds):
            for column in self.ast_wrapper.find_all_descendants_of_type(cond, "column"):
                table = self.schema.columns[column].table
                if table is None:
                    continue
                cond_indices_by_table[table.id].add(i)
                tables_involved_by_cond_idx[i].add(table.id)

        output_table_ids = set()
        output_cond_indices = set()
        tokens = [self.SQLToken(token="FROM")]
        for i, table_unit in enumerate(from_.get("table_units", [])):
            if i > 0:
                tokens += [self.SQLToken(token="JOIN")]

            if table_unit["_type"] == "TableUnitSql":
                tokens += [
                    self.SQLToken(token="("),
                    *self.unparse_sql(table_unit["s"]),
                    self.SQLToken(token=")"),
                ]
            elif table_unit["_type"] == "Table":
                table_id = table_unit["table_id"]
                tokens += [
                    self.SQLToken(
                        token=self.schema.tables[table_id].unsplit_name, tab_id=table_id
                    )
                ]
                output_table_ids.add(table_id)

                # Output "ON <cond>" if all tables involved in the condition have been output
                conds_to_output = []
                for cond_idx in sorted(cond_indices_by_table[table_id]):
                    if cond_idx in output_cond_indices:
                        continue
                    if tables_involved_by_cond_idx[cond_idx] <= output_table_ids:
                        conds_to_output.append(all_conds[cond_idx])
                        output_cond_indices.add(cond_idx)
                if conds_to_output:
                    tokens += [self.SQLToken(token="ON")]

                    _tokens = list(
                        intersperse(
                            self.SQLToken(token="AND"),
                            (self.unparse_cond(cond) for cond in conds_to_output),
                        )
                    )
                    for t in _tokens:
                        if t.isinstance(list) or t.is_distinct(tuple):
                            tokens += t
                        else:
                            tokens.append(t)
        return tokens

    def unparse_order_by(self, order_by):
        tokens = [self.SQLToken(token="ORDER BY")]
        for v in order_by["val_units"]:
            tokens += self.unparse_val_unit(v)
        tokens.append(self.SQLToken(token=order_by["order"]["_type"]))
        return tokens

    def unparse_sql(self, tree):
        result = []
        result += self.unparse_select(tree["select"])
        result += self.unparse_from(tree["from"])

        def find_subtree(_tree, name):
            return _tree, _tree[name]

        tree, target_tree = find_subtree(tree, "sql_where")
        # cond? where,
        if "where" in target_tree:
            result += [
                self.SQLToken(token="WHERE"),
                *self.unparse_cond(target_tree["where"]),
            ]

        tree, target_tree = find_subtree(tree, "sql_groupby")
        # col_unit* group_by,
        if "group_by" in target_tree:
            result += [self.SQLToken(token="GROUP BY")]
            for c in target_tree["group_by"]:
                result += self.unparse_col_unit(c)

        tree, target_tree = find_subtree(tree, "sql_orderby")
        if "order_by" in target_tree:
            result += self.unparse_order_by(target_tree["order_by"])

        tree, target_tree = find_subtree(tree, "sql_groupby")
        if "having" in target_tree:
            result += [
                self.SQLToken(token="HAVING"),
                *self.unparse_cond(target_tree["having"]),
            ]

        tree, target_tree = find_subtree(tree, "sql_orderby")
        if "limit" in target_tree:
            if isinstance(target_tree["limit"], bool):
                if target_tree["limit"]:
                    result += [self.SQLToken(token="LIMIT"), self.SQLToken(token="1")]
            else:
                result += [
                    self.SQLToken(token="LIMIT"),
                    self.SQLToken(token=str(target_tree["limit"])),
                ]

        tree, target_tree = find_subtree(tree, "sql_ieu")
        if "intersect" in target_tree:
            result += [
                self.SQLToken(token="INTERSECT"),
                *self.unparse_sql(target_tree["intersect"]),
            ]
        if "except" in target_tree:
            result += [
                self.SQLToken(token="EXCEPT"),
                *self.unparse_sql(target_tree["except"]),
            ]
        if "union" in target_tree:
            result += [self.SQLToken("UNION"), *self.unparse_sql(target_tree["union"])]

        return result
