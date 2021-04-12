import attr
import os
import enum
import tqdm
import random
import numpy as np
import pyrsistent
import collections
import networkx as nx

from tensor2struct.languages.ast import spider
from tensor2struct.utils import serialization
from tensor2struct.models.ast_decoder.utils import get_field_presence_info


class SpiderUnparser(spider.SpiderUnparser):
    """
    Unparse a ast tree into a str which will be used for encoding
    Known Issues:
    1) due to refine_from, table_units might be longer than what's been limited by grammar
    """

    def refine_from(self, tree):
        """
        1) Inferring tables from columns predicted 
        2) Mix them with the predicted tables if any
        3) Inferring conditions based on tables 

        Compared with original refine_from function, this refine_from take 
        either predicted tables or must_in_tables, instead the union of both
        """
        # nested query in from clause, recursively use the refinement
        if "from" in tree and tree["from"]["table_units"][0]["_type"] == "TableUnitSql":
            for table_unit in tree["from"]["table_units"]:
                subquery_tree = table_unit["s"]
                self.refine_from(subquery_tree)
            return

        # get predicted tables
        predicted_from_table_ids = set()
        if "from" in tree:
            table_unit_set = []
            for table_unit in tree["from"]["table_units"]:
                if "table_id" not in table_unit:
                    continue  # TODO: might be mix of table units and tableunit sql
                if table_unit["table_id"] not in predicted_from_table_ids:
                    predicted_from_table_ids.add(table_unit["table_id"])
                    table_unit_set.append(table_unit)
            tree["from"]["table_units"] = table_unit_set  # remove duplicate

        # Get all candidate columns
        candidate_column_ids = set(
            self.ast_wrapper.find_all_descendants_of_type(
                tree, "column", lambda field: field.type != "sql"
            )
        )
        candidate_columns = [self.schema.columns[i] for i in candidate_column_ids]
        must_in_from_table_ids = set(
            column.table.id for column in candidate_columns if column.table is not None
        )

        if len(must_in_from_table_ids) == 0:
            all_from_table_ids = predicted_from_table_ids
        else:
            all_from_table_ids = must_in_from_table_ids
        assert all_from_table_ids

        covered_tables = set()
        candidate_table_ids = sorted(all_from_table_ids)
        start_table_id = candidate_table_ids[0]
        conds = []
        for table_id in candidate_table_ids[1:]:
            if table_id in covered_tables:
                continue
            try:
                path = nx.shortest_path(
                    self.schema.foreign_key_graph,
                    source=start_table_id,
                    target=table_id,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                covered_tables.add(table_id)
                continue

            for source_table_id, target_table_id in zip(path, path[1:]):
                if target_table_id in covered_tables:
                    continue
                all_from_table_ids.add(target_table_id)
                col1, col2 = self.schema.foreign_key_graph[source_table_id][
                    target_table_id
                ]["columns"]
                conds.append(
                    {
                        "_type": "Eq",
                        "val_unit": {
                            "_type": "Column",
                            "col_unit1": {
                                "_type": "col_unit",
                                "agg_id": {"_type": "NoneAggOp"},
                                "col_id": col1,
                                "is_distinct": False,
                            },
                        },
                        "val1": {
                            "_type": "ColUnit",
                            "c": {
                                "_type": "col_unit",
                                "agg_id": {"_type": "NoneAggOp"},
                                "col_id": col2,
                                "is_distinct": False,
                            },
                        },
                    }
                )
        table_units = [
            {"_type": "Table", "table_id": i} for i in sorted(all_from_table_ids)
        ]

        tree["from"] = {"_type": "from", "table_units": table_units}
        cond_node = self.conjoin_conds(conds)

        if cond_node is not None:
            tree["from"]["conds"] = cond_node

    def unparse_cond(self, cond, negated=False):
        """
        Change: if negated is mistakely set, correct it
        """
        if cond["_type"] == "And":
            # assert negated
            if negated:
                negated == False
            return "{} AND {}".format(
                self.unparse_cond(cond["left"]), self.unparse_cond(cond["right"])
            )
        elif cond["_type"] == "Or":
            # assert negated
            if negated:
                negated == False
            return "{} OR {}".format(
                self.unparse_cond(cond["left"]), self.unparse_cond(cond["right"])
            )
        elif cond["_type"] == "Not":
            return self.unparse_cond(cond["c"], negated=True)
        elif cond["_type"] == "Between":
            tokens = [self.unparse_val_unit(cond["val_unit"])]
            if negated:
                tokens.append("NOT")
            tokens += [
                "BETWEEN",
                self.unparse_val(cond["val1"]),
                "AND",
                self.unparse_val(cond["val2"]),
            ]
            return " ".join(tokens)
        tokens = [self.unparse_val_unit(cond["val_unit"])]
        if negated:
            tokens.append("NOT")
        tokens += [self.COND_TYPES_B[cond["_type"]], self.unparse_val(cond["val1"])]
        return " ".join(tokens)

    def unparse_col_unit(self, col_unit):
        """
        Change: do not add table prefix to column
        TODO: this might not be necessary
        """
        if "col_id" in col_unit:
            column = self.schema.columns[col_unit["col_id"]]
            if column.table is None:
                column_name = column.orig_name
            else:
                # column_name = "{}.{}".format(column.table.orig_name, column.orig_name)
                column_name = column.orig_name
        else:
            column_name = "some_col"

        if col_unit["is_distinct"]:
            column_name = "DISTINCT {}".format(column_name)
        agg_type = col_unit["agg_id"]["_type"]
        if agg_type == "NoneAggOp":
            return column_name
        else:
            return "{}({})".format(agg_type, column_name)

    def unparse_str(self, raw_str):
        return raw_str


class SpiderUnparserSSP(SpiderUnparser):
    def refine_from(self, tree):
        super().refine_from(tree)

    def unparse_from(self, from_):
        tokens = ["FROM"]
        for i, table_unit in enumerate(from_.get("table_units", [])):
            if i > 0:
                tokens += [","]

            if table_unit["_type"] == "Table":
                table_id = table_unit["table_id"]
                tokens += [self.schema.tables[table_id].orig_name]
        from_str = " ".join(tokens)
        return from_str

def unparse(ast_wrapper, schema, tree, refine_from=True):
    if refine_from:
        unparser = SpiderUnparser(ast_wrapper, schema)
    else:
        unparser = SpiderUnparserSSP(ast_wrapper, schema)
    return unparser.unparse_sql(tree)


class PCFG:
    """
    PCFG wrapper for AST
    """

    def __init__(self, grammar, schema, use_seq_elem_rules=False):
        self.grammar = grammar
        self.schema = schema
        self.ast_wrapper = grammar.ast_wrapper
        self.use_seq_elem_rules = use_seq_elem_rules

        self.sum_type_constructors = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        self.field_presence_infos = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        self.seq_lengths = collections.defaultdict(lambda: collections.defaultdict(int))
        self.primitives = collections.defaultdict(lambda: collections.defaultdict(int))
        self.pointers = collections.defaultdict(lambda: collections.defaultdict(int))

    def record_productions(self, tree):
        queue = [(tree, False)]
        while queue:
            node, is_seq_elem = queue.pop()
            node_type = node["_type"]

            # sum type
            for type_name in [node_type] + node.get("_extra_types", []):
                if type_name in self.ast_wrapper.constructors:
                    sum_type_name = self.ast_wrapper.constructor_to_sum_type[type_name]
                    if is_seq_elem and self.use_seq_elem_rules:
                        self.sum_type_constructors[sum_type_name + "_seq_elem"][
                            type_name
                        ] += 1
                    else:
                        self.sum_type_constructors[sum_type_name][type_name] += 1

            # field
            assert node_type in self.ast_wrapper.singular_types
            field_presence_info = get_field_presence_info(
                self.ast_wrapper,
                node,
                self.ast_wrapper.singular_types[node_type].fields,
            )
            self.field_presence_infos[node_type][field_presence_info] += 1

            # seq elem
            for field_info in self.ast_wrapper.singular_types[node_type].fields:
                field_value = node.get(field_info.name, [] if field_info.seq else None)

                # a field that is not present
                if field_value is None:
                    continue

                to_enqueue = []
                if field_info.seq:
                    self.seq_lengths[field_info.type + "*"][len(field_value)] += 1
                    to_enqueue = field_value
                else:
                    to_enqueue = [field_value]

                # process each field
                for child in to_enqueue:
                    if isinstance(child, collections.abc.Mapping) and "_type" in child:
                        queue.append((child, field_info.seq))
                    elif field_info.type in self.grammar.pointers:
                        self.pointers[field_info.type][child] += 1
                    else:
                        # str, int, bool primitives
                        assert field_info.type in self.ast_wrapper.primitive_types
                        self.primitives[field_info.type][child] += 1

    def calculate_rules(self):
        all_rules = {}
        rules_prob = {}

        for sum_parent, children_with_freq in sorted(
            self.sum_type_constructors.items()
        ):
            assert not isinstance(children_with_freq, set)
            children_freq_pair = list(children_with_freq.items())
            all_rules[sum_parent] = [s[0] for s in children_freq_pair]
            freqs = [s[1] for s in children_freq_pair]
            rules_prob[sum_parent] = [f / sum(freqs) for f in freqs]

        for prod_name, field_presence_infos_with_freq in sorted(
            self.field_presence_infos.items()
        ):
            assert not isinstance(field_presence_infos_with_freq, set)
            field_freq_pair = list(field_presence_infos_with_freq.items())
            all_rules[prod_name] = [s[0] for s in field_freq_pair]
            freqs = [s[1] for s in field_freq_pair]
            rules_prob[prod_name] = [f / sum(freqs) for f in freqs]

        for seq_type_name, lengths_with_freq in sorted(self.seq_lengths.items()):
            assert not isinstance(lengths_with_freq, set)
            length_freq_pair = list(lengths_with_freq.items())
            all_rules[seq_type_name] = [s[0] for s in length_freq_pair]
            freqs = [s[1] for s in length_freq_pair]
            rules_prob[seq_type_name] = [f / sum(freqs) for f in freqs]

        for prim_type, prim_with_freq in sorted(self.primitives.items()):
            prim_freq_pair = list(prim_with_freq.items())
            all_rules[prim_type] = [s[0] for s in prim_freq_pair]
            freqs = [s[1] for s in prim_freq_pair]
            rules_prob[prim_type] = [f / sum(freqs) for f in freqs]

        for pointer_type, pointer_with_freq in sorted(self.pointers.items()):
            pointer_freq_pair = list(pointer_with_freq.items())
            all_rules[pointer_type] = [s[0] for s in pointer_freq_pair]
            freqs = [s[1] for s in pointer_freq_pair]
            rules_prob[pointer_type] = [f / sum(freqs) for f in freqs]

        params = {k: (all_rules[k], rules_prob[k]) for k in all_rules}
        return params

    def estimate(self):
        # rules and their probs
        self.params = self.calculate_rules()

    def sample(self, num_samples, max_actions=100):
        results = []
        for _ in tqdm.tqdm(range(num_samples), total=num_samples):
            ret = self._sample(max_actions=max_actions)
            if ret is not None:
                sql_tree, sql_str = ret
                if self._is_valid(sql_tree, sql_str):
                    results.append(ret)  # duplicate is fine
        return results

    def _sample(self, max_actions):
        traversal = TreeTraversal(self)
        choices, scores = traversal.step(None)

        for _ in range(max_actions):
            choice_idx = np.random.choice(len(choices), 1, p=scores)[0]
            choice = choices[choice_idx]
            ret = traversal.step(choice)

            if ret is not None:
                choices, scores = ret
            else:
                break
        return traversal.finalize(self.schema)

    def _is_valid(self, sql_tree, sql_str):
        # * column can only exists in SELECT clause
        column_ids = list(
            self.ast_wrapper.find_all_descendants_of_type(sql_tree, "column")
        )
        select_column_ids = list(
            self.ast_wrapper.find_all_descendants_of_type(sql_tree["select"], "column")
        )
        if select_column_ids.count(0) > 1 or column_ids.count(
            0
        ) > select_column_ids.count(0):
            return False

        # Orderby must have groupby
        if "ORDER BY" in sql_str and "GROUP BY" not in sql_str:
            return False

        # Empty groupby
        if "GROUP BY  " in sql_str or sql_str.endswith("GROUP BY "):
            return False

        return True


class TreeTraversal:
    """
    This is the minimal version of traversal used for training/inference of decoder 
    """

    class Handler:
        handlers = {}

        @classmethod
        def register_handler(cls, func_type):
            if func_type in cls.handlers:
                raise RuntimeError(f"{func_type} handler is already registered")

            def inner_func(func):
                cls.handlers[func_type] = func.__name__
                return func

            return inner_func

    @attr.s(frozen=True)
    class QueueItem:
        state = attr.ib()
        node_type = attr.ib()
        parent_field_name = attr.ib()

    class State(enum.Enum):
        SUM_TYPE_INQUIRE = 0
        SUM_TYPE_APPLY = 1
        CHILDREN_INQUIRE = 2
        CHILDREN_APPLY = 3
        LIST_LENGTH_INQUIRE = 4
        LIST_LENGTH_APPLY = 5
        GEN_TOKEN = 6
        POINTER_INQUIRE = 7
        POINTER_APPLY = 8
        NODE_FINISHED = 9

    class TreeAction:
        pass

    @attr.s(frozen=True)
    class SetParentField(TreeAction):
        parent_field_name = attr.ib()
        node_type = attr.ib()
        node_value = attr.ib(default=None)

    @attr.s(frozen=True)
    class CreateParentFieldList(TreeAction):
        parent_field_name = attr.ib()

    @attr.s(frozen=True)
    class AppendTerminalToken(TreeAction):
        parent_field_name = attr.ib()
        value = attr.ib()

    @attr.s(frozen=True)
    class FinalizeTerminal(TreeAction):
        parent_field_name = attr.ib()
        terminal_type = attr.ib()

    @attr.s(frozen=True)
    class NodeFinished(TreeAction):
        pass

    def __init__(self, pcfg):
        self.pcfg = pcfg
        self.params = pcfg.params

        self.actions = pyrsistent.pvector()
        self.queue = pyrsistent.pvector()
        root_type = pcfg.grammar.root_type
        if root_type in self.pcfg.ast_wrapper.sum_types:
            initial_state = TreeTraversal.State.SUM_TYPE_INQUIRE
        else:
            initial_state = TreeTraversal.State.CHILDREN_INQUIRE
        self.cur_item = TreeTraversal.QueueItem(
            state=initial_state, node_type=root_type, parent_field_name=None
        )

    def step(self, last_choice):
        while True:
            self.record_last_choice(last_choice)
            handler_name = TreeTraversal.Handler.handlers[self.cur_item.state]
            handler = getattr(self, handler_name)
            choices, continued = handler(last_choice)
            if continued:
                last_choice = choices
                continue
            else:
                return choices

    @Handler.register_handler(State.SUM_TYPE_INQUIRE)
    def process_sum_inquire(self, last_choice):
        self.cur_item = attr.evolve(
            self.cur_item, state=TreeTraversal.State.SUM_TYPE_APPLY
        )
        return self.params[self.cur_item.node_type], False

    @Handler.register_handler(State.SUM_TYPE_APPLY)
    def process_sum_apply(self, last_choice):
        singular_type = last_choice
        self.cur_item = attr.evolve(
            self.cur_item,
            node_type=singular_type,
            state=TreeTraversal.State.CHILDREN_INQUIRE,
        )
        return None, True

    @Handler.register_handler(State.CHILDREN_INQUIRE)
    def process_children_inquire(self, last_choice):
        type_info = self.pcfg.ast_wrapper.singular_types[self.cur_item.node_type]
        if not type_info.fields:
            if self.pop():
                last_choice = None
                return last_choice, True
            else:
                return None, False

        self.cur_item = attr.evolve(
            self.cur_item, state=TreeTraversal.State.CHILDREN_APPLY
        )

        return self.params[self.cur_item.node_type], False

    @Handler.register_handler(State.CHILDREN_APPLY)
    def process_children_apply(self, last_choice):
        children_presence = last_choice

        self.queue = self.queue.append(
            TreeTraversal.QueueItem(
                state=TreeTraversal.State.NODE_FINISHED,
                node_type=None,
                parent_field_name=None,
            )
        )
        for field_info, present in reversed(
            list(
                zip(
                    self.pcfg.ast_wrapper.singular_types[
                        self.cur_item.node_type
                    ].fields,
                    children_presence,
                )
            )
        ):
            if not present:
                continue

            child_type = field_type = field_info.type
            if field_info.seq:
                child_state = TreeTraversal.State.LIST_LENGTH_INQUIRE
            elif field_type in self.pcfg.ast_wrapper.sum_types:
                child_state = TreeTraversal.State.SUM_TYPE_INQUIRE
            elif field_type in self.pcfg.ast_wrapper.product_types:
                assert self.pcfg.ast_wrapper.product_types[field_type].fields
                child_state = TreeTraversal.State.CHILDREN_INQUIRE
            elif field_type in self.pcfg.grammar.pointers:
                child_state = TreeTraversal.State.POINTER_INQUIRE
            elif field_type in self.pcfg.ast_wrapper.primitive_types:
                child_state = TreeTraversal.State.GEN_TOKEN
            else:
                raise ValueError("Unable to handle field type {}".format(field_type))

            self.queue = self.queue.append(
                TreeTraversal.QueueItem(
                    state=child_state,
                    node_type=child_type,
                    parent_field_name=field_info.name,
                )
            )

        advanced = self.pop()
        assert advanced
        last_choice = None
        return last_choice, True

    @Handler.register_handler(State.LIST_LENGTH_INQUIRE)
    def process_list_length_inquire(self, last_choice):
        self.cur_item = attr.evolve(
            self.cur_item, state=TreeTraversal.State.LIST_LENGTH_APPLY
        )

        return self.params[self.cur_item.node_type + "*"], False

    @Handler.register_handler(State.LIST_LENGTH_APPLY)
    def process_list_length_apply(self, last_choice):
        num_children = last_choice
        elem_type = self.cur_item.node_type

        child_node_type = elem_type
        if elem_type in self.pcfg.ast_wrapper.sum_types:
            child_state = TreeTraversal.State.SUM_TYPE_INQUIRE
            if self.pcfg.use_seq_elem_rules:
                child_node_type = elem_type + "_seq_elem"
        elif elem_type in self.pcfg.ast_wrapper.product_types:
            child_state = TreeTraversal.State.CHILDREN_INQUIRE
        elif elem_type == "identifier":
            child_state = TreeTraversal.State.GEN_TOKEN
            child_node_type = "str"
        elif elem_type in self.pcfg.ast_wrapper.primitive_types:
            # TODO: Fix this
            raise ValueError("sequential builtin types not supported")
        else:
            raise ValueError("Unable to handle seq field type {}".format(elem_type))

        for i in range(num_children):
            self.queue = self.queue.append(
                TreeTraversal.QueueItem(
                    state=child_state,
                    node_type=child_node_type,
                    parent_field_name=self.cur_item.parent_field_name,
                )
            )

        advanced = self.pop()
        assert advanced
        last_choice = None
        return last_choice, True

    @Handler.register_handler(State.GEN_TOKEN)
    def process_gen_token(self, last_choice):
        if last_choice is not None:
            if self.pop():
                return None, True
            else:
                return None, False

        return self.params[self.cur_item.node_type], False

    @Handler.register_handler(State.POINTER_INQUIRE)
    def process_pointer_inquire(self, last_choice):
        self.cur_item = attr.evolve(
            self.cur_item, state=TreeTraversal.State.POINTER_APPLY
        )

        return self.params[self.cur_item.node_type], False

    @Handler.register_handler(State.POINTER_APPLY)
    def process_pointer_apply(self, last_choice):
        if self.pop():
            last_choice = None
            return last_choice, True
        else:
            return None, False

    @Handler.register_handler(State.NODE_FINISHED)
    def process_node_finished(self, last_choice):
        if self.pop():
            last_choice = None
            return last_choice, True
        else:
            return None, False

    def pop(self):
        if self.queue:
            self.cur_item = self.queue[-1]
            self.queue = self.queue.delete(-1)
            return True
        return False

    def record_last_choice(self, last_choice):
        # CHILDREN_INQUIRE
        if self.cur_item.state == TreeTraversal.State.CHILDREN_INQUIRE:
            self.actions = self.actions.append(
                self.SetParentField(
                    self.cur_item.parent_field_name, self.cur_item.node_type
                )
            )
            type_info = self.pcfg.ast_wrapper.singular_types[self.cur_item.node_type]
            if not type_info.fields:
                self.actions = self.actions.append(self.NodeFinished())

        # LIST_LENGTH_APPLY
        elif self.cur_item.state == TreeTraversal.State.LIST_LENGTH_APPLY:
            self.actions = self.actions.append(
                self.CreateParentFieldList(self.cur_item.parent_field_name)
            )

        # GEN_TOKEN
        elif self.cur_item.state == TreeTraversal.State.GEN_TOKEN:
            if last_choice is not None:
                self.actions = self.actions.append(
                    self.AppendTerminalToken(
                        self.cur_item.parent_field_name, last_choice
                    )
                )
                self.actions = self.actions.append(
                    self.FinalizeTerminal(
                        self.cur_item.parent_field_name, self.cur_item.node_type
                    )
                )

        elif self.cur_item.state == TreeTraversal.State.POINTER_APPLY:
            self.actions = self.actions.append(
                self.SetParentField(
                    self.cur_item.parent_field_name,
                    node_type=None,
                    node_value=last_choice,
                )
            )

        # NODE_FINISHED
        elif self.cur_item.state == TreeTraversal.State.NODE_FINISHED:
            self.actions = self.actions.append(self.NodeFinished())

    def finalize(self, schema):
        root = current = None
        stack = []
        for i, action in enumerate(self.actions):
            if isinstance(action, self.SetParentField):
                if action.node_value is None:
                    new_node = {"_type": action.node_type}
                else:
                    new_node = action.node_value

                if action.parent_field_name is None:
                    # Initial node in tree.
                    assert root is None
                    root = current = new_node
                    stack.append(root)
                    continue

                existing_list = current.get(action.parent_field_name)
                if existing_list is None:
                    current[action.parent_field_name] = new_node
                else:
                    assert isinstance(existing_list, list)
                    current[action.parent_field_name].append(new_node)

                if action.node_value is None:
                    stack.append(current)
                    current = new_node

            elif isinstance(action, self.CreateParentFieldList):
                current[action.parent_field_name] = []

            elif isinstance(action, self.AppendTerminalToken):
                tokens = current.get(action.parent_field_name)
                if tokens is None:
                    tokens = current[action.parent_field_name] = []
                tokens.append(action.value)

            elif isinstance(action, self.FinalizeTerminal):
                tokens = current.get(action.parent_field_name, [])
                assert len(tokens) == 1
                value = tokens[0]
                current[action.parent_field_name] = value

            elif isinstance(action, self.NodeFinished):
                current = stack.pop()

            else:
                raise ValueError(action)

        if stack:
            # not finished
            return None
        else:
            try:
                refine_from = self.pcfg.grammar.infer_from_conditions
                sql_str = unparse(
                    self.pcfg.ast_wrapper, schema, root, refine_from=refine_from
                )
                return root, sql_str
            except KeyError:
                return None
