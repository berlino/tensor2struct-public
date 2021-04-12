import os
import attr
import json
import inspect
import logging
import itertools
import functools
import tempfile
import subprocess
from nltk import tree
import collections

from typing import Dict, List, NamedTuple, Set, Tuple, Callable, Union, Any, Type

from tensor2struct.utils import registry
from tensor2struct.languages.dsl.common import (
    ParsingError,
    ExecutionError,
    START_SYMBOL,
    util,
)
from tensor2struct.languages.dsl.domain_language import (
    DomainLanguage,
    PredicateType,
    BasicType,
    FunctionType,
)

logger = logging.getLogger("tensor2struct")
logger.setLevel(logging.ERROR)


def to_lisp_like_string(node):
    if isinstance(node, tree.Tree):
        return f"({node.label()} {' '.join([to_lisp_like_string(child) for child in node])})"
    else:
        return node


class Value:
    pass


class SingletonValue:
    pass


class EntityValue(Value):
    pass


class NumericalValue(Value):
    pass


class DateValue(NumericalValue):
    pass


class NumberValue(NumericalValue):
    pass


class TimeValue(NumericalValue):
    pass


class Property:
    p: str


class Aggregate:
    mode: str


class Operator:
    op: str


_general_functions: Dict[str, Callable] = {}
_general_function_types: Dict[str, List[PredicateType]] = collections.defaultdict(list)


def predicate(func: Callable) -> Callable:
    signature = inspect.signature(func)
    argument_types = [param.annotation for name, param in signature.parameters.items()]
    argument_types = argument_types[1:]  # remove the type of self
    return_type = signature.return_annotation
    argument_nltk_types: List[PredicateType] = [
        PredicateType.get_type(arg_type) for arg_type in argument_types
    ]
    return_nltk_type = PredicateType.get_type(return_type)
    function_nltk_type = PredicateType.get_function_type(
        argument_nltk_types, return_nltk_type
    )
    name = func.__name__
    _general_functions[name] = func
    if function_nltk_type in _general_function_types[name]:
        raise ParsingError(f"duplicate definition of function {name}")

    _general_function_types[name].append(function_nltk_type)

    @functools.wraps(func)
    def wrap(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return wrap


@registry.register("grammar", "overnight")
class LambdaDCS(DomainLanguage):
    """
    The original DomainLanguage does not support function with mutliple function types, 
    this is instrinsically limited by the Python. 
    This class address this by registering each function when it's defined to capture
    the function with mutiple function types, like filter. 

    Basic types: string, Value, Property
    
    Lexical information is listed in overnight_terminals.json
    Each terminal will be assigned a Type class. 
    Function types and general terminals will be predicted as a part of the grammar;
    domain termainls will be predicted by pointing. 
    """

    _rule_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "grammar/overnight_terminals.json"
    )
    with open(_rule_path, "r") as f:
        overnight_rules = json.load(f)
    schema_relation_cache = {}
    schema_lexicon_cache = {}

    def __init__(self, domain):
        self.domain = domain
        self._general_function_types = _general_function_types

        self._functions = _general_functions
        self._function_types = _general_function_types
        self._start_types = {PredicateType.get_type(List[Value])}
        self._nonterminal_productions = None

        # function values of the terminal function will be used for denormalization
        _value_cache = []
        value_dict = self.overnight_rules["domain_terminals"][domain]["value"]
        for v in value_dict["singleton"]:
            self.add_constant(v, v, type_=SingletonValue)
            _value_cache.append((v, v))
        for v in value_dict["entity"]:
            self.add_constant(v, v, type_=EntityValue)
            self.add_constant(v, v, type_=Value)
            _value_cache.append((v, v))
        self.t_classes = {"number": NumberValue, "time": TimeValue, "date": DateValue}
        for t_type in self.t_classes:
            if t_type in value_dict:
                for v in value_dict[t_type]:
                    items = v.split()
                    assert items[0] == t_type
                    orig_v = f"( {v} )"
                    real_v = " ".join(items[1:])
                    norm_v = self.norm_value(real_v)
                    self.add_constant(norm_v, orig_v, type_=self.t_classes[t_type])
                    self.add_constant(norm_v, orig_v, type_=NumericalValue)
                    self.add_constant(norm_v, orig_v, type_=Value)
                    _value_cache.append((v, norm_v))

        _property_cache = []
        for p in self.overnight_rules["domain_terminals"][domain]["property"]:
            p_func = f"( string {p} )"
            self.add_constant(p, p_func, type_=Property)
            _property_cache.append((p, p))

        _g_cache = []
        for agg in self.overnight_rules["general_terminals"]["agg"]:
            orig_agg = f"( string {agg} )"
            self.add_constant(agg, orig_agg, type_=Aggregate)
            _g_cache.append(agg)

        for op in self.overnight_rules["general_terminals"]["op"]:
            orig_op = f"( string {op} )"
            self.add_constant(op, orig_op, type_=Operator)
            _g_cache.append(op)

        # general types are not counting inpoiters
        for p in self.overnight_rules["general_types"]:
            p_func = f"( string {p} )"
            self.add_constant(p, p_func, type_=Property)
            _g_cache.append(p)

        # the order matters for schema relation
        _value_cache = sorted(_value_cache, key=lambda x: x[0])
        self._values = [x[0] for x in _value_cache]
        self._ref_values = [x[1] for x in _value_cache]
        _property_cache = sorted(_property_cache, key=lambda x: x[0])
        self._properties = [x[0] for x in _property_cache]
        self._ref_properties = [x[1] for x in _property_cache]

        self._start_type = "List[Value]"
        self._nonterminal_productions = None
        self._general_terminals = _g_cache

        self._schema_relations = self.get_schema_relations()
        self._lex_values, self._lex_properties = self.get_schema_lexicons()

    def add_constant(self, name: str, value: Any, type_: Type = None):
        value_type = type_ if type_ else type(value)
        constant_type = PredicateType.get_type(value_type)
        # avoid duplicates
        if name in self._function_types and constant_type in self._function_types[name]:
            return
        self._functions[name] = lambda: value
        self._function_types[name].append(constant_type)

    def get_non_terminal_productions(self):
        """
        the original get_nonterminal_productions returns all the productions
        """
        if not self._nonterminal_productions:
            actions = collections.defaultdict(set)
            actions[START_SYMBOL].add(f"{START_SYMBOL} -> {self._start_type}")
            for name, function_type_list in self._function_types.items():
                for function_type in function_type_list:
                    if isinstance(function_type, FunctionType):
                        actions[str(function_type)].add(f"{function_type} -> {name}")
                        return_type = function_type.return_type
                        arg_types = function_type.argument_types
                        right_side = f"[{function_type}, {', '.join(str(arg_type) for arg_type in arg_types)}]"
                        actions[str(return_type)].add(f"{return_type} -> {right_side}")
            self._nonterminal_productions = {
                key: sorted(value) for key, value in actions.items()
            }
        return self._nonterminal_productions

    def get_general_terminal_productions(self):
        g_prods = collections.defaultdict(set)
        for func in self._function_types:
            if func in self._general_terminals:
                func_types = self._function_types[func]
                for func_type in func_types:
                    prod = f"{func_type} -> {func}"
                    g_prods[str(func_type)].add(prod)
        return g_prods

    def get_domain_terminal_productions(self):
        d_prods = collections.defaultdict(set)
        for func in self._function_types:
            if func in self._ref_values or func in self._ref_properties:
                func_types = self._function_types[func]
                for func_type in func_types:
                    prod = f"{func_type} -> {func}"
                    d_prods[str(func_type)].add(prod)
        return d_prods

    @staticmethod
    def norm_value(vp: str):
        # use '#' to pack and unpack terminals
        return "#".join(vp.split())

    @staticmethod
    def denorm_value(vp: str):
        return " ".join(vp.split("#"))

    def get_values(self):
        return self._lex_values, self._ref_values

    def get_properties(self):
        return self._lex_properties, self._ref_properties

    def get_schema_lexicons(self):
        if self.domain in self.schema_lexicon_cache:
            return self.schema_lexicon_cache[self.domain]
        logger.info("Extracting schema lexicons")
        lexicon_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"grammar/yushi_overnight_grammar/{self.domain}.grammar",
        )

        def node2str(node):
            if not isinstance(node, tree.Tree):
                return node  # presumbly string
            else:
                token_list = [node.label()]
                for child in node:
                    if child not in ["\""]:
                        token_list.append(node2str(child))
            return " ".join(token_list)

        lexicons = {}
        with open(lexicon_path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("(rule"):
                    continue
                g_tree = tree.Tree.fromstring(line)
                assert g_tree.label() == "rule"
                nl_node, lf_node = g_tree[1], g_tree[2]
                nl = node2str(nl_node)
                if isinstance(lf_node[0], tree.Tree):
                    r_lf_node = lf_node[0]
                    if r_lf_node.label() == "string":
                        # omit string
                        lf = " ".join(r_lf_node)
                    else:
                        lf = node2str(r_lf_node)
                else:
                    lf = lf_node[0]
                lexicons[lf] = nl

        # TODO: make a number mapping
        lexicons["number 2"] = "two"

        lex_values = []
        for val in self._values:
            if val in lexicons:
                lex = lexicons[val]
                lex_values.append(lex)
            else:
                if val.startswith("en."):  # remove en.
                    val_ = val.split(".")[-1]
                    if val_ in lexicons:
                        lex_val_ = lexicons[val_]
                    else:
                        lex_val_ = val_
                else:
                    lex_val_ = val
                lex_values.append(lex_val_)
                logger.warn(
                    f"Value '{val}' has no lexicons in {self.domain}, map with '{lex_val_}'"
                )

        lex_props = []
        for prop in self._properties:
            if prop in lexicons:
                lex = lexicons[prop]
                lex_props.append(lex)
            else:
                lex_prop_ = " ".join(prop.split("_"))
                lex_props.append(lex_prop_)
                logger.warn(
                    f"Property '{prop}' has no lexicons in {self.domain}, map with '{lex_prop_}'"
                )
        self.schema_lexicon_cache[self.domain] = (lex_values, lex_props)
        return (lex_values, lex_props)

    def get_schema_relations(self):
        if self.domain in self.schema_relation_cache:
            return self.schema_relation_cache[self.domain]
        # print("Extracting schema relations!")

        value2id = {v: k for k, v in enumerate(self._values)}
        pro2id = {v: k for k, v in enumerate(self._properties)}

        debugs = []

        schema_relations = set()
        value_dict = self.overnight_rules["domain_terminals"][self.domain]["value"]
        for s_v in value_dict["singleton"]:
            for v in value_dict["entity"]:
                if s_v in v:
                    rel = ("val", value2id[s_v], "subtype", "val", value2id[v])
                    debugs.append((s_v, "subtype", v))
                    schema_relations.add(rel)

        template = "(call SW.listValue ( call SW.getProperty (call SW.singleton {0} ) ( string {1} ) ) )"
        triples = []
        query_list = []
        for val_type in value_dict:
            for v in value_dict[val_type]:
                for p in self._properties:
                    if v.split()[0] in self.t_classes:
                        real_v = f"( {v} )"
                    else:
                        real_v = v
                    query = template.format(real_v, p)
                    query_list.append(query)
                    triple = (
                        "val",
                        value2id[v],
                        f"{val_type}_has_p",
                        "prop",
                        pro2id[p],
                    )
                    debugs.append((v, f"{val_type}_has_p", p))
                    triples.append(triple)

        denotations = self.execute(query_list, self.domain)
        assert len(denotations) == len(triples)
        for d, t in zip(denotations, triples):
            if d.startswith("(list"):
                schema_relations.add(t)
        self.schema_relation_cache[self.domain] = schema_relations
        return schema_relations

    @predicate
    def listValue(self, values: List[Value]) -> List[Value]:
        pass

    @predicate
    def listValue(self, value: Value) -> List[Value]:
        pass

    @predicate
    def domain(self, p: Property) -> List[Value]:
        pass

    @predicate
    def singleton(self, value: SingletonValue) -> List[Value]:
        pass

    @predicate
    def reverse(self, p: Property) -> Property:
        pass

    def concat(self, l1: Value, l2: Value) -> List[Value]:
        pass

    @predicate
    def concat(self, l1: EntityValue, l2: EntityValue) -> List[Value]:
        pass

    @predicate
    def concat(self, l1: TimeValue, l2: TimeValue) -> List[Value]:
        pass

    @predicate
    def concat(self, l1: DateValue, l2: DateValue) -> List[Value]:
        pass

    @predicate
    def concat(self, l1: NumberValue, l2: NumberValue) -> List[Value]:
        pass

    def concat(self, l1: List[Value], l2: List[Value]) -> List[Value]:
        # pylint: disable=function-redefined
        pass

    @predicate
    def ensureNumericProperty(self, p: Property) -> Property:
        pass

    @predicate
    def ensureNumericEntity(self, value: NumericalValue) -> List[Value]:
        pass

    @predicate
    def ensureNumericEntity(self, values: List[Value]) -> List[Value]:
        # pylint: disable=function-redefined
        pass

    @predicate
    def filter(self, entities: List[Value], p: Property) -> List[Value]:
        pass

    @predicate
    def filter(
        self, entities: List[Value], p: Property, compare: Operator
    ) -> List[Value]:
        pass

    @predicate
    def filter(
        self,
        entities: List[Value],
        p: Property,
        compare: Operator,
        refValues: List[Value],
    ) -> List[Value]:
        pass

    @predicate
    def filter(
        self, entities: List[Value], p: Property, compare: Operator, refValue: Value
    ) -> List[Value]:
        pass

    @predicate
    def superlative(
        self, entities: List[Value], mode: Aggregate, p: Property
    ) -> List[Value]:
        pass

    @predicate
    def superlative(
        self,
        entities: List[Value],
        mode: Aggregate,
        p: Property,
        restrictors: List[Value],
    ) -> List[Value]:
        pass

    @predicate
    def countSuperlative(
        self, entities: List[Value], mode: Aggregate, p: Property
    ) -> List[Value]:
        pass

    @predicate
    def countSuperlative(
        self,
        entities: List[Value],
        mode: Aggregate,
        p: Property,
        restrictors: List[Value],
    ) -> List[Value]:
        pass

    @predicate
    def countComparative(
        self, entities: List[Value], p: Property, op: Operator, threshhold: NumberValue
    ) -> List[Value]:
        pass

    @predicate
    def countComparative(
        self,
        entities: List[Value],
        p: Property,
        op: Operator,
        threshhold: NumberValue,
        restrictors: List[Value],
    ) -> List[Value]:
        pass

    @predicate
    def getProperty(self, objects: List[Value], p: Property) -> List[Value]:
        pass

    @predicate
    def getProperty(self, object: Value, p: Property) -> List[Value]:
        pass

    @predicate
    def _size(self, values: List[Value]) -> Value:
        pass

    @predicate
    def aggregate(self, mode: Aggregate, values: List[Value]) -> List[Value]:
        pass

    def action_seq_to_raw_lf(self, actions):
        _lf = self.action_sequence_to_logical_form(actions)
        lf = self.denormalize_lf(_lf)
        return lf

    def _get_transitions(
        self, expression: Any, expected_type: PredicateType
    ) -> Tuple[List[str], PredicateType]:
        """
        Adapt to support multiple function types
        """
        if isinstance(expression, (list, tuple)):
            ret_transition, ret_type = None, None
            for (
                function_transitions,
                return_type,
                argument_types,
            ) in self._get_function_transitions(expression[0], expected_type):
                try:
                    if len(argument_types) != len(expression[1:]):
                        raise ParsingError(
                            f"Wrong number of arguments for function in {expression}"
                        )
                    argument_transitions = []
                    for argument_type, subexpression in zip(
                        argument_types, expression[1:]
                    ):
                        argument_transitions.extend(
                            self._get_transitions(subexpression, argument_type)[0]
                        )
                    ret_transition, ret_type = (
                        function_transitions + argument_transitions,
                        return_type,
                    )
                except Exception as e:
                    logging.debug(e)
                    continue
            if ret_transition is None:
                raise ParsingError("Function parsing error")
            else:
                return ret_transition, ret_type

        elif isinstance(expression, str):
            if expression not in self._functions:
                raise ParsingError(f"Unrecognized constant: {expression}")
            constant_types = self._function_types[expression]
            if len(constant_types) == 1:
                constant_type = constant_types[0]
                # This constant had only one type; that's the easy case.
                if expected_type and expected_type != constant_type:
                    raise ParsingError(
                        f"{expression} did not have expected type {expected_type} "
                        f"(found {constant_type})"
                    )
                return [f"{constant_type} -> {expression}"], constant_type
            else:
                if not expected_type:
                    raise ParsingError(
                        "With no expected type and multiple types to pick from "
                        f"I don't know what type to use (constant was {expression})"
                    )
                if expected_type not in constant_types:
                    raise ParsingError(
                        f"{expression} did not have expected type {expected_type} "
                        f"(found these options: {constant_types}; none matched)"
                    )
                return [f"{expected_type} -> {expression}"], expected_type

        else:
            raise ParsingError(
                "Not sure how you got here. Please open an issue on github with details."
            )

    def _get_function_transitions(
        self, expression: Union[str, List], expected_type: PredicateType
    ) -> Tuple[List[str], PredicateType, List[PredicateType]]:
        if expression in self._functions:
            name = expression
            function_types = self._function_types[expression]
            for function_type in function_types:
                transitions = [f"{function_type} -> {name}"]
                argument_types = function_type.argument_types
                return_type = function_type.return_type
                right_side = f'[{function_type}, {", ".join(str(arg) for arg in argument_types)}]'
                first_transition = f"{return_type} -> {right_side}"
                transitions.insert(0, first_transition)
                if expected_type and expected_type != return_type:
                    raise ParsingError(
                        f"{expression} did not have expected type {expected_type} "
                        f"(found {return_type})"
                    )
                yield transitions, return_type, argument_types
        else:
            if isinstance(expression, str):
                raise ParsingError(f"Unrecognized function: {expression[0]}")
            else:
                raise ParsingError(f"Unsupported expression type: {expression}")
        if not isinstance(function_type, FunctionType):
            raise ParsingError(
                f"Zero-arg function or constant called with arguments: {name}"
            )

    def normalize_lf(self, raw_lf: str):
        """
        1. predicates and entities are realized by string in simpleword.java, we use property, value, string to differentiate them
        2. remove the terminal functions like string, number
        """
        lf_tree = tree.Tree.fromstring(raw_lf)

        def normalize(node):
            if isinstance(node, tree.Tree) and node.label() == "call":
                # map the SW functions to our defined ones
                if node[0].startswith("SW."):
                    node.set_label(node[0][3:])
                elif node[0].startswith("."):  # .size
                    node.set_label("_" + node[0][1:])  # _size
                else:
                    raise NotImplementedError
                node.remove(node[0])  # remove call

                for index, child in enumerate(node):
                    if (
                        isinstance(child, tree.Tree)
                        and child.label() in self.overnight_rules["terminal_types"]
                    ):
                        # particularly handle the terminals of properties
                        raw_child_str = " ".join(child)
                        norm_child_str = self.norm_value(raw_child_str)
                        node[index] = norm_child_str
                    elif isinstance(child, tree.Tree) and child.label() == "":
                        assert len(child) == 2

                        # replace the lambda with its first child
                        child[0] = child[0][1]
                        assert child[0][1].leaves() == ["s"]
                        # replace the variable
                        child[0][1] = child[1]
                        # replace the lambda node with new grounded node
                        node[index] = child[0]
                        normalize(node[index])
                    else:
                        normalize(child)

        normalize(lf_tree)
        normalized_lf = to_lisp_like_string(lf_tree)
        return normalized_lf

    def denormalize_lf(self, raw_lf: str):
        """
        Opposite of normalize_lf
        """
        lf_tree = tree.Tree.fromstring(raw_lf)

        def denormalize(node):
            if (
                isinstance(node, tree.Tree)
                and node.label() in self._general_function_types
            ):
                # change the way of calling functions
                if node.label().startswith("_"):  # _size
                    real_label = "." + node.label()[1:]
                else:
                    real_label = "SW." + node.label()
                node.set_label("call")
                node.insert(0, real_label)

                for index, child in enumerate(node):
                    if index == 0:
                        continue  # func name
                    if not isinstance(child, tree.Tree):
                        node[index] = self._functions[child]()
                    else:
                        denormalize(child)

        def to_lisp_like_string(node):
            if isinstance(node, tree.Tree):
                return f"( {node.label()} {' '.join([to_lisp_like_string(child) for child in node])} )"
            else:
                return node

        denormalize(lf_tree)
        denormalized_lf = to_lisp_like_string(lf_tree)
        return denormalized_lf

    @staticmethod
    def tostring(tree: tree.Tree) -> str:
        pass

    @staticmethod
    def execute(lfs, domain):
        def post_process(lf):
            replacements = [("SW", "edu.stanford.nlp.sempre.overnight.SimpleWorld")]
            for a, b in replacements:
                lf = lf.replace(a, b)
            return lf

        eval_path = "third_party/overnight"  # TODO: change this
        cur_dir = os.getcwd()
        os.chdir(eval_path)
        eval_script = "./evaluator/overnight"

        tf = tempfile.NamedTemporaryFile(suffix=".examples")
        for lf in lfs:
            p_lf = post_process(lf)
            tf.write(str.encode(p_lf + "\n"))
            tf.flush()
        FNULL = open(os.devnull, "w")
        msg = subprocess.check_output([eval_script, domain, tf.name], stderr=FNULL)
        tf.close()
        msg = msg.decode("utf-8")

        denotations = [
            line.split("\t")[1]
            for line in msg.split("\n")
            if line.startswith("targetValue\t")
        ]
        os.chdir(cur_dir)
        return denotations
