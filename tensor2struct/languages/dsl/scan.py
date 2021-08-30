import attr
import logging
import random
from nltk import CFG
from nltk.tree import Tree as NLTKTree
from nltk.parse import RecursiveDescentParser, ChartParser, generate
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union

from tensor2struct.utils import registry, tree_kernels, tree
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
    predicate,
)

logger = logging.getLogger("tensor2struct")
logger.setLevel(logging.ERROR)


class PP:
    pass


class Direction:
    pass


class Action:
    pass


class ActionLang(DomainLanguage):
    """
    Support list-like logical form:
        (twice (walk opposite left))
    """

    @predicate
    def twice(self, seq: List[Action]) -> List[Action]:
        return seq * 2

    @predicate
    def thrice(self, seq: List[Action]) -> List[Action]:
        return seq * 3

    @predicate
    def _and(self, seq1: List[Action], seq2: List[Action]) -> List[Action]:
        return seq1 + seq2

    @predicate
    def after(self, seq1: List[Action], seq2: List[Action]) -> List[Action]:
        return seq2 + seq1

    @predicate
    def walk_pd(self, p: PP, d: Direction) -> List[Action]:
        res = []
        if d == "left":
            d_action = "TURN_LEFT"
        else:
            assert d == "right"
            d_action = "TURN_RIGHT"

        if p == "opposite":
            res = [d_action, d_action, "WALK"]
        else:
            assert p == "around"
            res = [d_action, "WALK"] * 4
        return res

    @predicate
    def walk_d(self, d: Direction) -> List[Action]:
        if d == "left":
            return ["TURN_LEFT", "WALK"]
        else:
            assert d == "right"
            return ["TURN_RIGHT", "WALK"]

    @predicate
    def walk(self) -> List[Action]:
        return ["WALK"]

    @predicate
    def run_pd(self, p: PP, d: Direction) -> List[Action]:
        res = []
        if d == "left":
            d_action = "TURN_LEFT"
        else:
            assert d == "right"
            d_action = "TURN_RIGHT"

        if p == "opposite":
            res = [d_action, d_action, "RUN"]
        else:
            assert p == "around"
            res = [d_action, "RUN"] * 4
        return res

    @predicate
    def run_d(self, d: Direction) -> List[Action]:
        if d == "left":
            return ["TURN_LEFT", "RUN"]
        else:
            assert d == "right"
            return ["TURN_RIGHT", "RUN"]

    @predicate
    def run(self) -> List[Action]:
        return ["RUN"]

    @predicate
    def look_pd(self, p: PP, d: Direction) -> List[Action]:
        res = []
        if d == "left":
            d_action = "TURN_LEFT"
        else:
            assert d == "right"
            d_action = "TURN_RIGHT"

        if p == "opposite":
            res = [d_action, d_action, "LOOK"]
        else:
            assert p == "around"
            res = [d_action, "LOOK"] * 4
        return res

    @predicate
    def look_d(self, d: Direction) -> List[Action]:
        if d == "left":
            return ["TURN_LEFT", "LOOK"]
        else:
            assert d == "right"
            return ["TURN_RIGHT", "LOOK"]

    @predicate
    def look(self) -> List[Action]:
        return ["LOOK"]

    @predicate
    def jump_pd(self, p: PP, d: Direction) -> List[Action]:
        res = []
        if d == "left":
            d_action = "TURN_LEFT"
        else:
            assert d == "right"
            d_action = "TURN_RIGHT"

        if p == "opposite":
            res = [d_action, d_action, "JUMP"]
        else:
            assert p == "around"
            res = [d_action, "JUMP"] * 4
        return res

    @predicate
    def jump_d(self, d: Direction) -> List[Action]:
        if d == "left":
            return ["TURN_LEFT", "JUMP"]
        else:
            assert d == "right"
            return ["TURN_RIGHT", "JUMP"]

    @predicate
    def jump(self) -> List[Action]:
        return ["JUMP"]

    @predicate
    def turn_pd(self, p: PP, d: Direction) -> List[Action]:
        res = []
        if d == "left":
            d_action = "TURN_LEFT"
        else:
            assert d == "right"
            d_action = "TURN_RIGHT"

        if p == "opposite":
            res = [d_action, d_action]
        else:
            assert p == "around"
            res = [d_action] * 4
        return res

    @predicate
    def turn_d(self, d: Direction) -> List[Action]:
        if d == "left":
            return ["TURN_LEFT"]
        else:
            assert d == "right"
            return ["TURN_RIGHT"]

    def execute(self, logical_form: str):
        logical_form = logical_form.replace(",", " ")
        logical_form = logical_form.replace("and", "_and")
        expression = util.lisp_to_nested_expression(logical_form)
        self.manual_dispatch(expression)
        return self._execute_expression(expression)

    def manual_dispatch(self, expression):
        if isinstance(expression, list):
            if expression[0] in ["walk", "jump", "run", "look", "turn"]:
                if len(expression) == 2:
                    expression[0] += "_d"
                elif len(expression) == 3:
                    expression[0] += "_pd"
            else:
                for child in expression:
                    self.manual_dispatch(child)

    def is_nonterminal(self, symbol):
        return symbol in [
            "jump",
            "walk",
            "look",
            "run",
            "turn",
            "left",
            "right",
            "opposite",
            "around",
            "twice",
            "thrice",
            "and",
            "after",
        ]


CONSTANTS = {
    "left": Direction,
    "right": Direction,
    "opposite": PP,
    "around": PP,
}


@registry.register("grammar", "scan")
class ScanGrammar:
    def __init__(self):
        command_grammar = """
            S -> C
            S -> C AA C
            AA -> 'and' | 'after'
            C -> V | V T
            T -> 'twice' | 'thrice'
            V -> U | U D | U OA D
            OA -> "opposite" | "around"
            D -> "left" | "right"
            U -> "walk" | "look" | "run" | "jump" | "turn"
        """
        self.command_grammar = CFG.fromstring(command_grammar)
        # self.command_parser = RecursiveDescentParser(self.command_grammar)
        self.command_parser = ChartParser(self.command_grammar)

        self.action_lang = ActionLang(start_types=set([List[Action]]))
        for value, type_ in CONSTANTS.items():
            self.action_lang.add_constant(value, value, type_=type_)

        action_grammar = """
            S -> C
            S -> AA C C
            AA -> 'and' | 'after'
            C -> V | T V
            T -> 'twice' | 'thrice'
            V -> U | U D | U OA D
            OA -> "opposite" | "around"
            D -> "left" | "right"
            U -> "walk" | "look" | "run" | "jump" | "turn"
        """
        self.action_grammar = CFG.fromstring(action_grammar)
        self.action_parser = RecursiveDescentParser(self.action_grammar)

    def parse(self, command, code):
        """
        1) find lf from NL trees
        2) check if lf is correct by executing it
        3) remove bracketing of lf and tokenize it
        """
        parse = self.parse_command(command)
        lf = self.translate_command_to_programs(parse)
        assert self.action_lang.execute(lf) == code.split()
        unbracketed_lf = self.remove_bracketing(lf)
        assert self.add_bracketing(unbracketed_lf) == lf
        assert len(unbracketed_lf.split()) == len(command.split())
        return unbracketed_lf

    def sample(self, num_samples):
        samples_gen = generate.generate(self.action_grammar, depth=6)
        all_samples = list(samples_gen)
        assert len(all_samples) >= num_samples
        random.shuffle(all_samples)

        ret_examples = []
        for sample in all_samples:
            lf = self.add_bracketing(" ".join(sample))
            try:
                action_seqs = self.action_lang.execute(lf)
            except ExecutionError:
                # some bad cases, e.g., (thrice (turn))
                continue

            ret_examples.append((sample, action_seqs))
            if len(ret_examples) >= num_samples:
                break
        return ret_examples

    def remove_bracketing(self, lf):
        """
        Remove bracketing is safe for SCAN
        """
        unbracketed_lf = lf.replace("(", "").replace(")", "")
        return unbracketed_lf

    def add_bracketing(self, unbracketed_lf):
        """
        Convert unbracketed LF to bracketed ones which are executable
        """
        parses = self.action_parser.parse(unbracketed_lf.split())
        parse = list(parses)[0]

        def parse_s(tree):
            if len(tree) == 1:  # S -> C
                return parse_c(tree[0])
            elif len(tree) == 3:  # S -> AA C C
                lc, rc = parse_c(tree[1]), parse_c(tree[2])
                return f"({tree[0][0]} {lc} {rc})"

        def parse_c(tree):
            if len(tree) == 1:  # C -> V:
                return parse_v(tree[0])
            else:  # C -> T V
                v = parse_v(tree[1])
                return f"({tree[0][0]} {v})"

        def parse_v(tree):
            if len(tree) == 1:  # V -> U
                assert tree[0].label() == "U"
                token = tree[0][0]
                return f"({token})"
            elif len(tree) == 2:  # V -> U D
                token1 = tree[0][0]
                token2 = tree[1][0]
                return f"({token1} {token2})"
            elif len(tree) == 3:
                token1 = tree[0][0]  # UA
                token2 = tree[1][0]  # OA
                token3 = tree[2][0]  # D
                return f"({token1} {token2} {token3})"

        program = parse_s(parse)
        return program

    def parse_command(self, command):
        """
        Convert NL command to a CFG tree
        """
        tokens = command.split()
        parses = list(self.command_parser.parse(tokens))
        assert len(parses) > 0
        return parses[0]

    def kernel(self, nltk_tree1, nltk_tree2, norm=False):
        """
        Args:
            tree1, tree2: two nltk trees
        """

        def printree(nltk_tree):
            """
            print nltk tree into prolog style
            Example "predicate(arg1, arg2)"
            """
            if isinstance(nltk_tree, NLTKTree):
                leaves = [printree(leave) for leave in nltk_tree]
                return f"{nltk_tree.label()}({','.join(leaves)})"
            else:
                return nltk_tree

        def kernel_string(str1, str2, norm):
            """
            Args:
                str1, str2: strings that obtained from printree
            """
            tree1 = tree.Tree.fromPrologString(str1)
            tree2 = tree.Tree.fromPrologString(str2)
            sst_kernel = tree_kernels.KernelSST(1.0)
            k = sst_kernel.kernel(tree1, tree2)

            if norm and k != 0:
                denorm = (
                    sst_kernel.kernel(tree1, tree1) * sst_kernel.kernel(tree2, tree2)
                ) ** (0.5)
                k = k / denorm
            return k

        str1 = printree(nltk_tree1)
        str2 = printree(nltk_tree2)
        return kernel_string(str1, str2, norm)

    def translate_command_to_programs(self, parse):
        def parse_s(tree):
            if len(tree) == 1:  # S -> C
                return parse_c(tree[0])
            elif len(tree) == 3:  # S -> C AA C
                lc, rc = parse_c(tree[0]), parse_c(tree[2])
                return f"({tree[1][0]} {lc} {rc})"

        def parse_c(tree):
            if len(tree) == 1:  # C -> V:
                return parse_v(tree[0])
            else:  # C -> V T
                v = parse_v(tree[0])
                return f"({tree[1][0]} {v})"

        def parse_v(tree):
            if len(tree) == 1:  # V -> U
                assert tree[0].label() == "U"
                token = tree[0][0]
                return f"({token})"
            elif len(tree) == 2:  # V -> U D
                token1 = tree[0][0]
                token2 = tree[1][0]
                return f"({token1} {token2})"
            elif len(tree) == 3:
                token1 = tree[0][0]  # UA
                token2 = tree[1][0]  # OA
                token3 = tree[2][0]  # D
                return f"({token1} {token2} {token3})"

        program = parse_s(parse)
        return program
