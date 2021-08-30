import re
import attr
from nltk import PCFG
from nltk.tree import Tree as NLTKTree
from nltk.parse import DependencyGraph
from nltk.parse.pchart import InsideChartParser

from tensor2struct.languages.dsl import cogs_pcfg
from tensor2struct.utils import registry, tree_kernels, tree


@attr.s
class Node:
    raw_token = attr.ib(default="NULL_token")
    constant = attr.ib(default="NULL_constant")
    head_idx = attr.ib(default=0)
    edge_label = attr.ib(default="NULL_edge")

    def __str__(self):
        # return f"{self.raw_token}\t{self.constant}\t{self.head_idx}\t{self.edge_label}"
        return f"{self.constant}\t{self.raw_token}\t{self.head_idx}\t{self.edge_label}"


@registry.register("grammar", "cogs")
class CogsGrammar:
    def __init__(self, kernel_name="sst", lamb=1.0, mu=1.0):
        # parsing nl
        self.nl_parser = InsideChartParser(cogs_pcfg.main_grammar)
        self.nl_candidate_parsers = [
            InsideChartParser(g) for g in cogs_pcfg.candidate_grammars
        ]

        # parsing logical forms
        self.pattern = "|".join(map(re.escape, [";", "AND"]))

        if kernel_name == "sst":
            self._kernel = tree_kernels.KernelSST(lamb)
        elif kernel_name == "st":
            self._kernel = tree_kernels.KernelST(lamb)
        elif kernel_name == "pt":
            self._kernel = tree_kernels.KernelPT(lamb, mu)
        else:
            raise NotImplementedError

    def parse_nl(self, nl):
        tokens = nl.split()

        # remove punctuation
        if tokens[-1] in ["."]:
            tokens = tokens[:-1]

        # lowercase the first word
        if tokens[0] in ["A", "The"]:
            tokens[0] = tokens[0].lower()
        assert len(tokens) > 0

        # primitives
        if len(tokens) == 1:
            # TODO: add pre-terminal to primitives
            return NLTKTree.fromstring(f"({tokens[0]})")

        # try main parser
        parses = None
        try:
            parses = self.nl_parser.parse(tokens)
        except ValueError:
            for c_parser in self.nl_candidate_parsers:
                try:
                    parses = c_parser.parse(tokens)
                    break
                except ValueError:
                    continue

        if parses is None:
            raise ValueError("found no parses from COGS grammars")

        parses = list(parses)
        assert len(parses) > 0
        return parses[0]

    def parse(self, nl, code):
        """
        Output NLTK tree representation of logical forms on COGS. As the logical forms
        are converted from dependency trees, so the resulting representation is also a DependencyGraph.
        """
        raw_tokens = nl.lower().split()
        chunks = re.split(self.pattern, code)

        # atomic
        if len(raw_tokens) == 1:
            return NLTKTree.fromstring(f"({nl.lower()})")

        nodes = [Node(t) for t in raw_tokens]

        def resolve_arg(arg):
            if arg[-1].isdigit():
                idx = int(arg[-1])
            else:
                idx = raw_tokens.index(arg[0].lower())
                nodes[idx].constant = arg[0].lower()  # add to constant
                assert idx >= 0
            return idx

        for chunk in chunks:
            chunk = chunk.strip().split()
            left_para = chunk.index("(")
            right_para = chunk.index(")")

            if "." in chunk[:left_para]:
                dot_pos = chunk[:left_para].index(".")  # the first dot
                constant = "".join(chunk[:dot_pos])
                edge_label = "".join(chunk[dot_pos + 1 : left_para])

                nums = []
                split = chunk[left_para + 1 : right_para].index(",")
                arg1 = chunk[left_para + 1 : left_para + 1 + split]
                arg2 = chunk[left_para + 1 + split + 1 : right_para]
                head_idx = resolve_arg(arg1)
                dependent_idx = resolve_arg(arg2)

                nodes[head_idx].constant = constant
                nodes[head_idx].edge_label = "ROOT"  # root
                nodes[dependent_idx].head_idx = head_idx + 1  # root
                nodes[dependent_idx].edge_label = edge_label
            else:
                constant = "".join(chunk[:left_para])

                nums = []
                for t in chunk[left_para + 1 : right_para]:
                    if t.isdigit():
                        nums.append(int(t))
                assert len(nums) == 1
                idx = nums[0]
                nodes[idx].constant = constant

        tree = self.convert_nodes_to_depgraph(nodes)
        return tree

    def convert_nodes_to_depgraph(self, nodes):
        conll_str = []
        for node in nodes:
            conll_str.append(str(node))
        conll_str = "\n".join(conll_str)

        depgraph = DependencyGraph(conll_str)
        tree = depgraph.tree()
        return tree

    def printree(self, nltk_tree):
        """
        This function converts NLTK tree to a prolog style string which
        will used for computing tree kernels
        Example "predicate(arg1, arg2)"
        """
        if isinstance(nltk_tree, NLTKTree):
            leaves = [self.printree(leave) for leave in nltk_tree]
            return f"{nltk_tree.label()}({','.join(leaves)})"
        else:
            return nltk_tree

    def kernel(self, nltk_tree1, nltk_tree2, norm=False):
        """
        Args:
            tree1, tree2: two nltk trees
            norm: whether normalize kernel
        """
        str1 = self.printree(nltk_tree1)
        str2 = self.printree(nltk_tree2)
        return self.kernel_string(str1, str2, norm)

    def kernel_string(self, str1, str2, norm):
        """
        Args:
            str1, str2: strings that obtained from printree
        """
        tree1 = tree.Tree.fromPrologString(str1)
        tree2 = tree.Tree.fromPrologString(str2)
        k = self._kernel.kernel(tree1, tree2)

        if norm and k != 0:
            denorm = (
                self._kernel.kernel(tree1, tree1) * self._kernel.kernel(tree2, tree2)
            ) ** (0.5)
            k = k / denorm
        return k
