import logging
import itertools
from nltk.tree import Tree
from collections import defaultdict
from typing import List, Dict, Set

from tensor2struct.languages.dsl.common import (
    ParsingError,
    ExecutionError,
    START_SYMBOL,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ActionSpaceWalker:
    """
    ``ActionSpaceWalker`` takes a world, traverses all the valid paths driven by the valid action
    specification of the world to generate all possible logical forms (under some constraints). This
    class also has some utilities for indexing logical forms to efficiently retrieve required
    subsets.

    Parameters
    ----------
    world : ``World``
        The world from which valid actions will be taken.
    max_path_length : ``int``
        The maximum path length till which the action space will be explored. Paths longer than this
        length will be discarded.
    """

    def __init__(self) -> None:
        self._language = language

        self._row_selection_threshold = 32
        self._program_threshold = 1e4

    def _walk(self, start_types: List, actions: Dict, max_len: int) -> None:
        """
        Walk over action space to collect completed paths of at most ``_max_path_length`` steps.
        """
        # non_terminal_buffer, history
        incomplete_paths = [
            ([str(type_)], [f"{START_SYMBOL} -> {type_}"]) for type_ in start_types
        ]
        _completed_paths = []

        while incomplete_paths:
            next_paths = []
            # expand on terminal at a time
            for nonterminal_buffer, history in incomplete_paths:
                nonterminal = nonterminal_buffer.pop()
                next_actions = []
                if nonterminal not in actions:
                    # e.g. str is not in non_terminals
                    continue
                else:
                    next_actions.extend(actions[nonterminal])
                for action in next_actions:
                    new_history = history + [action]
                    new_nonterminal_buffer = nonterminal_buffer[:]
                    # expand the arguments first
                    for right_side_part in reversed(self._get_right_side_parts(action)):
                        if right_side_part in actions:
                            new_nonterminal_buffer.append(right_side_part)
                    next_paths.append((new_nonterminal_buffer, new_history))
            incomplete_paths = []

            for nonterminal_buffer, path in next_paths:
                if not nonterminal_buffer:
                    _completed_paths.append(path)
                elif len(path) <= max_len:
                    incomplete_paths.append((nonterminal_buffer, path))

            # avoid expensive search
            if len(_completed_paths) > self._program_threshold:
                break
        return _completed_paths

    @staticmethod
    def _get_right_side_parts(action: str) -> List[str]:
        _, right_side = action.split(" -> ")
        if "[" == right_side[0]:
            right_side_parts = right_side[1:-1].split(", ")
        else:
            right_side_parts = [right_side]
        return right_side_parts

    def prune_sketches(self, sketches: List) -> List:
        """
        Remove the dummy sketch
        """
        ret_sketches = []
        for sketch in sketches:
            sketch_lf = self._language.action_sequence_to_logical_form(sketch)
            if sketch_lf == "#PH#":
                continue

            ret_sketches.append(sketch)

        return ret_sketches

    def prune_single_row_selection(self, row_selection_cache: List):
        candidates = []
        for sin_ac in row_selection_cache:
            sin_ac_lf = self._language.action_sequence_to_logical_form(sin_ac)
            if sin_ac_lf.startswith("(filter"):  # allow all filter functions
                candidates.append(sin_ac)
        return candidates

    def prune_row_selections_by_kg(self, row_selection_cache: List):
        """
        1) Filter in should comply with table
        2) And takes two different expressions
        """
        kg = self._language.real_cv_map

        cond_cache = []
        for cond_candidate in row_selection_cache:
            r_lf = self._language.action_sequence_to_logical_form(cond_candidate)

            if r_lf == "all_rows":
                cond_cache.append(r_lf)
                continue
            r_lf_tree = Tree.fromstring(r_lf)

            filter_in_conds = []
            filter_conds = []
            queue = [r_lf_tree]
            while queue:
                item = queue.pop()
                if item.label() == "filter_in":
                    filter_in_conds.append((item[0], item[1]))
                    filter_conds.append((item[0], item[1]))
                elif item.label() == "conjunction":
                    queue.append(item[0])
                    queue.append(item[1])
                elif item.label().startswith("filter"):
                    filter_conds.append((item[0], item[1]))
                else:
                    raise NotImplementedError

            # duplicate conditions
            if len(set(filter_conds)) != len(filter_conds):
                continue

            # filter conds not right
            flag = True
            for filter_in_cond in filter_in_conds:
                col, val = filter_in_cond
                if val not in kg[col]:
                    flag = False
                    break
            if not flag:
                continue

            cond_cache.append(cond_candidate)
        return cond_cache

    def get_sketches(self, sketch_actions: Dict, max_sketch_len: int) -> List:
        sketches = self._walk(
            self._language._start_types, sketch_actions, max_sketch_len
        )
        sketches = self.prune_sketches(sketches)
        # print(f"{len(sketches)} sketches in total")
        return sketches

    def get_junctions(self, single_selections: List) -> List:
        prefix_2 = [
            "List[Row] -> [<List[Row],List<Row>:List[Row]>, List[Row], List[Row]]",
            "<List[Row],List<Row>:List[Row]> -> conjunction",
        ]

        candidates = single_selections

        ret_ac = []
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if i == j:
                    continue
                a1 = candidates[i]
                a2 = candidates[j]
                # conjunction
                ac = [a1[0]] + prefix_2 + a1[1:] + a2[1:]
                ret_ac.append(ac)
        return ret_ac

    def get_row_selection_cache(self, slot_actions: Dict) -> List:
        single_row_selection_cache = self._walk(["List[Row]"], slot_actions, 5)
        pruned_single_row_selections = self.prune_single_row_selection(
            single_row_selection_cache
        )

        threshold = self._row_selection_threshold
        if len(pruned_single_row_selections) > threshold:
            logger.warn(f"Found row selctions of size greater than {threshold}")
            pruned_single_row_selections = pruned_single_row_selections[
                :threshold
            ]  # pruning

        junctions = self.get_junctions(pruned_single_row_selections)
        pruned_row_selections = self.prune_row_selections_by_kg(
            pruned_single_row_selections + junctions
        )

        all_rows = self._walk(["List[Row]"], slot_actions, 1)[0]
        final_row_selection = pruned_row_selections + [all_rows]
        logger.info(f"{len(final_row_selection)} kinds of instantiations")
        return [ac[1:] for ac in final_row_selection]

    def get_action_seqs_from_sketch(
        self, slot_actions: Dict, sketch: List[str], row_selection_cache: List
    ) -> List:
        """
        check all the placeholders and then fill them
        """
        filler_dict = dict()
        for action_ind, action in enumerate(sketch):
            lhs, rhs = action.split(" -> ")
            if (
                lhs in ["Column", "StringColumn", "NumberColumn", "str", "Number"]
                and rhs == "#PH#"
            ):
                slot_candidates = slot_actions[lhs]
                if len(slot_candidates) == 0:
                    # if there is not candidates, the this sketch is not valid
                    return []
                filler_dict[action_ind] = slot_candidates
            elif lhs == "List[Row]" and rhs == "#PH#":
                if len(row_selection_cache) == 0:
                    # if there is not candidates, the this sketch is not valid
                    return []
                filler_dict[action_ind] = row_selection_cache

        # recursively fill all slots
        possible_paths = []

        def recur_find(prefix, i):
            if i == len(sketch):
                # check if action seq is valid
                possible_paths.append(prefix)
                return
            if i in filler_dict:
                for candidate in filler_dict[i]:
                    new_prefix = prefix[:]
                    if isinstance(candidate, list):
                        new_prefix += candidate
                    else:
                        new_prefix.append(candidate)
                    recur_find(new_prefix, i + 1)
            else:
                new_prefix = prefix[:]
                new_prefix.append(sketch[i])
                recur_find(new_prefix, i + 1)

        if len(filler_dict) > 0:
            recur_find([], 0)

        return possible_paths

    def get_all_logical_forms(
        self, max_path_length: int, max_num_logical_forms: int = None
    ) -> List[str]:
        actions = self._language.get_nonterminal_productions()

        _completed_paths = self._walk(
            self._language._start_types, actions, max_path_length
        )

        ret_paths = _completed_paths
        if max_num_logical_forms is not None:
            _length_sorted_paths = sorted(ret_paths, key=len)
            ret_paths = _length_sorted_paths[:max_num_logical_forms]

        logical_forms = []
        for path in ret_paths:
            try:
                lf = self._language.action_sequence_to_logical_form(path)
                logical_forms.append(lf)
            except ExecutionError:
                pass
        return logical_forms

    def get_action_seqs_by_sketches(
        self, max_path_length: int, max_num_sketches: int = None, sketches: List = None
    ) -> List[str]:
        """
        Collect action sequences by sketch
        if sketch is specified, only its instantiated programs are produced
        otherwise, use all possible sketches
        """
        actions = self._language.get_nonterminal_productions()
        sketch_actions = self._language._get_sketch_productions(actions)
        slot_actions = self._language._get_slot_productions(actions)
        row_selection_cache = self.get_row_selection_cache(slot_actions)

        if not sketches:
            assert max_path_length > 0  # otherwise it should be valid
            sketches = self.get_sketches(sketch_actions, max_path_length)
        sketches2prod = sketches
        if max_num_sketches is not None and max_num_sketches > 0:
            length_sorted_sketches = sorted(sketches2prod, key=len)
            sketches2prod = length_sorted_sketches[:max_num_sketches]

        ret_action_seqs = []
        for sketch in sketches2prod:
            action_seqs = self.get_action_seqs_from_sketch(
                slot_actions, sketch, row_selection_cache
            )
            for _seq in action_seqs:
                # make sure all palceholders arer resolved
                assert "#PH#" not in " ".join(_seq)
                ret_action_seqs.append((sketch, _seq))

        return ret_action_seqs

    def get_logical_forms_by_sketches(
        self, max_path_length: int, max_num_sketches: int = None, sketches: List = None
    ) -> List[str]:
        action_seqs = self.get_action_seqs_by_sketches(
            max_path_length, max_num_sketches, sketches
        )
        logical_forms = []
        for _sketch, _seq in action_seqs:
            try:
                lf = self._language.action_sequence_to_logical_form(_seq)
                logical_forms.append((_sketch, lf))
            except ParsingError:
                pass
        return logical_forms
