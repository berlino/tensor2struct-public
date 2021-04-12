import attr
import json
import enum
import tqdm
import heapq
import itertools
import collections

from typing import List
import pyrsistent
from tensor2struct.utils import vocab, gtrie

SPECIAL_SYMBOL = "@"


def split_bpe_subword(input_str):
    return input_str.split(SPECIAL_SYMBOL)


def is_bpe_subword(input_str):
    return SPECIAL_SYMBOL in input_str


@attr.s
class SearchState:
    class State(enum.IntEnum):
        SUBPREFIX = 0
        OTHER = 1

    tokens = attr.ib(default=None)
    prefix = attr.ib(default=None)
    pointer = attr.ib(default=None)
    state = attr.ib(default=State.OTHER)

    def add_token(self, token):
        self.prefix = self.prefix.append(token)
        self.tokens = self.tokens.delete(0)

    @property
    def cur_token(self):
        return self.tokens[self.pointer]

    @property
    def cur_pair(self):
        assert self.pointer < len(self.tokens) - 1
        pair = (self.tokens[self.pointer], self.tokens[self.pointer + 1])
        return pair

    def merge(self, token):
        self.tokens = self.tokens.delete(0)
        self.tokens = self.tokens.set(0, token)

    def is_last_token(self):
        return self.pointer == len(self.tokens) - 1

    def is_subprefix(self):
        return self.state == SearchState.State.SUBPREFIX

    def set_subprefix_state(self):
        self.state = SearchState.State.SUBPREFIX

    def unset_subprefix_state(self):
        self.state = SearchState.State.OTHER

    def finish(self):
        return tuple(self.prefix)

    def clone(self):
        other = self.__class__()
        other.tokens = self.tokens
        other.prefix = self.prefix
        other.pointer = self.pointer
        return other


class BPEncoder:
    """ 
    Word-level BPE encoding for extracting patterns/idioms 
    """

    def __init__(self, tokenize_func=None):
        self.special_symbol = SPECIAL_SYMBOL
        self.tokenize_func = tokenize_func
        self.SEG_THRESHOLD = 32

    def get_stats(self, text):
        pairs = collections.defaultdict(int)
        for sent_str in text:
            sent = self.tokenize_func(sent_str)
            for i in range(len(sent) - 1):
                pairs[sent[i], sent[i + 1]] += 1
        return pairs

    def merge_vocab(self, pair, text):
        bigram = " ".join(pair)
        merge = self.special_symbol.join(pair)

        new_text = []
        for sent_str in text:
            new_sent = sent_str.replace(bigram, merge)
            new_text.append(new_sent)
        return new_text

    def fit(self, text: List[str], num_iterations=20):
        _text = text[:]

        merge_table = []
        for _ in range(num_iterations):
            pairs = self.get_stats(_text)
            best_pair = max(pairs, key=pairs.get)
            merge_table.append(self.special_symbol.join(best_pair))

            _text = self.merge_vocab(best_pair, _text)

        self.merge_table = merge_table
        self.vocab = vocab.Vocab(
            set(itertools.chain.from_iterable([self.tokenize_func(t) for t in _text]))
        )

    def apply(self, sent_str):
        """
        Return all the possible segmentations (different from bpe) without recursion
        """
        results = []

        bpe_trie = gtrie.StringTrie(separator=self.special_symbol)
        for i, seg in enumerate(self.vocab):
            bpe_trie[seg] = i

        tokens = pyrsistent.pvector(self.tokenize_func(sent_str))
        prefix = pyrsistent.pvector()
        queue = [(0, SearchState(tokens=tokens, prefix=prefix, pointer=0))]
        while queue:
            if len(results) > self.SEG_THRESHOLD:
                break

            _, item = heapq.heappop(queue)

            if item.is_last_token():
                if not item.is_subprefix():
                    item.add_token(item.cur_token)
                    results.append(item.finish())
                continue

            # option 1: skip merge for current token
            if not item.is_subprefix():
                _item = item.clone()
                _item.add_token(item.cur_token)
                heapq.heappush(queue, (len(_item.prefix) + _item.state, _item))

            # option 2: try to merge to next word with subprefix
            pair = item.cur_pair
            merge_token = self.special_symbol.join(pair)
            if bpe_trie.has_subtrie(merge_token):
                _item = item.clone()
                _item.merge(merge_token)
                _item.set_subprefix_state()
                heapq.heappush(queue, (len(_item.prefix) + _item.state, _item))

            # option 3: try to merge to next word with prefix
            if bpe_trie.has_key(merge_token):
                _item = item.clone()
                _item.merge(merge_token)
                _item.unset_subprefix_state()
                heapq.heappush(queue, (len(_item.prefix) + _item.state, _item))

        # order-preserving set
        return sorted(set(results), key=results.index)

    def _apply_recur(self, sent_str):
        """
        Return all the possible segmentations (different from bpe)
        """
        results = []

        def recur_find(_prefix, _sent_str, _pointer):
            _sent = self.tokenize_func(_sent_str)
            _cur_token = _sent[_pointer]

            # all merges are done
            if _pointer == len(_sent) - 1:
                _new_seg = tuple(_prefix[:] + [_cur_token])
                results.append(tuple(_new_seg))
                return

            # option 1: skip merge for current token
            if _cur_token in self.vocab:
                recur_find(_prefix[:] + [_cur_token], _sent_str, _pointer + 1)

            # option 2: try to merge
            pair = (_sent[_pointer], _sent[_pointer + 1])
            merge = self.special_symbol.join(pair)
            if merge in self.vocab:
                bigram = " ".join(pair)
                _sent_str = _sent_str[:]
                _sent_str = _sent_str.replace(bigram, merge)
            else:
                _prefix = _prefix[:] + [_cur_token]
                _pointer += 1
            recur_find(_prefix, _sent_str, _pointer)

        recur_find([], sent_str[:], 0)
        return results
        # return list(set(results))


if __name__ == "__main__":
    text = ["JUMP JUMP", "WALK WALK", "WALK JUMP JUMP", "WALK WALK", "WALK JUMP"]
    bpe = BPEncoder(tokenize_func=lambda x: x.split())
    bpe.fit(text, num_iterations=3)
    print(bpe.vocab.id_to_elem)
    print(bpe.apply("WALK JUMP JUMP"))
    print(bpe.apply("JUMP JUMP WALK WALK"))
    print(bpe.apply("JUMP JUMP WALK WALK JUMP JUMP"))
