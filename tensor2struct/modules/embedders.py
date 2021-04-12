import abc
import os
import pickle

import numpy as np
import torch
import torchtext
from torch import nn

import stanza
from spacy_stanza import StanzaLanguage
from tensor2struct.utils import registry
from tensor2struct.utils import batched_sequence

import logging

logger = logging.getLogger("tensor2struct")


class Embedder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def tokenize(self, sentence):
        """Given a string, return a list of tokens suitable for lookup."""
        pass

    @abc.abstractmethod
    def untokenize(self, tokens):
        """Undo tokenize."""
        pass

    @abc.abstractmethod
    def lookup(self, token):
        """Given a token, return a vector embedding if token is in vocabulary.

        If token is not in the vocabulary, then return None."""
        pass

    @abc.abstractmethod
    def contains(self, token):
        pass

    @abc.abstractmethod
    def to(self, device):
        """Transfer the pretrained embeddings to the given device."""
        pass


@registry.register("word_emb", "glove_spacy")
class GloVe(Embedder):
    def __init__(self, kind, lemmatize=False):
        cache = os.path.join(os.environ.get("CACHE_DIR", os.getcwd()), ".vector_cache")
        self.glove = torchtext.vocab.GloVe(name=kind, cache=cache)
        self.dim = self.glove.dim
        self.vectors = self.glove.vectors
        self.lemmatize = lemmatize

        self.sp_nlp = None  # lazy initialization

    def tokenize(self, text):
        # might be better to keep the Token object to serve string matching
        if self.sp_nlp is None:
            # self.sp_nlp = spacy.load("en_core_web_sm")
            snlp = stanza.Pipeline(lang="en", use_gpu=True)
            self.sp_nlp = StanzaLanguage(snlp)
        tokens = self.sp_nlp(text)
        if self.lemmatize:
            return [tok.lemma_.lower() for tok in tokens]
        else:
            return [tok.text.lower() for tok in tokens]

    def tokenize_for_copying(self, text):
        tokens = self.tokenize(text)
        tokens_for_copying = tokens  # TODO, change this
        return tokens, tokens_for_copying

    def untokenize(self, tokens):
        return " ".join(tokens)

    def lookup(self, token):
        i = self.glove.stoi.get(token)
        if i is None:
            return None
        return self.vectors[i]

    def contains(self, token):
        return token in self.glove.stoi

    def to(self, device):
        self.vectors = self.vectors.to(device)


@registry.register("word_emb", "ch_baidu_embedder")
class BaiduEmbedder(Embedder):
    def __init__(self):
        cache = os.path.join(os.environ.get("CACHE_DIR", os.getcwd()), ".vector_cache")
        ch_w2v_path = os.path.join(cache, "sgns.baidubaike.bigram-char.pkl")
        if os.path.exists(ch_w2v_path):
            self.id2w, self.w2id, self.vectors, self.dim = self.load_stored_vectors(
                ch_w2v_path
            )
        else:
            raw_ch_w2v_path = os.path.join(cache, "sgns.baidubaike.bigram-char")
            self.id2w, self.w2id, _vectors, self.dim = self.load_vector(raw_ch_w2v_path)
            self.vectors = torch.Tensor(_vectors)
            self.save_vectors(ch_w2v_path)

        self.sp_nlp = None

    def load_stored_vectors(self, path):
        with open(path, "rb") as f:
            ret = pickle.load(f)
        return ret

    def save_vectors(self, path):
        with open(path, "wb") as f:
            pickle.dump([self.id2w, self.w2id, self.vectors, self.dim], f)

    def load_vector(self, path):
        id2w = []
        vectors = []
        num_words, dim = None, None
        with open(path, encoding="utf-8", errors="ignore") as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    items = line.rstrip().split()
                    num_words, dim = map(int, items)
                    continue
                tokens = line.rstrip().split(" ")
                _vec = np.asarray([float(x) for x in tokens[1:]])
                vectors.append(_vec)
                id2w.append(tokens[0])
        assert len(vectors) == num_words
        w2id = {v: k for k, v in enumerate(id2w)}
        return id2w, w2id, vectors, dim

    def tokenize(self, text):
        if self.sp_nlp is None:
            snlp = stanza.Pipeline(lang="zh", use_gpu=False)
            self.sp_nlp = StanzaLanguage(snlp)
        tokens = self.sp_nlp(text)
        return [token.lemma_ for token in tokens]

    def tokenize_for_copying(self, text):
        tokens = self.tokenize(text)
        return tokens, tokens

    def untokenize(self, tokens):
        return " ".join(tokens)

    def lookup(self, token):
        if token in self.w2id:
            ind = self.w2id[token]
            return self.vectors[ind]
        else:
            return None

    def contains(self, token):
        return token in self.w2id

    def to(self, device):
        self.vectors = self.vectors.to(device)


@registry.register("word_emb", "ch_tencent_embedder")
class TencentEmbedder(BaiduEmbedder):
    def __init__(self):
        cache = os.path.join(os.environ.get("CACHE_DIR", os.getcwd()), ".vector_cache")
        ch_w2v_path = os.path.join(cache, "filtered_tencent.pkl")
        if os.path.exists(ch_w2v_path):
            self.id2w, self.w2id, self.vectors, self.dim = self.load_stored_vectors(
                ch_w2v_path
            )
        else:
            raw_ch_w2v_path = os.path.join(cache, "filtered_tencent.txt")
            self.id2w, self.w2id, _vectors, self.dim = self.load_vector(raw_ch_w2v_path)
            self.vectors = torch.Tensor(_vectors)
            self.save_vectors(ch_w2v_path)

        self.sp_nlp = None


class VanillaEmbeddings(torch.nn.Module):
    def __init__(self, device, vocab, embedder, emb_size):
        super().__init__()
        self._device = device
        self.vocab = vocab
        self.embedder = embedder
        self.emb_size = emb_size

        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.vocab), embedding_dim=emb_size
        )

    def forward_unbatched(self, tokens):
        tok_ids = [self.vocab.index(t) for t in tokens]
        tok_ids_t = torch.LongTensor(tok_ids).to(self._device)
        return self.embedding(tok_ids_t)

    def forward(self, tokens):
        return self.forward_unbatched(tokens)


class LookupEmbeddings(torch.nn.Module):
    """
    Embed 3-d or 2-d list into a packed_sequence

    Attributes:
        embedder: an embedder object that houses pretrained embeddings
        emb_size: requires the same as the size of the pretrained embeddings
        learnable_words: a set of words whose embeddings should be updated, it's assumed to be a subset of vocab
    """

    def __init__(self, device, vocab, embedder, emb_size, learnable_words=None):
        super().__init__()
        self._device = device
        self.vocab = vocab
        self.embedder = embedder
        self.emb_size = emb_size

        # trainable embeddings
        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.vocab), embedding_dim=emb_size
        )

        # init embeddings with pretrained_embeddings
        if self.embedder:
            assert emb_size == self.embedder.dim

            init_embed_list = []
            count_not_in_embed = 0
            for i, word in enumerate(self.vocab):
                if self.embedder.contains(word):
                    init_embed_list.append(self.embedder.lookup(word))
                else:
                    count_not_in_embed += 1
                    init_embed_list.append(self.embedding.weight[i])
            logger.info(f"{count_not_in_embed} words not in fixed embedder")

            init_embed_weight = torch.stack(init_embed_list, 0)
            self.embedding.weight = nn.Parameter(init_embed_weight)

        # init learnable words
        if learnable_words is None:
            self.learnable_words = set()  # empty set
        else:
            self.learnable_words = learnable_words

    def forward_unbatched_3d(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.

        embs = []
        for tokens in token_lists:
            # token_indices shape: batch (=1) x length
            token_indices = torch.tensor(
                self.vocab.indices(tokens), device=self._device
            ).unsqueeze(0)

            # emb shape: batch (=1) x length x word_emb_size
            emb = self.embedding(token_indices)

            # emb shape: desc length x batch (=1) x word_emb_size
            emb = emb.transpose(0, 1)
            embs.append(emb)

        # all_embs shape: sum of desc lengths x batch (=1) x word_emb_size
        all_embs = torch.cat(embs, dim=0)

        # boundaries shape: num of descs + 1
        # If desc lengths are [2, 3, 4],
        # then boundaries is [0, 2, 5, 9]
        boundaries = np.cumsum([0] + [emb.shape[0] for emb in embs])

        return all_embs, boundaries

    def _compute_boundaries(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.
        boundaries = [
            np.cumsum([0] + [len(token_list) for token_list in token_lists_for_item])
            for token_lists_for_item in token_lists
        ]

        return boundaries

    def _embed_token(self, token, batch_idx=None):
        if (
            (token in self.learnable_words)
            or (self.embedder is None)
            or (not self.embedder.contains(token))
        ):
            return self.embedding.weight[self.vocab.index(token)]
        else:
            emb = self.embedder.lookup(token)
            return emb.to(self._device)

    def forward_batched_3d(self, token_lists):
        # token_lists: list of list of lists
        # [batch, num descs, desc length]
        # - each list contains tokens
        # - each list corresponds to a column name, table name, etc.
        # PackedSequencePlus, with shape: [batch, sum of desc lengths, emb_size]
        all_embs = batched_sequence.PackedSequencePlus.from_lists(
            lists=[
                [token for token_list in token_lists_for_item for token in token_list]
                for token_lists_for_item in token_lists
            ],
            item_shape=(self.emb_size,),
            device=self._device,
            item_to_tensor=self._embed_token,
        )
        all_embs = all_embs.apply(lambda d: d.to(self._device))

        return all_embs, self._compute_boundaries(token_lists)

    def forward_batched_2d(self, token_lists):
        all_embs = batched_sequence.PackedSequencePlus.from_lists(
            lists=[[token for token in token_list] for token_list in token_lists],
            item_shape=(self.emb_size,),
            device=self._device,
            item_to_tensor=self._embed_token,
        )
        all_embs = all_embs.apply(lambda d: d.to(self._device))

        return all_embs

    def forward(self, token_lists):
        if isinstance(token_lists[0][0], list) or isinstance(token_lists[0][0], tuple):
            return self.forward_batched_3d(token_lists)
        else:
            return self.forward_batched_2d(token_lists)
