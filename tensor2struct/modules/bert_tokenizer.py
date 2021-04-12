import os
import urllib.request

import stanza
from spacy_stanza import StanzaLanguage

from transformers import AutoTokenizer
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer

import logging

logger = logging.getLogger("tensor2struct")


class BERTokenizer:
    '''
    The basic element of preprocessing would be a pretrained tokenizer.
    This is a wrapper on top of such tokenizers to support preprocessing with spacy/stanza.
    Currently support: BERT, Electra
    TODO: Robera has not been support yet because its way of hanlding space
    '''
    sp_nlp = None

    def __init__(self, version):
        # download vocab files
        cache = os.path.join(os.environ.get("CACHE_DIR", os.getcwd()), ".vector_cache")
        vocab_dir = os.path.join(cache, f"{version}")
        if not os.path.exists(vocab_dir):
            pretrained_tokenizer = AutoTokenizer.from_pretrained(version)
            pretrained_tokenizer.save_pretrained(vocab_dir)
        
        if "uncased" in version or "cased" not in version:
            lowercase = True # roberta, electra, bert-base-uncased
        else:
            lowercase = False # bert-cased
        if version.startswith("bert") or "electra" in version:
            vocab_path = os.path.join(vocab_dir, "vocab.txt") 
            self.tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=lowercase)
        elif version.startswith("roberta"):
            vocab_path = os.path.join(vocab_dir, "vocab.json")
            merge_path = os.path.join(vocab_dir, "merges.txt")
            self.tokenizer = ByteLevelBPETokenizer(vocab_path, merge_path, lowercase=lowercase)
        else:
            raise NotImplementedError
        
        self.cls_token = self.tokenizer._parameters["cls_token"]
        self.cls_token_id = self.tokenizer.token_to_id(self.cls_token)
        self.sep_token = self.tokenizer._parameters["sep_token"]
        self.sep_token_id = self.tokenizer.token_to_id(self.sep_token)
        self.pad_token = self.tokenizer._parameters["pad_token"]
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)
    
    def _encode(self, input_):
        if isinstance(input_, list) or isinstance(input_, tuple):
            encodes = self.tokenizer.encode(input_, is_pretokenized=True)
        else:
            encodes = self.tokenizer.encode(input_)
        return encodes
    
    def tokenize(self, text):
        encodes = self._encode(text)
        tokens = encodes.tokens[1:-1]
        return tokens

    def tokenize_and_lemmatize(self, text, lang="en"):
        """
        This will be used for matching 
        1) remove cls and sep
        2) lemmatize 
        """
        if BERTokenizer.sp_nlp is None:
            snlp = stanza.Pipeline(lang=lang, use_gpu=True, tokenize_pretokenized=True)
            BERTokenizer.sp_nlp = StanzaLanguage(snlp)
        encodes = self._encode(text)
        tokens = encodes.tokens[1:-1]
        norm_tokens = [t.lemma_ for t in self.sp_nlp([tokens])]
        return norm_tokens

    def tokenize_with_orig(self, text):
        """
        Tokenize but return the original chars, this would be helpful for copying operations.
        """
        # TODO: if text is a list, change accordingly how the offset is computed
        assert isinstance(text, str)
        encodes = self._encode(text)
        orig_tokens = [text[i:j] for i,j in encodes.offsets[1:-1]]
        return orig_tokens

    def tokenize_and_spacy(self, text, lang="en"):
        """
        Keep meta information from spacy, used for matching
        """
        if BERTokenizer.sp_nlp is None:
            snlp = stanza.Pipeline(lang=lang, use_gpu=True, tokenize_pretokenized=True)
            BERTokenizer.sp_nlp = StanzaLanguage(snlp)

        tokens = self.tokenizer.encode(text).tokens[1:-1]
        return self.sp_nlp([tokens])

    def check_bert_input_seq(self, toks):
        if toks[0] == self.cls_token_id and toks[-1] == self.sep_token_id:
            return True
        else:
            return False

    def pieces_to_words(self, pieces):
        """
        TODO: use general variable of prefix
        """
        words = []
        cur_word = None
        for piece in pieces:
            if piece.startswith("##"):
                cur_word = cur_word + piece[2:]
            else:
                if cur_word is not None:
                    words.append(cur_word)
                cur_word = piece
        return words

    def text_to_ids(self, sent, cls=True):
        """
        This function is primarily used convert text to bpe token ids
        """
        encs = self._encode(sent)
        if cls:
            return encs.ids
        else:
            assert encs.tokens[0] == self.cls_token
            return encs.ids[1:]  # remove CLS

    def pad_sequence_for_bert_batch(self, tokens_lists):
        """
        1) Pad with pad token
        2) Generate token_type_list
        """
        pad_id = self.pad_token_id
        max_len = max([len(it) for it in tokens_lists])
        assert max_len <= 512
        toks_ids = []
        att_masks = []
        tok_type_lists = []
        for item_toks in tokens_lists:
            padded_item_toks = item_toks + [pad_id] * (max_len - len(item_toks))
            toks_ids.append(padded_item_toks)

            _att_mask = [1] * len(item_toks) + [0] * (max_len - len(item_toks))
            att_masks.append(_att_mask)

            first_sep_id = padded_item_toks.index(self.sep_token_id)
            assert first_sep_id > 0
            _tok_type_list = [0] * (first_sep_id + 1) + [1] * (
                max_len - first_sep_id - 1
            )
            tok_type_lists.append(_tok_type_list)
        return toks_ids, att_masks, tok_type_lists
