import pdb
import torch
from transformers import BertModel

from tensor2struct.modules import bert_tokenizer
from tensor2struct.utils import batched_sequence

import logging

logger = logging.getLogger("tensor2struct")


class BERTEncoder(torch.nn.Module):
    def __init__(self, device, bert_version):
        super().__init__()
        self._device = device
        self.bert_model = BertModel.from_pretrained(bert_version)
        self.tokenizer = bert_tokenizer.BERTokenizer(bert_version)

    def forward(self, tokens_list):
        """
        Remove CLS and SEP representation
        """
        token_ids_list = []
        lengths = []
        for tokens in tokens_list:
            lengths.append(len(tokens))
            ids = self.tokenizer.text_to_ids(tokens)
            token_ids_list.append(ids)

        (
            padded_token_lists,
            att_mask_lists,
            tok_type_lists,
        ) = self.tokenizer.pad_sequence_for_bert_batch(token_ids_list)

        tokens_tensor = torch.LongTensor(padded_token_lists).to(self._device)
        att_masks_tensor = torch.LongTensor(att_mask_lists).to(self._device)

        # token type is not used
        bert_output = self.bert_model(tokens_tensor, attention_mask=att_masks_tensor)[0]

        def map_index(batch_idx, seq_idx):
            return (batch_idx, seq_idx + 1)  # because of CLS

        def gather_from_indices(indices):
            # TODO: better indexing
            res = []
            for inds in indices:
                t = bert_output[inds]
                res.append(t)

            res = torch.stack(res, dim=0)
            return res

        packed_seq = batched_sequence.PackedSequencePlus.from_gather(
            lengths, map_index, gather_from_indices
        )
        return packed_seq


class BERT2Embed(torch.nn.Module):
    """
    Map BERT embeddings to a small vector
    """

    def __init__(self, device, bert_version, emb_size):
        super().__init__()

        if "base" in bert_version:
            input_size = 768
        else:
            input_size = 1024
        self.map = torch.nn.Linear(input_size, emb_size)

    def forward(self, packed_seq):
        return packed_seq.apply(self.map)

