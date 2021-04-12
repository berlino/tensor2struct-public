import einops
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from tensor2struct.utils import batched_sequence
from tensor2struct.modules import transformer


def clamp(value, abs_max):
    value = max(-abs_max, value)
    value = min(abs_max, value)
    return value


def get_attn_mask(seq_lengths):
    """attention mask for encoder"""
    max_length, batch_size = int(max(seq_lengths)), len(seq_lengths)
    attn_mask = torch.LongTensor(batch_size, max_length, max_length).fill_(0)
    for batch_idx, seq_length in enumerate(seq_lengths):
        attn_mask[batch_idx, :seq_length, :seq_length] = 1
    return attn_mask

def get_src_attn_mask(seq_lengths):
    """encoder memory attention mask for decoder, second dim is 1 for broadcasting"""
    max_length, batch_size = int(max(seq_lengths)), len(seq_lengths)
    attn_mask = torch.LongTensor(batch_size, 1, max_length).fill_(0)
    for batch_idx, seq_length in enumerate(seq_lengths):
        attn_mask[batch_idx, 0, :seq_length] = 1
    return attn_mask

def subsequent_mask(size):
    "Mask out subsequent positions for transformer decoding"
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask

class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        device,
        num_layers,
        num_heads,
        hidden_size,
        tie_layers=False,
        ff_size=None,
        dropout=0.1,
    ):
        super().__init__()
        self._device = device
        self.num_heads = num_heads

        if ff_size is None:
            ff_size = hidden_size * 4

        self.encoder = transformer.Encoder(
            lambda: transformer.EncoderLayer(
                hidden_size,
                transformer.MultiHeadedAttention(num_heads, hidden_size, dropout),
                transformer.PositionwiseFeedForward(hidden_size, ff_size, dropout),
                dropout,
            ),
            hidden_size,
            num_layers,
            tie_layers,
        )

        # init with xavier
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_unbatched(self, embeds):
        """
        Args:
            embeds: a list of tensors return by Glue layers
        """
        ret = []
        for embed in embeds:
            enc = self.encoder(embed, mask=None)
            ret.append(enc)
        return ret

    def forward(self, embeds):
        """
        Args:
            embeds: a pair of tensor and lengths
        """
        enc_memory, lengths = embeds
        attn_mask = get_attn_mask(lengths).to(self._device)
        new_enc_memory = self.encoder(enc_memory, mask=attn_mask)
        return new_enc_memory, lengths

class TransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        device,
        num_layers,
        num_heads,
        hidden_size,
        ff_size=None,
        dropout=0.1,
    ):
        super().__init__()
        self._device = device
        self.num_heads = num_heads

        if ff_size is None:
            ff_size = hidden_size * 4

        self.decoder = transformer.Decoder(
            lambda: transformer.DecoderLayer(
                hidden_size,
                transformer.MultiHeadedAttention(num_heads, hidden_size, dropout),
                transformer.MultiHeadedAttention(num_heads, hidden_size, dropout),
                transformer.PositionwiseFeedForward(hidden_size, ff_size, dropout),
                dropout,
            ),
            num_layers
        )

        # init with xavier
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, src_memory, src_mask, tgt_mask):
        tgt_enc = self.decoder(tgt, src_memory, src_mask, tgt_mask)
        return tgt_enc

class PadCLS(torch.nn.Module):
    """
    Glue layer by the following steps (in order):
    1. convert packedsequence to a list of encode with batch 1
    2. add pos encoding
    3. add CLS token as the first token
    """

    def __init__(self, device, hidden_size, pos_encode=False):
        super().__init__()
        self._device = device
        self.cls_token = nn.Parameter(
            torch.randn(1, hidden_size).to(self._device).requires_grad_()
        )

        self.pos_encode = pos_encode
        if pos_encode:
            self.pos = transformer.PositionalEncoding(hidden_size, dropout=0.1)

    def forward_unbatched(self, embeds):
        """
        Return a list of tensors (with batch size one)
        """
        encs = []
        for i in range(len(embeds.lengths)):
            enc = embeds.select(i).unsqueeze(0)
            if self.pos_encode:
                enc = self.pos(enc)
            cls_enc = torch.cat([self.cls_token.unsqueeze(0), enc], dim=1)
            encs.append(cls_enc)  # batch 1
        return encs

    def forward(self, embeds):
        enc_memory, lengths = embeds.pad(batch_first=True)

        # add pos embedding
        if self.pos_encode:
            enc_memory = self.pos(enc_memory)

        # add cls token
        cls_emb = self.cls_token.unsqueeze(0).expand(len(lengths), 1, -1)
        for i in range(len(lengths)):
            lengths[i] = lengths[i] + 1
        enc_memory = torch.cat([cls_emb, enc_memory], dim=1)
        return enc_memory, lengths


class RAT(torch.nn.Module):
    def __init__(
        self,
        device,
        num_layers,
        num_heads,
        hidden_size,
        relations2id,
        enable_latent_relations=False,
        num_latent_relations=None,
        combine_latent_relations=False,
        tie_layers=False,
        ff_size=None,
        dropout=0.1,
    ):
        super().__init__()
        self._device = device
        self.num_heads = num_heads

        if ff_size is None:
            ff_size = hidden_size * 4

        if combine_latent_relations:
            assert enable_latent_relations
        encoder_class = transformer.EncoderLayerWithLatentRelations
        self.encoder = transformer.RATEncoder(
            lambda: encoder_class(
                hidden_size,
                transformer.MultiHeadedAttentionWithRelations(
                    num_heads, hidden_size, dropout
                ),
                transformer.PositionwiseFeedForward(hidden_size, ff_size, dropout),
                relations2id,
                num_latent_relations,
                enable_latent_relations=enable_latent_relations,
                combine_latent_relations=combine_latent_relations,
            ),
            hidden_size,
            num_layers,
            tie_layers,
        )

    def forward_unbatched(self, desc, q_enc, c_enc, t_enc, relation):
        # enc shape: total len x batch (=1) x recurrent size
        enc = torch.cat((q_enc, c_enc, t_enc), dim=0)

        # enc shape: batch (=1) x total len x recurrent size
        enc = enc.transpose(0, 1)
        enc_new = self.encoder(enc, relation, mask=None)

        c_base = q_enc.shape[0]
        t_base = q_enc.shape[0] + c_enc.shape[0]
        q_enc_new = enc_new[:, :c_base]
        c_enc_new = enc_new[:, c_base:t_base]
        t_enc_new = enc_new[:, t_base:]

        return q_enc_new, c_enc_new, t_enc_new

    def forward(self, descs, q_enc, c_enc, t_enc, relations):
        return self.forward_unbatched(descs, q_enc, c_enc, t_enc, relations)


class AlignmentWithRAT(torch.nn.Module):
    """
    Yet another RAT, which compute the alignment matrix 
    """

    def __init__(
        self,
        hidden_size,
        relations2id,
        device,
        enable_latent_relations=False,
        num_latent_relations=None,
        dropout=0.1,
        combine_latent_relations=False,
    ):
        super().__init__()
        self._device = device
        self.enable_latent_relations = enable_latent_relations
        self.align_attn = transformer.PointerWithLatentRelations(
            hidden_size,
            relations2id,
            enable_latent_relations=enable_latent_relations,
            num_latent_relations=num_latent_relations,
            combine_latent_relations=combine_latent_relations,
        )

        if combine_latent_relations:
            assert enable_latent_relations

    def forward_unbatched(self, desc, q_enc, c_enc, t_enc, relation):
        enc = torch.cat((q_enc, c_enc, t_enc), dim=1)
        m2c_align_mat = self.align_attn(enc, c_enc, c_enc, relation, "column")
        m2t_align_mat = self.align_attn(enc, t_enc, t_enc, relation, "table")
        return (m2c_align_mat, m2t_align_mat)

    def forward(self, desc, q_enc, c_enc, t_enc, relations):
        return self.forward_unbatched(desc, q_enc, c_enc, t_enc, relations)


class NoOpUpdate:
    def __init__(self, device, hidden_size):
        pass

    def __call__(self, desc, q_enc, c_enc, t_enc):
        # return q_enc.transpose(0, 1), c_enc.transpose(0, 1), t_enc.transpose(0, 1)
        return q_enc, c_enc, t_enc

    def forward_unbatched(self, desc, q_enc, c_enc, t_enc):
        """
        The same interface with RAT
        return: encodings with size: length * embed_size, alignment matrix
        """
        return (q_enc.transpose(0, 1), c_enc.transpose(0, 1), t_enc.transpose(0, 1))

    def forward(self, desc, q_enc, c_enc, t_enc):
        return self.forward_unbatched(desc, q_enc, c_enc, t_enc)
