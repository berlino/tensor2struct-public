import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import entmax

"""
Currently it contains three encoder layer: EncoderLayer, RATEncoderLayer, EncoderLayerWithLatentRelations 
"""

# Adapted from
# https://github.com/tensorflow/tensor2tensor/blob/0b156ac533ab53f65f44966381f6e147c7371eee/tensor2tensor/layers/common_attention.py
def relative_attention_logits(query, key, relation):
    # We can't reuse the same logic as tensor2tensor because we don't share relation vectors across the batch.
    # In this version, relation vectors are shared across heads.
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [batch, num queries, num kvs, depth].

    # qk_matmul is [batch, heads, num queries, num kvs]
    qk_matmul = torch.matmul(query, key.transpose(-2, -1))

    # q_t is [batch, num queries, heads, depth]
    q_t = query.permute(0, 2, 1, 3)

    # r_t is [batch, num queries, depth, num kvs]
    r_t = relation.transpose(-2, -1)

    #   [batch, num queries, heads, depth]
    # * [batch, num queries, depth, num kvs]
    # = [batch, num queries, heads, num kvs]
    # For each batch and query, we have a query vector per head.
    # We take its dot product with the relation vector for each kv.
    q_tr_t_matmul = torch.matmul(q_t, r_t)

    # qtr_t_matmul_t is [batch, heads, num queries, num kvs]
    q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3)

    # [batch, heads, num queries, num kvs]
    return (qk_matmul + q_tr_tmatmul_t) / math.sqrt(query.shape[-1])

    # Sharing relation vectors across batch and heads:
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [num queries, num kvs, depth].
    #
    # Then take
    # key reshaped
    #   [num queries, batch * heads, depth]
    # relation.transpose(-2, -1)
    #   [num queries, depth, num kvs]
    # and multiply them together.
    #
    # Without sharing relation vectors across heads:
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [batch, heads, num queries, num kvs, depth].
    #
    # Then take
    # key.unsqueeze(3)
    #   [batch, heads, num queries, 1, depth]
    # relation.transpose(-2, -1)
    #   [batch, heads, num queries, depth, num kvs]
    # and multiply them together:
    #   [batch, heads, num queries, 1, depth]
    # * [batch, heads, num queries, depth, num kvs]
    # = [batch, heads, num queries, 1, num kvs]
    # and squeeze
    # [batch, heads, num queries, num kvs]


def relative_attention_values(weight, value, relation):
    # In this version, relation vectors are shared across heads.
    # weight: [batch, heads, num queries, num kvs].
    # value: [batch, heads, num kvs, depth].
    # relation: [batch, num queries, num kvs, depth].

    # wv_matmul is [batch, heads, num queries, depth]
    wv_matmul = torch.matmul(weight, value)

    # w_t is [batch, num queries, heads, num kvs]
    w_t = weight.permute(0, 2, 1, 3)

    #   [batch, num queries, heads, num kvs]
    # * [batch, num queries, num kvs, depth]
    # = [batch, num queries, heads, depth]
    w_tr_matmul = torch.matmul(w_t, relation)

    # w_tr_matmul_t is [batch, heads, num queries, depth]
    w_tr_matmul_t = w_tr_matmul.permute(0, 2, 1, 3)

    return wv_matmul + w_tr_matmul_t


# Adapted from The Annotated Transformer
def clones(module_fn, N):
    return nn.ModuleList([module_fn() for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return torch.matmul(p_attn, value), scores.squeeze(1).squeeze(1)
    return torch.matmul(p_attn, value), p_attn


def sparse_attention(query, key, value, alpha, mask=None, dropout=None):
    "Use sparse activation function"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    if alpha == 2:
        p_attn = entmax.sparsemax(scores, -1)
    elif alpha == 1.5:
        p_attn = entmax.entmax15(scores, -1)
    else:
        raise NotImplementedError
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return torch.matmul(p_attn, value), scores.squeeze(1).squeeze(1)
    return torch.matmul(p_attn, value), p_attn


def attention_with_relations(
    query, key, value, relation_k, relation_v, mask=None, dropout=None
):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = relative_attention_logits(query, key, relation_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn_orig = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn_orig)
    return relative_attention_values(p_attn, value, relation_v), p_attn_orig


# Adapted from The Annotated Transformers
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(lambda: nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if query.dim() == 3:
            x = x.squeeze(1)
        return self.linears[-1](x)


# Adapted from The Annotated Transformer
class MultiHeadedAttentionWithRelations(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionWithRelations, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(lambda: nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, relation_k, relation_v, mask=None):
        # query shape: [batch, num queries, d_model]
        # key shape: [batch, num kv, d_model]
        # value shape: [batch, num kv, d_model]
        # relations_k shape: [batch, num queries, num kv, (d_model // h)]
        # relations_v shape: [batch, num queries, num kv, (d_model // h)]
        # mask shape: [batch, num queries, num kv]
        if mask is not None:
            # Same mask applied to all h heads.
            # mask shape: [batch, 1, num queries, num kv]
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # x shape: [batch, heads, num queries, depth]
        x, self.attn = attention_with_relations(
            query, key, value, relation_k, relation_v, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# Adapted from The Annotated Transformer
class RATEncoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, layer_size, N, tie_layers=False):
        super(RATEncoder, self).__init__()
        if tie_layers:
            self.layer = layer()
            self.layers = [self.layer for _ in range(N)]
        else:
            self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer_size)

        # TODO initialize using xavier

    def forward(self, x, relation, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, relation, mask)
        return self.norm(x)


# Adapted from The Annotated Transformer
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, layer_size, N, tie_layers=False):
        super(Encoder, self).__init__()
        if tie_layers:
            self.layer = layer()
            self.layers = [self.layer for _ in range(N)]
        else:
            self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer_size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# Adapted from The Annotated Transformer
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# Adapted from The Annotated Transformer
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(lambda: SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# Adapted from The Annotated Transformer
class RATEncoderLayer(nn.Module):
    "Encoder with RAT"

    def __init__(self, size, self_attn, feed_forward, num_relation_kinds, dropout):
        super(RATEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(lambda: SublayerConnection(size, dropout), 2)
        self.size = size

        self.relation_k_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)
        self.relation_v_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)

    def forward(self, x, relation, mask):
        "Follow Figure 1 (left) for connections."
        relation_k = self.relation_k_emb(relation)
        relation_v = self.relation_v_emb(relation)

        x = self.sublayer[0](
            x, lambda x: self.self_attn(x, x, x, relation_k, relation_v, mask)
        )
        return self.sublayer[1](x, self.feed_forward)


# Adapted from The Annotated Transformer
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# Adapted from The Annotated Transformer
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# Adapted from The Annotated Transformer
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(self.layers[0].size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# Adapted from The Annotated Transformer
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(lambda: SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class EncoderLayerWithLatentRelations(nn.Module):
    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        relations2id,
        num_latent_relations=3,
        dropout=0.1,
        enable_latent_relations=False,
        combine_latent_relations=False,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(lambda: SublayerConnection(size, dropout), 2)
        self.size = size
        self.relations2id = relations2id
        num_relation_kinds = len(relations2id)

        self.default_qq = "q:q-default"
        self.default_qq_id = self.relations2id[self.default_qq]
        self.relation_k_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)
        self.relation_v_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)

        self.enable_latent_relations = enable_latent_relations
        self.combine_latent_relations = combine_latent_relations
        if enable_latent_relations:
            num_qc_relations = num_latent_relations
            num_qt_relations = num_latent_relations
            num_cq_relations = num_latent_relations
            num_tq_relations = num_latent_relations
            self.latent_qc_relation_k_emb = nn.Embedding(
                num_qc_relations, self.self_attn.d_k
            )
            self.latent_qc_relation_v_emb = nn.Embedding(
                num_qc_relations, self.self_attn.d_k
            )
            self.latent_cq_relation_k_emb = nn.Embedding(
                num_cq_relations, self.self_attn.d_k
            )
            self.latent_cq_relation_v_emb = nn.Embedding(
                num_cq_relations, self.self_attn.d_k
            )
            self.latent_qt_relation_k_emb = nn.Embedding(
                num_qt_relations, self.self_attn.d_k
            )
            self.latent_qt_relation_v_emb = nn.Embedding(
                num_qt_relations, self.self_attn.d_k
            )
            self.latent_tq_relation_k_emb = nn.Embedding(
                num_tq_relations, self.self_attn.d_k
            )
            self.latent_tq_relation_v_emb = nn.Embedding(
                num_tq_relations, self.self_attn.d_k
            )

    def encode_merge_relations(self, relations):
        ct_relation, qc_relation, cq_relation, qt_relation, tq_relation = (
            relations.ct_relation,
            relations.qc_relation,
            relations.cq_relation,
            relations.qt_relation,
            relations.tq_relation,
        )

        _device = ct_relation.device
        # qq relation
        q_len = qc_relation.size(0)
        qq_relation_t = (
            torch.LongTensor(q_len, q_len).fill_(self.default_qq_id).to(_device)
        )
        qq_relation_k = self.relation_k_emb(qq_relation_t)
        qq_relation_v = self.relation_v_emb(qq_relation_t)

        # ct relation
        ct_relation_k = self.relation_k_emb(ct_relation)
        ct_relation_v = self.relation_v_emb(ct_relation)

        # qc relation
        qc_relation_k = torch.einsum(
            "qcf,fl->qcl", [qc_relation, self.latent_qc_relation_k_emb.weight]
        )
        qc_relation_v = torch.einsum(
            "qcf,fl->qcl", [qc_relation, self.latent_qc_relation_v_emb.weight]
        )

        # cq relation
        cq_relation_k = torch.einsum(
            "cqf,fl->cql", [cq_relation, self.latent_cq_relation_k_emb.weight]
        )
        cq_relation_v = torch.einsum(
            "cqf,fl->cql", [cq_relation, self.latent_cq_relation_v_emb.weight]
        )

        # qt relation
        qt_relation_k = torch.einsum(
            "qtf,fl->qtl", [qt_relation, self.latent_qt_relation_k_emb.weight]
        )
        qt_relation_v = torch.einsum(
            "qtf,fl->qtl", [qt_relation, self.latent_qt_relation_v_emb.weight]
        )

        # cq relation
        tq_relation_k = torch.einsum(
            "tqf,fl->tql", [tq_relation, self.latent_tq_relation_k_emb.weight]
        )
        tq_relation_v = torch.einsum(
            "tqf,fl->tql", [tq_relation, self.latent_tq_relation_v_emb.weight]
        )

        q_relation_k = torch.cat([qq_relation_k, qc_relation_k, qt_relation_k], 1)
        q_relation_v = torch.cat([qq_relation_v, qc_relation_v, qt_relation_v], 1)

        q_ct_relation_k = torch.cat([cq_relation_k, tq_relation_k], 0)
        q_ct_relation_v = torch.cat([cq_relation_v, tq_relation_v], 0)
        qct_relation_k = torch.cat([q_ct_relation_k, ct_relation_k], 1)
        qct_relation_v = torch.cat([q_ct_relation_v, ct_relation_v], 1)

        relation_k = torch.cat([q_relation_k, qct_relation_k], 0)
        relation_v = torch.cat([q_relation_v, qct_relation_v], 0)
        return relation_k, relation_v

    def forward(self, x, relations, mask):
        """
        x: 1 * len * feat_size
        ct_relation: ct_len * ct_len
        """
        if self.enable_latent_relations:
            relation_k_latent, relation_v_latent = self.encode_merge_relations(
                relations
            )
            if self.combine_latent_relations:
                relation_k_fixed = self.relation_k_emb(relations.predefined_relation)
                relation_v_fixed = self.relation_v_emb(relations.predefined_relation)

                relation_k = relation_k_fixed + relation_k_latent
                relation_v = relation_v_fixed + relation_v_latent
            else:
                relation_k = relation_k_latent
                relation_v = relation_v_latent
        else:
            relation_k = self.relation_k_emb(relations.predefined_relation)
            relation_v = self.relation_v_emb(relations.predefined_relation)

        relation_k, relation_v = relation_k.unsqueeze(0), relation_v.unsqueeze(0)

        x = self.sublayer[0](
            x, lambda x: self.self_attn(x, x, x, relation_k, relation_v, mask)
        )
        return self.sublayer[1](x, self.feed_forward)


class PointerWithLatentRelations(nn.Module):
    def __init__(
        self,
        hidden_size,
        relations2id,
        dropout=0.1,
        enable_latent_relations=False,
        num_latent_relations=None,
        combine_latent_relations=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.relations2id = relations2id
        num_relation_kinds = len(relations2id)
        self.linears = clones(lambda: nn.Linear(hidden_size, hidden_size), 3)
        self.dropout = nn.Dropout(p=dropout)

        self.default_qq = "q:q-default"
        self.default_qq_id = self.relations2id[self.default_qq]
        self.relation_k_emb = nn.Embedding(num_relation_kinds, hidden_size)
        self.relation_v_emb = nn.Embedding(num_relation_kinds, hidden_size)

        self.enable_latent_relations = enable_latent_relations
        self.combine_latent_relations = combine_latent_relations
        if enable_latent_relations:
            num_qc_relations = num_latent_relations
            num_qt_relations = num_latent_relations
            num_cq_relations = num_latent_relations
            num_tq_relations = num_latent_relations
            self.latent_qc_relation_k_emb = nn.Embedding(
                num_qc_relations, self.hidden_size
            )
            self.latent_qc_relation_v_emb = nn.Embedding(
                num_qc_relations, self.hidden_size
            )
            self.latent_cq_relation_k_emb = nn.Embedding(
                num_cq_relations, self.hidden_size
            )
            self.latent_cq_relation_v_emb = nn.Embedding(
                num_cq_relations, self.hidden_size
            )
            self.latent_qt_relation_k_emb = nn.Embedding(
                num_qt_relations, self.hidden_size
            )
            self.latent_qt_relation_v_emb = nn.Embedding(
                num_qt_relations, self.hidden_size
            )
            self.latent_tq_relation_k_emb = nn.Embedding(
                num_tq_relations, self.hidden_size
            )
            self.latent_tq_relation_v_emb = nn.Embedding(
                num_tq_relations, self.hidden_size
            )

    def encode_merge_column_relations(self, relations):
        ct_relation, qc_relation = (relations.ct_relation, relations.qc_relation)

        # sc relation
        t_base = relations.c_len
        sc_relation = ct_relation[:, :t_base]
        sc_relation_k = self.relation_k_emb(sc_relation)
        sc_relation_v = self.relation_v_emb(sc_relation)

        # qc relation
        qc_relation_k = torch.einsum(
            "qcf,fl->qcl", [qc_relation, self.latent_qc_relation_k_emb.weight]
        )
        qc_relation_v = torch.einsum(
            "qcf,fl->qcl", [qc_relation, self.latent_qc_relation_v_emb.weight]
        )

        mc_relation_k = torch.cat([qc_relation_k, sc_relation_k], 0)
        mc_relation_v = torch.cat([qc_relation_v, sc_relation_v], 0)
        return mc_relation_k, mc_relation_v

    def encode_merge_table_relations(self, relations):
        ct_relation, qt_relation = (relations.ct_relation, relations.qt_relation)

        # st relation
        t_base = relations.c_len
        st_relation = ct_relation[:, t_base:]
        st_relation_k = self.relation_k_emb(st_relation)
        st_relation_v = self.relation_v_emb(st_relation)

        # qt relation
        qt_relation_k = torch.einsum(
            "qtf,fl->qtl", [qt_relation, self.latent_qt_relation_k_emb.weight]
        )
        qt_relation_v = torch.einsum(
            "qtf,fl->qtl", [qt_relation, self.latent_qt_relation_v_emb.weight]
        )

        mt_relation_k = torch.cat([qt_relation_k, st_relation_k], 0)
        mt_relation_v = torch.cat([qt_relation_v, st_relation_v], 0)
        return mt_relation_k, mt_relation_v

    def get_fixed_column_relation(self, relations):
        c_base = relations.q_len
        t_base = relations.q_len + +relations.c_len
        mc_relation = relations.predefined_relation[:, c_base:t_base]  # 1 * len * len
        relation_k = self.relation_k_emb(mc_relation)
        relation_v = self.relation_v_emb(mc_relation)
        return relation_k, relation_v

    def get_fixed_table_relation(self, relations):
        t_base = relations.q_len + +relations.c_len
        mc_relation = relations.predefined_relation[:, t_base:]
        relation_k = self.relation_k_emb(mc_relation)
        relation_v = self.relation_v_emb(mc_relation)
        return relation_k, relation_v

    def get_latent_relation(self, relations, kind):
        if kind == "column":
            return self.encode_merge_column_relations(relations)
        else:
            return self.encode_merge_table_relations(relations)

    def get_fixed_relation(self, relations, kind):
        if kind == "column":
            return self.get_fixed_column_relation(relations)
        else:
            return self.get_fixed_table_relation(relations)

    def forward(self, query, key, value, relations, kind="column"):
        if self.enable_latent_relations:
            relation_k, relation_v = self.get_latent_relation(relations, kind)

            if self.combine_latent_relations:
                relation_k_fixed, relation_v_fixed = self.get_fixed_relation(
                    relations, kind
                )
                relation_k = relation_k + relation_k_fixed
                relation_v = relation_v + relation_v_fixed
        else:
            relation_k, relation_v = self.get_fixed_relation(relations, kind)

        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, 1, self.hidden_size).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        assert nbatches == 1  # TODO, support batching
        relation_k, relation_v = relation_k.unsqueeze(0), relation_v.unsqueeze(0)

        _, self.attn = attention_with_relations(
            query, key, value, relation_k, relation_v, mask=None, dropout=self.dropout
        )

        return self.attn[0, 0]
