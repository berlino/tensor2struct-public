import collections
import itertools
import json
import os

import attr
import torch
import torch.nn.functional as F
import numpy as np

from tensor2struct.models import abstract_preproc, encoder, batched_encoder
from tensor2struct.modules import embedders, lstm, attention, permutation
from tensor2struct.utils import serialization, vocab, registry

import logging

logger = logging.getLogger("tensor2struct")


@attr.s
class EncoderState:
    src_memory = attr.ib()
    lengths = attr.ib()

    # only useful for conventional attention-based decoding
    src_summary = attr.ib(default=None)

    # for lexical model
    src_embedding = attr.ib(default=None)

    # for debugging
    permutation = attr.ib(default=None)

    # heuristic loss e.g., constraining the latent permutations
    enc_loss = attr.ib(default=None)


@registry.register("encoder", "latper_enc")
class LatPerEncoder(batched_encoder.Encoder):
    batched = True
    Preproc = encoder.EncPreproc

    """
    Latent permuttion produced by syntax encoder + 
    semantic encoder  ==> postorder encoder to obtain 
    the final representation
    """

    def __init__(
        self,
        device,
        preproc,
        dropout=0.1,
        word_emb_size=128,
        recurrent_size=256,
        num_heads=4,
        use_native_lstm=True,
        bert_version="bert-base-uncased",
        syntax_encoder=("emb", "bilstm"),
        semantic_encoder=("emb",),
        postorder_encoder=None,
        forward_relaxed=True,
        gumbel_temperature=None,
        use_map_decode=False,
    ):
        super().__init__(
            device=device,
            preproc=preproc,
            dropout=dropout,
            word_emb_size=word_emb_size,
            recurrent_size=recurrent_size,
            encoder=syntax_encoder,
            num_heads=num_heads,
            use_native_lstm=use_native_lstm,
            bert_version=bert_version,
        )

        # another encoder for obtain semantic info
        self.semantic_encoder_modules = semantic_encoder
        if self.semantic_encoder_modules is not None:
            self.semantic_encoder = self._build_modules(self.semantic_encoder_modules)

        self.postorder_encoder_modules = postorder_encoder
        if postorder_encoder is not None:
            self.postorder_encoder = self._build_modules(self.postorder_encoder_modules)

        if self.postorder_encoder_modules:
            self.last_enc_module = self.postorder_encoder_modules[-1]
        elif self.semantic_encoder_modules:
            self.last_enc_module = self.semantic_encoder_modules[-1]
        else:
            self.last_enc_module = self.encoder_modules[-1]

        self.permutator = permutation.BinarizableTree(
            device=device,
            input_size=recurrent_size,
            forward_relaxed=forward_relaxed,
            gumbel_temperature=gumbel_temperature,
            use_map_decode=use_map_decode,
            dropout=dropout
        )

    def _pad(self, tokens_list):
        """
        Add BOS and EOS to use LSTM-minus features
        """
        res = []
        for tokens in tokens_list:
            res.append([vocab.BOS] + tokens + [vocab.EOS])
        return res

    def compute_encoding(self, tokens_list):
        res = self.compute_encoding_batched(tokens_list)
        # res = self.compute_encoding_unbatched(tokens_list)
        return res

    def extract_lstm_enc(self, src_enc, enc_module):
        assert enc_module in ["bilstm", "unilstm"]
        src_memory, lengths = src_enc.pad(batch_first=True)
        bidirectional = enc_module == "bilstm"
        src_summary = lstm.extract_last_hidden_state_batched(
            src_memory, lengths, bidirectional=bidirectional
        )
        return src_memory, lengths, src_summary

    def extract_trans_enc(self, src_enc, enc_module):
        assert enc_module in ["transformer"]
        raw_src_enc_memory, lengths = src_enc

        # unpack CLS representation as the summary, recover original lengths
        src_summary = raw_src_enc_memory[:, 0, :]
        src_memory = raw_src_enc_memory[:, 1:, :]
        for i in range(len(lengths)):
            lengths[i] = lengths[i] - 1

        return src_memory, lengths, src_summary

    def extract_enc(self, src_enc, enc_module):
        if enc_module in ["bilstm", "unilstm"]:
            return self.extract_lstm_enc(src_enc, enc_module)
        elif enc_module in ["transformer"]:
            return self.extract_trans_enc(src_enc, enc_module)
        elif enc_module in ["emb"]:
            src_memory, lengths = src_enc.pad(batch_first=True)
            return src_memory, lengths, None

    def compute_encoding_batched(self, tokens_list):
        """
        For syntax encoding, each sentence is padded with bos and eos to obtain
        the LSTM-minus span-level features.
        """
        # 1. obtain permutation from syntax representations
        padded_tokens_list = self._pad(tokens_list)
        syntax_src_enc = self.encoder(padded_tokens_list)
        syntax_src_enc_batched, padded_lengths = syntax_src_enc.pad(batch_first=True)

        if self.semantic_encoder_modules is None:
            # 2.a baseline without any the reodering
            permutation_matrix = None
            permuted_memory = syntax_src_enc_batched
            lengths = padded_lengths
        else:
            syntax_span_rep = lstm.SpanRepresentation(
                syntax_src_enc_batched, padded_lengths
            )
            permutation_matrix, _ = self.permutator(
                syntax_span_rep
            )  # use span_rep to handle bos and eos

            # 2.b use permutation matrix to obtain reordered semantic representations
            # optional: postorder encoder is applied after permutation
            if self.postorder_encoder_modules:
                preorder_src_enc = self.semantic_encoder(tokens_list)
                postorder_input = preorder_src_enc.apply_raw(
                    lambda x: torch.bmm(permutation_matrix, x)
                )
                postorder_src_enc = self.postorder_encoder(postorder_input)
                permuted_memory, lengths, src_summary = self.extract_enc(
                    postorder_src_enc, self.last_enc_module
                )
            else:
                semantic_src_enc = self.semantic_encoder(tokens_list)
                semantic_src_enc_batched, lengths, src_summary = self.extract_enc(
                    semantic_src_enc, self.last_enc_module
                )
                permuted_memory = torch.bmm(
                    permutation_matrix, semantic_src_enc_batched
                )

            # optional: check lengths
            # span_rep.get_length() remove bos and eos
            lengths = [int(l) for l in lengths]  # tensor to int
            for l1, l2 in zip(syntax_span_rep.get_lengths(), lengths):
                assert l1 == l2

        res = EncoderState(
            src_memory=permuted_memory,
            lengths=lengths,
            src_summary=src_summary,
            permutation=permutation_matrix,
        )
        return res


@registry.register("encoder", "ssnt_latper_enc")
class LatPerSSNTEncoder(LatPerEncoder):
    batched = True
    Preproc = encoder.EncPreproc

    """
    Compared with latper_enc, this encoder 
        1. adds an additional EOS token at the end of every input utterance. 
        2. In addition, it also supports semantic dropout (not very effective).
        3. support posteriro control of straignt/invert operations
    """

    def __init__(
        self,
        device,
        preproc,
        dropout=0.1,
        word_emb_size=128,
        recurrent_size=256,
        num_heads=4,
        use_native_lstm=True,
        bert_version="bert-base-uncased",
        syntax_encoder=("emb", "bilstm"),
        semantic_encoder=("emb",),
        postorder_encoder=None,
        forward_relaxed=True,
        gumbel_temperature=None,
        use_map_decode=False,
        semantic_dropout=None,
    ):
        super().__init__(
            device,
            preproc,
            dropout=dropout,
            word_emb_size=word_emb_size,
            recurrent_size=recurrent_size,
            num_heads=num_heads,
            use_native_lstm=use_native_lstm,
            bert_version=bert_version,
            syntax_encoder=syntax_encoder,
            semantic_encoder=semantic_encoder,
            postorder_encoder=postorder_encoder,
            forward_relaxed=forward_relaxed,
            gumbel_temperature=gumbel_temperature,
            use_map_decode=use_map_decode,
        )

        self.semantic_dropout = semantic_dropout
        self.eos_emb = torch.nn.Parameter(torch.randn(word_emb_size).to(device))

    def compute_encoding_batched(self, tokens_list):
        """
        Add a special token at the end of each sentence
        """
        padded_tokens_list = self._pad(tokens_list)
        syntax_src_enc = self.encoder(padded_tokens_list)
        syntax_src_enc_batched, padded_lengths = syntax_src_enc.pad(batch_first=True)

        # 1. syntax rep
        syntax_span_rep = lstm.SpanRepresentation(
            syntax_src_enc_batched, padded_lengths
        )
        permutation_matrix, reorder_loss = self.permutator(
            syntax_span_rep
        )  # use span_rep to handle bos and eos

        # 2. use permutation matrix to obtain reordered semantic representations
        assert self.postorder_encoder_modules

        preorder_src_enc = self.semantic_encoder(tokens_list)
        postorder_input = preorder_src_enc.apply_raw(
            lambda x: torch.bmm(permutation_matrix, x)
        )

        # 3. add EOS to the permuted embedding
        def add_eos(x):
            padded_x, lengths = x.pad()
            bs, _, rs = padded_x.size()
            zero_pad = torch.zeros([bs, 1, rs]).to(padded_x.device)
            x_with_zero_padded = torch.cat([padded_x, zero_pad], dim=1)

            aux_t = torch.zeros_like(x_with_zero_padded)
            for batch_idx, eos_idx in enumerate(lengths):
                aux_t[batch_idx, eos_idx] = self.eos_emb
            new_x = x_with_zero_padded + aux_t

            # increase the sorted length of packed seq by 1
            sorted_lengths = [length + 1 for length in x.lengths]
            per_idx_t = torch.LongTensor(x.orig_to_sort).to(self._device)
            per_data = new_x[per_idx_t]
            new_ps = torch.nn.utils.rnn.pack_padded_sequence(
                per_data, sorted_lengths, batch_first=True
            )
            return attr.evolve(x, ps=new_ps, lengths=sorted_lengths)

        postorder_input_with_eos = add_eos(postorder_input)

        # 4. apply postoder update
        postorder_src_enc = self.postorder_encoder(postorder_input_with_eos)
        permuted_memory, lengths, src_summary = self.extract_enc(
            postorder_src_enc, self.last_enc_module
        )

        # 5. optional: apply semantic dropout
        postorder_emb, _ = postorder_input_with_eos.pad(batch_first=True)
        if self.training and self.semantic_dropout:
            p_mask = self.semantic_dropout * torch.ones(permuted_memory.size()[:2]).to(
                self._device
            )
            mask = torch.bernoulli(p_mask)
            batch_mask = mask.unsqueeze(-1).expand(-1, -1, permuted_memory.size()[-1])

            permuted_memory = permuted_memory * (1 - batch_mask) + postorder_emb * batch_mask
        elif self.semantic_dropout == 1.0:
            # if semantic_dropout is 1.0, we skip the postordering model
            permuted_memory = postorder_emb

        # optional: check lengths
        # span_rep.get_length() remove bos and eos
        lengths = [int(l) for l in lengths]  # tensor to int
        for l1, l2 in zip(syntax_span_rep.get_lengths(), lengths):
            assert l1 + 1 == l2

        res = EncoderState(
            src_memory=permuted_memory,
            lengths=lengths,
            permutation=permutation_matrix,
            src_summary=src_summary,
            src_embedding=postorder_emb,
            enc_loss=reorder_loss,
        )
        return res


@registry.register("encoder", "latper_semi_batched_enc")
class LatPerSemiBatchedEncoder(encoder.Encoder):
    """
    Used for SemiBatchedEncDec 
    """

    batched = True
    Preproc = encoder.EncPreproc

    def compute_encoding(self, tokens_list):
        res = self.compute_encoding_unbatched(tokens_list)
        return res

    def compute_encoding_unbatched(self, tokens_list):
        tokens_list = self._pad(tokens_list)
        src_enc = self.encoder(tokens_list)

        ret_list = []
        for i in range(len(tokens_list)):
            # does not transformer for now
            assert "transformer" not in self.encoder_modules
            assert self.encoder_modules[-1] == "bilstm"
            src_memory = src_enc.select(i)
            src_summary = lstm.extract_last_hidden_state(src_memory)

            # extract and apply latent permutation
            span_rep = lstm.SpanRepresentation(src_memory)
            permutation_matrix = self.permutator(span_rep)
            real_src_memory = src_memory[1:-1, :]  # remove bos and eos
            permuted_memory = torch.matmul(permutation_matrix, real_src_memory)

            # attach a batch dimension
            permuted_memory = permuted_memory.unsqueeze(0)
            src_summary = src_summary.unsqueeze(0)

            ret_list.append(
                EncoderState(src_memory=permuted_memory, src_summary=src_summary)
            )
        return ret_list

@registry.register("encoder", "sinkhorn_batched_enc")
class SinkhornEncoder(batched_encoder.Encoder):
    batched = True
    Preproc = encoder.EncPreproc

    def __init__(
        self,
        device,
        preproc,
        dropout=0.1,
        word_emb_size=128,
        recurrent_size=256,
        encoder=("emb", "bilstm"),
        semantic_encoder=("emb",),
    ):
        super().__init__(
            device, preproc, dropout, word_emb_size, recurrent_size, encoder
        )

        query_size = recurrent_size
        key_size = recurrent_size
        self.query_proj = torch.nn.Linear(recurrent_size, query_size)
        self.key_proj = torch.nn.Linear(recurrent_size, key_size)
        # self.temp = np.power(key_size, 0.5)
        self.temp = 1
        self.num_sh_it = 32

        self.semantic_encoder_modules = semantic_encoder
        self.semantic_encoder = self._build_modules(self.semantic_encoder_modules)

    def sinkhorn_attention(self, input_v):
        """ input_v: sent_len * recurent_size """
        query_v = self.query_proj(input_v)
        key_v = self.key_proj(input_v)
        score_mat = torch.einsum("ij,kj->ik", [key_v, query_v]) / self.temp

        it_scores = score_mat
        for _ in range(self.num_sh_it):
            it_scores = it_scores - torch.logsumexp(it_scores, dim=1, keepdim=True)
            it_scores = it_scores - torch.logsumexp(it_scores, dim=0, keepdim=True)
        prob_m = torch.exp(it_scores)
        return prob_m
    
    def compute_encoding(self, tokens_list):
        syntax_enc = self.encoder(tokens_list)
        semantic_enc = self.semantic_encoder(tokens_list)
        max_len = max(len(tokens) for tokens in tokens_list)

        memory_list = []
        length_list = []
        for i in range(len(tokens_list)):
            src_memory = syntax_enc.select(i)
            semantic_memory = semantic_enc.select(i)

            permutation_mat = self.sinkhorn_attention(src_memory)
            permutated_memory = torch.einsum("ji,jk->ik", [permutation_mat, semantic_memory])

            cur_length = len(tokens_list[i])
            reshaped_permutated_memory = F.pad(permutated_memory, (0, 0, 0, max_len - cur_length), "constant", 0)

            memory_list.append(reshaped_permutated_memory)
            length_list.append(cur_length)

        src_enc_memory = torch.stack(memory_list, dim=0)
        return EncoderState(src_enc_memory, length_list, None)
