import operator
from typing import Tuple, List

import torch
from torch import layer_norm, nn
from torch.nn.init import xavier_uniform_
from tensor2struct.modules import variational_lstm  # jit version
from tensor2struct.utils import batched_sequence


def extract_last_hidden_state(hidden_state):
    """
    Extract last hidden state from output of lstm
    Args:
        hidden_state: length * hidden_size, the current version is
        not batched yet
    """
    seq_len, hidden_size = hidden_state.size()
    assert hidden_size % 2 == 0
    split_point = hidden_size // 2
    last_hidden_state = torch.cat(
        [hidden_state[-1, :split_point], hidden_state[0, split_point:]], dim=0
    )
    return last_hidden_state


def extract_last_hidden_state_batched(hidden_state, lengths, bidirectional):
    """
    Use torch.gather for efficient retrieving last hidden states
    Args:
        hidden_state: batch_size * length * hidden_size
    """
    bs, seq_len, hidden_size = hidden_state.size()
    assert bs == len(lengths)
    assert hidden_size % 2 == 0
    split_point = hidden_size // 2

    length_v = torch.stack(lengths, dim=0)
    if bidirectional:
        last_idx = (length_v - 1).unsqueeze(-1).unsqueeze(-1).expand(bs, 1, split_point)
        first_idx = torch.zeros([bs, 1, split_point], dtype=torch.long)
        batched_idx = torch.cat([last_idx, first_idx], dim=-1).to(hidden_state.device)
        last_hidden_state = torch.gather(
            hidden_state, dim=1, index=batched_idx
        ).squeeze(1)
    else:
        last_idx = (
            (length_v - 1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(bs, 1, hidden_size)
            .to(hidden_state.device)
        )
        last_hidden_state = torch.gather(hidden_state, dim=1, index=last_idx).squeeze(1)

    return last_hidden_state


def extract_lstm_minus_feature(hidden_state, i, j):
    """
    Extract span representation using lstm-minus feature,
    
    Args:
        hidden_state: Length * Batch * Hidden_size
        i, j: start and end pos, note that i, j is the 
        real index of words, discarding bos and eos:
        ... i, [i+1, ... , j+1], j+2, ...
    """
    seq_len, bs, hidden_size = hidden_state.size()
    assert hidden_size % 2 == 0
    split_point = hidden_size // 2
    hidden_f = hidden_state[j + 1, :, :split_point] - hidden_state[i, :, :split_point]
    hidden_b = (
        hidden_state[i + 1, :, split_point:] - hidden_state[j + 2, :, split_point:]
    )
    span_v = torch.cat([hidden_f, hidden_b], dim=-1)
    return span_v


def extract_all_span_features(hidden_state):
    """
    Return: ret[i] of size  n * bs * hidden_size, where n = seq_len - i
    """
    _seq_len, bs, hidden_size = hidden_state.size()
    assert hidden_size % 2 == 0  # bilstm by default
    seq_len = _seq_len - 2  # discard bos and eos

    ret = []
    for i in range(seq_len):
        temp_list = []
        for j in range(i, seq_len):
            span_v = extract_lstm_minus_feature(hidden_state, i, j)
            temp_list.append(span_v)
        temp_v = torch.stack(temp_list, dim=0)
        ret.append(temp_v)
    return ret


class SpanRepresentation(object):
    """
    Hidden state is the encoding from a LSTM; note that bos and eos 
    is attached to each sentence
    """

    def __init__(self, hidden_state, lengths):
        """
        Args:
            hidden_state: bs * length * hidden_size; we need to transform it
            into bs second for the convenience of parsing

            lengths: lengths of sequences including bos and eos, but get_lengths 
            function should return the real length
        """
        self.hidden_state = hidden_state.transpose(0, 1)  # Length * Bath * Hidden_size
        self.lengths = lengths
        self.span_v = extract_all_span_features(self.hidden_state)

    def num_tokens(self):
        # discard bos and eos
        return self.hidden_state.size()[0] - 2

    def num_batches(self):
        return self.hidden_state.size()[1]

    def get_lengths(self):
        # convert tensor to int, remove bos and eos
        lengths = [int(l) - 2 for l in self.lengths]
        return lengths

    def get_span(self, i, j):
        """
        Obtain vector for span from i to j (exclusive)
        """
        assert j > i
        return self.span_v[i][j - i - 1]


class BiLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout,
        summarize,
        bidirectional=True,
        use_native=False,
    ):
        """
        A wrapper over lstm that handles batched_sequence as input
        This is mostly used for encoding. For decoding, use the UniLSTM below

        summarize:
        - True: return Tensor of 1 x batch x emb size
        - False: return Tensor of seq len x batch x emb size

        Native vs VarLSTM
        1. in native lstm, dropout is applied in the input, whereas varlstm use Gal. dropout
        2. ValLSTM use layernorm by default
        """
        super().__init__()

        if use_native:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=output_size // 2 if bidirectional else output_size,
                bidirectional=bidirectional,
                dropout=0.0,
            )
            self.dropout = torch.nn.Dropout(dropout)
        else:
            if bidirectional:
                self.lstm = variational_lstm.BiLSTM(
                    input_size=input_size,
                    hidden_size=int(output_size // 2),
                    dropout=dropout,
                    layernorm=True,
                )
            else:
                self.lstm = variational_lstm.UniLSTM(
                    input_size=input_size,
                    hidden_size=output_size,
                    dropout=dropout,
                    layernorm=True,
                )
        self.summarize = summarize
        self.use_native = use_native

    def forward_unbatched_3d(self, input_):
        # all_embs shape: sum of desc lengths x batch (=1) x input_size
        all_embs, boundaries = input_

        new_boundaries = [0]
        outputs = []
        for left, right in zip(boundaries, boundaries[1:]):
            # state shape:
            # - h: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
            # - c: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
            # output shape: seq len x batch size x output_size
            if self.use_native:
                inp = self.dropout(all_embs[left:right])
                output, (h, c) = self.lstm(inp)
            else:
                output, (h, c) = self.lstm(all_embs[left:right])

            if self.summarize:
                seq_emb = torch.cat((h[0], h[1]), dim=-1).unsqueeze(0)
                new_boundaries.append(new_boundaries[-1] + 1)
            else:
                seq_emb = output
                new_boundaries.append(new_boundaries[-1] + output.shape[0])
            outputs.append(seq_emb)

        return torch.cat(outputs, dim=0), new_boundaries

    def forward_batched_3d(self, input_):
        # all_embs shape: PackedSequencePlus with shape [batch, sum of desc lengths, input_size]
        # boundaries: list of lists with shape [batch, num descs + 1]
        all_embs, boundaries = input_

        # List of the following:
        # (batch_idx, desc_idx, length)
        desc_lengths = []
        batch_desc_to_flat_map = {}
        for batch_idx, boundaries_for_item in enumerate(boundaries):
            for desc_idx, (left, right) in enumerate(
                zip(boundaries_for_item, boundaries_for_item[1:])
            ):
                desc_lengths.append((batch_idx, desc_idx, right - left))
                batch_desc_to_flat_map[batch_idx, desc_idx] = len(
                    batch_desc_to_flat_map
                )

        # Recreate PackedSequencePlus into shape
        # [batch * num descs, desc length, input_size]
        # with name `rearranged_all_embs`
        remapped_ps_indices = []

        def rearranged_all_embs_map_index(desc_lengths_idx, seq_idx):
            batch_idx, desc_idx, _ = desc_lengths[desc_lengths_idx]
            return batch_idx, boundaries[batch_idx][desc_idx] + seq_idx

        def rearranged_all_embs_gather_from_indices(indices):
            batch_indices, seq_indices = zip(*indices)
            remapped_ps_indices[:] = all_embs.raw_index(batch_indices, seq_indices)
            return all_embs.ps.data[torch.LongTensor(remapped_ps_indices)]

        rearranged_all_embs = batched_sequence.PackedSequencePlus.from_gather(
            lengths=[length for _, _, length in desc_lengths],
            map_index=rearranged_all_embs_map_index,
            gather_from_indices=rearranged_all_embs_gather_from_indices,
        )
        rev_remapped_ps_indices = tuple(
            x[0]
            for x in sorted(enumerate(remapped_ps_indices), key=operator.itemgetter(1))
        )

        # output shape: PackedSequence, [batch * num_descs, desc length, output_size]
        # state shape:
        # - h: [num_layers (=1) * num_directions (=2), batch, output_size / 2]
        # - c: [num_layers (=1) * num_directions (=2), batch, output_size / 2]
        if self.use_native:
            rearranged_all_embs = rearranged_all_embs.apply(self.dropout)
        output, (h, c) = self.lstm(rearranged_all_embs.ps)
        if self.summarize:
            # h shape: [batch * num descs, output_size]
            h = torch.cat((h[0], h[1]), dim=-1)

            # new_all_embs: PackedSequencePlus, [batch, num descs, input_size]
            new_all_embs = batched_sequence.PackedSequencePlus.from_gather(
                lengths=[
                    len(boundaries_for_item) - 1 for boundaries_for_item in boundaries
                ],
                map_index=lambda batch_idx, desc_idx: rearranged_all_embs.sort_to_orig[
                    batch_desc_to_flat_map[batch_idx, desc_idx]
                ],
                gather_from_indices=lambda indices: h[torch.LongTensor(indices)],
            )

            new_boundaries = [
                list(range(len(boundaries_for_item)))
                for boundaries_for_item in boundaries
            ]
        else:
            new_all_embs = all_embs.apply(
                lambda _: output.data[torch.LongTensor(rev_remapped_ps_indices)]
            )
            new_boundaries = boundaries

        return new_all_embs, new_boundaries

    def forward_batched_2d(self, all_embs):
        if self.use_native:
            all_embs = all_embs.apply(self.dropout)
        output, (h, c) = self.lstm(all_embs.ps)
        assert not self.summarize  # return the full output
        new_all_embs = all_embs.with_new_ps(output)
        return new_all_embs

    def forward(self, input_):
        """
        3d input is a tuple (data, boundaries)
        2d input does not need bondaries
        """
        if isinstance(input_, list) or isinstance(input_, tuple):
            return self.forward_batched_3d(input_)
        else:
            return self.forward_batched_2d(input_)


class VarLSTMCell(torch.nn.Module):
    """
    Copy from seq2struct.models.variational_lstm, since higher package does not work 
    with torch.jit.Module, so this is the plain nn.Module version
    """

    def __init__(self, input_size, hidden_size, dropout=0.0, layernorm=False):
        super(VarLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W_i = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_i = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.W_f = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_f = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.W_c = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_c = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.W_o = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_o = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_ih = torch.nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.empty(4 * hidden_size))

        self.layernorm = layernorm
        if self.layernorm:
            self.ln_xi2h = torch.nn.LayerNorm(hidden_size)
            self.ln_xf2h = torch.nn.LayerNorm(hidden_size)
            self.ln_xc2h = torch.nn.LayerNorm(hidden_size)
            self.ln_xo2h = torch.nn.LayerNorm(hidden_size)

            self.ln_hi2h = torch.nn.LayerNorm(hidden_size)
            self.ln_hf2h = torch.nn.LayerNorm(hidden_size)
            self.ln_hc2h = torch.nn.LayerNorm(hidden_size)
            self.ln_ho2h = torch.nn.LayerNorm(hidden_size)

            self.ln_cell = torch.nn.LayerNorm(hidden_size)

        self._input_dropout_mask = torch.empty((), requires_grad=False)
        self._h_dropout_mask = torch.empty((), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.W_i)
        torch.nn.init.orthogonal_(self.U_i)
        torch.nn.init.orthogonal_(self.W_f)
        torch.nn.init.orthogonal_(self.U_f)
        torch.nn.init.orthogonal_(self.W_c)
        torch.nn.init.orthogonal_(self.U_c)
        torch.nn.init.orthogonal_(self.W_o)
        torch.nn.init.orthogonal_(self.U_o)
        self.bias_ih.data.fill_(0.0)
        # forget gate set to 1.
        self.bias_ih.data[self.hidden_size : 2 * self.hidden_size].fill_(1.0)
        self.bias_hh.data.fill_(0.0)

    def set_dropout_masks(self, batch_size):
        def constant_mask(v):
            return (
                torch.tensor(v)
                .reshape(1, 1, 1)
                .expand(4, batch_size, -1)
                .to(self.W_i.device)
            )

        if self.dropout:
            if self.training:
                new_tensor = self.W_i.data.new
                self._input_dropout_mask = torch.bernoulli(
                    new_tensor(4, batch_size, self.input_size).fill_(1 - self.dropout)
                )
                self._h_dropout_mask = torch.bernoulli(
                    new_tensor(4, batch_size, self.hidden_size).fill_(1 - self.dropout)
                )
            else:
                mask = constant_mask(1 - self.dropout)
                self._input_dropout_mask = mask
                self._h_dropout_mask = mask
        else:
            mask = constant_mask(1.0)
            self._input_dropout_mask = mask
            self._h_dropout_mask = mask

    def forward(self, input, hidden_state):
        h_tm1, c_tm1 = hidden_state

        xi_t = torch.nn.functional.linear(
            input * self._input_dropout_mask[0, : input.shape[0]], self.W_i
        )
        xf_t = torch.nn.functional.linear(
            input * self._input_dropout_mask[1, : input.shape[0]], self.W_f
        )
        xc_t = torch.nn.functional.linear(
            input * self._input_dropout_mask[2, : input.shape[0]], self.W_c
        )
        xo_t = torch.nn.functional.linear(
            input * self._input_dropout_mask[3, : input.shape[0]], self.W_o
        )

        hi_t = torch.nn.functional.linear(
            h_tm1 * self._h_dropout_mask[0, : input.shape[0]], self.U_i
        )
        hf_t = torch.nn.functional.linear(
            h_tm1 * self._h_dropout_mask[1, : input.shape[0]], self.U_f
        )
        hc_t = torch.nn.functional.linear(
            h_tm1 * self._h_dropout_mask[2, : input.shape[0]], self.U_c
        )
        ho_t = torch.nn.functional.linear(
            h_tm1 * self._h_dropout_mask[3, : input.shape[0]], self.U_o
        )

        if self.layernorm:
            xi_t = self.ln_xi2h(xi_t)
            xf_t = self.ln_xf2h(xf_t)
            xc_t = self.ln_xc2h(xc_t)
            xo_t = self.ln_xo2h(xo_t)

            hi_t = self.ln_hi2h(hi_t)
            hf_t = self.ln_hf2h(hf_t)
            hc_t = self.ln_hc2h(hc_t)
            ho_t = self.ln_ho2h(ho_t)

        i_t = torch.sigmoid(
            xi_t
            + self.bias_ih[: self.hidden_size]
            + hi_t
            + self.bias_hh[: self.hidden_size]
        )
        f_t = torch.sigmoid(
            xf_t
            + self.bias_ih[self.hidden_size : 2 * self.hidden_size]
            + hf_t
            + self.bias_hh[self.hidden_size : 2 * self.hidden_size]
        )
        c_t = f_t * c_tm1 + i_t * torch.tanh(
            xc_t
            + self.bias_ih[2 * self.hidden_size : 3 * self.hidden_size]
            + hc_t
            + self.bias_hh[2 * self.hidden_size : 3 * self.hidden_size]
        )
        o_t = torch.sigmoid(
            xo_t
            + self.bias_ih[3 * self.hidden_size : 4 * self.hidden_size]
            + ho_t
            + self.bias_hh[3 * self.hidden_size : 4 * self.hidden_size]
        )

        if self.layernorm:
            h_t = o_t * torch.tanh(self.ln_cell(c_t))
        else:
            h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class UniLSTM(torch.nn.Module):
    """
    Used for decoding, only forward is defined. This is a duplicate version
    for lstm in variational_lstm, and it does not use torch.jit.ScriptModule
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        dropout=0.0,
        layernorm=False,
        cell_factory=VarLSTMCell,
    ):
        super(UniLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.cell_factory = cell_factory
        self.lstm_cells = []

        cell = cell_factory(
            input_size, hidden_size, dropout=dropout, layernorm=layernorm
        )
        self.lstm_cells.append(cell)

        suffix = ""
        cell_name = "cell{}".format(suffix)
        self.add_module(cell_name, cell)

    def forward(self, input, hidden_state=None):
        is_packed = isinstance(input, batched_sequence.PackedSequencePlus)
        if not is_packed:
            raise NotImplementedError

        max_batch_size = input.ps.batch_sizes[0]
        for cell in self.lstm_cells:
            cell.set_dropout_masks(max_batch_size)

        if hidden_state is None:
            num_directions = 1
            hx = input.ps.data.new_zeros(
                num_directions, max_batch_size, self.hidden_size, requires_grad=False
            )
            hidden_state = (hx, hx)

        forward_hidden_state = tuple(v[0] for v in hidden_state)
        output, next_hidden = self._forward_packed(
            input.ps.data, input.ps.batch_sizes, forward_hidden_state
        )

        output_ps = torch.nn.utils.rnn.PackedSequence(output, input.ps.batch_sizes,)
        return input.with_new_ps(output_ps)

    def _forward_packed(
        self,
        input: torch.Tensor,
        batch_sizes: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ):
        # Derived from
        # https://github.com/pytorch/pytorch/blob/6a4ca9abec1c18184635881c08628737c8ed2497/aten/src/ATen/native/RNN.cpp#L589

        step_outputs = []
        hs = []
        cs = []
        input_offset = torch.zeros((), dtype=torch.int64)  # scalar zero
        num_steps = batch_sizes.shape[0]
        last_batch_size = batch_sizes[0]

        # Batch sizes is a sequence of decreasing lengths, which are offsets
        # into a 1D list of inputs. At every step we slice out batch_size elements,
        # and possibly account for the decrease in the batch size since the last step,
        # which requires us to slice the hidden state (since some sequences
        # are completed now). The sliced parts are also saved, because we will need
        # to return a tensor of final hidden state.
        h, c = hidden_state
        for i in range(num_steps):
            batch_size = batch_sizes[i]
            step_input = input.narrow(0, input_offset, batch_size)
            input_offset += batch_size
            dec = last_batch_size - batch_size
            if dec > 0:
                hs.append(h[last_batch_size - dec : last_batch_size])
                cs.append(c[last_batch_size - dec : last_batch_size])
                h = h[: last_batch_size - dec]
                c = c[: last_batch_size - dec]
            last_batch_size = batch_size
            h, c = self.cell(step_input, (h, c))
            step_outputs.append(h)

        hs.append(h)
        cs.append(c)
        hs.reverse()
        cs.reverse()

        concat_h = torch.cat(hs)
        concat_c = torch.cat(cs)

        return (torch.cat(step_outputs, dim=0), (concat_h, concat_c))

    def _set_dropout_masks(self, batch_size):
        """
        For decoding where we need to manually set dropout masks
        """
        self.lstm_cells[0].set_dropout_masks(batch_size)

    def _step(self, input, rnn_state):
        """
        Used in decoding at test time, make sure set_dropout_mask 
        is used during decoding
        """
        assert len(self.lstm_cells) == 1
        rnn_cell = self.lstm_cells[0]
        new_rnn_state = rnn_cell(input, rnn_state)
        return new_rnn_state
