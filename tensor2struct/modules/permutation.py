import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensor2struct.modules import energys
from tensor2struct.utils import gumbel

import logging

logger = logging.getLogger("tensor2struct")


def _anti_block_diag(m1, m2):
    """
    m1, m2:  2-d tensors
    """
    return torch.flip(
        torch.block_diag(torch.flip(m1, dims=[0]), torch.flip(m2, dims=[0])), dims=[0]
    )


def anti_block_diag(m1, m2):
    """
    Returns: [0, m2; m1, 0]
    TODO: test which one works faster vs. _anti_block_diag
    """
    if len(m1.size()) == 1:
        m1 = m1.unsqueeze(0)
    l1, _ = m1.size()

    if len(m2.size()) == 1:
        m2 = m2.unsqueeze(0)
    l2, _ = m2.size()

    res = torch.zeros([l1 + l2, l1 + l2]).to(m1.device)
    res[l2:, :l1] = m1
    res[:l2, l1:] = m2
    return res


def batched_block_diag_forloop(m1, m2, f):
    """
    Args:
        m1, m2: 3-d tensors

        f: either torch.block_diag or anti_block_diag
    """
    bs, _, _ = m1.size()
    res_list = []
    for i in range(bs):
        res_list.append(f(m1[i], m2[i]))
    res = torch.stack(res_list, dim=0)
    return res


def batched_block_diag(m1, m2):
    """
    Args:
        m1, m2: 3-d tensors
    """
    bs, l1, _ = m1.size()
    _bs, l2, _ = m2.size()
    assert bs == _bs
    res = torch.zeros([bs, l1 + l2, l1 + l2]).to(m1.device)
    res[:, :l1, :l1] = m1
    res[:, l1:, l1:] = m2
    return res


def batched_anti_block_diag(m1, m2):
    """
    Args:
        m1, m2: 3-d tensors
    """
    bs, l1, _ = m1.size()
    _bs, l2, _ = m2.size()
    assert bs == _bs
    res = torch.zeros([bs, l1 + l2, l1 + l2]).to(m1.device)
    res[:, l2:, :l1] = m1
    res[:, :l2, l1:] = m2
    return res


def sort_span_dict(score_dic):
    sorted_score_dic = collections.OrderedDict(
        {k: score_dic[k] for k in sorted(score_dic.keys(), key=lambda x: x[1] - x[0])}
    )
    return sorted_score_dic


class BinarizableTree(nn.Module):
    """
    Binary tree to represent separable permutations
    """

    def __init__(
        self,
        device,
        input_size,
        forward_relaxed=True,
        gumbel_temperature=None,
        use_map_decode=True,
        wcfg=True,
        dropout=0.1,
        invert_prior=None,
    ):
        super().__init__()
        self._device = device
        self.input_size = input_size
        self.score_f = energys.MLP(
            input_size=input_size, output_size=2, dropout=dropout
        )

        self.forward_relaxed = forward_relaxed  # whether straight-through
        self.forward_hard = not self.forward_relaxed
        self.add_gumbel_noise = (
            gumbel_temperature is not None
        )  # whether add Gumbel noise for sampling
        self.gumbel_temperature = gumbel_temperature
        self.use_map_decode = use_map_decode  # whether use discrete during decoding
        self.wcfg = wcfg  # if false, use pcfg parameterization
        self.invert_prior = invert_prior

    def inside(self, span_rep, lengths):
        """
        Args:
            span_rep: an object which support get_span(i,j)
            lengths: the lenght of input sentences

        Return:
            partition: a vector of parition values
            decision_scores: a dict that maps i,j to a vector of size j-i-1 * bs * 2
            beta: a dict that maps i,j to a vector of score bs
            gamma: a `global' version of decision_scores, size j-i-1 * bs * 2
        """
        beta = {}  # scores in log space
        gamma = {}  # intermedia beta scores
        decision_scores = {}  # scores for straight or invert

        n = span_rep.num_tokens()
        bs = len(lengths)
        for i in range(n):
            beta[i, i + 1] = torch.zeros([bs]).to(self._device)

        for w in range(2, n + 1):  # span length
            for i in range(0, n - w + 1):  # span start
                k = i + w  # span end

                # compute decision scores
                left_spans, right_spans = [], []
                for j in range(i + 1, k):
                    left_spans.append(span_rep.get_span(i, j))
                    right_spans.append(span_rep.get_span(j, k))
                left_span_m = torch.stack(left_spans, dim=0)
                right_span_m = torch.stack(right_spans, dim=0)
                score_v = self.score_f(left_span_m, right_span_m)
                decision_scores[i, k] = score_v

                # compute beta and gamma
                gamma_score_list = []
                for j in range(i + 1, k):
                    score_idx = j - (i + 1)
                    beta_ijk = beta[i, j] + beta[j, k]
                    gamma_score_list.append(
                        score_v[score_idx] + beta_ijk.unsqueeze(-1).expand(-1, 2)
                    )
                gamma[i, k] = torch.stack(gamma_score_list, dim=0)  # j-i-1 * bs * 2
                beta[i, k] = torch.logsumexp(
                    torch.cat(gamma_score_list, dim=-1), dim=-1
                )

        # sort the span-based score dict
        beta = sort_span_dict(beta)
        gamma = sort_span_dict(gamma)
        decision_scores = sort_span_dict(decision_scores)

        # extract the partition value
        partitions = []
        for batch_idx, length in enumerate(lengths):
            partition = beta[0, length][batch_idx]
            partitions.append(partition)
        partition_v = torch.stack(partitions, dim=0)

        return partition_v, beta, gamma, decision_scores

    def marginals(self, partition, decision_scores):
        """
        Args:
            decision_scores: local decision scores
        Returns the marginal for each production rule, the marginals of scores 
        are computed by backpropagation.
        """
        idtotuple = list(decision_scores.keys())
        grad_inputs = [decision_scores[i] for i in idtotuple]
        raw_marginals = torch.autograd.grad(
            partition,
            grad_inputs,
            create_graph=True,
            only_inputs=True,
            allow_unused=False,
        )
        marginals = {}
        for i, _marginal in enumerate(raw_marginals):
            marginals[idtotuple[i]] = _marginal
        return marginals

    def coverto_pcfg(self, decision_scores):
        """
        Args:
            decision_scores: this can be a global (gamma) or local score (decision_scores)
            If perturb is True, we use gumbel-softmax with temperature tau
            if invert_prior is not None, we control the posterior of p(inverted_prior) and output a rule_loss
        """
        rule_probs = {}
        op_probs_list = []  # p(straight) and p(inverted) for compute rule_loss
        for i, k in decision_scores:
            num_split = k - i - 1
            _num_split, bs, _d = decision_scores[i, k].size()
            assert _num_split == num_split and _d == 2
            unnorm_score = decision_scores[i, k].transpose(0, 1).reshape(bs, -1)

            # essentially soft_norm_score below
            cat_dist = (
                torch.softmax(unnorm_score, dim=1).view([bs, num_split, 2]).sum(1)
            )
            op_probs_list.append(cat_dist)

            if self.training:
                if self.add_gumbel_noise:
                    norm_score = F.gumbel_softmax(
                        unnorm_score,
                        tau=self.gumbel_temperature,
                        hard=self.forward_hard,
                        dim=1,
                    )
                else:
                    # non-stochastic relaxation using straight-through estimation
                    soft_norm_score = torch.softmax(unnorm_score, dim=1)
                    if self.forward_hard:
                        _, max_ind = soft_norm_score.max(dim=1)
                        one_hot = torch.zeros_like(soft_norm_score).to(self._device)
                        one_hot.scatter_(1, max_ind.view(-1, 1), 1)
                        norm_score = (
                            one_hot - soft_norm_score
                        ).detach() + soft_norm_score
                    else:
                        norm_score = soft_norm_score
            else:
                norm_score = torch.softmax(unnorm_score, dim=1)
            norm_score = norm_score.view([bs, num_split, 2]).transpose(0, 1)
            rule_probs[i, k] = norm_score

        # compute the rule loss
        if self.invert_prior:
            log_mean_rule_prob = torch.cat(op_probs_list, dim=0).mean(0).log()
            rule_loss = (
                -self.invert_prior * log_mean_rule_prob[1]
                - (1 - self.invert_prior) * log_mean_rule_prob[0]
            )
        else:
            rule_loss = None

        return rule_probs, rule_loss

    def compute_entropy(self, rule_probs, lengths):
        raise NotImplementedError

    def bottom_up_compute_permutation(self, rule_probs, lengths):
        """
        Compute permutation matrix in a bottom-up manner
        """
        p_matrices = {}
        bs = len(lengths)
        _, seq_len = list(rule_probs.keys())[-1]
        for i in range(seq_len):
            p_matrices[i, i + 1] = torch.ones([bs, 1, 1]).to(self._device)
        for i, k in rule_probs:
            rule_prob_v = rule_probs[i, k]
            m_list = []
            for j in range(i + 1, k):
                idx = j - i - 1  # split point starts from i + 1
                rule_prob = rule_prob_v[idx]

                left_m = p_matrices[i, j]
                right_m = p_matrices[j, k]

                #  batch-wise product, some test code included
                s_mat = batched_block_diag(left_m, right_m)
                # _s_mat = batched_block_diag_forloop(left_m, right_m, torch.block_diag)
                # assert torch.all(torch.eq(s_mat, _s_mat))
                s_prob = rule_prob[:, 0:1].unsqueeze(dim=-1).expand_as(s_mat)
                straight = s_prob * s_mat
                m_list.append(straight)

                i_mat = batched_anti_block_diag(left_m, right_m)
                # _i_mat = batched_block_diag_forloop(left_m, right_m, anti_block_diag)
                # assert torch.all(torch.eq(i_mat, _i_mat))
                i_prob = rule_prob[:, 1:2].unsqueeze(dim=-1).expand_as(i_mat)
                inverted = i_prob * i_mat
                m_list.append(inverted)
            p_matrices[i, k] = sum(m_list)

        res_list = []
        for batch_idx, length in enumerate(lengths):
            real_p_mat = p_matrices[0, length][batch_idx]
            if length < seq_len:
                pad_mat = torch.zeros([seq_len, seq_len]).to(self._device)
                pad_mat[:length, :length] = real_p_mat
                res_list.append(pad_mat)
            else:
                res_list.append(real_p_mat)
        res = torch.stack(res_list, dim=0)
        return res

    def bottom_up_map_decode(self, rule_probs, lengths):
        p_matrices = {}
        bs = len(lengths)
        _, seq_len = list(rule_probs.keys())[-1]
        for i in range(seq_len):
            p_matrices[i, i + 1] = torch.ones([bs, 1, 1]).to(self._device)
        for i, k in rule_probs:
            rule_prob_v = rule_probs[i, k]
            m_list = []
            for j in range(i + 1, k):
                idx = j - i - 1  # split point starts from i + 1
                rule_prob = rule_prob_v[idx]

                left_m = p_matrices[i, j]
                right_m = p_matrices[j, k]

                #  batch-wise product, some test code included
                s_mat = batched_block_diag(left_m, right_m)
                s_prob = rule_prob[:, 0:1].unsqueeze(dim=-1).expand_as(s_mat)
                straight = s_prob * s_mat
                m_list.append(straight)

                i_mat = batched_anti_block_diag(left_m, right_m)
                i_prob = rule_prob[:, 1:2].unsqueeze(dim=-1).expand_as(i_mat)
                inverted = i_prob * i_mat
                m_list.append(inverted)

            cat_mat = torch.stack(m_list, dim=1)  # bs * num_splits * n * n
            _, max_ids = cat_mat.sum(dim=-1).sum(dim=-1).max(dim=-1)  # batch_size
            p_matrices[i, k] = cat_mat[
                torch.arange(bs), max_ids
            ].ceil()  # ceil prob to 1

        res_list = []
        for batch_idx, length in enumerate(lengths):
            real_p_mat = p_matrices[0, length][batch_idx]
            if length < seq_len:
                pad_mat = torch.zeros([seq_len, seq_len]).to(self._device)
                pad_mat[:length, :length] = real_p_mat
                res_list.append(pad_mat)
            else:
                res_list.append(real_p_mat)
        res = torch.stack(res_list, dim=0)
        return res

    def top_down_compute_permutation(self, rule_probs, lengths):
        """
        Previous unbatched version support top-k inference to save time; for 
        the current batched version, we might also do this
        """
        bs = len(lengths)
        _, seq_len = list(rule_probs.keys())[-1]

        p_matrices = {}

        def recur_compute(i, k):
            if i + 1 == k:
                p_matrices[i, k] = torch.ones([bs, 1, 1]).to(self._device)

            if (i, k) in p_matrices:
                return p_matrices[i, k]

            rule_prob_v = rule_probs[i, k]
            assert k - i - 1 == rule_prob_v.size()[0]  # from i+1 to k-1

            m_list = []
            for j in range(i + 1, k):
                idx = j - i - 1
                rule_prob = rule_prob_v[idx]

                left_m = recur_compute(i, j)
                right_m = recur_compute(j, k)

                s_mat = batched_block_diag(left_m, right_m)
                s_prob = rule_prob[:, 0:1].unsqueeze(dim=-1).expand_as(s_mat)
                straight = s_prob * s_mat
                m_list.append(straight)

                i_mat = batched_anti_block_diag(left_m, right_m)
                i_prob = rule_prob[:, 1:2].unsqueeze(dim=-1).expand_as(i_mat)
                inverted = i_prob * i_mat
                m_list.append(inverted)

            p_matrices[i, k] = sum(m_list)
            return p_matrices[i, k]

        recur_compute(0, seq_len)

        res_list = []
        for batch_idx, length in enumerate(lengths):
            real_p_mat = p_matrices[0, length][batch_idx]
            if length < seq_len:
                pad_mat = torch.zeros([seq_len, seq_len]).to(self._device)
                pad_mat[:length, :length] = real_p_mat
                res_list.append(pad_mat)
            else:
                res_list.append(real_p_mat)
        res = torch.stack(res_list, dim=0)
        return res

    def forward(self, span_rep):
        """
        Args:
            span_rep: an object defined in lstm.py

        Return: 
            a n*n matrix M, M[i,j] means the marginal 
            prob that j_th token is reordered to i. 
        """
        bs = span_rep.num_batches()
        lengths = span_rep.get_lengths()

        if span_rep.num_tokens == 1:
            return torch.ones([bs, 1]).to(self._device)

        partition, beta, gamma, decision_scores = self.inside(span_rep, lengths)
        # marginals = self.marginals(partition, decision_scores)

        # it can als be locally normalized
        if self.wcfg:
            rule_probs, rule_loss = self.coverto_pcfg(gamma)
        else:
            rule_probs, rule_loss = self.coverto_pcfg(decision_scores)

        # use map for decoding if set; otherwise use marginal
        if not self.training and self.use_map_decode:
            permutation_matrix = self.bottom_up_map_decode(rule_probs, lengths)
        else:
            #  use marginal for forward, could be 1) marginal as attention 2) soft sample with gumbel noise
            #  equivalent two ways to compute the marginal exactly
            # TODO: use top-down to filter out paths that are with probs 0
            permutation_matrix = self.bottom_up_compute_permutation(rule_probs, lengths)
            # permutation_matrix = self.top_down_compute_permutation(rule_probs, lengths)

        return permutation_matrix, rule_loss
