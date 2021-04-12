import attr
import copy
import operator

import torch
import torch.nn.functional as F
from tensor2struct.utils import registry, gumbel


@attr.s
class Hypothesis:
    inference_state = attr.ib()
    next_choices = attr.ib()
    score = attr.ib(default=0)

    choice_history = attr.ib(factory=list)
    score_history = attr.ib(factory=list)  # store tensors


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


@registry.register("infer_method", "beam_search_sampling")
def top_k_top_p_sampling(
    model,
    orig_item,
    preproc_item,
    sample_size=1,
    max_steps=100,
    top_p=0.0,
):
    """
    Beam search, if top_p > 0.0, we also do filtering based on nuclues sampling
    Score history stores log-potential in the form of tensors, rather than scalars
    """
    ret_state = model(orig_item, preproc_item, compute_loss=False, infer=True)
    inference_state, next_choices = (
        ret_state["initial_state"],
        ret_state["initial_choices"],
    )
    beam = [Hypothesis(inference_state, next_choices)]
    finished = []

    for step in range(max_steps):
        if len(finished) == sample_size:
            break

        candidates = []
        for hyp in beam:
            # collect scores for sorting
            scores = [choice_score for _, choice_score in hyp.next_choices]
            score_v = torch.stack(scores, dim=0)
            if top_p > 0:
                score_v = top_k_top_p_filtering(score_v, top_p=top_p)

            for i, (choice, choice_score) in enumerate(hyp.next_choices):
                filtered_score = score_v[i].item()
                if filtered_score == -float("Inf"):
                    continue
                candidates.append(
                    (
                        hyp,
                        choice,
                        choice_score,
                        hyp.score + choice_score.item(),
                    )
                )

        # Keep the top K expansions
        candidates.sort(key=operator.itemgetter(3), reverse=True)
        candidates = candidates[: sample_size - len(finished)]

        # Create the new hypotheses from the expansions
        beam = []
        for hyp, choice, choice_score, cum_score in candidates:
            inference_state = hyp.inference_state.clone()
            next_choices = inference_state.step(choice)
            new_hyp = Hypothesis(
                inference_state=inference_state,
                next_choices=next_choices,
                score=cum_score,
                choice_history=hyp.choice_history + [choice],
                score_history=hyp.score_history + [choice_score],
            )
            if next_choices is None:
                finished.append(new_hyp)
            else:
                beam.append(new_hyp)
    finished.sort(key=operator.attrgetter("score"), reverse=True)
    return finished


@registry.register("infer_method", "stochastic_beam_search_sampling")
def stochastic_beam_search_sampling(
    model,
    orig_item,
    preproc_item,
    sample_size=1,
    max_steps=100,
    early_stop=False,
):
    """
    Accoding to stochastic beam search paper.
    Note that theorectically early stopping should not be used, so max_steps matters in this method.
    Score history will still store the original logits but score will store perturbed scores for searching
    """
    ret_state = model(orig_item, preproc_item, compute_loss=False, infer=True)
    inference_state, next_choices = (
        ret_state["initial_state"],
        ret_state["initial_choices"],
    )
    beam = [Hypothesis(inference_state, next_choices)]
    finished = []

    for step in range(max_steps):
        if early_stop and len(finished) == sample_size:
            break
        elif len(finished) > sample_size * 6:  # to avoid too many samples
            break

        candidates = []
        for hyp in beam:
            # collect scores for sorting
            scores = [choice_score for _, choice_score in hyp.next_choices]
            score_v = torch.stack(scores, dim=0).detach()

            if step == 0:
                gumbel_score_v = gumbel.gumbel_like(score_v) + score_v
            else:
                parent_score = torch.Tensor([hyp.score]).to(score_v.device)
                score_v = score_v.unsqueeze(0)  # 1 * num_cand
                gumbel_score_v, _ = gumbel.gumbel_with_maximum(score_v, parent_score)
                gumbel_score_v = gumbel_score_v.squeeze(0)

            for i, (choice, choice_score) in enumerate(hyp.next_choices):
                gumbel_score = gumbel_score_v[i].item()
                candidates.append(
                    (
                        hyp,
                        choice,
                        choice_score,
                        hyp.score + gumbel_score,
                    )
                )
        # Keep the top K expansions
        candidates.sort(key=operator.itemgetter(3), reverse=True)
        if early_stop:
            candidates = candidates[: sample_size - len(finished)]
        else:
            candidates = candidates[: sample_size]

        # Create the new hypotheses from the expansions
        beam = []
        for hyp, choice, choice_score, cum_score in candidates:
            inference_state = hyp.inference_state.clone()
            next_choices = inference_state.step(choice)
            new_hyp = Hypothesis(
                inference_state=inference_state,
                next_choices=next_choices,
                score=cum_score,
                choice_history=hyp.choice_history + [choice],
                score_history=hyp.score_history + [choice_score],
            )
            if next_choices is None:
                finished.append(new_hyp)
            else:
                beam.append(new_hyp)
    finished.sort(key=operator.attrgetter("score"), reverse=True)
    return finished[: sample_size]  # not returning all beams
