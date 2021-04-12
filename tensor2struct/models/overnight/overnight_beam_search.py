import attr
import copy
import operator

import torch
import torch.nn.functional as F

from tensor2struct.datasets import overnight
from tensor2struct.utils import registry
import tensor2struct.languages.dsl.common.errors as lf_errors
import tensor2struct.languages.dsl.common.util as lf_util


@attr.s
class Hypothesis:
    inference_state = attr.ib()
    next_choices = attr.ib()
    score = attr.ib(default=0)

    choice_history = attr.ib(factory=list)
    score_history = attr.ib(factory=list)


@attr.s
class Candidate:
    hyp = attr.ib()
    choice = attr.ib()
    choice_score = attr.ib()
    cum_score = attr.ib()


@registry.register("infer_method", "overnight_beam_search")
def overnight_beam_search(model, orig_item, preproc_item, beam_size, max_steps):
    """
    Beam search and finally filtered with execution 
    """
    orig_beam_size = beam_size
    beam_size = beam_size * 2

    ret_state = model(orig_item, preproc_item, compute_loss=False, infer=True)
    inference_state, next_choices = (
        ret_state["initial_state"],
        ret_state["initial_choices"],
    )
    beam = [Hypothesis(inference_state, next_choices)]
    finished = []

    for step in range(max_steps):
        if len(finished) == beam_size:
            break

        candidates = []
        for hyp in beam:
            candidates += [
                Candidate(
                    hyp, choice, choice_score.item(), hyp.score + choice_score.item()
                )
                for choice, choice_score in hyp.next_choices
            ]

        # Keep the top K expansions
        candidates.sort(key=operator.attrgetter("cum_score"), reverse=True)
        candidates = candidates[: beam_size - len(finished)]

        # Create the new hypotheses from the expansions
        beam = []
        for candidate in candidates:
            inference_state = candidate.hyp.inference_state.clone()
            next_choices = inference_state.step(candidate.choice)
            new_hyp = Hypothesis(
                inference_state,
                next_choices,
                candidate.cum_score,
                candidate.hyp.choice_history + [candidate.choice],
                candidate.hyp.score_history + [candidate.choice_score],
            )
            if next_choices is None:
                finished.append(new_hyp)
            else:
                beam.append(new_hyp)

    # filter by execution
    lfs = []
    for hyp in finished:
        _, lf = hyp.inference_state.finalize()
        lfs.append(lf)
    denotations = overnight.execute(lfs, orig_item.domain)

    executables = []
    for beam, d in zip(finished, denotations):
        if d is not None:
            executables.append(beam)

    executables.sort(key=operator.attrgetter("score"), reverse=True)
    executables = executables[:orig_beam_size]
    return executables


def have_mentioned_vp(prods, mentions):
    """
    Heursitics to make sure that mentioned entities and propertied are predicted 
    """
    if len(mentions["exact"]["property"]) > 0 and not all(
        any(v in prod for prod in prods) for v in mentions["exact"]["property"]
    ):
        em_p_flag = False
    else:
        em_p_flag = True

    if len(mentions["exact"]["value"]) > 0 and not all(
        any(v in prod for prod in prods) for v in mentions["exact"]["value"]
    ):
        em_v_flag = False
    else:
        em_v_flag = True

    if len(mentions["partial"]["property"]) > 0 and not all(
        any(v in prod for prod in prods) for v in mentions["partial"]["property"]
    ):
        pa_p_flag = False
    else:
        pa_p_flag = True

    if len(mentions["partial"]["value"]) > 0 and not all(
        any(v in prod for prod in prods) for v in mentions["partial"]["value"]
    ):
        pa_v_flag = False
    else:
        pa_v_flag = True

    # if all([em_v_flag, em_p_flag, pa_p_flag, pa_v_flag]):
    if em_p_flag:
        return True
    else:
        return False
