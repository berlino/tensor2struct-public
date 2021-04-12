import attr
import copy
import operator

import torch
import torch.nn.functional as F
from tensor2struct.utils import registry


@attr.s
class Hypothesis:
    inference_state = attr.ib()
    next_choices = attr.ib()
    score = attr.ib(default=0)

    choice_history = attr.ib(factory=list)
    score_history = attr.ib(factory=list)


@registry.register("infer_method", "beam_search")
def beam_search(model, orig_item, preproc_item, beam_size, max_steps):
    # the following line only works for unbatched 
    # ret_state = model(orig_item, preproc_item, compute_loss=False, infer=True)
    # inference_state, next_choices = (
    #     ret_state["initial_state"],
    #     ret_state["initial_choices"],
    # )

    inference_state, next_choices = model.begin_inference(orig_item, preproc_item)

    beam = [Hypothesis(inference_state, next_choices)]
    finished = []

    for step in range(max_steps):
        # Check if all beams are finished
        if len(finished) == beam_size:
            break

        candidates = []

        # For each hypothesis, get possible expansions
        # Score each expansion
        for hyp in beam:
            candidates += [
                (hyp, choice, choice_score.item(), hyp.score + choice_score.item())
                for choice, choice_score in hyp.next_choices
            ]

        # Keep the top K expansions
        candidates.sort(key=operator.itemgetter(3), reverse=True)
        candidates = candidates[: beam_size - len(finished)]

        # Create the new hypotheses from the expansions
        beam = []
        for hyp, choice, choice_score, cum_score in candidates:
            inference_state = hyp.inference_state.clone()
            next_choices = inference_state.step(choice)
            if next_choices is None:
                finished.append(
                    Hypothesis(
                        inference_state,
                        None,
                        cum_score,
                        hyp.choice_history + [choice],
                        hyp.score_history + [choice_score],
                    )
                )
            else:
                beam.append(
                    Hypothesis(
                        inference_state,
                        next_choices,
                        cum_score,
                        hyp.choice_history + [choice],
                        hyp.score_history + [choice_score],
                    )
                )

    if len(finished) == 1 and len(finished[0].score_history) <= 2:
        import pdb; pdb.set_trace()
    finished.sort(key=operator.attrgetter("score"), reverse=True)
    return finished
