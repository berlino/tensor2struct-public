import attr
import copy
import operator

import torch
import torch.nn.functional as F
from tensor2struct.utils import registry, vocab


@attr.s
class Hypothesis:
    inference_state = attr.ib()
    next_choices = attr.ib()
    score = attr.ib(default=0)

    choice_history = attr.ib(factory=list)
    score_history = attr.ib(factory=list)


@registry.register("infer_method", "length_beam_search")
def beam_search(model, orig_item, preproc_item, beam_size, max_steps):
    """
    Enfore that the number of decoded items is equal to the number of encoded items
    """
    ret_state = model(orig_item, preproc_item, compute_loss=False, infer=True)
    inference_state, next_choices = (
        ret_state["initial_state"],
        ret_state["initial_choices"],
    )
    num_tokens = len(preproc_item[0]["tokens"])
    beam = [Hypothesis(inference_state, next_choices)]
    finished = []

    for step in range(max_steps):
        # Check if all beams are finished
        if len(finished) == beam_size:
            break

        candidates = []
        for hyp in beam:
            for choice, choice_score in hyp.next_choices:
                if (step < num_tokens and choice != vocab.EOS) or (
                    step == num_tokens and choice == vocab.EOS
                ):
                    candidates.append(
                        (
                            hyp,
                            choice,
                            choice_score.item(),
                            hyp.score + choice_score.item(),
                        )
                    )

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

    finished.sort(key=operator.attrgetter("score"), reverse=True)
    return finished
