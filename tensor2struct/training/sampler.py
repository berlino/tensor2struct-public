import torch
import copy
import operator

import attr
from tensor2struct.models.beam_search import Hypothesis


def sample_seq_topk(
    model,
    orig_item,
    preproc_item,
    sample_size=1,
    max_steps=1000,
    gumbel_temperature=0.1,
):
    inference_state, next_choices = model.begin_inference(orig_item, preproc_item)
    beam = [Hypothesis(inference_state, next_choices)]
    finished = []

    for step in range(max_steps):
        # Check if all beams are finished
        if len(finished) == sample_size:
            break

        candidates = []
        for hyp in beam:
            for i, (choice, choice_score) in enumerate(hyp.next_choices):
                candidates.append(
                    (hyp, choice, choice_score, hyp.score + choice_score.item())
                )

        # Keep the top K expansions
        candidates.sort(key=operator.itemgetter(3), reverse=True)
        candidates = candidates[: sample_size - len(finished)]

        # Create the new hypotheses from the expansions
        beam = []
        for hyp, choice, choice_score, hyp_score in candidates:
            inference_state = hyp.inference_state.clone()
            next_choices = inference_state.step(choice)
            if next_choices is None:
                finished.append(
                    Hypothesis(
                        inference_state,
                        None,
                        hyp_score,
                        hyp.choice_history + [choice],
                        hyp.score_history + [choice_score],
                    )
                )
            else:
                beam.append(
                    Hypothesis(
                        inference_state,
                        next_choices,
                        hyp_score,
                        hyp.choice_history + [choice],
                        hyp.score_history + [choice_score],
                    )
                )

    finished.sort(key=operator.attrgetter("score"), reverse=True)
    return finished


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -1 * torch.log(-torch.log(U + eps) + eps)


def gumbel_log_softmax_sample(logits, temperature, device):
    y = logits + sample_gumbel(logits.shape, device)
    return torch.nn.functional.log_softmax(y / temperature, dim=-1)


def sample_seq_with_gumbel(
    model, orig_item, preproc_item, sample_size=1, max_steps=1000, gumbel_temperature=1
):
    inference_state, next_choices = model.begin_inference(orig_item, preproc_item)
    beam = [Hypothesis(inference_state, next_choices)]
    finished = []

    assert sample_size == 1

    for step in range(max_steps):
        # Check if all beams are finished
        if len(finished) == sample_size:
            break

        candidates = []
        for hyp in beam:
            orig_score = [choice_score for _, choice_score in hyp.next_choices]
            orig_score_v = torch.stack(orig_score, dim=0)
            perturbed_score_v = gumbel_log_softmax_sample(
                orig_score_v, gumbel_temperature, model.decoder._device
            )

            for i, (choice, real_choice_score) in enumerate(hyp.next_choices):
                # gumbel_loss = -1 * torch.exp(perturbed_score_v) * orig_score_v
                candidates.append(
                    (hyp, choice, real_choice_score, perturbed_score_v[i].item())
                )

        # Keep the top K expansions
        candidates.sort(key=operator.itemgetter(3), reverse=True)
        candidates = candidates[: sample_size - len(finished)]

        # Create the new hypotheses from the expansions
        beam = []
        for hyp, choice, choice_score, perturbed_score in candidates:
            inference_state = hyp.inference_state.clone()
            next_choices = inference_state.step(choice)
            cum_score = hyp.score + choice_score.item()
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
