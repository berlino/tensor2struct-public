import pdb
import attr
import copy
import operator

import torch
import torch.nn.functional as F
from tensor2struct.utils import registry


@attr.s
class Hypothesis:
    inference_state = attr.ib()

    score_list = attr.ib(default=None)
    choices_history = attr.ib(default=None)

    def add_actions(self, actions):
        if self.choices_history is None:
            self.choices_history = [[] for _ in range(len(actions))]

        for history, action in zip(self.choices_history, actions):
            if action is not None:
                history.append(action)

    def add_scores(self, scores):
        if self.score_list is None:
            self.score_list = [[0] for _ in range(scores)]

        for i in range(len(scores)):
            self.score_list[i] += scores[i]


@registry.register("infer_method", "batched_greedy_search")
def batched_greedy_search(model, orig_items, preproc_items, beam_size, max_steps):
    assert beam_size == 1
    enc_items, dec_items = preproc_items
    ret_state = model(enc_items, compute_loss=False, infer=True)
    inference_state, next_choices_list = (
        ret_state["initial_state"],
        ret_state["initial_choices_list"],
    )
    assert getattr(inference_state, "batched")

    bs = len(enc_items)
    hyp = Hypothesis(inference_state, score_list=[0.0] * bs)
    for _ in range(max_steps):
        if next_choices_list is None:
            break

        chosen_candidates = []
        for batch_idx in range(bs):
            candidates = [
                (choice, hyp.score_list[batch_idx] + choice_score.item())
                for choice, choice_score in next_choices_list[batch_idx]
            ]
            best_one = max(candidates, key=lambda x: x[1])
            chosen_candidates.append(best_one)

        cur_actions = [item[0] for item in chosen_candidates]
        cur_scores = [item[1] for item in chosen_candidates]
        hyp.add_actions(cur_actions)
        hyp.add_scores(cur_scores)

        next_choices_list = inference_state.step(cur_actions)

    codes_list = hyp.inference_state.finalize()
    return codes_list


@registry.register("infer_method", "batched_greedy_search_v")
def batched_greedy_search_v(model, orig_items, preproc_items, beam_size, max_steps):
    """
    Compared with batched_greedy_search, intermedia representation is vectorized
    """
    assert beam_size == 1

    enc_items, dec_items = preproc_items
    ret_state = model(enc_items, compute_loss=False, infer=True)
    inference_state, next_choices_list = (
        ret_state["initial_state"],
        ret_state["initial_choices_list"],
    )
    assert getattr(inference_state, "batched")

    for _ in range(max_steps):
        if next_choices_list is None:
            break

        # bs * candidate_num
        candidates_v, candidates_score_v = next_choices_list
        next_choices_list = inference_state.step(candidates_v)

    codes_list = inference_state.finalize()
    return codes_list

@registry.register("infer_method", "batched_classify")
def beam_search(model, orig_items, preproc_items, beam_size, max_steps):
    enc_items, dec_items = preproc_items
    ret_state = model(enc_items, compute_loss=False, infer=True)
    return ret_state["predictions"].tolist()
