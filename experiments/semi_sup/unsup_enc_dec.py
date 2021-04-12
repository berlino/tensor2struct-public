import os
import attr
import math
import logging
import collections

import torch
from torch import nn
import torch.utils.data

from tensor2struct.utils import registry
from tensor2struct.models import enc_dec
from tensor2struct.datasets import overnight

logger = logging.getLogger("tensor2struct")


@attr.s
class Example:
    """
    Inference requires the orig data but model does not have that,
    so we have a fake example created. But this logic should be changed.
    """

    domain = attr.ib()


@attr.s
class Buffer:
    """
    Use factory, instead of default=[], which is actually a global variable
    """

    prods_set = attr.ib(default=attr.Factory(set))
    prods2probs = attr.ib(default=attr.Factory(dict))
    max_size = attr.ib(default=32)

    def add_item(self, prods, prob):
        if prods not in self.prods2probs:
            self.prods_set.add(prods)
        self.prods2probs[prods] = prob

    def get_items(self):
        if len(self.prods_set) < self.max_size:
            return list(self.prods_set)
        else:
            all_prods = list(self.prods_set)
            sorted_prods = sorted(all_prods, key=lambda x: -self.prods2probs[x])

            ret_list = []
            for i, prods in enumerate(sorted_prods):
                if i < self.max_size:
                    ret_list.append(prods)
                else:
                    self.prods_set.remove(prods)
                    del self.prods2probs[prods]
            return ret_list


@registry.register("search_scheduler", "online")
class OnlineSearchScheduler:
    """
    It's designed to use different search strategy based on training steps,
    but it turns out that it makes trainig much more complex, so currently,
    the search config is fixed.
    """

    def __init__(
        self, sample_size=16, use_cache=False, use_gumbel=False, top_p=0.99, ratio=1.2,
    ):
        self.sample_size = sample_size
        self.use_gumbel = use_gumbel
        self.ratio = ratio
        self.top_p = top_p

        self.counter = collections.Counter()

        self.use_cache = use_cache
        if use_cache:
            self.buffer = {}

    def step(self, question):
        infer_config = {}
        infer_config["need_new_sample"] = True
        self.counter[question] += 1
        cur_step = self.counter[question]
        infer_config["num_epoch"] = cur_step

        infer_config["use_gumbel"] = self.use_gumbel
        infer_config["ratio"] = self.ratio
        infer_config["top_p"] = self.top_p
        infer_config["sample_size"] = self.sample_size
        return infer_config

    def update(self, question, prods_list):
        if self.use_cache:
            # only keep a fixed-size list of programs for efficiency
            self.buffer[question] = prods_list[: self.sample_size]

    def get_cached_samples(self, question):
        if self.use_cache and question in self.buffer:
            return self.buffer[question]
        else:
            return []

    def need_warmup(self, question):
        return self.use_cache and question not in self.buffer


@registry.register("model", "UnSupEncDec")
class UnSupEncDecModel(enc_dec.SemiBatchedEncDecModel):
    """
    Scheduler to collect pseudo labels, see test case for usage
    """

    def __init__(self, preproc, device, encoder, decoder, search_scheduler):
        super().__init__(preproc, device, encoder, decoder)
        self._device = device
        self.search_scheduler = registry.construct("search_scheduler", search_scheduler)
        self.debug_stat = collections.Counter()

    @staticmethod
    def _summarize(debug_stat):
        ret_dic = {}

        if debug_stat["num_all_exs"] > 0:
            ret_dic["coverage_by_search"] = (
                debug_stat["num_of_covered_exs_by_search"] / debug_stat["num_all_exs"]
            )
            ret_dic["coverage_after_filter"] = (
                debug_stat["num_of_covered_exs_after_filter"]
                / debug_stat["num_all_exs"]
            )

        if debug_stat["num_of_pseudo_labels"] > 0:
            ret_dic["percents_of_executable_programs"] = (
                debug_stat["num_of_pseudo_labels_after_filter"]
                / debug_stat["num_of_pseudo_labels"]
            )

        ret_dic["avg_gold_ratio"] = debug_stat["avg_gold_ratio"]
        ret_dic["avg_ratio"] = debug_stat["avg_ratio"]
        ret_dic["avg_prob_mass"] = debug_stat["avg_prob_mass"]
        ret_dic["avg_prob_mass_filtered"] = debug_stat["avg_prob_mass_filtered"]

        return ret_dic

    def forward(self, *input_items, compute_loss=True, infer=False):
        "The only entry point of encdec"
        ret_dic = {}
        if compute_loss:
            assert len(input_items) == 1  # it's a batched version
            loss = self.compute_loss(input_items[0])
            ret_dic["loss"] = loss

            if self.training:
                summary = self._summarize(self.debug_stat)
                logger.info(f"Global stat: {summary}")
                ret_dic["summary"] = summary

        if infer:
            len(input_items) == 2  # unbatched version of inference
            orig_item, preproc_item = input_items
            infer_dic = self.begin_inference(orig_item, preproc_item)
            ret_dic = {**ret_dic, **infer_dic}
        return ret_dic

    def get_executable_seqs(self, example, preproc_example):
        """
        Sampling programs according beam search, etc. 
        Then filter programs based on executability and ratio
        """
        # assert not self.training

        enc_item, dec_item = preproc_example
        gold_prods = dec_item["productions"]  # for debugging purpose
        question_tokens = enc_item["question"]
        question = " ".join(question_tokens)  # used as the identity of the example

        infer_config = self.search_scheduler.step(question)
        assert infer_config["need_new_sample"]
        # if not infer_config["need_new_sample"]:
        #     return list(self.search_scheduler.get_cached_samples(question))

        if not infer_config["use_gumbel"] or self.search_scheduler.need_warmup(
            question
        ):
            infer_method = registry.lookup("infer_method", "beam_search_sampling")
            beams = infer_method(
                self,
                example,
                preproc_example,
                sample_size=infer_config["sample_size"],
                top_p=infer_config["top_p"],
            )
        else:
            infer_method = registry.lookup(
                "infer_method", "stochastic_beam_search_sampling"
            )
            beams = infer_method(
                self,
                example,
                preproc_example,
                sample_size=infer_config["sample_size"],
                max_steps=len(question_tokens) * 3,  # ratio maximally 3
                early_stop=True,
            )

        # collect logical form, production rules and log probs
        prods_list = []
        lfs = []
        log_probs = []
        for beam in beams:
            prods, lf = beam.inference_state.finalize()
            lfs.append(lf)
            prods_list.append(prods)
            log_probs.append(sum(beam.score_history))

        # filter examples by executability and ratio
        s1_prods = []
        s1_lfs = []
        s1_log_probs = []
        s2_log_probs = []
        denotations = overnight.execute(lfs, example.domain)
        for i, (prods, d) in enumerate(zip(prods_list, denotations)):
            if (
                d is not None
                and (len(prods) / len(question_tokens)) >= infer_config["ratio"]
            ):
                s1_prods.append(tuple(prods))
                s1_lfs.append(lfs[i])
                s1_log_probs.append(log_probs[i])
            else:
                s2_log_probs.append(log_probs[i])

        # retrieve cached programs
        cached_seqs = self.search_scheduler.get_cached_samples(question)
        if len(cached_seqs) > 0:
            logger.info(f"Obtain {len(cached_seqs)} seqs from cache")

        more_s1_log_probs = []
        more_s1_prods = []
        for cached_seq in cached_seqs:
            assert isinstance(cached_seq, tuple)
            if cached_seq not in s1_prods:
                enc_state = self.encoder([enc_item])[0]  # batch size 1
                _dec_item = {"domain": dec_item["domain"], "productions": cached_seq}
                ret_dict = self.decoder(_dec_item, enc_state)

                more_s1_prods.append(cached_seq)
                more_s1_log_probs.append(-ret_dict["loss"])

        # merge and sort
        s1_prods += more_s1_prods
        s1_log_probs += more_s1_log_probs

        if len(s1_prods) > 0:
            s1_prods, s1_log_probs = zip(
                *sorted(zip(s1_prods, s1_log_probs), key=lambda x: -x[1])
            )

            # update the cache
            self.search_scheduler.update(question, s1_prods)

        logger.info(
            f"Epoch {infer_config['num_epoch']}, collected {len(s1_prods)} plausible programs"
        )

        # debug information
        self.debug_stat["num_all_exs"] += 1
        self.debug_stat["num_of_pseudo_labels"] += len(prods_list)
        self.debug_stat["num_of_pseudo_labels_after_filter"] += len(s1_prods)
        self.debug_stat["num_of_tokens_for_gold_labels"] += len(question_tokens)
        self.debug_stat["num_of_actions_for_gold_labels"] += len(gold_prods)
        self.debug_stat["num_of_tokens_for_filtered_pseudo_labels"] += len(
            question_tokens
        ) * len(s1_prods)
        self.debug_stat["num_of_actions_for_filtered_pseudo_labels"] += sum(
            len(prods) for prods in s1_prods
        )
        self.debug_stat["acc_prob_mass_for_pseudo_labels"] += sum(
            math.exp(float(log_prob)) for log_prob in log_probs
        )
        self.debug_stat["acc_prob_mass_for_filtered_pseudo_labels"] += sum(
            math.exp(float(log_prob)) for log_prob in s1_log_probs
        )

        if gold_prods in prods_list:
            self.debug_stat["num_of_covered_exs_by_search"] += 1
        logger.info(
            f"Catched gold programs after beam search: {gold_prods in prods_list}"
        )

        if tuple(gold_prods) in s1_prods:
            self.debug_stat["num_of_covered_exs_after_filter"] += 1
        logger.info(
            f"Catched gold programs after filter(exe, ratio): {tuple(gold_prods) in s1_prods}"
        )

        self.debug_stat["avg_gold_ratio"] = (
            self.debug_stat["num_of_actions_for_gold_labels"]
            / self.debug_stat["num_of_tokens_for_gold_labels"]
        )
        self.debug_stat["avg_prob_mass"] = (
            self.debug_stat["acc_prob_mass_for_pseudo_labels"]
            / self.debug_stat["num_all_exs"]
        )

        if len(s1_prods) > 0:
            self.debug_stat["avg_ratio"] = (
                self.debug_stat["num_of_actions_for_filtered_pseudo_labels"]
                / self.debug_stat["num_of_tokens_for_filtered_pseudo_labels"]
            )

            self.debug_stat["avg_prob_mass_filtered"] = (
                self.debug_stat["acc_prob_mass_for_filtered_pseudo_labels"]
                / self.debug_stat["num_all_exs"]
            )

        # return
        return s1_prods, s1_log_probs, s2_log_probs

    def _compute_loss_enc_batched(self, batch):
        if not self.training:
            # eval on train set
            return super()._compute_loss_enc_batched(batch)
        return self.compute_unsup_loss_by_beam_search(batch)

    def compute_unsup_loss_by_beam_search(self, batch, use_top1=False):
        losses = []
        enc_states = self.encoder([enc_input for enc_input, dec_output in batch])

        for enc_state, (enc_input, _dec_output) in zip(enc_states, batch):
            example = Example(_dec_output["domain"])
            with torch.no_grad():
                self.eval()
                # seqs = self.get_executable_seqs(example, (enc_input, None))
                seqs, _ = self.get_executable_seqs(example, (enc_input, _dec_output))
                # seqs = [_dec_output["productions"]]
                self.train()

            _loss = []
            for seq in seqs:
                dec_output = {"domain": _dec_output["domain"], "productions": seq}
                ret_dict = self.decoder(dec_output, enc_state)
                _loss.append(ret_dict["loss"])
            if len(_loss) == 0:
                continue

            if use_top1:
                loss = min(_loss)
            else:
                loss = -1 * torch.logsumexp(-1 * torch.stack(_loss, 0), dim=0)
            losses.append(loss)
        if len(losses) == 0:
            return torch.Tensor([1]).requires_grad_()
        return torch.mean(torch.stack(losses, dim=0), dim=0)
