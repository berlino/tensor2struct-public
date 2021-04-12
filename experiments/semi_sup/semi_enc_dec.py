import os
import copy
import attr
import logging
import collections

import torch
import higher
from torch import nn
import torch.utils.data

from entmax import sparsemax

from tensor2struct.utils import registry
from tensor2struct.models import enc_dec
from tensor2struct.datasets import overnight

from experiments.semi_sup import unsup_enc_dec

logger = logging.getLogger("tensor2struct")


@registry.register("model", "SemiSupEncDec")
class SemiSupEncDecModel(unsup_enc_dec.UnSupEncDecModel):
    """
    Scheduler to collect pseudo labels, see test case for usage
    """

    def __init__(
        self, preproc, device, encoder, decoder, search_scheduler, unsup_config
    ):
        super().__init__(preproc, device, encoder, decoder, search_scheduler)

        # unsupervised loss config
        self.enable_unsup_loss = unsup_config["enable_unsup_loss"]
        self.alpha = unsup_config["alpha"]
        self.unsup_loss_type = unsup_config["unsup_loss_type"]
        self.eps = 1e-6

    def forward(self, *input_items, compute_loss=True, infer=False):
        "The only entry point of encdec"
        ret_dic = {}
        if compute_loss:
            batch = input_items[0]
            unlabel_batch = None if len(input_items) == 1 else input_items[1]
            loss = self._compute_loss_enc_batched(batch, unlabel_batch)
            ret_dic["loss"] = loss

            if unlabel_batch:
                summary = self._summarize(self.debug_stat)
                logger.info(f"Global stat: {summary}")
                ret_dic["summary"] = summary

        if infer:
            len(input_items) == 2  # unbatched version of inference
            orig_item, preproc_item = input_items
            infer_dic = self.begin_inference(orig_item, preproc_item)
            ret_dic = {**ret_dic, **infer_dic}
        return ret_dic

    def _compute_loss_enc_batched(self, batch, unlabel_batch=None):
        """
        Compute supervised loss for batch and unsupervised loss for unlabel batch

        """
        if batch:
            sup_loss = super(
                unsup_enc_dec.UnSupEncDecModel, self
            )._compute_loss_enc_batched(batch)
        else:
            assert self.enable_unsup_loss and unlabel_batch is not None
            sup_loss = None

        # eval on train set
        if not self.training or not self.enable_unsup_loss or unlabel_batch is None:
            return sup_loss

        assert self.unsup_loss_type != "gradsim"
        unsup_loss = self.compute_unsup_loss_by_beam_search(
            unlabel_batch, self.unsup_loss_type
        )

        if sup_loss:
            return self.alpha * unsup_loss + sup_loss
        else:
            return self.alpha * unsup_loss

    def compute_unsup_loss_by_beam_search(self, batch, unsup_loss_type="topk"):
        """
        Compute unsup loss using beam search
        s1, s2: beam search retrieved programs
        s1, s3: plausible/executable programs
        """
        losses = []

        for enc_item, dec_item in batch:
            example = unsup_enc_dec.Example(dec_item["domain"])

            # obtain seqs in eval mode
            # with torch.no_grad():
            #     self.eval()
            #     sampled_seqs, sampled_seq_log_probs = self.get_executable_seqs(
            #         example, (enc_input, _dec_output)
            #     )
            #     self.train()

            # obtain seqs in train mode, more efficient
            s1_seqs, s1_log_probs, s2_log_probs = self.get_executable_seqs(
                example, (enc_item, dec_item)
            )
            assert len(s1_seqs) == len(s1_log_probs)

            if len(s1_log_probs) > 0 and len(s2_log_probs) > 0:
                s1_log_prob = torch.logsumexp(torch.stack(s1_log_probs, dim=0), dim=0)
                s2_log_prob = torch.logsumexp(torch.stack(s2_log_probs, dim=0), dim=0)
                s1s2_log_prob = torch.logsumexp(
                    torch.stack([s1_log_prob, s2_log_prob], dim=0), dim=0
                )
                s3s4_log_prob = torch.log(1 - torch.exp(s1s2_log_prob))

                # compute different q
                l0 = -s1_log_probs[0]
                l1 = -s1_log_prob

                s1_34_logits = [s1_log_prob, s3s4_log_prob]
                s1_34_log_p_v = torch.stack(s1_34_logits, dim=0)
                q_l2 = torch.softmax(s1_34_log_p_v.detach(), dim=0)
                l2 = (-q_l2 * s1_34_log_p_v).sum()

                q_l3 = torch.exp(s1s2_log_prob.detach())
                l3 = -q_l3 * s1_log_prob - (1 - q_l3) * s3s4_log_prob

                if self.unsup_loss_type == "self-train":
                    losses.append(l0)
                elif self.unsup_loss_type == "top-k":
                    losses.append(l1)
                elif self.unsup_loss_type == "repulsion":
                    losses.append(l2)
                elif self.unsup_loss_type == "gentle":
                    losses.append(l3)
                elif self.unsup_loss_type == "sparse":
                    l5_logits = torch.stack(s1_log_probs, dim=0)
                    q_l5 = sparsemax(l5_logits.detach(), dim=0)
                    l5 = (-q_l5 * l5_logits).sum()
                    losses.append(l5)
                else:
                    raise NotImplementedError

            elif len(s1_log_probs) == 0 and len(s2_log_probs) > 0:
                s2_log_prob = torch.logsumexp(torch.stack(s2_log_probs, dim=0), dim=0)
                s3s4_log_prob = torch.log(1 - torch.exp(s2_log_prob))

                l2 = -s3s4_log_prob
                losses.append(l2)  # which means this is the only valid loss
            elif len(s1_log_probs) > 0 and len(s2_log_probs) == 0:
                s1_log_prob = torch.logsumexp(torch.stack(s1_log_probs, dim=0), dim=0)
                s3s4_log_prob = torch.log(1 - torch.exp(s1_log_prob))

                l0 = -s1_log_probs[0]
                l1 = -s1_log_prob

                if self.unsup_loss_type == "self-train":
                    losses.append(l0)
                elif self.unsup_loss_type == "top-k":
                    losses.append(l1)
                elif self.unsup_loss_type == "replusion":
                    # l2, l3 loss would result in zero gradient
                    continue
                elif self.unsup_loss_type == "gentle":
                    continue
                elif self.unsup_loss_type == "sparse":
                    l5_logits = torch.stack(s1_log_probs, dim=0)
                    q_l5 = sparsemax(l5_logits.detach(), dim=0)
                    l5 = (-q_l5 * l5_logits).sum()
                    losses.append(l5)
                else:
                    raise NotImplementedError
            else:
                logger.warn("semi_enc_dec obtains empty seqs from searching")
                continue

        if len(losses) == 0:
            return torch.Tensor([1]).to(self._device).requires_grad_()
        return torch.mean(torch.stack(losses, dim=0), dim=0)

    def collect_seq_and_orig_losses(self, batch):
        seqs = []
        losses = []
        enc_states = self.encoder([enc_input for enc_input, dec_output in batch])

        for enc_state, (enc_input, _dec_output) in zip(enc_states, batch):
            example = unsup_enc_dec.Example(_dec_output["domain"])
            with torch.no_grad():
                seqs_of_one_example, _ = self.get_executable_seqs(
                    example, (enc_input, _dec_output)
                )
            seqs.append(seqs_of_one_example)

            _loss = []
            for seq in seqs_of_one_example:
                dec_output = {"domain": _dec_output["domain"], "productions": seq}
                ret_dict = self.decoder(dec_output, enc_state)
                _loss.append(ret_dict["loss"])
            losses.append(_loss)
        return seqs, losses

    def collect_update_losses(self, model, batch, seqs):
        """
        Inefficient way of obtaining loss
        """
        losses = []
        for seqs_of_one_example, (enc_input, _dec_output) in zip(seqs, batch):
            loss_l_of_one_example = []
            for seq in seqs_of_one_example:
                _loss = self.get_loss_of_single_example(
                    model, enc_input, _dec_output, seq
                )
                loss_l_of_one_example.append(_loss)
            losses.append(loss_l_of_one_example)
        return losses

    @staticmethod
    def get_loss_of_single_example(model, enc_input, _dec_output, seq):
        dec_output = {"domain": _dec_output["domain"], "productions": seq}
        loss_dict = model([[enc_input, dec_output]])
        return loss_dict["loss"]
