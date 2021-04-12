import argparse
import collections
import datetime
import json
import os
import wandb
import logging

import _jsonnet
import attr
import random
import torch
import pickle

from tensor2struct.utils import random_state, registry
from tensor2struct.utils import saver as saver_mod
from tensor2struct.commands import train
from tensor2struct.models.enc_dec import ZippedDataset


@attr.s
class TrainConfig(train.TrainConfig):
    use_bert_training = attr.ib(kw_only=True)
    use_kd_train = attr.ib(kw_only=True, default=False)
    lambda_mixture = attr.ib(kw_only=True, default=None)
    check_syn_consistency = attr.ib(kw_only=True, default=False)
    data_scheduler = attr.ib(kw_only=True)


class Trainer(train.Trainer):
    def load_train_config(self):
        self.train_config = registry.instantiate(TrainConfig, self.config["train"])

    def load_optimizer(self, config):
        with self.init_random:
            if self.train_config.use_bert_training:
                bert_params = self.model.get_bert_parameters()
                non_bert_params = self.model.get_non_bert_parameters()
                assert len(non_bert_params) + len(bert_params) == len(
                    list(self.model.parameters())
                )
                optimizer = registry.construct(
                    "optimizer",
                    config["optimizer"],
                    non_bert_params=non_bert_params,
                    bert_params=bert_params,
                )
                lr_scheduler = registry.construct(
                    "lr_scheduler",
                    config.get("lr_scheduler", {"name": "noop"}),
                    param_groups=[
                        optimizer.non_bert_param_group,
                        optimizer.bert_param_group,
                    ],
                )
            else:
                optimizer = registry.construct(
                    "optimizer",
                    config["optimizer"],
                    params=self.model.get_trainable_parameters(),
                )
                lr_scheduler = registry.construct(
                    "lr_scheduler",
                    config.get("lr_scheduler", {"name": "noop"}),
                    param_groups=optimizer.param_groups,
                )
            return optimizer, lr_scheduler

    def load_train_data(self):
        with self.data_random:
            train_data = self.model_preproc.dataset("train")
            syn_train_data = self.model_preproc.dataset("syn_train")
            self.logger.info(
                f"Load {len(train_data)} orig examples and {len(syn_train_data)} synthetic examples"
            )

            assert not self.train_config.use_kd_train
            assert not self.train_config.check_syn_consistency

            train_data_scheduler = registry.construct(
                "syn_data_scheduler",
                self.train_config.data_scheduler,
                examples=train_data,
                syn_examples=syn_train_data,
                batch_size=self.train_config.batch_size,
                warm_up_steps=self.config["lr_scheduler"]["num_warmup_steps"],
                decay_steps=self.config["train"]["max_steps"]
                - 2 * self.config["lr_scheduler"]["num_warmup_steps"],
            )
        return train_data_scheduler

    def _inner_forward(self, model, batch):
        """
        Compute loss with pretrained-logits if using KD
        """
        if not self.train_config.use_kd_train:
            return model(batch)

        enc_states = model.encoder([enc_input for enc_input, dec_output in batch])
        losses = []
        for enc_state, (enc_input, dec_output) in zip(enc_states, batch):
            assert dec_output.kd_logits is not None
            try:
                loss = model.decoder.compute_mle_loss(
                    dec_output,
                    enc_state,
                    kd_logits=dec_output.kd_logits,
                    lambda_mixture=self.train_config.lambda_mixture,
                )
            except Exception as e:
                self.logger.warn(f"foward error {str(e)}")
                loss = torch.Tensor([0.0]).to(model.decoder._device)
            losses.append(loss)
        avg_loss = torch.mean(torch.stack(losses, dim=0), dim=0)
        return {"loss": avg_loss}

    def step(self, config, train_data_scheduler, optimizer, lr_scheduler, last_step):
        with self.model_random:
            for _i in range(self.train_config.num_batch_accumulated):
                batch, ratio = train_data_scheduler.get_batch(last_step)
                ret_dic = self._inner_forward(self.model, batch)
                loss = ratio * ret_dic["loss"]
                norm_loss = loss / self.train_config.num_batch_accumulated
                norm_loss.backward()

            # clip grad for both bert and non-bert params, as syn data would break the gradients somehow
            if self.train_config.clip_grad and self.train_config.use_bert_training:
                for param_group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(
                        param_group["params"], self.train_config.clip_grad,
                    )

            optimizer.step()
            new_lr = lr_scheduler.update_lr(last_step)
            optimizer.zero_grad()

            if last_step % self.train_config.report_every_n == 0:
                self.logger.info("Step {}: loss={:.4f}".format(last_step, loss.item()))
                self.logger.info(f"Step {last_step}, lr={new_lr}")
                wandb.log({"train_loss": loss.item()}, step=last_step)
                for i in range(len(new_lr)):
                    wandb.log({f"lr_{i}": new_lr[i]}, step=last_step)


def main(args):
    # setup logger etc
    config, logger = train.setup(args)

    # Construct trainer and do training
    trainer = Trainer(logger, config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == "__main__":
    args = train.add_parser()
    main(args)
