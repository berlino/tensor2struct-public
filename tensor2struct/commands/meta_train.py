import argparse
import collections
import datetime
import json
import os
import logging
import itertools

import _jsonnet
import attr
import numpy as np
import torch
import wandb

from tensor2struct.commands import train
from tensor2struct.training import maml
from tensor2struct.utils import registry, random_state, vocab
from tensor2struct.utils import saver as saver_mod


@attr.s
class MetaTrainConfig(train.TrainConfig):
    # kw_only is required for inheritance
    inner_opt = attr.ib(kw_only=True)
    first_order = attr.ib(kw_only=True, default=False)
    data_scheduler = attr.ib(kw_only=True)


class MetaTrainer(train.Trainer):
    def load_train_config(self):
        self.train_config = registry.instantiate(
            MetaTrainConfig, self.config["meta_train"]
        )

        if self.train_config.num_batch_accumulated > 1:
            self.logger.warn("Batch accumulation is used only at MAML-step level")
            raise NotImplementedError

    def load_optimizer(self, config):
        with self.init_random:
            # 0. create inner_optimizer
            inner_parameters = self.model.get_trainable_parameters()
            inner_optimizer = registry.construct(
                "optimizer", self.train_config.inner_opt, params=inner_parameters
            )

            # 1. MAML trainer, might add new parameters to the optimizer, e.g., step size
            maml_trainer = maml.MAML(
                model=self.model,
                inner_opt=inner_optimizer,
                device=self.device,
                first_order=self.train_config.first_order,
            )
            maml_trainer.to(self.device)

            opt_params = maml_trainer.get_inner_opt_params()
            self.logger.info(f"{len(opt_params)} opt meta parameters")

            # 2. Outer optimizer
            optimizer = registry.construct(
                "optimizer",
                config["optimizer"],
                params=itertools.chain(
                    self.model.get_trainable_parameters(), opt_params
                ),
            )

            lr_scheduler = registry.construct(
                "lr_scheduler",
                config.get("lr_scheduler", {"name": "noop"}),
                param_groups=optimizer.param_groups,
            )
            return inner_optimizer, maml_trainer, optimizer, lr_scheduler

    def load_train_data(self):
        with self.data_random:
            train_data = self.model_preproc.dataset("train")
            train_data_scheduler = registry.construct(
                "data_scheduler",
                self.train_config.data_scheduler,
                examples=train_data,
                max_train_step=self.train_config.max_steps,
            )
        return train_data_scheduler

    def step(
        self,
        config,
        train_data_scheduler,
        maml_trainer,
        optimizer,
        lr_scheduler,
        last_step,
    ):
        task = train_data_scheduler.get_batch(last_step)
        with self.model_random:
            inner_batch, outer_batches = task
            ret_dic = maml_trainer.meta_train(self.model, inner_batch, outer_batches)
            loss = ret_dic["loss"]

            if self.train_config.clip_grad:
                self.logger.warn("Clip grad is only designed for BERT finetune")

            optimizer.step()
            optimizer.zero_grad()

            # log lr for each step
            outer_lr = lr_scheduler.update_lr(last_step)
            if outer_lr is None:
                outer_lr = [param["lr"] for param in optimizer.param_groups]
            inner_lr = [param["lr"] for param in maml_trainer.inner_opt.param_groups]

        # Report metrics and lr
        if last_step % self.train_config.report_every_n == 0:
            self.logger.info("Step {}: loss={:.4f}".format(last_step, loss))
            self.logger.info(f"Step {last_step}, lr={inner_lr, outer_lr}")
            wandb.log({"train_loss": loss}, step=last_step)
            for idx, lr in enumerate(inner_lr):
                wandb.log({f"inner_lr_{idx}": lr}, step=last_step)
            for idx, lr in enumerate(outer_lr):
                wandb.log({f"outer_lr_{idx}": lr}, step=last_step)

    def train(self, config, modeldir):
        inner_optimizer, maml_trainer, optimizer, lr_scheduler = self.load_optimizer(
            config
        )
        saver, last_step = self.load_saver(
            config,
            modeldir,
            inner_optimizer=inner_optimizer,
            optimizer=optimizer,
            maml_trainner=maml_trainer,
        )

        train_data_scheduler = self.load_train_data()
        train_eval_data_loader, val_data_loader = self.load_eval_data()

        # 5. Start training loop
        with self.data_random:
            while last_step < self.train_config.max_steps:
                self.eval_model(last_step, train_eval_data_loader, val_data_loader)

                try:
                    self.step(
                        config,
                        train_data_scheduler,
                        maml_trainer,
                        optimizer,
                        lr_scheduler,
                        last_step,
                    )
                    last_step += 1
                    self.save_state(saver, modeldir, last_step)
                except RuntimeError as e:
                    # it seems to work for meta-train
                    err_msg = str(e)
                    self.logger.warn(f"Step Failed {err_msg}")

            saver.save(modeldir, last_step)


def main(args):
    # setup logger etc
    config, logger = train.setup(args)

    # Construct trainer and do training
    trainer = MetaTrainer(logger, config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == "__main__":
    args = train.add_parser()
    main(args)
