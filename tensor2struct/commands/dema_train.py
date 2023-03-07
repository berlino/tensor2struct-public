import argparse
import collections
import copy
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
from tensor2struct.training import dema
from tensor2struct.utils import registry, random_state, vocab
from tensor2struct.utils import saver as saver_mod


@attr.s
class DEMATrainConfig(train.TrainConfig):
    # kw_only is required for inheritance
    num_particles = attr.ib(default=2)


class DEMATrainer(train.Trainer):
    def load_train_config(self):
        self.train_config = registry.instantiate(
            DEMATrainConfig, self.config["dema_train"]
        )

        if self.train_config.num_batch_accumulated > 1:
            self.logger.warn("Batch accumulation is used only at MAML-step level")
            raise NotImplementedError

    def load_optimizer(self, config):
        with self.init_random:
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
            dema_trainer = dema.DEMA(
                device=self.device,
                num_particles=self.train_config.num_particles
            )
            return optimizer, lr_scheduler, dema_trainer

    def step(
        self,
        config,
        train_data_loader,
        optimizer,
        lr_scheduler,
        last_step,
        dema_trainer,
    ):
        with self.model_random:
            
            for p in self.model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            model_encoder_params = []
            for i in range(self.train_config.num_particles):
                model_encoder_params.append(list(self.model.list_of_encoders[i].parameters()))
            
            # alignment matrix params
            model_aligner_params = list(self.model.aligner.parameters())
            model_decoder_params = list(self.model.decoder.parameters())
            for _i in range(self.train_config.num_batch_accumulated):
                batch = next(train_data_loader)
                ret_dic = dema_trainer(
                    self.model,
                    model_encoder_params,
                    model_aligner_params,
                    model_decoder_params,
                    batch,
                    self.train_config.num_batch_accumulated
                )

            if self.train_config.clip_grad:
                self.logger.warn("Clip grad is only designed for BERT finetune")

            optimizer.step()
            new_lr = lr_scheduler.update_lr(last_step)
            optimizer.zero_grad()

            if new_lr is None:
                new_lr = [param["lr"] for param in optimizer.param_groups]

            if last_step % self.train_config.report_every_n == 0:
                self.logger.info("Step {}: loss={:.4f}".format(last_step, loss.item()))
                self.logger.info(f"Step {last_step}, lr={new_lr}")
                wandb.log({"train_loss": ret_dic["loss"]}, step=last_step)
                for i in range(len(new_lr)):
                    wandb.log({f"lr_{i}": new_lr[i]}, step=last_step)

    def train(self, config, modeldir):
        optimizer, lr_scheduler, dema_trainer = self.load_optimizer(
            config
        )
        saver, last_step = self.load_saver(
            config,
            modeldir,
            optimizer=optimizer,
            dema_trainer=dema_trainer,
        )

        train_data_loader = self.load_train_data()
        train_eval_data_loader, val_data_loader = self.load_eval_data()

        # 5. Start training loop
        with self.data_random:
            while last_step < self.train_config.max_steps:

                self.eval_model(last_step, train_eval_data_loader, val_data_loader)
                self.step(config, train_data_loader, optimizer, lr_scheduler, last_step)
                last_step += 1
                self.save_state(saver, modeldir, last_step)

            saver.save(modeldir, last_step)


def main(args):
    # setup logger etc
    config, logger = train.setup(args)

    # Construct trainer and do training
    trainer = DEMATrainer(logger, config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == "__main__":
    args = train.add_parser()
    main(args)
