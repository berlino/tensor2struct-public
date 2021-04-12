import argparse
import collections
import datetime
import json
import os
import wandb

import _jsonnet
import attr
import torch

from tensor2struct.commands import train
from tensor2struct.training import maml
from tensor2struct.utils import registry, random_state, vocab
from tensor2struct.utils import saver as saver_mod

from experiments.semi_sup import semi_enc_dec, unsup_enc_dec


@attr.s
class TrainConfig(train.TrainConfig):
    pretrain_threshold = attr.ib(default=300)


class Trainer(train.Trainer):
    def load_train_config(self):
        self.train_config = registry.instantiate(TrainConfig, self.config["train"])

    def load_train_data(self):
        with self.data_random:
            try:
                train_data = self.model_preproc.dataset("train")
                train_data_loader = self._yield_batches_from_epochs(
                    torch.utils.data.DataLoader(
                        train_data,
                        batch_size=self.train_config.batch_size,
                        shuffle=True,
                        drop_last=False,
                        collate_fn=lambda x: x,
                    )
                )
            except FileNotFoundError:
                # unsupervised learning does not have training data at all
                train_data_loader = None

            unlabel_train_data = self.model_preproc.dataset("unlabel_train")
            unlabel_train_data_loader = self._yield_batches_from_epochs(
                torch.utils.data.DataLoader(
                    unlabel_train_data,
                    batch_size=self.train_config.batch_size,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=lambda x: x,
                )
            )
        return (train_data_loader, unlabel_train_data_loader)

    def load_eval_data(self):
        with self.data_random:
            try:
                train_data = self.model_preproc.dataset("train")
                train_eval_data_loader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=self.train_config.eval_batch_size,
                    collate_fn=lambda x: x,
                )
            except FileNotFoundError:
                # unsupervised learning does not have training data at all
                train_eval_data_loader = None

            val_data = self.model_preproc.dataset("val")
            val_data_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.train_config.eval_batch_size,
                collate_fn=lambda x: x,
            )
        return train_eval_data_loader, val_data_loader

    def step(self, config, train_data_loader, optimizer, lr_scheduler, last_step):
        labelled_data_loader, unlabelled_data_loader = train_data_loader
        with self.model_random:
            for _i in range(self.train_config.num_batch_accumulated):
                if labelled_data_loader:
                    labelled_batch = next(labelled_data_loader)
                else:
                    labelled_batch = None

                if last_step < self.train_config.pretrain_threshold:
                    ret_dic = self.model(labelled_batch)
                    loss = ret_dic["loss"]
                else:
                    unlabelled_batch = next(unlabelled_data_loader)
                    ret_dic = self.model(labelled_batch, unlabelled_batch)
                    loss = ret_dic["loss"]
                norm_loss = loss / self.train_config.num_batch_accumulated
                norm_loss.backward()

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

                if "summary" in ret_dic:
                    wandb.log({"train_loss": loss.item(), **ret_dic["summary"]}, step=last_step)
                else:
                    wandb.log({"train_loss": loss.item()}, step=last_step)


def main(args):
    # setup logger etc
    config, logger = train.setup(args)

    # Construct trainer and do training
    trainer = Trainer(logger, config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == "__main__":
    args = add_parser()
    main(args)
