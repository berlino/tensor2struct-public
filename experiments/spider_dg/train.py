import argparse
import collections
import datetime
import json
import os
import wandb
import logging

import _jsonnet
import attr
import torch

from tensor2struct.utils import random_state, registry
from tensor2struct.utils import saver as saver_mod
from tensor2struct.commands import train

@attr.s
class TrainConfig(train.TrainConfig):
    use_bert_training = attr.ib(kw_only=True)

class Trainer(train.Trainer):
    def load_train_config(self):
        self.train_config = registry.instantiate(TrainConfig, self.config["train"])

        if self.train_config.use_bert_training:
            if self.train_config.clip_grad is None:
                self.logger.info("Grad clipping is recommended for BERT training")

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

    def step(self, config, train_data_loader, optimizer, lr_scheduler, last_step):
        with self.model_random:
            for _i in range(self.train_config.num_batch_accumulated):
                batch = next(train_data_loader)
                ret_dic = self.model(batch)
                loss = ret_dic["loss"]
                norm_loss = loss / self.train_config.num_batch_accumulated
                norm_loss.backward()

            # clip grad for both bert and non-bert params
            if self.train_config.clip_grad and self.train_config.use_bert_training:
                for param_group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(
                        param_group["params"], self.train_config.clip_grad,
                    )

            optimizer.step()
            new_lr = lr_scheduler.update_lr(last_step)
            optimizer.zero_grad()

            if new_lr is None:
                new_lr = [param["lr"] for param in optimizer.param_groups]

            if last_step % self.train_config.report_every_n == 0:
                self.logger.info("Step {}: loss={:.4f}".format(last_step, loss.item()))
                self.logger.info(f"Step {last_step}, lr={new_lr}")
                wandb.log({"train_loss": loss.item()}, step=last_step)
                for idx, lr in enumerate(new_lr):
                    wandb.log({f"lr_{idx}": lr}, step=last_step)

    def train(self, config, modeldir):
        optimizer, lr_scheduler = self.load_optimizer(config)
        saver, last_step = self.load_saver(config, modeldir, optimizer=optimizer)

        train_data_loader = self.load_train_data()
        train_eval_data_loader, val_data_loader = self.load_eval_data()

        with self.data_random:
            while last_step < self.train_config.max_steps:
                oom = False
                try:
                    self.eval_model(last_step, train_eval_data_loader, val_data_loader)
                    self.step(config, train_data_loader, optimizer, lr_scheduler, last_step)
                    last_step += 1
                    self.save_state(saver, modeldir, last_step)
                except RuntimeError as e:
                    err_msg = str(e)
                    self.logger.warn(f"Forward Failed: {err_msg}")
                    oom = True
                
                if oom:
                    ## the basic idea is to save and load the current model again
                    ## but it turns out the oom is still not resolved by this
                    # save the checkpoints
                    tmp_step = int(1e8)
                    saver.save(modeldir, step=tmp_step)
                    self.model.to("cpu")
                    del self.model
                    _optimizer_to(optimizer, "cpu")
                    del optimizer, lr_scheduler
                    torch.cuda.empty_cache()
                    import gc; gc.collect()

                    # load again
                    self.load_model(config)
                    optimizer, lr_scheduler = self.load_optimizer(config)
                    saver, _ = self.load_saver(config, modeldir, optimizer=optimizer)
                    ## remove the tmp checkpoint
                    os.unlink(os.path.join(modeldir, f"model_checkpoint-{tmp_step}"))

            saver.save(modeldir, last_step)

def _optimizer_to(optimizer, device):
    """ 
    Move optimizer state to cpu
    """
    for param in optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def main(args):
    # setup logger etc
    config, logger = train.setup(args)

    # Construct trainer and do training
    trainer = Trainer(logger, config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == "__main__":
    args = train.add_parser()
    main(args)
