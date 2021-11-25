import datetime
import json
import os
import wandb
import logging

import _jsonnet
import attr
import collections
import torch
import random

from tensor2struct.commands import train
from tensor2struct.utils import registry

logger = logging.getLogger("tensor2struct")

@attr.s
class TrainConfig(train.TrainConfig):
    use_bert_training = attr.ib(kw_only=True, default=False)

class Trainer(train.Trainer):
    """
    Specialized training module
    1) support bert training
    2) support "lagging inference"
    """
    def load_train_config(self):
        self.train_config = registry.instantiate(TrainConfig, self.config["train"])

        if self.train_config.use_bert_training:
            if self.train_config.clip_grad is None:
                self.logger.info("Grad clipping is recommended for BERT training")

    @staticmethod
    def _eval_model(
        logger, model, last_step, eval_data_loader, eval_section, num_eval_items=None
    ):
        """
        Marginals requires autograd, so no_grad during eval does not work for this model
        """
        stats = collections.defaultdict(float)
        model.eval()
        for eval_batch in eval_data_loader:
            ret_dic = model(eval_batch)
            stats["loss"] += ret_dic["loss"].item() * len(eval_batch)
            stats["total"] += len(eval_batch)
            if num_eval_items and stats["total"] > num_eval_items:
                break
        model.train()

        # Divide each stat by 'total'
        for k in stats:
            if k != "total":
                stats[k] /= stats["total"]
        if "total" in stats:
            del stats["total"]

        logger.info(
            "Step {} stats, {}: {}".format(
                last_step,
                eval_section,
                ", ".join("{} = {}".format(k, v) for k, v in stats.items()),
            )
        )
        wandb.log(
            {f"{eval_section}_eval_{k}": v for k, v in stats.items()}, step=last_step
        )

    def load_optimizer(self, config):
        with self.init_random:
            if self.train_config.use_bert_training:
                logger.info("Use a separate optimizer for bert parameters") 
                bert_params = self.model.get_bert_parameters()
                non_bert_params = self.model.get_non_bert_parameters()
                assert len(non_bert_params) + len(bert_params) == len(
                    list(self.model.parameters())
                )
                logger.info(f"Found {len(bert_params)} bert parameters") 
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

def main(args):
    # setup logger etc
    config, logger = train.setup(args)

    # Construct trainer and do training
    trainer = Trainer(logger, config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == "__main__":
    args = train.add_parser()
    main(args)
