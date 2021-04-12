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


@attr.s
class TrainConfig:
    device = attr.ib(default="cpu")
    eval_every_n = attr.ib(default=100)
    report_every_n = attr.ib(default=100)
    save_threshold = attr.ib(default=1000)
    save_every_n = attr.ib(default=100)
    keep_every_n = attr.ib(default=1000)

    batch_size = attr.ib(default=32)
    eval_batch_size = attr.ib(default=32)
    max_steps = attr.ib(default=100000)
    num_eval_items = attr.ib(default=None)
    eval_on_train = attr.ib(default=True)
    eval_on_val = attr.ib(default=True)

    data_seed = attr.ib(default=None)
    init_seed = attr.ib(default=None)
    model_seed = attr.ib(default=None)

    num_batch_accumulated = attr.ib(default=1)
    clip_grad = attr.ib(default=None)


class Trainer:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.load_train_config()

        self.device = torch.device(self.train_config.device)
        self.data_random = random_state.RandomContext(self.train_config.data_seed)
        self.model_random = random_state.RandomContext(self.train_config.model_seed)
        self.init_random = random_state.RandomContext(self.train_config.init_seed)

        self.load_model(config)
    
    def load_train_config(self):
        self.train_config = registry.instantiate(TrainConfig, self.config["train"])

    def load_model(self, config):
        with self.init_random:
            # 0. Construct preprocessors
            self.model_preproc = registry.instantiate(
                registry.lookup("model", config["model"]).Preproc,
                config["model"],
                unused_keys=("name",),
            )
            self.model_preproc.load()

            # 1. Construct model
            self.model = registry.construct(
                "model",
                config["model"],
                unused_keys=("encoder_preproc", "decoder_preproc"),
                preproc=self.model_preproc,
                device=self.device,
            )
            self.model.to(self.device)

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
            return optimizer, lr_scheduler

    def load_saver(self, config, modeldir, **kwargs):
        """ Initialize saver, support loading from pretrained models """
        # Restore model parameters
        saver = saver_mod.Saver(
            {"model": self.model, **kwargs}, keep_every_n=self.train_config.keep_every_n
        )
        # optimizer state is not stored to save space!
        last_step = saver.restore(
            modeldir, item_keys=["model"] + list(kwargs.keys()), map_location=self.device
        )

        # support loading from pretrained models, remember not use in debug mode
        # where the _step might be zero, not the pretrained steps
        if "pretrain" in config and last_step == 0:
            pretrain_config = config["pretrain"]
            _path = pretrain_config["pretrain_path"]

            if _path is not None:
                _step = pretrain_config["checkpoint_step"]
                pretrain_step = saver.restore(
                    _path, step=_step, map_location=self.device, item_keys=["model"]
                )
                # saver.save(modeldir, last_step)  # for evaluating pretrained models
                # last_step = pretrain_step
        return saver, last_step

    def load_train_data(self):
        with self.data_random:
            train_data = self.model_preproc.dataset("train")
            train_data_loader = self._yield_batches_from_epochs(
                torch.utils.data.DataLoader(
                    train_data,
                    batch_size=self.train_config.batch_size,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=lambda x: x,
                )
            )
        return train_data_loader

    def load_eval_data(self):
        with self.data_random:
            train_data = self.model_preproc.dataset("train")
            train_eval_data_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=self.train_config.eval_batch_size,
                collate_fn=lambda x: x,
            )

            val_data = self.model_preproc.dataset("val")
            val_data_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.train_config.eval_batch_size,
                collate_fn=lambda x: x,
            )
        return train_eval_data_loader, val_data_loader

    def step(self, config, train_data_loader, optimizer, lr_scheduler, last_step):
        with self.model_random:
            for _i in range(self.train_config.num_batch_accumulated):
                batch = next(train_data_loader)
                ret_dic = self.model(batch)
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
                wandb.log({"train_loss": loss.item()}, step=last_step)
                for i in range(len(new_lr)):
                    wandb.log({f"lr_{i}": new_lr[i]}, step=last_step)

    def save_state(self, saver, modeldir, last_step):
        if (
            last_step % self.train_config.save_every_n == 0
            and last_step >= self.train_config.save_threshold
        ):
            saver.save(modeldir, last_step)

    def train(self, config, modeldir):
        optimizer, lr_scheduler = self.load_optimizer(config)
        saver, last_step = self.load_saver(config, modeldir, optimizer=optimizer)

        train_data_loader = self.load_train_data()
        train_eval_data_loader, val_data_loader = self.load_eval_data()

        with self.data_random:
            while last_step < self.train_config.max_steps:
                self.eval_model(last_step, train_eval_data_loader, val_data_loader)
                self.step(config, train_data_loader, optimizer, lr_scheduler, last_step)
                last_step += 1
                self.save_state(saver, modeldir, last_step)

            saver.save(modeldir, last_step)

    @staticmethod
    def _yield_batches_from_epochs(loader):
        while True:
            for batch in loader:
                yield batch

    def eval_model(self, last_step, train_eval_data_loader, val_data_loader):
        if last_step % self.train_config.eval_every_n == 0:
            if self.train_config.eval_on_train:
                self._eval_model(
                    self.logger,
                    self.model,
                    last_step,
                    train_eval_data_loader,
                    "train",
                    num_eval_items=self.train_config.num_eval_items,
                )
            if self.train_config.eval_on_val:
                self._eval_model(
                    self.logger,
                    self.model,
                    last_step,
                    val_data_loader,
                    "val",
                    num_eval_items=self.train_config.num_eval_items,
                )

    @staticmethod
    def _eval_model(
        logger, model, last_step, eval_data_loader, eval_section, num_eval_items=None
    ):
        stats = collections.defaultdict(float)
        model.eval()
        with torch.no_grad():
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


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config_args")
    args = parser.parse_args()
    return args


def setup(args):
    if args.config_args:
        config = json.loads(
            _jsonnet.evaluate_file(args.config, tla_codes={"args": args.config_args})
        )
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    # logdir
    if "model_name" in config:
        args.logdir = os.path.join(args.logdir, config["model_name"])

    # Initialize the logger
    logfile_path = os.path.join(args.logdir, "log.txt")
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    logger = logging.getLogger("tensor2struct")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile_path)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("Logging to {}".format(args.logdir))

    # Save the config info
    with open(
        os.path.join(
            args.logdir,
            "config-{}.json".format(
                datetime.datetime.now().strftime("%Y%m%dT%H%M%S%Z")
            ),
        ),
        "w",
    ) as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # save to wandb
    wandb.config.update(config)
    return config, logger


def main(args):
    # setup logger etc
    config, logger = setup(args)

    # Construct trainer and do training
    trainer = Trainer(logger, config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == "__main__":
    args = add_parser()
    main(args)
