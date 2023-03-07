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

from tensor2struct.utils import registry, random_state, vocab
from tensor2struct.utils import saver as saver_mod
from tensor2struct.commands import train, dema_train
from tensor2struct.training import maml


@attr.s
class MetaTrainConfig(dema_train.DEMATrainConfig):
    use_bert_training = attr.ib(kw_only=True)
    
class DEMATrainer(dema_train.DEMATrainer):
    def load_train_config(self):
        self.train_config = registry.instantiate(
            MetaTrainConfig, self.config["meta_train"]
        )

        if self.train_config.num_batch_accumulated > 1:
            self.logger.warn("Batch accumulation is used only at MAML-step level")

        if self.train_config.use_bert_training:
            if self.train_config.clip_grad is None:
                self.logger.info("Gradient clipping is recommended for BERT training")

def main(args):
    # setup logger etc
    config, logger = train.setup(args)

    # Construct trainer and do training
    trainer = DEMATrainer(logger, config)
    trainer.train(config, modeldir=args.logdir)

if __name__ == "__main__":
    args = train.add_parser()
    main(args)
