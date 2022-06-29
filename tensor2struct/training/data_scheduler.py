import random
import collections
import numpy as np
import torch
import itertools
import logging
import abc
from copy import deepcopy

from tensor2struct.utils import registry
from tensor2struct.training import spider_eval
import spacy

logger = logging.getLogger("tensor2struct")


class DataScheduler(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def get_batch(self, step=None):
        """
        Return inner_batch and a list of outter batches for MAML-style training
        """
        pass


@registry.register("data_scheduler", "random_scheduler")
class RandScheduler(DataScheduler):
    """
    Random sample k batches of data, despite of their db_ids
    """

    def __init__(self, examples, batch_size, num_batch_per_train):
        self.examples = examples
        self.batch_size = batch_size
        self.num_batch_per_train = num_batch_per_train
        self.generator = self._get_generator()

    def _get_generator(self):
        while True:
            dataloader = torch.utils.data.DataLoader(
                self.examples,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=lambda x: x,
            )
            yield from dataloader

    def get_batch(self, step=None):
        inner_batch = next(self.generator)
        outer_batches = [
            next(self.generator) for _ in range(self.num_batch_per_train - 1)
        ]
        return inner_batch, outer_batches


@registry.register("data_scheduler", "db_scheduler")
class DBScheduler(DataScheduler):
    """
    Sample k batches of data from k different distinct databases
    """

    def __init__(
        self,
        examples,
        batch_size,
        num_batch_per_train,
        group_func=None,
        use_similarity=False,
        filter_large_db=True,
        sampler=None,
    ):
        self.examples = examples
        self.batch_size = batch_size
        self.num_batch_per_train = num_batch_per_train
        self.sampler_config = sampler  # for distributed training

        # sample config
        self.sp_nlp = None
        self.temperature = 0.8
        self.use_similarity = use_similarity

        # group examples
        if group_func is None:
            self.group_func = lambda x: x[0]["db_id"]
        else:
            self.group_func = group_func
        self.iterators_by_db, self.dbid2count = self._create_iterators(
            self.examples, self.batch_size
        )
        assert sum(self.dbid2count.values()) == len(self.examples)

        # filter the large db
        self.filter_large_db = filter_large_db
        if filter_large_db:
            large_db_set = self.obtain_large_db()
            self.db_list = list(self.iterators_by_db.keys() - large_db_set)
        else:
            self.db_list = list(self.iterators_by_db.keys())
        logger.info(f"{len(self.db_list)} dbs loaded for training")

        if self.use_similarity:
                self.sp_nlp = spacy.load("en_core_web_md")
                self.db_similarity = self._compute_cached_sim_matrix()
        self._task_generator = self._yield_tasks_by_id()

    def obtain_large_db(self):
        large_db_set = set()
        visited = set()
        for example in self.examples:
            db_id = self.group_func(example)
            if db_id not in visited:
                visited.add(db_id)
                if len(example[0]["columns"]) > 200:
                    large_db_set.add(db_id)
        logger.info(f"Detected large dbs: {large_db_set}")
        return large_db_set

    def _compute_cached_sim_matrix(self):
        """
        Pre-compute the similarity matrix for faster batch loading
        """
        if self.sp_nlp is None:
            self.sp_nlp = spacy.load("en_core_web_md")
        logger.info("Pre-computing the Database similarities")
        dbs = self.db_list
        db_sim = {}
        for db1 in dbs:
            other_dbs = deepcopy(self.db_list)
            other_dbs.remove(db1)
            db1_ = db1.replace("_", " ")
            db1_sim = {}
            for db2 in other_dbs:
                db2_ = db2.replace("_", " ")
                v1 = self.sp_nlp(db1_)
                v2 = self.sp_nlp(db2_)
                db1_sim[db2] = v1.similarity(v2)

            db_sim[db1] = db1_sim

        return db_sim

    def compute_sim(self, db1, db2):
        return self.db_similarity[db1][db2]

    def _yield_tasks_by_id(self):
        id2iterators = self.iterators_by_db
        id2count = self.dbid2count

        _counts = [id2count[_id] for _id in self.db_list]
        all_count = sum(_counts)
        _p = [_c / all_count for _c in _counts]

        outer_batch_per_train = self.num_batch_per_train - 1
        while True:
            # sampled_ids = np.random.choice(self.db_list, num_batch_per_train, p=_p, replace=True)
            inner_db_id = random.sample(self.db_list, 1)[0]
            inner_task = next(id2iterators[inner_db_id])
            other_ids = self.db_list[:]
            other_ids.remove(inner_db_id)
            if self.use_similarity:
                cos_sims = [
                    self.compute_sim(inner_db_id, _outer_id) for _outer_id in other_ids
                ]
                # scores = [ (cos_sim + 1) / self.temperature for cos_sim in cos_sims]
                # p_sim = scores / sum(scores)
                scores = [cos_sim / self.temperature for cos_sim in cos_sims]
                p_sim = torch.softmax(torch.Tensor(scores), dim=0).numpy()
                outer_ids = np.random.choice(
                    other_ids, outer_batch_per_train, p=p_sim, replace=False
                )
            else:
                outer_ids = np.random.choice(
                    other_ids, outer_batch_per_train, replace=False
                )
            outer_tasks = [next(id2iterators[_db_id]) for _db_id in outer_ids]
            logger.info(f"Inner DB: {inner_db_id}, Outer DB: {outer_ids}")
            yield inner_task, outer_tasks

    def _create_iterators(self, data, bs):
        def _yield_batches(x, bs):
            if self.sampler_config is not None:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    x, **self.sampler_config,
                )
            else:
                sampler = None

            dataloader = torch.utils.data.DataLoader(
                x,
                batch_size=bs,
                shuffle=(sampler is None),
                drop_last=False,
                collate_fn=lambda x: x,
                sampler=sampler,
            )
            while True:
                yield from dataloader

        db_groups = itertools.groupby(
            sorted(data, key=self.group_func), self.group_func
        )
        dbid2examples_iterators = collections.defaultdict(dict)
        dbid2count = {}
        for db_id, g in db_groups:
            exs = list(g)
            dbid2examples_iterators[db_id] = _yield_batches(exs, bs)
            dbid2count[db_id] = len(exs)
        return dbid2examples_iterators, dbid2count

    def get_batch(self, step=None):
        return next(self._task_generator)
