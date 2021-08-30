import random
import collections
import numpy as np
import torch
import itertools
import logging
import os
import abc
import attr
import tqdm
import pickle
import sys

from tensor2struct.utils import registry
from tensor2struct.training import data_scheduler
from tensor2struct.models.scan import edit_utils
from tensor2struct.languages.dsl import cogs, scan


def load_ssk_module():
    import pyximport

    pyximport.install()
    import tensor2struct.utils.string_kernel as ssk

logger = logging.getLogger("tensor2struct")


@registry.register("data_scheduler", "length_scheduler")
class CompDataSchedule(data_scheduler.DataScheduler):
    """
    Sample data according to the length of actions
    """

    def __init__(self, examples, batch_size, num_batch_per_train):
        self.examples = examples
        self.batch_size = batch_size
        self.num_batch_per_train = num_batch_per_train
        self.iterator = self._create_iterator()

    def _get_comp_tasks(self, step):
        examples = next(self.iterator)
        sorted_examples = sorted(examples, key=lambda x: len(x[1]["actions"]))
        inner_task = sorted_examples[: self.batch_size]
        outer_tasks = [sorted_examples[self.batch_size :]]
        return inner_task, outer_tasks

    def _create_iterator(self):
        def _yield_batches(x, bs):
            dataloader = torch.utils.data.DataLoader(
                x, batch_size=bs, shuffle=True, drop_last=True, collate_fn=lambda x: x
            )
            while True:
                yield from dataloader

        return _yield_batches(self.examples, self.batch_size * self.num_batch_per_train)

    def get_batch(self, step):
        return self._get_comp_tasks(step)

@attr.s
class GeneralizationSet:
    src_example = attr.ib(default=None)
    tgt_examples = attr.ib(default=attr.Factory(list))
    edists = attr.ib(default=attr.Factory(list))
    probs = attr.ib(default=None)


@registry.register("data_scheduler", "cogs_edist_scheduler")
class CogsEdistDataScheduler(data_scheduler.DataScheduler):
    def __init__(
        self,
        examples,
        batch_size,
        num_batch_per_train,
        topk=100,
        temp=1,
        cache_dir=".vector_cache/cogs",
        cache_file_prefix="edist",
    ):
        self.examples = examples
        self.batch_size = batch_size
        self.num_batch_per_train = num_batch_per_train

        # build cache
        self.topk = topk
        self.cache_dir = cache_dir
        self.cache_file = f"{self.cache_dir}/{cache_file_prefix}_top{self.topk}.pkl"
        if not os.path.exists(self.cache_file):
            if not os.path.exists(self.cache_dir):
                os.mkdir(self.cache_dir)
            self.build_cache(examples, self.cache_file)
        self.neighbours_list, self.edists_list = self.load_from_cache(self.cache_file)

        self.temp = temp
        self.src_examples, self.probs_list = self.build_sampling_prob(
            self.neighbours_list, self.edists_list, self.temp
        )
        self.generator = self._get_generator(self.src_examples)

    def _get_generator(self, examples):
        while True:
            dataloader = torch.utils.data.DataLoader(
                examples,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=lambda x: x,
            )
            yield from dataloader

    def _get_comp_tasks(self, step):
        inner_batch_ids = next(self.generator)
        inner_batch = []
        outer_batch = []

        for src_id in inner_batch_ids:
            src_example = self.examples[src_id]
            neighbours = self.neighbours_list[src_id]
            sampled_ids = np.random.choice(
                len(neighbours),
                self.num_batch_per_train - 1,
                replace=False,
                p=self.probs_list[src_id],
            )
            tgt_examples = [self.examples[neighbours[i]] for i in sampled_ids]

            inner_batch.append(src_example)
            outer_batch += tgt_examples

        return inner_batch, [outer_batch]

    def build_cache(self, examples, cache_file):
        # for testing only
        # _examples = [examples[i] for i in range(100)]
        # examples = _examples

        # compute token edist
        def attr_f(example):
            return example[0]["tokens"]

        # compute action edist
        # def attr_f(example):
        #     return example[1]["actions"][1:-1]

        window_size = 1000  # deactive it with a large number
        neighbours_list = collections.defaultdict(list)
        edists_list = collections.defaultdict(list)
        for i, src_example in tqdm.tqdm(
            enumerate(examples), desc="edist", total=len(examples)
        ):
            src_text_tokens = attr_f(src_example)

            for j in range(i + 1, len(examples)):
                tgt_example = examples[j]
                tgt_text_tokens = attr_f(tgt_example)

                if (
                    src_text_tokens != tgt_text_tokens
                    and len(tgt_text_tokens) >= len(src_text_tokens) - window_size
                    and len(tgt_text_tokens) <= len(src_text_tokens) + window_size
                ):
                    edist, _ = edit_utils.compute_levenshtein_distance(
                        src_text_tokens, tgt_text_tokens
                    )

                    neighbours_list[i].append(j)
                    edists_list[i].append(edist)

                    neighbours_list[j].append(i)
                    edists_list[j].append(edist)

        self.filter_topk(neighbours_list, edists_list, self.topk)

        with open(cache_file, "wb") as f:
            pickle.dump([neighbours_list, edists_list], f)
        return neighbours_list, edists_list

    def load_from_cache(self, cache_file):
        with open(cache_file, "rb") as f:
            neighbours_list, edists_list = pickle.load(f)
        return neighbours_list, edists_list

    def filter_topk(self, neighbours_list, edists_list, k):
        for i in range(len(self.examples)):
            orig_neighbours = neighbours_list[i]
            orig_edists = edists_list[i]

            if len(orig_neighbours) < k:
                continue

            sorted_pairs = sorted(zip(orig_neighbours, orig_edists), key=lambda x: x[1])
            topk_pairs = sorted_pairs[:k]
            filtered_neighbours, filtered_edists = zip(*topk_pairs)

            neighbours_list[i] = filtered_neighbours
            edists_list[i] = filtered_edists

    def build_sampling_prob(self, neighbours_list, edists_list, temp):
        probs_list = {}
        src_examples = list(neighbours_list)
        logger.info(
            f"Obtain {len(src_examples)} examples out of {len(self.examples)} that have neighbours"
        )
        for src_example in src_examples:
            edists = edists_list[src_example]
            edists_v = torch.Tensor(edists)
            probs = torch.softmax(-edists_v / temp, dim=0).numpy()
            probs_list[src_example] = probs
        return src_examples, probs_list

    def get_batch(self, step):
        return self._get_comp_tasks(step)


@registry.register("data_scheduler", "cogs_rand_edist_scheduler")
class CogsRandEdistDataScheduler(CogsEdistDataScheduler):
    def _get_comp_tasks(self, step):
        inner_batch_ids = next(self.generator)
        inner_batch = []
        outer_batch = []

        for src_id in inner_batch_ids:
            src_example = self.examples[src_id]

            eta = 0.5
            if random.random() >= eta:
                sampled_ids = np.random.choice(
                    len(self.src_examples), self.num_batch_per_train - 1, replace=False,
                )
                tgt_examples = [self.examples[i] for i in sampled_ids]
            else:
                neighbours = self.neighbours_list[src_id]
                sampled_ids = np.random.choice(
                    len(neighbours),
                    self.num_batch_per_train - 1,
                    replace=False,
                    p=self.probs_list[src_id],
                )
                tgt_examples = [self.examples[neighbours[i]] for i in sampled_ids]

            inner_batch.append(src_example)
            outer_batch += tgt_examples

        return inner_batch, [outer_batch]


@registry.register("data_scheduler", "cogs_tree_kernel_scheduler")
class CogsKernelDataScheduler(CogsEdistDataScheduler):
    def __init__(
        self,
        examples,
        batch_size,
        num_batch_per_train,
        topk=100,
        temp=1,
        cache_dir=".vector_cache/cogs",
        cache_file_prefix="kernel",
        kernel_name="sst",
        norm_kernel=False,
        lamb=1.0,
        mu=1.0,
    ):
        self.kernel_name, self.lamb, self.mu = kernel_name, lamb, mu
        self.norm_kernel = norm_kernel

        if not self.norm_kernel:
            prefix = f"{cache_file_prefix}-{kernel_name}-{lamb}-{mu}"
        else:
            prefix = f"{cache_file_prefix}-norm-{kernel_name}-{lamb}-{mu}"

        super().__init__(
            examples,
            batch_size,
            num_batch_per_train,
            topk=topk,
            temp=temp,
            cache_dir=cache_dir,
            cache_file_prefix=prefix,
        )

    def save_parse(self, parse_list):
        f = open(f"{self.cache_dir}/parse.tsv", "w")
        for example, parse in zip(self.examples, parse_list):
            f.write(
                f"{' '.join(example[0]['tokens'])}\t{' '.join(example[1]['actions'][1:-1])}\t{str(parse)}\n"
            )
        f.close()

    def build_cache(self, examples, cache_file):
        # for testing only
        # _examples = [examples[i] for i in range(120)]
        # examples = _examples

        parser = cogs.CogsGrammar(self.kernel_name, self.lamb, self.mu)

        def attr_f(example):
            return " ".join(example[0]["tokens"]), " ".join(example[1]["actions"][1:-1])

        parse_list = []  # grammar reprentation of a neighbour
        for i, src_example in tqdm.tqdm(
            enumerate(examples), desc="parsing", total=len(examples)
        ):
            parse_list.append(parser.parse(*attr_f(src_example)))
        self.save_parse(parse_list)  # for debug

        neighbours_list = collections.defaultdict(list)
        edists_list = collections.defaultdict(list)
        for i, src_example in tqdm.tqdm(
            enumerate(examples), desc="kernel", total=len(examples)
        ):
            src_parse = parse_list[i]

            for j in range(i + 1, len(examples)):
                tgt_parse = parse_list[j]

                sstkerenl = parser.kernel(src_parse, tgt_parse, self.norm_kernel)
                if self.norm_kernel:
                    assert sstkerenl >= 0 and sstkerenl <= 1
                edist = -1 * sstkerenl  # make it compatible with edist

                neighbours_list[i].append(j)
                edists_list[i].append(edist)

                neighbours_list[j].append(i)
                edists_list[j].append(edist)

        self.filter_topk(neighbours_list, edists_list, self.topk)

        with open(cache_file, "wb") as f:
            pickle.dump([neighbours_list, edists_list], f)
        return neighbours_list, edists_list


@registry.register("data_scheduler", "cogs_rand_tree_kernel_scheduler")
class CogsRandKerenlDataScheduler(CogsKernelDataScheduler):
    def _get_comp_tasks(self, step):
        inner_batch_ids = next(self.generator)
        inner_batch = []
        outer_batch = []

        for src_id in inner_batch_ids:
            src_example = self.examples[src_id]

            eta = 0.5
            if random.random() >= eta:
                sampled_ids = np.random.choice(
                    len(self.src_examples), self.num_batch_per_train - 1, replace=False,
                )
                tgt_examples = [self.examples[i] for i in sampled_ids]
            else:
                neighbours = self.neighbours_list[src_id]
                sampled_ids = np.random.choice(
                    len(neighbours),
                    self.num_batch_per_train - 1,
                    replace=False,
                    p=self.probs_list[src_id],
                )
                tgt_examples = [self.examples[neighbours[i]] for i in sampled_ids]

            inner_batch.append(src_example)
            outer_batch += tgt_examples

        return inner_batch, [outer_batch]


@registry.register("data_scheduler", "cogs_string_kernel_scheduler")
class CogsStringKernelDataScheduler(CogsEdistDataScheduler):
    def __init__(
        self,
        examples,
        batch_size,
        num_batch_per_train,
        topk=100,
        temp=1,
        cache_dir=".vector_cache/cogs",
        cache_file_prefix="string-kernel",
        norm_kernel=True,
        max_subseq=4,
        lamb=1.0,
    ):
        self.lamb = lamb
        self.max_subseq = 4
        self.norm_kernel = norm_kernel

        if not self.norm_kernel:
            prefix = f"{cache_file_prefix}-{lamb}"
        else:
            prefix = f"{cache_file_prefix}-norm-{lamb}"
        
        # load ssk module when needed
        if "tensor2struct.utils.string_kernel" not in sys.modules:
            load_ssk_module()

        super().__init__(
            examples,
            batch_size,
            num_batch_per_train,
            topk=topk,
            temp=temp,
            cache_dir=cache_dir,
            cache_file_prefix=prefix,
        )

    def build_vocab(self, examples):
        vocab = set()
        for example in examples:
            for token in example[0]["tokens"]:
                vocab.add(token)
        id2word = list(vocab)
        word2id = {v: k for k, v in enumerate(id2word)}
        return id2word, word2id

    def build_cache(self, examples, cache_file):
        # for testing only
        # _examples = [examples[i] for i in range(120)]
        # examples = _examples

        id2word, word2id = self.build_vocab(examples)

        def attr_f(example):
            return example[0]["tokens"]

        def kernel(l1, l2):
            # remove puntuation, only works for COGS
            if l1[-1] == ".":
                l1 = l1[:-1]
            if l2[-1] == ".":
                l2 = l2[:-1]

            ids_1 = [word2id[w] for w in l1]
            ids_2 = [word2id[w] for w in l2]

            k = ssk.ssk_list(ids_1, ids_2, self.max_subseq, self.lamb)
            if self.norm_kernel:
                n1 = ssk.ssk_list(ids_1, ids_1, self.max_subseq, self.lamb)
                n2 = ssk.ssk_list(ids_2, ids_2, self.max_subseq, self.lamb)
                dnorm = (n1 * n2) ** 0.5
                k = k / dnorm
            return k

        neighbours_list = collections.defaultdict(list)
        edists_list = collections.defaultdict(list)
        for i, src_example in tqdm.tqdm(
            enumerate(examples), desc="string-kernel", total=len(examples)
        ):
            for j in range(i + 1, len(examples)):
                tgt_example = examples[j]
                str_kerenl = kernel(attr_f(src_example), attr_f(tgt_example))
                if self.norm_kernel:
                    assert str_kerenl >= 0 and str_kerenl <= 1
                edist = -1 * str_kerenl  # make it compatible with edist

                neighbours_list[i].append(j)
                edists_list[i].append(edist)

                neighbours_list[j].append(i)
                edists_list[j].append(edist)

        self.filter_topk(neighbours_list, edists_list, self.topk)

        with open(cache_file, "wb") as f:
            pickle.dump([neighbours_list, edists_list], f)
        return neighbours_list, edists_list


@registry.register("data_scheduler", "cogs_rand_string_kernel_scheduler")
class CogsRandStringKernelDataScheduler(CogsStringKernelDataScheduler):
    def _get_comp_tasks(self, step):
        inner_batch_ids = next(self.generator)
        inner_batch = []
        outer_batch = []

        for src_id in inner_batch_ids:
            src_example = self.examples[src_id]

            eta = 0.5
            if random.random() >= eta:
                sampled_ids = np.random.choice(
                    len(self.src_examples), self.num_batch_per_train - 1, replace=False,
                )
                tgt_examples = [self.examples[i] for i in sampled_ids]
            else:
                neighbours = self.neighbours_list[src_id]
                sampled_ids = np.random.choice(
                    len(neighbours),
                    self.num_batch_per_train - 1,
                    replace=False,
                    p=self.probs_list[src_id],
                )
                tgt_examples = [self.examples[neighbours[i]] for i in sampled_ids]

            inner_batch.append(src_example)
            outer_batch += tgt_examples

        return inner_batch, [outer_batch]


@registry.register("data_scheduler", "cogs_nl_kernel_scheduler")
class CogsNLKernelDataScheduler(CogsEdistDataScheduler):
    def __init__(
        self,
        examples,
        batch_size,
        num_batch_per_train,
        topk=100,
        temp=1,
        cache_dir=".vector_cache/cogs",
        cache_file_prefix="nl-kernel",
        kernel_name="sst",
        norm_kernel=False,
        lamb=1.0,
        mu=1.0,
    ):
        self.kernel_name, self.lamb, self.mu = kernel_name, lamb, mu
        self.norm_kernel = norm_kernel

        if not self.norm_kernel:
            prefix = f"{cache_file_prefix}-{kernel_name}-{lamb}-{mu}"
        else:
            prefix = f"{cache_file_prefix}-norm-{kernel_name}-{lamb}-{mu}"

        super().__init__(
            examples,
            batch_size,
            num_batch_per_train,
            topk=topk,
            temp=temp,
            cache_dir=cache_dir,
            cache_file_prefix=prefix,
        )

    def save_parse(self, parse_list):
        f = open(f"{self.cache_dir}/nl-parse.tsv", "w")
        for example, parse in zip(self.examples, parse_list):
            f.write(
                f"{' '.join(example[0]['tokens'])}\t{' '.join(example[1]['actions'][1:-1])}\t{str(parse)}\n"
            )
        f.close()

    def build_cache(self, examples, cache_file):
        # for testing only
        # _examples = [examples[i] for i in range(120)]
        # examples = _examples

        parser = cogs.CogsGrammar(self.kernel_name, self.lamb, self.mu)

        def attr_f(example):
            return " ".join(example[0]["tokens"])

        parse_list = []  # grammar reprentation of a neighbour
        for i, src_example in tqdm.tqdm(
            enumerate(examples), desc="parsing", total=len(examples)
        ):
            parse_list.append(parser.parse_nl(attr_f(src_example)))
        self.save_parse(parse_list)  # for debug

        neighbours_list = collections.defaultdict(list)
        edists_list = collections.defaultdict(list)
        for i, src_example in tqdm.tqdm(
            enumerate(examples), desc="nl-kernel", total=len(examples)
        ):
            src_parse = parse_list[i]

            for j in range(i + 1, len(examples)):
                tgt_parse = parse_list[j]

                sstkerenl = parser.kernel(src_parse, tgt_parse, self.norm_kernel)
                if self.norm_kernel:
                    assert sstkerenl >= 0 and sstkerenl <= 1
                edist = -1 * sstkerenl  # make it compatible with edist

                neighbours_list[i].append(j)
                edists_list[i].append(edist)

                neighbours_list[j].append(i)
                edists_list[j].append(edist)

        self.filter_topk(neighbours_list, edists_list, self.topk)

        with open(cache_file, "wb") as f:
            pickle.dump([neighbours_list, edists_list], f)
        return neighbours_list, edists_list


@registry.register("data_scheduler", "cogs_nl_rand_kernel_scheduler")
class CogsNLRandKerenlDataScheduler(CogsNLKernelDataScheduler):
    def _get_comp_tasks(self, step):
        inner_batch_ids = next(self.generator)
        inner_batch = []
        outer_batch = []

        for src_id in inner_batch_ids:
            src_example = self.examples[src_id]

            eta = 0.5
            if random.random() >= eta:
                sampled_ids = np.random.choice(
                    len(self.src_examples), self.num_batch_per_train - 1, replace=False,
                )
                tgt_examples = [self.examples[i] for i in sampled_ids]
            else:
                neighbours = self.neighbours_list[src_id]
                sampled_ids = np.random.choice(
                    len(neighbours),
                    self.num_batch_per_train - 1,
                    replace=False,
                    p=self.probs_list[src_id],
                )
                tgt_examples = [self.examples[neighbours[i]] for i in sampled_ids]

            inner_batch.append(src_example)
            outer_batch += tgt_examples

        return inner_batch, [outer_batch]


@registry.register("data_scheduler", "cogs_rand_scheduler")
class CogsRandScheduler(data_scheduler.DataScheduler):
    """
    This is slightly different from random_scheduler in that 
    the outer batch is indeed is sampled
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

        outer_batch = []
        for _ in range(self.batch_size):
            example = random.choice(self.examples)
            outer_batch.append(example)
        return inner_batch, [outer_batch]


@registry.register("data_scheduler", "scan_rand_scheduler")
class SCANRandScheduler(CogsRandScheduler):
    pass


@registry.register("data_scheduler", "scan_edist_scheduler")
class ScanEdistDataScheduler(CogsEdistDataScheduler):
    def __init__(
        self,
        split,
        examples,
        batch_size,
        num_batch_per_train,
        topk=100,
        temp=1,
        cache_dir=".vector_cache/scan",
        cache_file_prefix="edist",
    ):
        super().__init__(
            examples,
            batch_size,
            num_batch_per_train,
            topk=topk,
            temp=temp,
            cache_dir=os.path.join(cache_dir, split),
            cache_file_prefix=cache_file_prefix,
        )


@registry.register("data_scheduler", "scan_rand_edist_scheduler")
class ScanRandEdistDataScheduler(CogsRandEdistDataScheduler):
    def __init__(
        self,
        split,
        examples,
        batch_size,
        num_batch_per_train,
        topk=100,
        temp=1,
        cache_dir=".vector_cache/scan",
        cache_file_prefix="edist",
    ):
        super().__init__(
            examples,
            batch_size,
            num_batch_per_train,
            topk=topk,
            temp=temp,
            cache_dir=os.path.join(cache_dir, split),
            cache_file_prefix=cache_file_prefix,
        )

@registry.register("data_scheduler", "scan_string_kernel_scheduler")
class ScanStringKernelDataScheduler(CogsStringKernelDataScheduler):
    def __init__(
        self,
        split,
        examples,
        batch_size,
        num_batch_per_train,
        topk=100,
        temp=1,
        cache_dir=".vector_cache/scan",
        cache_file_prefix="string-kernel",
    ):
        super().__init__(
            examples,
            batch_size,
            num_batch_per_train,
            topk=topk,
            temp=temp,
            cache_dir=os.path.join(cache_dir, split),
            cache_file_prefix=cache_file_prefix,
        )


@registry.register("data_scheduler", "scan_rand_string_kernel_scheduler")
class ScanRandStringDataScheduler(CogsRandStringKernelDataScheduler):
    def __init__(
        self,
        split,
        examples,
        batch_size,
        num_batch_per_train,
        topk=100,
        temp=1,
        cache_dir=".vector_cache/scan",
        cache_file_prefix="string-kernel",
    ):
        super().__init__(
            examples,
            batch_size,
            num_batch_per_train,
            topk=topk,
            temp=temp,
            cache_dir=os.path.join(cache_dir, split),
            cache_file_prefix=cache_file_prefix,
        )

@registry.register("data_scheduler", "scan_tree_kernel_scheduler")
class ScanKernelDataScheduler(CogsEdistDataScheduler):
    def __init__(
        self,
        examples,
        batch_size,
        num_batch_per_train,
        topk=100,
        temp=1,
        cache_dir=".vector_cache/scan",
        cache_file_prefix="kernel",
    ):
        super().__init__(
            examples,
            batch_size,
            num_batch_per_train,
            topk=topk,
            temp=temp,
            cache_dir=cache_dir,
            cache_file_prefix=cache_file_prefix,
        )

    def save_parse(self, parse_list):
        f = open(f"{self.cache_dir}/parse.tsv", "w")
        for example, parse in zip(self.examples, parse_list):
            f.write(f"{' '.join(example[0]['tokens'])}\t{str(parse)}\n")
        f.close()

    def build_cache(self, examples, cache_file):
        # for testing only
        # _examples = [examples[i] for i in range(120)]
        # examples = _examples

        parser = scan.ScanGrammar()

        def attr_f(example):
            return " ".join(example[0]["tokens"])

        parse_list = []  # grammar reprentation of a neighbour
        for i, src_example in tqdm.tqdm(
            enumerate(examples), desc="parsing", total=len(examples)
        ):
            parse_list.append(parser.parse_command(attr_f(src_example)))
        self.save_parse(parse_list)  # for debug

        neighbours_list = collections.defaultdict(list)
        edists_list = collections.defaultdict(list)
        for i, src_example in tqdm.tqdm(
            enumerate(examples), desc="kernel", total=len(examples)
        ):
            src_parse = parse_list[i]

            for j in range(i + 1, len(examples)):
                tgt_parse = parse_list[j]

                sstkerenl = parser.kernel(src_parse, tgt_parse)
                edist = -1 * sstkerenl  # make it compatible with edist

                neighbours_list[i].append(j)
                edists_list[i].append(edist)

                neighbours_list[j].append(i)
                edists_list[j].append(edist)

        self.filter_topk(neighbours_list, edists_list, self.topk)

        with open(cache_file, "wb") as f:
            pickle.dump([neighbours_list, edists_list], f)
        return neighbours_list, edists_list
