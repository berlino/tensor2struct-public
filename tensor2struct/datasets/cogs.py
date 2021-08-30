import attr
import torch
import os
import collections

from tensor2struct.utils import registry, dataset


@attr.s
class Stat:
    sketch_cor_num = attr.ib(default=0)
    lf_cor_num = attr.ib(default=0)
    denotation_cor_num = attr.ib(default=0)
    num_examples = attr.ib(default=0)

    def __str__(self):
        if self.num_examples > 0:
            str_builder = []
            str_builder.append(
                f"sketch eval: {self.sketch_cor_num/self.num_examples}, {self.sketch_cor_num}/{self.num_examples}"
            )
            str_builder.append(
                f"lf eval: {self.lf_cor_num/self.num_examples}, {self.lf_cor_num}/{self.num_examples}"
            )
            str_builder.append(
                f"denotation eval: {self.denotation_cor_num/self.num_examples}, {self.denotation_cor_num}/{self.num_examples}"
            )
            return "\n".join(str_builder)
        else:
            return "Empty stat"

    def to_dict(self):
        if self.num_examples > 0:
            rep = {}
            rep["sketch_eval"] = self.sketch_cor_num / self.num_examples
            rep["sketch_eval_detail"] = f"{self.sketch_cor_num}/{self.num_examples}"
            rep["lf_accuracy"] = self.lf_cor_num / self.num_examples
            rep["lf_eval_detail"] = f"{self.lf_cor_num}/{self.num_examples}"
            rep["exe_accuracy"] = self.denotation_cor_num / self.num_examples
            rep["exe_eval_detail"] = f"{self.denotation_cor_num}/{self.num_examples}"
            return rep
        else:
            return {}


@attr.s
class CogsItem:
    text = attr.ib()
    code = attr.ib()
    category = attr.ib()


@registry.register("dataset", "cogs")
class CogsDataset(dataset.Dataset):
    def __init__(self, path):
        self.path = path
        self.examples = []

        with open(path, "r") as f:
            for line in f:
                question, lf, category = line.strip().split("\t")
                item = CogsItem(question, lf, category)
                self.examples.append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    class Metrics:
        def __init__(self, dataset, etype=None):
            self.dataset = dataset
            self.stat = Stat(num_examples=len(dataset))
            self.categorized_stat = collections.defaultdict(Stat)
            self.results = []

        def add_one(self, item, inferred_code):
            ret_dict = self.eval_one(item.code, inferred_code, item.category)
            ret_dict["question"] = item.text  # for debug
            self.results.append(ret_dict)

        def add_beams(self, item, inferred_codes):
            raise NotImplementedError

        def eval_one(self, gold_code, inferred_code, category):
            ret_dic = {}
            ret_dic["gold_code"] = gold_code
            ret_dic["inferred_code"] = inferred_code
            ret_dic["lf_eval"] = gold_code == inferred_code
            ret_dic["denotation_eval"] = None

            self.categorized_stat[category].num_examples += 1

            if ret_dic["lf_eval"]:
                self.stat.lf_cor_num += 1
                self.categorized_stat[category].lf_cor_num += 1
            if ret_dic["denotation_eval"]:
                self.stat.denotation_cor_num += 1
                self.categorized_stat[category].denotation_cor_num += 1
            return ret_dic

        def finalize(self):
            ret_stats = {"per_item": self.results, "total_scores": self.stat.to_dict()}
            for category in self.categorized_stat:
                ret_stats[category] = self.categorized_stat[category].to_dict()
            return ret_stats

@registry.register("dataset", "cogs_grammar")
class CogsDatasetGrammar(CogsDataset):
    def __init__(self, path):
        pass