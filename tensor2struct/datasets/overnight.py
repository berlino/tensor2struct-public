import attr
import torch
import os
import tempfile
import subprocess

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
class OvernightItem:
    question = attr.ib()
    lf = attr.ib()
    domain = attr.ib()


DOMAINS = [
    "blocks",
    "calendar",
    "publications",
    "basketball",
    "housing",
    "recipes",
    "restaurants",
    "socialnetwork",
]


def execute(lfs, domain, eval_path="third_party/overnight"):
    def post_process(lf):
        if lf is None:
            lf = "None"
        replacements = [("SW", "edu.stanford.nlp.sempre.overnight.SimpleWorld")]
        for a, b in replacements:
            lf = lf.replace(a, b)
        return lf

    def is_error(d):
        return "FAILED" in d or "Exception" in d

    cur_dir = os.getcwd()
    os.chdir(eval_path)
    eval_script = "./evaluator/overnight"

    tf = tempfile.NamedTemporaryFile(suffix=".examples")
    for lf in lfs:
        p_lf = post_process(lf)
        tf.write(str.encode(p_lf + "\n"))
        tf.flush()
    FNULL = open(os.devnull, "w")
    msg = subprocess.check_output([eval_script, domain, tf.name], stderr=FNULL)
    tf.close()
    msg = msg.decode("utf-8")

    denotations = [
        line.split("\t")[1]
        for line in msg.split("\n")
        if line.startswith("targetValue\t")
    ]
    denotations = [None if is_error(d) else d for d in denotations]
    os.chdir(cur_dir)
    return denotations


@registry.register("dataset", "overnight")
class OvernightDataset(dataset.Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.examples = []

        for path in paths:
            filename = os.path.basename(path)
            domain = filename.split("_")[0]
            assert domain in DOMAINS
            with open(path, "r") as f:
                for line in f:
                    items = line.strip().split("\t")
                    if len(items) == 3:
                        question, lf, _domain = line.strip().split("\t")
                        assert _domain == f"overnight-{domain}"
                    else:
                        question, lf = line.strip().split("\t")
                    p_lf = self.preprocess_lf(lf)
                    item = OvernightItem(question, p_lf, domain)
                    self.examples.append(item)

    @staticmethod
    def preprocess_lf(lf):
        # jonathan split it up
        lf = lf.replace("! ", "!")
        return lf

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    class Metrics:
        def __init__(self, dataset, etype=None):
            self.dataset = dataset
            self.stat = Stat(num_examples=len(dataset))
            self.results = []

            self.gold_code_cache = []
            self.pred_code_cache = []

        def add_one(self, item, inferred_code):
            ret_dict = self.eval_one(item.lf, inferred_code, item.domain)
            self.results.append(ret_dict)

        def add_beams(self, item, inferred_codes):
            raise NotImplementedError

        def eval_one(self, gold_code, inferred_code, domain):
            ret_dic = {}
            ret_dic["gold_code"] = gold_code
            ret_dic["inferred_code"] = inferred_code
            ret_dic["lf_eval"] = gold_code == inferred_code

            self.gold_code_cache.append(gold_code)
            self.pred_code_cache.append(inferred_code)
            ret_dic["denotation_eval"] = None  # lazy execution

            if ret_dic["lf_eval"]:
                self.stat.lf_cor_num += 1
            if ret_dic["denotation_eval"]:
                self.stat.denotation_cor_num += 1
            return ret_dic

        def finalize(self):
            # TODO: group examples by their domain
            domain = self.dataset[0].domain
            gold_denotations = execute(self.gold_code_cache, domain)
            pred_denotations = execute(self.pred_code_cache, domain)
            for i, (g, p) in enumerate(zip(gold_denotations, pred_denotations)):
                if g == p:
                    self.stat.denotation_cor_num += 1
                self.results[i]["denotation_eval"] = g == p
            return {"per_item": self.results, "total_scores": self.stat.to_dict()}
