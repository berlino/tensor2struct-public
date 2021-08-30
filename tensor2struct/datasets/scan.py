import re
import json
import attr
import tqdm
import torch

from nltk import CFG
from nltk.parse import RecursiveDescentParser

from tensor2struct.languages.dsl import scan
from tensor2struct.utils import registry, dataset


@attr.s
class ScanItem:
    text = attr.ib()
    code = attr.ib()

    # orig text
    orig = attr.ib(default=None)


@registry.register("dataset", "scan")
class ScanDataset(dataset.Dataset):
    def __init__(self, paths):
        self.examples = []

        for path in paths:
            if path[-4:] == ".txt":
                self.examples += self.load_from_txt(path)
            else:
                self.examples += self.load_from_json(path)
    
    def load_from_json(self, path):
        examples = []
        with open(path) as f:
            f_json = json.load(f)
            for item in f_json:
                text = " ".join(item['inp'])
                code = " ".join(item['out'])
                norm_text = self.normalize(text)
                norm_code = self.normalize(code)
                examples.append(ScanItem(norm_text, norm_code))
        return examples

    def load_from_txt(self, path):
        examples = []
        with open(path) as f:
            for line in f:
                text, code = line.strip().split("IN: ")[-1].split(" OUT: ")
                norm_code = self.normalize(code)
                norm_text = self.normalize(text)
                examples.append(ScanItem(norm_text, norm_code, line))
        return examples

    @staticmethod
    def normalize(s):
        # s += '.'
        s = re.sub(r"I_", r"", s)
        s = re.sub(r"([.!?])", r" \1", s)
        return s

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    class Metrics:
        def __init__(self, dataset, etype=None):
            self.dataset = dataset
            self.num_examples = len(dataset)
            self.counter = 0
            self.predictions = []

        def add_one(self, item, prediction, orig_question=None):
            if item.code == prediction:
                self.counter += 1
            summary = {
                "text": item.text,
                "gold_code": item.code,
                "predicted_code": prediction,
            }
            self.predictions.append(summary)

        def finalize(self):
            results = {
                "predictions": self.predictions,
                "total_scores": {
                    "lf_accuracy": self.counter / self.num_examples,
                    "exe_accuracy": self.counter / self.num_examples,
                },
            }
            return results


@registry.register("dataset", "scan_lf")
class ScanGrammarDataset(ScanDataset):
    def __init__(self, path):
        super().__init__([path])
        self.parser = scan.ScanGrammar()
        self.parse_all_example()

    def parse_all_example(self):
        """
        Use logical form tokens rather than action sequences
        """
        for example in tqdm.tqdm(self.examples, desc="parsing"):
            lf = self.parser.parse(example.text, example.code)
            example.code = lf

@registry.register("dataset", "scan_aug")
class ScanAugDataset(ScanDataset):
    def __init__(self, path, num_aug=None):
        super().__init__([path])
        self.num_aug = num_aug
        self.tag_orig_data = "<ORIG>"
        self.tag_aug_data = "<AUG>"
        self.tag_aug_token = "@"

        # for original data, we tag it so that a model is aware of this
        for example in self.examples:
            example.text = f"{self.tag_orig_data} {example.text}"

        if num_aug:
            self.generate_aug_example(num_aug)
    
    def generate_aug_example(self, num_aug):
        parser = scan.ScanGrammar()
        sampled_examples = parser.sample(num_aug)

        for sample_program, sample_action_seqs in sampled_examples:
            aug_tokens = [f"{self.tag_aug_token}{t}" for t in sample_program]
            program_text = " ".join([self.tag_aug_data] + aug_tokens)
            action_text = " ".join(sample_action_seqs)
            aug_example = ScanItem(text=program_text, code=action_text)
            self.examples.append(aug_example)
