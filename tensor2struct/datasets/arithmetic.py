import attr
import torch

from tensor2struct.utils import registry, dataset


@attr.s
class ArithmeticItem:
    src = attr.ib()
    tgt = attr.ib()


@registry.register("dataset", "arithmetic")
class ArithmeticDataset(dataset.Dataset):
    def __init__(self, path, mode):
        self.examples = []
        with open(path) as f:
            for line in f:
                infix, postfix, prefix = line.strip().split("\t")
                infix_chars, prefix_chars, postfix_chars = list(infix), list(postfix), list(prefix)
                if mode == "infix2postfix":
                    item = ArithmeticItem(infix_chars, postfix_chars)
                elif mode == "infix2prefix":
                    item = ArithmeticItem(infix_chars, prefix_chars)
                else:
                    raise NotImplementedError
                self.examples.append(item)

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
            gold_tgt = " ".join(item.tgt)
            if gold_tgt == prediction:
                self.counter += 1
            summary = {
                "src": item.src,
                "gold_tgt": gold_tgt,
                "predicted_tgt": prediction,
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
