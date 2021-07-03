#!/usr/bin/env python
import sys
import os
from tensor2struct.utils import registry, pcfg

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ./collect_spider_src_tgt.py use_ssp output_path")
        sys.exit(0) 
    
    use_ssp = sys.argv[1] == "true"
    data_dir = sys.argv[2]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # data loading
    data_path = "../../data/spider/raw"
    if use_ssp:
        paths = [f"{data_path}/train_{s}.json" for s in ["spider", "others"]]
    else:
        paths = [f"{data_path}/train_{s}.json" for s in ["spider"]] 

    train_data_config = {
        "name": "spider",
        "paths": paths,
        "tables_paths": [f"{data_path}/tables.json"],
        "db_path": f"{data_path}/database"
    }

    dev_data_config = {
        "name": "spider",
        "paths": [f"{data_path}/dev.json"],
        "tables_paths": [f"{data_path}/tables.json"],
        "db_path": f"{data_path}/database"
    }

    grammar_config = {
        "name": "spiderv2",
        "include_literals": True,
        "end_with_from": True,
        "infer_from_conditions": True
    }

    train_dataset = registry.construct("dataset", train_data_config)
    dev_dataset = registry.construct("dataset", dev_data_config)
    grammar = registry.construct("grammar", grammar_config)

    with open(f"{data_dir}/train.source", "w") as f1, open(f"{data_dir}/train.target", "w") as f2:
        for example in train_dataset.examples:
            astree = grammar.parse(example.code)
            sql = pcfg.unparse(grammar.ast_wrapper, example.schema, astree)
            # sql = example.orig["query"]
            f1.write(sql)
            f1.write("\n")

            f2.write(example.orig["question"])
            f2.write("\n")

    with open(f"{data_dir}/dev.source", "w") as f1, open(f"{data_dir}/dev.target", "w") as f2:
        for example in dev_dataset.examples:
            astree = grammar.parse(example.code)
            sql = pcfg.unparse(grammar.ast_wrapper, example.schema, astree)
            # sql = example.orig["query"]
            f1.write(sql)
            f1.write("\n")

            f2.write(example.orig["question"])
            f2.write("\n")