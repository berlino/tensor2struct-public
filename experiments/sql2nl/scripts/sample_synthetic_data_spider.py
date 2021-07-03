#!/usr/bin/env python
import json
import sys
import os
import collections
from tensor2struct.utils import registry, pcfg, random_state

data_path = "../../data/spider/raw"
train_data_config = {
    "name": "spider",
    "paths": [f"{data_path}/train_{s}.json" for s in ["spider", "others"]],
    "tables_paths": [f"{data_path}/tables.json"],
    "db_path": f"{data_path}/database",
}

grammar_config = {
    "name": "spiderv2",
    "include_literals": True,
    "end_with_from": True,
    "infer_from_conditions": True,
}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ./sample_synthetic_data_spider.py ratio max_actions")

    data_dir = "data-synthetic"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    synthetic_program_ratio = int(sys.argv[1])
    max_action_for_sampling = int(sys.argv[2])

    # loading path
    train_dataset = registry.construct("dataset", train_data_config)
    grammar = registry.construct("grammar", grammar_config)
    sample_state = random_state.RandomContext(seed=1)
    # ast_pcfg = pcfg.PCFG(grammar, True)

    db2examples = collections.defaultdict(list)
    for example in train_dataset.examples:
        db2examples[example.schema.db_id].append(example)

    # could also use a general pcfg for all dbs
    new_examples = []
    for db, examples in db2examples.items():
        schema = examples[0].schema
        db_pcfg = pcfg.PCFG(grammar, schema, use_seq_elem_rules=True)

        for example in examples:
            astree = grammar.parse(example.code)
            db_pcfg.record_productions(astree)

        db_pcfg.estimate()

        with sample_state:
            db_new_sqls = db_pcfg.sample(
                num_samples=len(examples) * synthetic_program_ratio,
                max_actions=max_action_for_sampling,
            )

        print(
            f"Collect {len(db_new_sqls)} ({len(set([s[1] for s in db_new_sqls]))} distinct) new sqls for {db}"
        )
        for sql_tree, sql_str in db_new_sqls:
            new_example = {"db_id": db, "sql": sql_tree, "query": sql_str}
            new_examples.append(new_example)

    # store the sampled examples
    with open(
        f"{data_dir}/new-examples-no-question-{synthetic_program_ratio}-{max_action_for_sampling}.json",
        "w",
    ) as f1, open(
        f"{data_dir}/sampled-sql-{synthetic_program_ratio}-{max_action_for_sampling}.source",
        "w",
    ) as f2:
        json.dump(new_examples, f1)
        for example in new_examples:
            f2.write(example["query"])
            f2.write("\n")
