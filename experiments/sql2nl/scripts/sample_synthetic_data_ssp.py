#!/usr/bin/env python
import json
import sys
import os
import collections
from tensor2struct.utils import registry, pcfg, random_state


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "Usage: ./sample_synthetic_data_ssp.py domain data_split sample_new_sql(bool) ratio max_actions"
        )
        sys.exit(0)

    data_dir = "data-ssp-synthetic"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    domain = sys.argv[1]
    data_split = sys.argv[2]
    sample_new_sql = True if sys.argv[3] == "true" else False
    synthetic_program_ratio = int(sys.argv[4])
    max_action_for_sampling = int(sys.argv[5])

    # check domain
    assert domain in ["geography", "atis"]

    # data config
    data_path = "../../data/ssp"
    train_data_config = {
        "name": "ssp",
        "split": "train",
        "data_split": data_split,
        "domain": domain,
        "path": f"{data_path}/raw/data/{domain}.json",
        "tables_path": f"{data_path}/database/converted_nh/{domain}/{domain}_tables.json",
        "db_path": f"{data_path}/database/converted_nh/",
    }
    dev_data_config = {
        "name": "ssp",
        "split": "dev",
        "data_split": data_split,
        "domain": domain,
        "path": f"{data_path}/raw/data/{domain}.json",
        "tables_path": f"{data_path}/database/converted_nh/{domain}/{domain}_tables.json",
        "db_path": f"{data_path}/database/converted_nh/",
    }
    test_data_config = {
        "name": "ssp",
        "split": "test",
        "data_split": data_split,
        "domain": domain,
        "path": f"{data_path}/raw/data/{domain}.json",
        "tables_path": f"{data_path}/database/converted_nh/{domain}/{domain}_tables.json",
        "db_path": f"{data_path}/database/converted_nh/",
    }

    infer_from_conditions = True

    grammar_config = {
        "name": "spiderv2",
        "include_literals": True,
        "end_with_from": True,
        "infer_from_conditions": infer_from_conditions,
    }

    # load data
    train_dataset = registry.construct("dataset", train_data_config)
    dev_dataset = registry.construct("dataset", dev_data_config)
    test_dataset = registry.construct("dataset", test_data_config)
    grammar = registry.construct("grammar", grammar_config)
    sample_state = random_state.RandomContext(seed=1)
    # ast_pcfg = pcfg.PCFG(grammar, True)

    # learn PCFG
    examples = train_dataset.examples
    schema = examples[0].schema
    collected_sqls = []

    db_pcfg = pcfg.PCFG(grammar, schema, use_seq_elem_rules=True)
    for example in train_dataset.examples + dev_dataset.examples:
        assert example.schema.db_id == domain

        if not example.is_sql_json_valid:
            continue

        try:
            astree = grammar.parse(example.code)
            unparsed_sql_str = pcfg.unparse(
                grammar.ast_wrapper, schema, astree, refine_from=infer_from_conditions
            )
            collected_sqls.append((astree, unparsed_sql_str))
        except Exception as e:
            # print(str(e))
            continue

        if sample_new_sql:
            db_pcfg.record_productions(astree)

    print(f"Collect {len(collected_sqls)} SQLs from training set of {domain}")

    # collect test SQLs for evaluating coverage
    collected_test_sqls = []
    for example in test_dataset.examples:
        assert example.schema.db_id == domain
        if not example.is_sql_json_valid:
            continue

        try:
            astree = grammar.parse(example.code)
            unparsed_sql_str = pcfg.unparse(
                grammar.ast_wrapper, schema, astree, refine_from=infer_from_conditions
            )
            collected_test_sqls.append((astree, unparsed_sql_str))
        except Exception as e:
            # print(str(e))
            # test example code is unparsable
            collected_test_sqls.append((None, example.orig["query"]))

    print(f"Collect {len(collected_test_sqls)} test SQLs for evaluation of {domain}")

    # collect new SQLs
    orig_sql_set = set([s[1] for s in collected_sqls])
    if sample_new_sql:
        db_pcfg.estimate()
        with sample_state:
            db_new_sqls = db_pcfg.sample(
                num_samples=len(examples) * synthetic_program_ratio,
                max_actions=max_action_for_sampling,
            )

        num_unseen_sqls = sum(s[1] not in orig_sql_set for s in db_new_sqls)
        num_distinct_sqls = len(set([s[1] for s in db_new_sqls]))
        print(
            f"Collect {len(db_new_sqls)} ({num_distinct_sqls} distinct, {num_unseen_sqls} unseen) new SQLs for {domain}"
        )

        # evaluating coverage
        # import pdb; pdb.set_trace()
        db_new_sqls_set = set([s[1] for s in db_new_sqls])
        num_overlap = sum(s[1] in db_new_sqls_set for s in collected_test_sqls)

        # debug only, print recalled instances
        # for _s in collected_test_sqls:
        #     if _s[1] in db_new_sqls_set:
        #         print(_s[1])

        coverage = num_overlap / len(collected_test_sqls)
        print(
            f"Coverage of test SQLs for {domain}: {coverage} {num_overlap}:{len(collected_test_sqls)}"
        )

        # two options, whether to include original sqls or not
        output_sqls = db_new_sqls
        # output_sqls = collected_sqls + db_new_sqls
    else:
        output_sqls = collected_sqls

    new_examples = []
    for sql_tree, sql_str in output_sqls:
        new_example = {"db_id": domain, "sql": sql_tree, "query": sql_str}
        new_examples.append(new_example)

    # store the sampled examples
    with open(
        f"{data_dir}/{domain}-{data_split}-sampling-{str(sample_new_sql).lower()}-no-question-ratio-{synthetic_program_ratio}-max-ac-{max_action_for_sampling}.json",
        "w",
    ) as f1, open(
        f"{data_dir}/sql-{domain}-{data_split}-sampling-{str(sample_new_sql).lower()}-ratio-{synthetic_program_ratio}-max-ac{max_action_for_sampling}.source",
        "w",
    ) as f2:
        json.dump(new_examples, f1)
        for example in new_examples:
            f2.write(example["query"])
            f2.write("\n")
