#!/usr/bin/env python
import json
import sys

def remove_from_conditions(astree):
    astree = astree.copy()
    queue = [astree]
    while len(queue) > 0:
        node = queue.pop()

        if not isinstance(node, dict) or "_type" not in node:
            continue

        empty_children = []
        for child_name, child_node in node.items():
            if child_name == "_type":
                continue
            elif child_name == "from":
                if "conds" in child_node:
                    del child_node["conds"]
            elif isinstance(child_node, (list, tuple)) and len(child_node) == 0:
                empty_children.append(child_name)  # empty child
            else:
                queue.append(child_node)
        
        # delete empty nodes
        for child_name in empty_children:
            del node[child_name]
    return astree

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: ./create_spider_json.py example_json target_file output_file")
        sys.exit(0)

    example_json = sys.argv[1]
    target_file = sys.argv[2]
    output_file = sys.argv[3]

    with open(example_json) as f1, open(target_file) as f2, open(output_file, "w") as f3:
        examples = json.load(f1)
        q_lines = f2.readlines()
        assert len(examples) == len(q_lines)

        for example, q_line in zip(examples, q_lines):
            q = q_line.strip()
            example["question"] = q
            example["question_toks"] = q.split()

            # rat-sql does not predict conditions for FROM clause 
            astree = example["sql"]
            example["sql_with_from_cond"] = astree.copy()
            tree_no_from_cond = remove_from_conditions(astree)
            example["sql"] = tree_no_from_cond
        
        json.dump(examples, f3)