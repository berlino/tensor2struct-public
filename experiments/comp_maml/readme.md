# Experiments on COGS

## Setup Data 

The director tree should be like

```
data/cogs
└── raw
    ├── cogs
    │   ├── dev.tsv
    │   ├── gen_cp_recursion.tsv
    │   ├── gen_pp_recursion.tsv
    │   ├── gen_samples.tsv
    │   ├── gen.tsv
    │   ├── test.tsv
    │   ├── train_100.tsv
    │   └── train.tsv
    ├── LICENSE
    └── README.md
```

where gen_samples.tsv contains 2100 examples sampled from gen.tsv for model selection and reproducibility. Please see the paper for details.

## COGS training

1. Preprocessing

```
tensor2struct preprocess configs/cogs/run_config/run_cogs_comp.jsonnet
```


2. Standard training using

```
tensor2struct train configs/cogs/run_config/run_cogs_comp.jsonnet
```

or training with meta-learning:

```
tensor2struct meta_train configs/cogs/run_config/run_cogs_comp.jsonnet
```

## COGS inference

To obtain overall accuracy, simply run:

```
tensor2struct batched_eval configs/cogs/run_config/run_cogs_comp.jsonnet
```

To obtain a detailed accuracy for each category, run

```
python experiments/comp-maml/run.py batched_eval_cogs configs/cogs/run_config/run_cogs_comp.jsonnet
```

By default, tensor2struct uses wandb for visualization of the results.

## Random Seeds for Reproducibility

Random seed has a significant impact on the performance of a model. In our paper, we choose 0-4 as the random seeds for hyperparameter tuning on gen\_samples. We finally use 5-14 to for testing on the final gen set of COGS. Note that during the latter stage, gen\_samples set is not touched at any step. 

Random seed is specified via the variable `att` in `run_cogs_comp.jsonnet`; the test set (either val, test, gen or gen\_samples) is specified via `eval_section` in the same config file.