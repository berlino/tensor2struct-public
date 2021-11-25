# Experiment on Inducing Latent Reorderings

The minimal experiments to run is to induce the reordering 
for the synthetic task of converting arithmetic expressions in infix format to the ones in postfix format.

## Setup Data 

The data director tree should be like

```
data/arithmetic
└── raw
    ├── gen_5k.tsv
    ├── test_5k.tsv
    ├── train_10k.tsv
    └── val_5k.tsv
```

Apart from the standar train/val/test set, we also have a generalization set where expressions are deeper in depth. See the paper for details.

## Training

1. Preprocessing

```
tensor2struct preprocess configs/arithmetic/run_config/run_arithmetic_latper.jsonnet
```


2. Training the model using

```
tensor2struct train configs/arithmetic/run_config/run_arithmetic_latper.jsonnet
```


3. Evaluation

```
python experiments/permutation/run.py  eval_tagging configs/arithmetic/run_config/run_arithmetic_latper.jsonnet
```

The accuracy of 0.84485 is the expected output. The loss converges pretty quickly, and training longer will lead to slightly better performance. The accuracy reported in the paper can be reproduced by setting max\_steps to be 2k in the config.
