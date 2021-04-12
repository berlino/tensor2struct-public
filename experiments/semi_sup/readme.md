# Learning from Executions for Semantic Parsing


## Supervised Models

Run the following commands
```
# preprocess data
tensor2struct preprocess configs/overnight/run_config/run_overnight_supervised.jsonnet
# train it
tensor2struct train configs/overnight/run_config/run_overnight_supervised.jsonnet
# eval it
tensor2struct eval configs/overnight/run_config/run_overnight_supervised.jsonnet
```

## Semi-Supervised Models

Run the following commands
```
# preprocess data
tensor2struct preprocess configs/overnight/run_config/run_overnight_semi_supervised.jsonnet 
# train it
python experiments/semi_sup/run.py semi_train configs/overnight/run_config/run_overnight_semi_supervised.jsonnet  
# eval it
tensor2struct eval configs/overnight/run_config/run_overnight_semi_supervised.jsonnet
```

## Scripts

`scripts/` contains scripts to preprocess/train/eval models efficiently for all domains of Overnight.