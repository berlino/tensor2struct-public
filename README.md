# Tensor2Struct 

tensor2struct is a package contains a set of neural semantic parsers based on encoder-decoder framework. Currently, it contains the code and data for the following papers:

* [Meta-Learning for Domain Generalization in Semantic Parsing, NAACL 2021](https://arxiv.org/abs/2010.11988)
* [Learning from Executions for Semantic Parsing, NAACL 2021](https://arxiv.org/abs/2104.05819)
* [Learning to Synthesize Data for Semantic Parsing, NAACL 2021](https://arxiv.org/abs/2104.05827)


## Setup

Create a virtual environment and run the setup script.

```
conda create --name tensor2struct python=3.7
conda activate tensor2struct
./setup.sh
```

[wandb](https://www.wandb.com/) is used for logging. To enable it you can create your own account and `wandb login` to enable logging.
Or you could just `wandb off` to only allow dryrun locally.


In general, the raw data is expected to be placed under "/data/TASK\_NAME/raw" where TASK\_NAME could be spider/ssp/overnight.

Make `log/` and `ie_dir/` which will be used for storing checkpoints and predictions (during inference).


##  Experiments

* [Meta-Learning for Domain Generalization](experiments/spider_dg/)
* [Learning from Executions for Semantic Parsing](experiments/semi_sup/)
* [Learning to Synthesize Data for Semantic Parsing](experiments/sql2nl/)


## Acknowledge

Tensor2struct is a generalization of [RAT-SQL](https://github.com/microsoft/rat-sql).
