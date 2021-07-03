
## Prepare Data

### Spider 

Download original data, see [another project](../spider_dg/)for instructions.
The repo contains the synthesized data at `data-spider-with-ssp-synthetic`
To reproduce the data,
please see the final step for detailed instructions.

## Pretraining

```
python experiments/sql2nl/run.py train configs/spider/run_config/run_en_spider_pretrain_syn.jsonnet
```

## Fine-tuning

The location of the pre-trained checkpoints is specified in the config file for fine-tuning.


```
python experiments/spider_dg/run.py train configs/spider/run_config/run_en_spider_finetune_syn.jsonnet
```

## Evaluation

```
tensor2struct eval configs/spider/run_config/run_en_spider_finetune_syn.jsonnet
```

This will print out the exact-match and execution accuarcy.

## Process of Synthesizing Data (Optional)

Attention, this will take many steps. Please stay patient :).

### i. Setup BART 


1) Install fairseq. It's recommended to setup another virutal environment for finetuning and data synthesizing.
It's also assumed that this environment has tensor2struct installed.

2) Download GPT2 BPE

    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'


3) Download BART pretrained model

    wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz

Unzip it to the current directory. Or you could place it in another directory; just make use the checkpoint path is correctly configured in the `./scripts/finetune.sh`

### ii. Prepara Data for BART Finetuning

Run the following commands to preprocess data for finetuning.

    ./scripts/collect_spider_src_tgt.py use_ssp data_dir # 1) collect spider sql-nl pairs, use_ssp in {false, true}
    ./scripts/run_bpe_tokenize.sh data_dir # 2) use bpe to tokenize the pairs 
    ./scripts/binarize_data.sh data_dir data-bin # 3) binarize data in data_dir/ and store in data-bin/

where you need to specify `data_dir` which is the directory for storing the utterance-SQL pairs. `use_ssp` indicates whether to use additional data (e.g., geoquery); these data are also provided by the original Spider dataset. For example, the first command can be:

    ./scripts/collect_spider_src_tgt.py true spider-raw-data

### iii. Finetune BART

    ./scripts/finetune.sh data-bin

The checkpoints will be saved into checkpoints/

### iv. Synthesized Data using Finetuned BART


Sample SQLs for Spider source domains and translate them into natural language utterances.

    ./scripts/sample_synthetic_data_spider.py ratio max_actions # 1) sample sql and store to data-synthetic
    ./bart_infer.py checkpoints/ data-bin data-synthetic/sampled-sql-{ratio}-{max_actions}.source data-synthetic/sampled-sql-{ratio}-{max_actions}.target # 2) generate corresponding nl
    ./create_spider_json.py data-synthetic/new-examples-no-questions-{ratio}-{max_actions}.json data-synthetic/sampled-sql-{ratio}-{max_actions}.target new-examples-{ratio}-{max_actions}.json  # 3) merge into a spider-format json

where `ratio` is chosen from {3, 6}, `max_actions` is chosen from {64, 128}. By default, the sampled and synthesized data are placed under `data-synthetic`.
For sanity check, you should be able to reproduce the data 
in `data-spider-with-ssp-synthetic` with `ratio` 3 and `max_actions` 128.