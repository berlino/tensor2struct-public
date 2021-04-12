# Domain Generalization via Meta-Learning

## Prepare Data

### (English) Spider 

Download from [Spider](https://yale-lily.github.io/spider) or using bash:

```
pip install gdown
gdown https://drive.google.com/uc\?export\=download\&id\=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0
cd data/spider/
unzip spider.zip
mv spider raw
```


### Chinese Spider 

Download [data](https://taolusi.github.io/CSpider-explorer/) and place it at `data/ch_spider/raw`.


## Train BERT models for Spider

First, preprocess the data:
```
tensor2struct preprocess configs/spider/run_config/run_en_spider_bert_baseline.jsonnet
```

To train a supervised model, run

```
python experiments/spider_dg/run.py train configs/spider/run_config/run_en_spider_bert_baseline.jsonnet
```

To meta-train a supervised model, run

```
python experiments/spider_dg/run.py meta_train configs/spider/run_config/run_en_spider_bert_dgmaml.jsonnet
```

## Evaluation 

Run 

```
tensor2struct eval config_file
```

where config_file can be the baseline config or bert\_dgmaml config.
For Chinese Spider, you need the following configs:

* configs/spider/run_config/run_ch_spider_bert_baseline.jsonnet
* configs/spider/run_config/run_ch_spider_bert_dgmaml.jsonnet