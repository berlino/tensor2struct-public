# Tensor2Struct 

tensor2struct is a package that contains a set of neural semantic parsers based on the encoder-decoder framework. Currently, it supports the following datasets:

* Overnight 
* Spider
* Chinese Spider
* SCAN
* COGS


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

To run experiments in a paticular paper, please see the corresponding directory listed below for detailed instructions. 

* [Meta-Learning for Domain Generalization](experiments/spider_dg/)
* [Learning from Executions for Semantic Parsing](experiments/semi_sup/)
* [Learning to Synthesize Data for Semantic Parsing](experiments/sql2nl/)
* [Meta-Learning to Compositionally Generalize](experiments/comp_maml)

## Citations

If you use tensor2struct, please cite one of the following papers.

* [Meta-Learning for Domain Generalization in Semantic Parsing, NAACL 2021](https://arxiv.org/abs/2010.11988)

``` bibtex
@inproceedings{wang-etal-2021-meta,
    title = "Meta-Learning for Domain Generalization in Semantic Parsing",
    author = "Wang, Bailin  and
      Lapata, Mirella  and
      Titov, Ivan",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.33",
    doi = "10.18653/v1/2021.naacl-main.33",
    pages = "366--379",
}
```

* [Learning from Executions for Semantic Parsing, NAACL 2021](https://arxiv.org/abs/2104.05819)

``` bibtex
@inproceedings{wang-etal-2021-learning-executions,
    title = "Learning from Executions for Semantic Parsing",
    author = "Wang, Bailin  and
      Lapata, Mirella  and
      Titov, Ivan",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.219",
    doi = "10.18653/v1/2021.naacl-main.219",
    pages = "2747--2759",
}
```

* [Learning to Synthesize Data for Semantic Parsing, NAACL 2021](https://arxiv.org/abs/2104.05827)

``` bibtex
@inproceedings{wang-etal-2021-learning-synthesize,
    title = "Learning to Synthesize Data for Semantic Parsing",
    author = "Wang, Bailin  and
      Yin, Wenpeng  and
      Lin, Xi Victoria  and
      Xiong, Caiming",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.220",
    doi = "10.18653/v1/2021.naacl-main.220",
    pages = "2760--2766",
}
```

* [Meta-Learning to Compositionally Generalize, ACL 2021](https://arxiv.org/abs/2106.04252)

``` bibtex
@inproceedings{conklin-etal-2021-meta,
    title = "Meta-Learning to Compositionally Generalize",
    author = "Conklin, Henry  and
      Wang, Bailin  and
      Smith, Kenny  and
      Titov, Ivan",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.258",
    doi = "10.18653/v1/2021.acl-long.258",
    pages = "3322--3335",
    abstract = "Natural language is compositional; the meaning of a sentence is a function of the meaning of its parts. This property allows humans to create and interpret novel sentences, generalizing robustly outside their prior experience. Neural networks have been shown to struggle with this kind of generalization, in particular performing poorly on tasks designed to assess compositional generalization (i.e. where training and testing distributions differ in ways that would be trivial for a compositional strategy to resolve). Their poor performance on these tasks may in part be due to the nature of supervised learning which assumes training and testing data to be drawn from the same distribution. We implement a meta-learning augmented version of supervised learning whose objective directly optimizes for out-of-distribution generalization. We construct pairs of tasks for meta-learning by sub-sampling existing training data. Each pair of tasks is constructed to contain relevant examples, as determined by a similarity metric, in an effort to inhibit models from memorizing their input. Experimental results on the COGS and SCAN datasets show that our similarity-driven meta-learning can improve generalization performance.",
}
```

## Acknowledgement

Tensor2struct is a generalization of [RAT-SQL](https://github.com/microsoft/rat-sql).
