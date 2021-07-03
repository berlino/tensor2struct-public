#!/bin/bash

SRCDIR=$1
TGTDIR=$2

fairseq-preprocess \
    --source-lang "source" \
    --target-lang "target" \
    --trainpref "$1/train.bpe" \
    --validpref "$1/dev.bpe" \
    --destdir $2 \
    --workers 60 \
    --srcdict gpt2-bpe/dict.txt \
    --tgtdict gpt2-bpe/dict.txt