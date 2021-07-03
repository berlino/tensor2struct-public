#!/bin/bash

DIR=$1

for SPLIT in train dev
do
    for LANG in source target
    do
        python -m bpe_encoder \
            --encoder-json gpt2-bpe/encoder.json \
            --vocab-bpe gpt2-bpe/vocab.bpe \
            --inputs "$1/$SPLIT.$LANG" \
            --outputs "$1/$SPLIT.bpe.$LANG" \
            --workers 60 \
            --keep-empt
    done
done