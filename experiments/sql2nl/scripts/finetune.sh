#!/bin/bash


# Note: use --fp16 for faster training

TOTAL_NUM_UPDATES=4000 # 4000 for spider, 300 for geo
WARMUP_UPDATES=50      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=2
BART_PATH="./bart.large/model.pt"

BIN_DIR=$1

CUDA_VISIBLE_DEVICES=0 fairseq-train $BIN_DIR \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR \
    --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-update $TOTAL_NUM_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
