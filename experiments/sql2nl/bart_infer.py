#!/usr/bin/env python
import torch
import tqdm
import sys
from fairseq.models.bart import BARTModel

if len(sys.argv) < 5:
    print("Usage: python bart_infer.py ckp_path bin_path source_file target_file")
    sys.exit(0)

ckp_path = sys.argv[1]
bin_path = sys.argv[2]
source_file = sys.argv[3]
target_file = sys.argv[4]

bart = BARTModel.from_pretrained(
    ckp_path,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path=bin_path,
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 2  # make it small (2) for atis, 32 for others
with open(source_file) as source, open(target_file, 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in tqdm.tqdm(source):
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=20, min_len=6, no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    
    # leftover
    if len(slines) != 0:
        with torch.no_grad():
            hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=20, min_len=6, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
