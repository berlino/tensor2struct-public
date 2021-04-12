#!/bin/bash

python experiments/semi_sup/run.py semi_train \
    configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
    --config_args "{\"target_domain\": \"calendar\"}" &
python experiments/semi_sup/run.py semi_train \
    configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
    --config_args "{\"target_domain\": \"publications\"}" &
python experiments/semi_sup/run.py semi_train \
    configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
    --config_args "{\"target_domain\": \"recipes\"}" &
python experiments/semi_sup/run.py semi_train \
    configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
    --config_args "{\"target_domain\": \"housing\"}" &

# python experiments/semi_sup/run.py semi_train \
#     configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
#     --config_args "{\"target_domain\": \"restaurants\"}" &
# python experiments/semi_sup/run.py semi_train \
#     configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
#     --config_args "{\"target_domain\": \"blocks\"}" &
# python experiments/semi_sup/run.py semi_train \
#     configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
#     --config_args "{\"target_domain\": \"socialnetwork\"}" &
# python experiments/semi_sup/run.py semi_train \
#     configs/overnight/run_config/run_overnight_semi_supervised.jsonnet \
#     --config_args "{\"target_domain\": \"basketball\"}" &