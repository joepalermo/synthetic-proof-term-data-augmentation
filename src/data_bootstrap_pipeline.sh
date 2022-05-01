#!/usr/bin/env bash

# load config
source src/data_bootstrap_pipeline/config.py

# reset output directory
rm -rf output
mkdir output
mkdir output/proofs
mkdir output/type_universe_variables

# run pipeline
lean --run src/data_bootstrap_pipeline/extract_subproofs.lean $max_num_declarations_to_take $max_proof_length
python3 src/data_bootstrap_pipeline/cleanup_proofs.py
python3 src/data_bootstrap_pipeline/typecheck_all_proofs.py
python3 src/data_bootstrap_pipeline/create_lm_tasks.py
