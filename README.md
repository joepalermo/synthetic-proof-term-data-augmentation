# lean-proof-term-augmentation

## setup

```bash
# Download pre-built binaries and build the project (targeting mathlib).
bash ./setup.sh
```

## summary

Here we show how to use language models and the Lean kernel to generate Lean proof terms and to use these proof terms to augment an existing dataset. This is useful since datasets augmented with generated proof terms can be used to re-train models to higher performance levels.

We train a language model initially on examples of the form: "PROOF {proof} EOT". Then we can prompt the trained language model with the "PROOF" token to generate candidate proof terms. Candidate proof terms are then typechecked using the Lean kernel to filter out malformed proofs. 

Note also that we can train a theorem proving model by conditioning proof generation on theorem statements by using examples of form: "THEOREM {theorem} PROOF {proof} EOT". Such a model can be used to prove theorems at inference time by prompting with "THEOREM {theorem} PROOF".

With generated proof terms on-hand we can augment the original dataset by creating more examples of form "PROOF {proof} EOT". Note we can also create additional conditioned training examples of form: "THEOREM {theorem} PROOF {proof} EOT", since during the typechecking process we also extract the types of the proof terms (i.e. theorem statements).

Here is a summary of the stages in this process:
1. run the data bootstrap pipeline to create an initial dataset of proof terms
2. train a model on examples of the form: "PROOF {proof} EOT"
3. create new proof term candidates by sampling from this model with prompt "PROOF"
4. typecheck the proof term candidates to filter out malformed proofs
5. combine the generated proof terms with the original proof terms to create an augmented dataset
6. train new models on the augmented dataset

Optionally, one can repeat the process from step 2 (starting this time from an augmented dataset) to further improve results.

## data bootstrap pipeline

### run pipeline

The data bootstrap pipeline can be used to create an initial "bootstrap" dataset. The following command will run the data bootstrap pipeline end-to-end:

```bash -x src/data_bootstrap_pipeline.sh```

Configuration for the pipeline can be set in [`src/data_bootstrap_pipeline/config.py`](https://github.com/joepalermo/lean-theorem-gen/blob/master/src/data_bootstrap_pipeline/config.py).

### stages in the pipeline

See below descriptions of each stage in the pipeline.

#### 1 - extract subproofs

```lean --run src/data_bootstrap_pipeline/extract_subproofs.lean $max_num_declarations_to_take $max_proof_length```

- extracts pretty-printed subproofs from proofs in mathlib by traversing the expr trees and caching pretty-printed proofs as it goes
- also caches the set of all type universe variable names used in each subproof to simplify the following stage of the pipeline  
  
- input files: None (extracts proofs from mathlib)
- output files: 
  - output/proofs/{declaration-name}.txt
  - output/type_universe_variables/{declaration-name}.txt
  
#### 2 - cleanup proofs

```python src/data_bootstrap_pipeline/cleanup_proofs.py```

- replace type universe variable names
  - Lean has very few string processing utilities so we use Python
  - Alternatively this could be done by finding and replacing type universe variable names directly in exprs before pretty-printing (in extract_subproofs.lean)
- remove duplicate proofs
  
- input files: 
  - output/proofs/{declaration-name}>.txt
  - output/type_universe_variables/{declaration-name}>.txt
- output files: 
  - output/modified_proofs/{declaration-name}>.txt

#### 3 - filter proofs that fail to parse and typecheck

```python src/data_bootstrap_pipeline/typecheck_all_proofs.py```

- invokes `typecheck_proofs.lean` in parallel threads on the proofs in each input file
- output contains only pretty-printed proofs (and corresponding theorem statements) that can be parsed and typechecked

- input files: 
  - output/modified_proofs/{declaration-name}>.txt
- output files: 
  - output/checked_proofs/{declaration-name}>.txt
  - output/checked_theorems/{declaration-name}>.txt

#### 4 - generate language modelling tasks

```python src/data_bootstrap_pipeline/create_lm_task.py```

- creates examples for training and validating language models
- there are "forwards" examples and "backwards examples":
    - forwards examples have the form: "PROOF {proof} EOT"
    - backwards examples have the form: "GOAL {goal} PROOF {proof} EOT"
- splits the examples into train/val/test sets by:
    - split on declaration names
    - remove any examples from val and test that have too much overlap with the most similar training example (identify via cosine similarity)

- input files: 
  - output/checked_proofs/{declaration-name}.txt
  - output/checked_theorems/{declaration-name}.txt
- output files: 
  - output/forwards_text/{train/val/test.txt}
  - output/backwards_text/{train/val/test.txt}
  - output/forwards_json/{train/val/test.json}
  - output/backwards_json/{train/val/test.json}
  
## train language models

We use [fairseq](https://github.com/pytorch/fairseq) to train language models.

After training models the following products should exist (which are required in the next step):

- {root_dirpath}/preprocessed_data/{dataset_name}_preprocessed/*
- {root_dirpath}/checkpoints/{dataset_name}_{model_name}_checkpoints/checkpoint_best.pt

## sample proof terms from trained model 

Generate proof term candidates by sampling repeatedly with the prompt "PROOF":

```python src/generate_proof_term_candidates.py --root_dirpath {root_dirpath} --dataset_name {dataset_name} --model_name {model_name} --gpu_id {gpu_id}```

## filter out malformed proofs

Filter out the proof term candidates which fail to typecheck:

```python src/filter_proof_term_candidates.py --max_num_to_typecheck {max_num_to_typecheck} --num_shards {num_shards}```

## create augmented dataset

Finally we can combine the original bootstrap data with generated proof terms to create augmented datasets.

### create data controlled augmented dataset

The following script creates augmented datasets that are controlled for size to be 100% larger the bootstrap data (either by counting examples or total characters).

```python src/create_data_controlled_augmented_datasets.py --root_dirpath {root_dirpath} --dataset_name {forwards_dataset_name}```

### create maximum augmented datasets

The following script creates two augmented datasets.

The first dataset contains:
- bootstrap data
- all type correct generated proof terms

The second dataset contains:
 - bootstrap data
 - all generated proof terms (whether they be type correct or not)

```python src/create_maximum_augmented_datasets.py```

## acknowledgements

We include lean utility files from https://github.com/openai/lean-gym