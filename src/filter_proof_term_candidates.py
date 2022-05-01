import os
import shutil
import random
import numpy as np
from sqlitedict import SqliteDict
from data_bootstrap_pipeline.utils import typecheck_all_proofs, split_theorems, load_sets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dirpath", help="path to root directory containing data and checkpoints", type=str)
parser.add_argument("--dataset_name", help="name of dataset", type=str)
parser.add_argument("--max_num_to_typecheck", help="max number to typecheck", type=int, default=int(1e7))
parser.add_argument("--num_shards", help="number of shards to use when splitting candidate proofs", type=int, default=20)
args = parser.parse_args()

# load cache of candidates
data_dirpath = f'{args.root_dirpath}/data/{args.dataset_name}'
generated_dirpath = f'{args.root_dirpath}/data/{args.dataset_name}/generated'
generated_cache = list(load_sets(generated_dirpath))
random.shuffle(generated_cache)
filtered_cache = SqliteDict(f'{data_dirpath}/filtered_candidates.sqlite', autocommit=True)

# empty and recreate output directories
if os.path.isdir(f'{data_dirpath}/shards_to_typecheck'):
    shutil.rmtree(f'{data_dirpath}/shards_to_typecheck')
os.mkdir(f'{data_dirpath}/shards_to_typecheck')
if os.path.isdir(f'{data_dirpath}/typechecked_shards'):
    shutil.rmtree(f'{data_dirpath}/typechecked_shards')
os.mkdir(f'{data_dirpath}/typechecked_shards')
os.mkdir(f'{data_dirpath}/typechecked_shards/checked_proofs')
os.mkdir(f'{data_dirpath}/typechecked_shards/checked_theorems')

# split into shards
all_candidates = list(generated_cache)[:args.max_num_to_typecheck]
shards = np.array_split(np.array(all_candidates), args.num_shards)
shards = [arr.tolist() for arr in shards]
print(f'{len(all_candidates)} generated, split into {args.num_shards} shards of length aproximately '
      f'{len(all_candidates)//args.num_shards} each')

# write shards to file
generated_passed_examples = []
for i, shard in enumerate(shards):
    with open(f'{data_dirpath}/shards_to_typecheck/shard_{i}.txt', 'w') as f:
        shard_text = ";\n".join([candidate for candidate in shard])
        f.write(shard_text)

# typecheck all
source_dirpath = f'{data_dirpath}/shards_to_typecheck'
output_dirpath = f'{data_dirpath}/typechecked_shards'
typecheck_all_proofs(source_dirpath, output_dirpath, args.num_shards)

# process typechecked output
num_passed = 0
for filename in [f'shard_{i}.txt' for i in range(args.num_shards)]:
    with open(f'{output_dirpath}/checked_proofs/{filename}') as f:
        proofs = f.read().split(';\n')
    proofs = [p.strip() for p in proofs if p.strip() != ""]
    with open(f'{output_dirpath}/checked_theorems/{filename}') as f:
        text = f.read()
    text = text.strip()[1:-1]  # remove '[' and ']' at start and end
    theorems = split_theorems(text)
    assert len(proofs) == len(theorems)
    num_passed += len(proofs)
    for proof, theorem in zip(proofs, theorems):
        filtered_cache[proof] = theorem
print(f"pass rate: {num_passed}/{len(all_candidates)} = {round(num_passed/len(all_candidates)*100, 2)} %")