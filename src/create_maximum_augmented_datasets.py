from sqlitedict import SqliteDict
from data_bootstrap_pipeline.utils import normalize_whitespace, create_new_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dirpath", help="path to root directory containing data and checkpoints", type=str)
parser.add_argument("--forwards_dataset_name", help="name of dataset", type=str)
parser.add_argument("--forwards", help="whether to augment forwards or backwards datasets", type=bool)
args = parser.parse_args()

# settings
forwards_dirpath = f'{args.root_dirpath}/data/{args.forwards_dataset_name}'
backwards_dirpath = forwards_dirpath.replace('forwards', 'backwards')
toggled_dirpath = forwards_dirpath if args.forwards else backwards_dirpath
bootstrap_train_filepath = f'{toggled_dirpath}/train.txt'
generated_passed_cache = SqliteDict(f'{forwards_dirpath}/filtered_candidates.sqlite')
generated_cache = SqliteDict(f'{forwards_dirpath}/all_candidates.sqlite')

# extract proof terms that have passed filtering
generated_passed_examples = []
for proof, theorem in list(generated_passed_cache.items()):
    if args.forwards:
        example = "PROOF {} EOT".format(proof)
    else:
        example = "THEOREM {} PROOF {} EOT".format(theorem[:-1], proof)
    example = normalize_whitespace(example)
    generated_passed_examples.append(example)
print(f'# total generated passed examples: {len(generated_passed_examples)}')

# load bootstrap train examples
with open(bootstrap_train_filepath) as f:
    bootstrap_train_examples = f.read().split('\n\n')
num_bootstrap_train_examples = len(bootstrap_train_examples)
print(f'# bootstrap train examples: {num_bootstrap_train_examples}')

# extract proof terms that have failed filtering
generated_examples = [normalize_whitespace("PROOF {} EOT".format(proof))
                             for proof in list(generated_cache)]
print(f'# generated examples: {len(generated_examples)}')

combined_examples = bootstrap_train_examples + generated_examples
print('bootstrap_examples + generated_examples = combined_examples')
print(len(bootstrap_train_examples), ' + ', len(generated_examples), ' = ', len(combined_examples))
with open(f'{toggled_dirpath}/combined_train.txt', 'w') as f:
    for example in combined_examples:
        f.write(f'{example}\n\n')
create_new_dataset(toggled_dirpath, f'{toggled_dirpath}_augmented_all', 'combined_train.txt')
