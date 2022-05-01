import os
import shutil
import random
import argparse
from sqlitedict import SqliteDict
from data_bootstrap_pipeline.utils import normalize_whitespace, create_new_dataset, load_sets, write_examples_as_json, \
                                          total_character_length

parser = argparse.ArgumentParser()
parser.add_argument("--root_dirpath", help="path to root directory containing data and checkpoints", type=str)
parser.add_argument("--dataset_name", help="name of dataset", type=str)
args = parser.parse_args()

# configure directory paths
forwards = 'forwards' in args.dataset_name
if forwards:
    forwards_dirpath = f'{args.root_dirpath}/data/{args.dataset_name}'
    backwards_dirpath = forwards_dirpath.replace('forwards', 'backwards')
    toggled_dirpath = forwards_dirpath
elif 'backwards' in args.dataset_name:
    backwards_dirpath = f'{args.root_dirpath}/data/{args.dataset_name}'
    forwards_dirpath = backwards_dirpath.replace('backwards', 'forwards')
    toggled_dirpath = backwards_dirpath
else:
    raise Exception(f'{args.dataset_name} should contain either forwards or backwards')
bootstrap_train_filepath = f'{toggled_dirpath}/train.txt'

# load data
generated_dirpath = f'{forwards_dirpath}/generated'
generated_cache = list(load_sets(generated_dirpath))
generated_passed_cache = SqliteDict(f'{forwards_dirpath}/filtered_candidates.sqlite')

# load bootstrap train examples
with open(bootstrap_train_filepath) as f:
    bootstrap_train_examples = f.read().split('\n\n')
num_bootstrap_train_examples = len(bootstrap_train_examples)
print(f'# bootstrap train examples: {num_bootstrap_train_examples}')

# extract proof terms that have passed filtering
generated_passed_examples = []
for proof, theorem in list(generated_passed_cache.items()):
    if forwards:
        example = "PROOF {} EOT".format(proof)
    else:
        example = "THEOREM {} PROOF {} EOT".format(theorem, proof)
    example = normalize_whitespace(example)
    generated_passed_examples.append(example)
print(f'# total generated passed examples: {len(generated_passed_examples)}')

# extract proofs terms which have not been typechecked
generated_non_typechecked_examples = [normalize_whitespace("NON_TYPECHECKED_PROOF {} EOT".format(proof))
                             for proof in list(generated_cache)]
print(f'# total non-typechecked examples: {len(generated_cache)}')

# shuffle
random.shuffle(generated_passed_examples)
random.shuffle(generated_non_typechecked_examples)

# assess total character lengths
generated_passed_string = "\n\n".join(generated_passed_examples)
generated_non_typechecked_string = "\n\n".join(generated_non_typechecked_examples)
total_bootstrap_character_length = total_character_length(bootstrap_train_examples)
print(f'total bootstrap character length (in millions): {round(total_bootstrap_character_length/1_000_000, 2)}')
print(f'total passed character length (in millions): {round(len(generated_passed_string)/1_000_000, 2)}')
print(f'total non-typechecked character length (in millions): {round(len(generated_non_typechecked_string)/1_000_000, 2)}')

# select character-controlled
char_weighted_100_pct_generated_passed_examples = \
    generated_passed_string[:total_bootstrap_character_length].split("\n\n")[:-1]
char_weighted_100_pct_generated_non_typechecked_examples = \
    generated_non_typechecked_string[:total_bootstrap_character_length].split("\n\n")[:-1]
char_weighted_50_pct_generated_passed_examples = \
    generated_passed_string[:total_bootstrap_character_length//2].split("\n\n")[:-1]
char_weighted_50_pct_generated_non_typechecked_examples = \
    generated_non_typechecked_string[:total_bootstrap_character_length//2].split("\n\n")[:-1]

# select example-controlled subsets
example_weighted_100_pct_passed_examples = generated_passed_examples[:len(bootstrap_train_examples)]
example_weighted_100_pct_non_typechecked_examples = generated_non_typechecked_examples[:len(bootstrap_train_examples)]
example_weighted_50_pct_passed_examples = generated_passed_examples[:len(bootstrap_train_examples)//2]
example_weighted_50_pct_non_typechecked_examples = generated_non_typechecked_examples[:len(bootstrap_train_examples)//2]

# build bootstrap + 100% (by examples) passed
combined_examples = bootstrap_train_examples + example_weighted_100_pct_passed_examples
print('\nbuild bootstrap + 100% (by examples) passed = total')
print(len(bootstrap_train_examples), ' + ', len(example_weighted_100_pct_passed_examples), ' = ',
      len(combined_examples))
with open(f'{toggled_dirpath}/combined_train.txt', 'w') as f:
    for example in combined_examples:
        f.write(f'{example}\n\n')
write_examples_as_json(combined_examples, f'{toggled_dirpath}/combined_train.json')
create_new_dataset(toggled_dirpath, f'{toggled_dirpath}_augmented_passed', 'combined_train.txt')

# build bootstrap + 100% (by characters) passed
combined_examples = bootstrap_train_examples + char_weighted_100_pct_generated_passed_examples
print('\nbootstrap + 100% (by characters) passed = total')
print(len(bootstrap_train_examples), ' + ', len(char_weighted_100_pct_generated_passed_examples), ' = ',
      len(combined_examples))
with open(f'{toggled_dirpath}/combined_train.txt', 'w') as f:
    for example in combined_examples:
        f.write(f'{example}\n\n')
write_examples_as_json(combined_examples, f'{toggled_dirpath}/combined_train.json')
create_new_dataset(toggled_dirpath, f'{toggled_dirpath}_augmented_passed_char_weighted', 'combined_train.txt')

# build bootstrap + 50% (by examples) passed + 50% (by examples) non-typechecked
combined_examples = bootstrap_train_examples + example_weighted_50_pct_passed_examples + \
    example_weighted_50_pct_non_typechecked_examples
print('\nbuild bootstrap + 50% (by examples) passed + 50% (by examples) non-typechecked')
print(len(bootstrap_train_examples), ' + ', len(example_weighted_50_pct_passed_examples), ' + ',
      len(example_weighted_50_pct_non_typechecked_examples), ' = ', len(combined_examples))
with open(f'{toggled_dirpath}/combined_train.txt', 'w') as f:
    for example in combined_examples:
        f.write(f'{example}\n\n')
write_examples_as_json(combined_examples,
                       f'{toggled_dirpath}/combined_train.json')
create_new_dataset(toggled_dirpath, f'{toggled_dirpath}_augmented_half_and_half', 'combined_train.txt')

# build bootstrap + 50% (by characters) passed + 50% (by characters) non-typechecked
combined_examples = bootstrap_train_examples + char_weighted_50_pct_generated_passed_examples + \
                                              char_weighted_50_pct_generated_non_typechecked_examples
print('\nbuild bootstrap + 50% (by characters) passed + 50% (by characters) non-typechecked')
print(len(bootstrap_train_examples), ' + ', len(char_weighted_50_pct_generated_passed_examples), ' + ',
      len(char_weighted_50_pct_generated_non_typechecked_examples) // 2, ' = ', len(combined_examples))
with open(f'{toggled_dirpath}/combined_train.txt', 'w') as f:
    for example in combined_examples:
        f.write(f'{example}\n\n')
write_examples_as_json(combined_examples, f'{toggled_dirpath}/combined_train.json')
create_new_dataset(toggled_dirpath, f'{toggled_dirpath}_augmented_half_and_half_char_weighted', 'combined_train.txt')

# build bootstrap + 100% (by examples) non-typechecked
combined_examples = bootstrap_train_examples + example_weighted_100_pct_non_typechecked_examples
print('\nbuild bootstrap + 100% (by examples) non-typechecked')
print(len(bootstrap_train_examples), ' + ', len(example_weighted_100_pct_non_typechecked_examples), ' = ',
      len(combined_examples))
with open(f'{toggled_dirpath}/combined_train.txt', 'w') as f:
    for example in combined_examples:
        f.write(f'{example}\n\n')
write_examples_as_json(combined_examples, f'{toggled_dirpath}/combined_train.json')
create_new_dataset(toggled_dirpath, f'{toggled_dirpath}_augmented_non_typechecked', 'combined_train.txt')

# build bootstrap + 100% (by characters) non-typechecked
combined_examples = bootstrap_train_examples + char_weighted_100_pct_generated_non_typechecked_examples
print('\nbuild bootstrap + 100% (by characters) non-typechecked')
print(len(bootstrap_train_examples), ' + ', len(char_weighted_100_pct_generated_non_typechecked_examples), ' = ',
      len(combined_examples))
with open(f'{toggled_dirpath}/combined_train.txt', 'w') as f:
    for example in combined_examples:
        f.write(f'{example}\n\n')
write_examples_as_json(combined_examples, f'{toggled_dirpath}/combined_train.json')
create_new_dataset(toggled_dirpath, f'{toggled_dirpath}_augmented_non_typechecked_char_weighted', 'combined_train.txt')