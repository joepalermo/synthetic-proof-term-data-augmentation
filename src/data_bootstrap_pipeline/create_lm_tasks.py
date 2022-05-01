import os
import matplotlib.pyplot as plt
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import filter_examples, get_forwards_example, get_backwards_example, normalize_whitespace, \
                  merge_files, write_text, split_theorems, filter_by_indices, write_examples_as_json
from config import subsample_rate, max_proof_length, max_theorem_length, distance_threshold, train_test_split_size, \
    test_val_split_size

# reset output directories ---------------------------------------------------------------------------------------------

output_dirpath = 'output'
forwards_text_dirpath = f'{output_dirpath}/forwards_text'
backwards_text_dirpath = f'{output_dirpath}/backwards_text'
forwards_text_decls_dirpath = f'{output_dirpath}/forwards_text/declarations'
backwards_text_decls_dirpath = f'{output_dirpath}/backwards_text/declarations'
forwards_json_dirpath = f'{output_dirpath}/forwards_json'
backwards_json_dirpath = f'{output_dirpath}/backwards_json'

# recreate forwards dirs
if os.path.isdir(forwards_text_dirpath):
    shutil.rmtree(forwards_text_dirpath)
os.mkdir(forwards_text_dirpath)
os.mkdir(forwards_text_decls_dirpath)

# recreate backwards dirs
if os.path.isdir(backwards_text_dirpath):
    shutil.rmtree(backwards_text_dirpath)
os.mkdir(backwards_text_dirpath)
os.mkdir(backwards_text_decls_dirpath)

# recreate forwards json dirs
if os.path.isdir(forwards_json_dirpath):
    shutil.rmtree(forwards_json_dirpath)
os.mkdir(forwards_json_dirpath)

# recreate backwards json dirs
if os.path.isdir(backwards_json_dirpath):
    shutil.rmtree(backwards_json_dirpath)
os.mkdir(backwards_json_dirpath)

# extract declaration filepaths
proofs_dirpath = f'{output_dirpath}/checked_proofs'
proofs_filenames = [fn for fn in os.listdir(proofs_dirpath) if '.txt' in fn]
proofs_filepaths = [os.path.join(proofs_dirpath, fn) for fn in proofs_filenames]
assert proofs_filepaths  # check proofs_filepaths is non-empty
proofs_filepaths = random.sample(proofs_filepaths, k=int(subsample_rate * len(proofs_filepaths)))

all_proofs = []
all_theorems = []
empty_declaration_file_count = proof_count = 0
# iterate through all checked proofs
for proofs_filepath in tqdm(proofs_filepaths):
    filename = proofs_filepath.split('/')[-1]
    declaration_name = filename.split('.txt')[0]

    # attempt to read proofs/theorems in paired proof/theorem files
    try:
        with open(proofs_filepath) as f:
            proofs = f.read().split(';\n')
        proofs = [p.strip() for p in proofs if p.strip() != ""]
        proof_count += len(proofs)
        if len(proofs) == 0:
            empty_declaration_file_count += 1
            continue
        else:
            # since there are proofs, extract the corresponding theorems
            theorems_filepath = proofs_filepath.replace('proofs', 'theorems')
            with open(theorems_filepath) as f:
                text = f.read()
            text = text.strip()[1:-1]  # remove '[' and ']' at start and end
            theorems = split_theorems(text)
            # if number of theorems doesn't match number of proofs then ignore this file
            if len(theorems) != len(proofs):
                print(f"{len(theorems)} == {len(proofs)} in {proofs_filepath}")
                continue
    except:
        print(f"failed to process proof/theorem {proofs_filepath}")
        continue

    # filter and cleanup proofs/theorems
    proof_theorem_tuples = [(p, t) for p, t in zip(proofs, theorems)
                            if len(normalize_whitespace(p)) < max_proof_length
                            and len(normalize_whitespace(t)) < max_theorem_length]
    proofs = [normalize_whitespace(p) for p, _ in proof_theorem_tuples]
    theorems = [normalize_whitespace(t) for _, t in proof_theorem_tuples]

    # write fairseq_scripts forwards LM task format
    filepath = os.path.join(forwards_text_decls_dirpath, f'{declaration_name}.txt')
    text = "\n\n".join([get_forwards_example(proof) for proof in proofs])
    write_text(filepath, text)

    # write fairseq_scripts backwards LM task format
    filepath = os.path.join(backwards_text_decls_dirpath, f'{declaration_name}.txt')
    text = "\n\n".join([get_backwards_example(proof, theorem) for proof, theorem in zip(proofs, theorems)])
    write_text(filepath, text)

    # cache all to extract stats at end
    all_proofs.extend(proofs)
    all_theorems.extend(theorems)

# gather basic stats
assert len(all_theorems) == len(all_proofs)
context_lengths = [len(t) for t in all_theorems]
target_lengths = [len(p) for p in all_proofs]
mean_observed_context_length = int(sum(context_lengths)/len(all_theorems))
max_observed_context_length = int(max(context_lengths))
mean_observed_target_length = int(sum(target_lengths)/len(all_proofs))
max_observed_target_length = int(max(target_lengths))

# print basic stats
non_empty_declaration_file_count = len(proofs_filepaths) - empty_declaration_file_count
print(f'non-empty declaration files: {non_empty_declaration_file_count}/{empty_declaration_file_count+non_empty_declaration_file_count}')
print(f'num typechecked proofs: {proof_count}')
print(f'mean context length: {mean_observed_context_length}, max context length: {max_observed_context_length}')
print(f'mean target length: {mean_observed_target_length}, max target length: {max_observed_target_length}')

# save histogram of lengths
fig, ax = plt.subplots()
ax.hist(context_lengths, bins=100, label='context')
ax.hist(target_lengths, bins=100, label='target')
plt.legend(loc="upper right")
plt.savefig('example_length_histogram.png')

# train-val-test split ----------------------------------------------------------

# split on declaration names
declaration_names = [fn.split('.txt')[0] for fn in os.listdir(forwards_text_decls_dirpath) if '.txt' in fn]
train_declaration_names, test_declaration_names = train_test_split(declaration_names, train_size=train_test_split_size)
test_declaration_names, val_declaration_names = train_test_split(test_declaration_names, train_size=test_val_split_size)

assert len(train_declaration_names + val_declaration_names + test_declaration_names) == len(declaration_names)
train_declaration_names = set(train_declaration_names)
val_declaration_names = set(val_declaration_names)
test_declaration_names = set(test_declaration_names)
assert len(train_declaration_names.intersection(val_declaration_names)) == 0
assert len(train_declaration_names.intersection(test_declaration_names)) == 0

# forwards LM task -------------------------------------------------------------------------------------------------

# get filepaths corresponding to each set of declaration names (fairseq_scripts forwards LM format)
train_text_filepaths = [os.path.join(forwards_text_decls_dirpath, decl_name + '.txt') for decl_name in train_declaration_names]
val_text_filepaths = [os.path.join(forwards_text_decls_dirpath, decl_name + '.txt') for decl_name in val_declaration_names]
test_text_filepaths = [os.path.join(forwards_text_decls_dirpath, decl_name + '.txt') for decl_name in test_declaration_names]

# merge the contents of the files in each set (fairseq_scripts forwards LM format)
merge_files(train_text_filepaths, os.path.join(forwards_text_dirpath, 'train.txt'))
merge_files(val_text_filepaths, os.path.join(forwards_text_dirpath, 'val.txt'))
merge_files(test_text_filepaths, os.path.join(forwards_text_dirpath, 'test.txt'))

# backwards LM task ------------------------------------------------------------------------------------------------

# get filepaths corresponding to each set of declaration names (fairseq_scripts forwards LM format)
train_text_filepaths = [os.path.join(backwards_text_decls_dirpath, decl_name + '.txt') for decl_name in train_declaration_names]
val_text_filepaths = [os.path.join(backwards_text_decls_dirpath, decl_name + '.txt') for decl_name in val_declaration_names]
test_text_filepaths = [os.path.join(backwards_text_decls_dirpath, decl_name + '.txt') for decl_name in test_declaration_names]

# merge the contents of the files in each set (fairseq_scripts forwards LM format)
merge_files(train_text_filepaths, os.path.join(backwards_text_dirpath, 'train.txt'))
merge_files(val_text_filepaths, os.path.join(backwards_text_dirpath, 'val.txt'))
merge_files(test_text_filepaths, os.path.join(backwards_text_dirpath, 'test.txt'))

# remove declarations files (which are now redundant)
shutil.rmtree(forwards_text_decls_dirpath)
shutil.rmtree(backwards_text_decls_dirpath)

# filter test and val sets ---------------------------------------------------------------------------------------------

# load forwards -------------------------------------------

with open(os.path.join(forwards_text_dirpath, 'train.txt')) as f:
    train_text = f.read()

forwards_train_examples = train_text.split('\n\n')

with open(os.path.join(forwards_text_dirpath, 'val.txt')) as f:
    val_text = f.read()
forwards_val_examples = val_text.split('\n\n')

with open(os.path.join(forwards_text_dirpath, 'test.txt')) as f:
    test_text = f.read()
forwards_test_examples = test_text.split('\n\n')

# load backwards -------------------------------------------

with open(os.path.join(backwards_text_dirpath, 'train.txt')) as f:
    train_text = f.read()
backwards_train_examples = train_text.split('\n\n')

with open(os.path.join(backwards_text_dirpath, 'val.txt')) as f:
    val_text = f.read()
backwards_val_examples = val_text.split('\n\n')

with open(os.path.join(backwards_text_dirpath, 'test.txt')) as f:
    test_text = f.read()
backwards_test_examples = test_text.split('\n\n')

assert len(forwards_train_examples) == len(backwards_train_examples)
assert len(forwards_val_examples) == len(backwards_val_examples)
assert len(forwards_test_examples) == len(backwards_test_examples)

# filter based on backwards -------------------------------------------

val_idxs = filter_examples(backwards_val_examples, backwards_train_examples, backwards_text_dirpath, 'filtered_val.txt',
    distance_threshold=distance_threshold, limit='max')

test_idxs = filter_examples(backwards_test_examples, backwards_train_examples, backwards_text_dirpath, 'filtered_test.txt',
    distance_threshold=distance_threshold, limit='max')

# apply filter to forwards -------------------------------------------

print('pre-filter forwards (train/val/test):')
print(len(forwards_train_examples), len(forwards_val_examples), len(forwards_test_examples))
print('pre-filter backwards (train/val/test):')
print(len(backwards_train_examples), len(backwards_val_examples), len(backwards_test_examples))

forwards_val_examples = filter_by_indices(forwards_val_examples, val_idxs)
forwards_test_examples = filter_by_indices(forwards_test_examples, test_idxs)
backwards_val_examples = filter_by_indices(backwards_val_examples, val_idxs)
backwards_test_examples = filter_by_indices(backwards_test_examples, test_idxs)

assert len(forwards_train_examples) == len(backwards_train_examples)
assert len(forwards_val_examples) == len(backwards_val_examples)
assert len(forwards_test_examples) == len(backwards_test_examples)

print('post-filter forwards (train/val/test):')
print(len(forwards_train_examples), len(forwards_val_examples), len(forwards_test_examples))
print('post-filter backwards (train/val/test):')
print(len(backwards_train_examples), len(backwards_val_examples), len(backwards_test_examples))

# write to file -------------------------------------------

shutil.copyfile(os.path.join(backwards_text_dirpath, 'val.txt'), os.path.join(backwards_text_dirpath, 'unfiltered_val.txt'))
shutil.copyfile(os.path.join(backwards_text_dirpath, 'test.txt'), os.path.join(backwards_text_dirpath, 'unfiltered_test.txt'))
shutil.copyfile(os.path.join(forwards_text_dirpath, 'val.txt'), os.path.join(forwards_text_dirpath, 'unfiltered_val.txt'))
shutil.copyfile(os.path.join(forwards_text_dirpath, 'test.txt'), os.path.join(forwards_text_dirpath, 'unfiltered_test.txt'))

with open(f'{backwards_text_dirpath}/val.txt', 'w') as f:
    for example in backwards_val_examples:
        f.write(example + '\n\n')
with open(f'{backwards_text_dirpath}/test.txt', 'w') as f:
    for example in backwards_test_examples:
        f.write(example + '\n\n')
with open(f'{forwards_text_dirpath}/val.txt', 'w') as f:
    for example in forwards_val_examples:
        f.write(example + '\n\n')
with open(f'{forwards_text_dirpath}/test.txt', 'w') as f:
    for example in forwards_test_examples:
        f.write(example + '\n\n')

# write examples as json as well
write_examples_as_json(forwards_train_examples, f'{forwards_json_dirpath}/train.txt')
write_examples_as_json(forwards_val_examples, f'{forwards_json_dirpath}/val.txt')
write_examples_as_json(forwards_test_examples, f'{forwards_json_dirpath}/test.txt')
write_examples_as_json(backwards_train_examples, f'{backwards_json_dirpath}/train.txt')
write_examples_as_json(backwards_val_examples, f'{backwards_json_dirpath}/val.txt')
write_examples_as_json(backwards_test_examples, f'{backwards_json_dirpath}/test.txt')

