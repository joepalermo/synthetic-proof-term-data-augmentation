import difflib
import json
import os
import pickle
import re
import glob
import shutil
import datetime
import subprocess
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import Pool
from pathos.multiprocessing import ProcessingPool
from string_grouper import match_most_similar, compute_pairwise_similarities


def read_pickle(filepath):
    return pickle.load(open(filepath, "rb"))


def write_pickle(filepath, obj):
    pickle.dump(obj, open(filepath, "wb"))


def write_json(filepath, obj, pact_style=False):
    if pact_style:
        json_string = json.dumps(obj)
        json_string = json_string.replace('"}, {"', '"}\n{"')  # put one dict on each line without commas
        json_string = json_string[1:-1]  # remove '[' and ']'
    else:
        json_string = json.dumps(obj, indent=4)
    with open(filepath, 'w') as f:
        f.write(json_string)


def write_text(filepath, text):
    with open(filepath, 'w') as f:
        f.write(text)


def recreate_dirpath(dirpath):
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)


def get_current_datetime_string():
    now = datetime.datetime.now()
    datetime_components = [str(now.year), str(now.month).zfill(2), str(now.day).zfill(2),
                           str(now.hour).zfill(2), str(now.minute).zfill(2), str(now.second).zfill(2)]
    return "-".join(datetime_components)


def flatten(list_of_lists):
    return [x for ls in list_of_lists for x in ls]


def total_overlap(source, target):
    matching_chars = [source_ch == target_ch for source_ch, target_ch in zip(source, target[:len(source)])]
    return sum(matching_chars)/len(source)


def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


def filter_examples(test_examples, train_examples, cache_dirpath, output_filename,
                    distance_threshold=0.25, limit='max', num_shards=50):

    def find_most_similar_train_example(test_shard):
        matches_series = match_most_similar(pd.Series(train_examples), pd.Series(test_shard), ignore_index=True)
        # sims = compute_pairwise_similarities(pd.Series(test_shard), matches_series).tolist()
        distance_scores = [levenshtein(source, target)/len(source) for source, target in zip(test_shard, matches_series.tolist())]
        return [(test_ex, match, distance_score) for test_ex, match, distance_score in
                zip(test_shard, matches_series.tolist(), distance_scores)]

    # shard test_examples
    if limit == 'max':
        limit = len(test_examples)
    shards = np.array_split(np.array(test_examples[:limit]), num_shards)
    shards = [arr.tolist() for arr in shards]

    # run get_max_shard_overlaps on each shard
    t = time()
    most_similar_examples = ProcessingPool().map(find_most_similar_train_example, shards)
    most_similar_examples = flatten(most_similar_examples)
    print(f"\tprocessed all shards in {round(time() - t, 1)}s")

    # cache max overlaps info
    write_pickle(f'{cache_dirpath}/max_overlaps_{output_filename}'.replace('.txt', '.pkl'), most_similar_examples)
    fair_test_examples = []
    fair_idxs = []
    with open(f'{cache_dirpath}/max_overlaps_{output_filename}', 'w') as f:
        for i, (test_example, closest_train_example, distance_score) in enumerate(most_similar_examples):
            f.write(f'similarity score: {distance_score}\n')
            f.write(f'test example: {test_example}\n')
            f.write(f'train example: {closest_train_example}\n\n')
            if distance_score > distance_threshold:
                fair_test_examples.append(test_example)
                fair_idxs.append(i)
    print(f'{len(most_similar_examples)-len(fair_idxs)} filtered out of {len(most_similar_examples)}')
    return fair_idxs


def normalize_pairs(pairs):
    contexts = normalize_whitespace([pair['context'] for pair in pairs])
    targets = normalize_whitespace([pair['target'] for pair in pairs])
    return [{'context': context, 'target': target} for context, target in zip(contexts, targets)]


def get_forwards_example(proof):
    return ' '.join(['PROOF', proof, 'EOT'])


def get_backwards_example(proof, theorem):
    return ' '.join(['THEOREM', theorem, 'PROOF', proof, 'EOT'])


def extract_training_pairs(proofs):

    # sort proofs by ascending order in length
    proofs = sorted(proofs, key=len)

    # for each proof, find the proofs which prefix it
    prefix_dict = {}  # maps an index j to the list of indices for which i is a prefix of j
    for i in range(len(proofs)-1):
        for j in range(i+1, len(proofs)):
            if proofs[j].startswith(proofs[i]):
                if i not in prefix_dict:
                    prefix_dict[j] = [i]
                else:
                    prefix_dict[j].append(i)

    # extract the largest prefix for each proof
    index_pairs = []
    for j in prefix_dict:
        sorted(prefix_dict[j])
        index_pairs.append((max(prefix_dict[j]), j))

    ## debugging prints
    # print(prefix_dict)
    # print(sorted(index_pairs, key=lambda x: x[1]))

    # each proof with a prefix proof can be converted into a context-target pair where both typecheck
    pairs = []
    for prefix_i, other_i in index_pairs:
        context = proofs[prefix_i]
        assert proofs[other_i].startswith(context)
        target = proofs[other_i][len(context):]
        pair = {'context': context, 'target': target}
        assert context + target == proofs[other_i]
        pairs.append(pair)

    # proofs without a prefix proof can be converted into a context-target pair where the context is bindings-only
    proofs_idxs_without_prefix = [proof_i for proof_i in range(len(proofs)) if proof_i not in prefix_dict]
    for proof_i in proofs_idxs_without_prefix:
        context = extract_bindings(proofs[proof_i])
        target = proofs[proof_i][len(context):]
        pair = {'context': context, 'target': target}
        assert context + target == proofs[proof_i]
        pairs.append(pair)

    # add binding gen task
    all_bindings = [extract_bindings(proof) for proof in proofs]
    all_unique_bindings = set([bs for bs in all_bindings if len(bs) > 0])
    binding_gen_pairs = [{'context': '', 'target': bindings} for bindings in all_unique_bindings]
    pairs.extend(binding_gen_pairs)

    return pairs


def clip_lengths(pairs, max_length):
    return [pair for pair in pairs if len(pair['context']) + len(pair['target']) < max_length]


def merge_files(filepaths, destination_filepath):
    all_contents = []
    for fp in filepaths:
        with open(fp) as f:
            content = f.read()
        if len(content) > 0:
            all_contents.append(content)
    content = '\n\n'.join(all_contents)
    with open(destination_filepath, 'w') as f:
        f.write(content)


def extract_bindings(proof):
    bracket_stack = list()
    opening_brackets = ['[', '(', '{']
    closing_brackets = [']', ')', '}']
    for i, ch in enumerate(proof):
        if ch == ',' and len(bracket_stack) == 0:
            return proof[:i+1]  # return bindings
        if ch in opening_brackets:
            bracket_stack.append(None)
        elif ch in closing_brackets:
            bracket_stack.pop()
    return ""  # no bindings found


def normalize_whitespace(inpt):
    """replace any sequnce of whitespace characters with a single space"""
    if type(inpt) == list:
        return [' '.join(s.split()) for s in inpt]
    elif type(inpt) == str:
        return ' '.join(inpt.split())


def split_theorems(text):
    pattern = ',\n [^\s]'  # this pattern is observed between theorems
    match_indices = [m.start(0) for m in re.finditer(pattern, text)]
    if len(match_indices) > 0:
        start_index = match_indices[0]
        theorems = [text[0:start_index]] + [text[i+3:j].strip() for i, j in zip(match_indices, match_indices[1:] + [None])]
    else:
        theorems = [text]
    return theorems


def process_options(inpts):
    processed_inpts = []
    for inpt in inpts:
        if inpt == 'none':
            processed_inpts.append(None)
        elif inpt.startswith('(some '):
            processed_inpts.append(inpt[len('(some '):])
    return processed_inpts


def create_new_dataset(source_dirpath, destination_dirpath, new_train_filename):
    if os.path.isdir(destination_dirpath):
        shutil.rmtree(destination_dirpath)
    os.mkdir(destination_dirpath)
    # copy txt files
    shutil.copyfile(f'{source_dirpath}/{new_train_filename}', f'{destination_dirpath}/train.txt')
    shutil.copyfile(f'{source_dirpath}/val.txt', f'{destination_dirpath}/val.txt')
    shutil.copyfile(f'{source_dirpath}/test.txt', f'{destination_dirpath}/test.txt')
    # copy json files
    shutil.copyfile(f'{source_dirpath}/{new_train_filename}'.replace('txt', 'json'),
                    f'{destination_dirpath}/train.json')
    shutil.copyfile(f'{source_dirpath}/val.json', f'{destination_dirpath}/val.json')
    shutil.copyfile(f'{source_dirpath}/test.json', f'{destination_dirpath}/test.json')


def typecheck_all_proofs(source_dirpath, output_dirpath, num_processes):
    print('typechecking...')
    with Pool(num_processes) as pool:
        t = time()
        source_filenames = [fn for fn in os.listdir(source_dirpath) if '.txt' in fn]
        pool_args = [(source_dirpath, output_dirpath, source_filename) for source_filename in source_filenames]
        finished_count = 0
        for _ in tqdm(pool.imap_unordered((lambda x: typecheck_proofs(*x)), pool_args), total=len(source_filenames)):
            finished_count += 1
            print(f"{finished_count} files typechecked")
    print(f'{time() - t} seconds taken to typecheck all proofs')


def typecheck_proofs(source_dirpath, output_dirpath, source_filename):
    lean_cmd = ['lean', '--run', './src/data_bootstrap_pipeline/typecheck_proofs.lean', source_dirpath, output_dirpath,
                source_filename]
    stdout_dest = f'{output_dirpath}/checked_theorems/{source_filename}'
    with open(stdout_dest, "w") as f:
        subprocess.run(lean_cmd, stdout=f, stderr=f)


def filter_by_indices(ls, idxs):
    return [x for i, x in enumerate(ls) if i in set(idxs)]


def example_to_dict(example):
    if 'THEOREM' in example:
        theorem, proof = example.split('PROOF')
        theorem = theorem.replace('THEOREM', '').strip()
        proof = proof.replace('EOT', '').strip()
        return {'context': theorem, 'target': proof}
    else:
        example = example.replace('PROOF', '')
        example = example.replace('NON_TYPECHECKED_PROOF', '')
        example = example.replace('EOT', '')
        example = example.strip()
        return {'context': '', 'target': example}


def write_examples_as_json(examples, filepath):
    pairs = [example_to_dict(ex) for ex in examples]
    write_json(filepath, pairs, pact_style=True)


def load_sets(dirpath):
    filepaths = glob.glob(f"{dirpath}/*")
    print("reading")
    sets = [read_pickle(fp) for fp in tqdm(filepaths)]
    print("taking union of all sets of generated proof terms")
    union_set = set.union(*sets)
    print("done loading generated proof terms")
    return union_set


def total_character_length(dataset):
    return sum([len(example) for example in dataset])