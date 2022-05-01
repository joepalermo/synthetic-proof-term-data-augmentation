import argparse
import os
import random
from time import time
from data_bootstrap_pipeline.utils import normalize_whitespace, write_pickle, get_current_datetime_string
from utils.generation_utils import infer, initiate_model, build_batch

parser = argparse.ArgumentParser()
parser.add_argument("--root_dirpath", help="path to root directory containing data and checkpoints", type=str)
parser.add_argument("--dataset_name", help="name of dataset", type=str)
parser.add_argument("--model_name", help="name of model", type=str)
parser.add_argument("--gpu_id", help="gpu to use", type=int, default=0)
parser.add_argument("--batch_size", help="batch size", type=int, default=640, required=False)
parser.add_argument("--num_batches", help="total number of inference batches to run", type=int, default=10000)
parser.add_argument("--temperature", help="temperature to sample with", type=float, default=1.3)
parser.add_argument("--min_cache_size", help="number of unique generated proof terms to accumulate before caching",
                    type=int, default=50000)
parser.add_argument("--verbose", help="whether to print samples", type=bool, default=False)
args = parser.parse_args()

# load model
preprocessed_data_dirpath = f"{args.root_dirpath}/preprocessed_data/{args.dataset_name}_preprocessed"
checkpoint_filepath = f"{args.root_dirpath}/checkpoints/{args.dataset_name}_{args.model_name}_checkpoints/checkpoint_best.pt"
cfg, task, max_positions, align_dict, tokenizer, bpe, generator, models, src_dict, tgt_dict = \
    initiate_model(preprocessed_data_dirpath, checkpoint_filepath, args.temperature, gpu_id=args.gpu_id, nbest=1)

# init proof caches
data_dirpath = f'{args.root_dirpath}/data/{args.dataset_name}'
generated_dirpath = f'{data_dirpath}/generated'
if not os.path.isdir(generated_dirpath):
    os.mkdir(generated_dirpath)

round_i = 0
candidates_set = set()
while round_i < args.num_batches:

    # sample from trained model ----------------------------------------------------------------------------------------
    print(f'\nstarting round {round_i}\n')
    # construct context strings for model inference
    batch = build_batch(args.batch_size)

    # sample outputs for each context string
    print("\tcalling inference...")
    t = time()
    context_samples = infer(batch, cfg, task, max_positions, align_dict, tokenizer, bpe, generator, models, src_dict,
                            tgt_dict, args.batch_size, args.gpu_id)
    print(f"\tinference complete in {round(time()-t, 1)}s")

    # inspect samples
    if args.verbose:
        random_idx = random.randint(0, len(batch)-1)
        context, samples = batch[random_idx], context_samples[random_idx]
        print(f"\tinspect sample from inference:")
        print(f"\tcontext: {context}")
        print(f"\t\tsample: {samples[0]}")

    # extract proof term candidates from samples
    candidates = []
    for samples in context_samples:
        for sample in samples:
            try:
                candidate = sample.split('EOT')[0]
                candidates.append(normalize_whitespace(candidate))
            except:
                pass

    # cache candidates
    candidates_set = candidates_set.union(set(candidates))
    if len(candidates_set) > args.min_cache_size:
        dt = get_current_datetime_string()
        write_pickle(f'{generated_dirpath}/{dt}.pkl', candidates_set)
        candidates_set = set()
    round_i += 1
