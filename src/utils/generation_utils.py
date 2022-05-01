import ast
import fileinput
import logging
import math
import os
import subprocess
import sys
import time
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def initiate_model(preprocessed_data_dirpath, checkpoint_filepath, temperature, gpu_id=0, nbest=1):
    # Get args
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser, input_args=[preprocessed_data_dirpath, "--task",
                                                           "language_modeling", "--path",
                                                           checkpoint_filepath,
                                                           "--sampling", "--beam", f"{nbest}", "--nbest", f"{nbest}",
                                                           "--temperature", f"{temperature}"])
    cfg = convert_namespace_to_omegaconf(args)

    # Setup task, e.g., translation
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda(gpu_id)
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    return cfg, task, max_positions, align_dict, tokenizer, bpe, generator, models, src_dict, tgt_dict


def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if cfg.generation.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    if cfg.generation.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )


def infer(input, cfg, task, max_positions, align_dict, tokenizer, bpe, generator, models, src_dict, tgt_dict,
          batch_size, gpu_id):
    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x
    use_cuda = torch.cuda.is_available()
    results = []
    cfg.dataset.batch_size = batch_size

    for batch in make_batches(input, cfg, task, max_positions, encode_fn):
        bsz = batch.src_tokens.size(0)
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        constraints = batch.constraints
        if use_cuda:
            src_tokens = src_tokens.cuda(gpu_id)
            src_lengths = src_lengths.cuda(gpu_id)
            if constraints is not None:
                constraints = constraints.cuda(gpu_id)

        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
        }

        translations = task.inference_step(
            generator, models, sample, constraints=constraints
        )


        list_constraints = [[] for _ in range(bsz)]
        if cfg.generation.constraints:
            list_constraints = [unpack_constraints(c) for c in constraints]
        for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
            src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
            constraints = list_constraints[i]
            results.append(
                (
                    0 + id,
                    src_tokens_i,
                    hypos,
                    {
                        "constraints": constraints,
                    },
                )
            )

    hypo_str_batch = []
    # sort output to match input order
    for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
        hypo_str_id = []
        src_str = ''
        if src_dict is not None:
            src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)

        # Process top predictions
        for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str=src_str,
                alignment=hypo["alignment"],
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=cfg.common_eval.post_process,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
            )
            detok_hypo_str = decode_fn(hypo_str)
            detok_hypo_str_without_src = detok_hypo_str[len(src_str):]
            score = hypo["score"] / math.log(2)  # convert to base 2
            hypo_str_id.append(detok_hypo_str_without_src)

        hypo_str_batch.append(hypo_str_id)
    return hypo_str_batch


def build_batch(batch_size):
    return pad_with_empty_context_examples([], batch_size, token="PROOF")


def pad_with_empty_context_examples(batch, batch_size, token):
    assert len(batch) <= batch_size
    # if len(batch) < batch_size:
    empty_context_strings = [token for _ in range(batch_size - len(batch))]
    batch.extend(empty_context_strings)
    return batch


def typecheck_shard(shard_filename, candidates_dirpath, results_dirpath, theorems_dirpath):
    stdout_dest = f'{theorems_dirpath}/{shard_filename}'
    with open(stdout_dest, "w") as f:
        subprocess.run(['lean', '--run', 'src/typecheck_generated_proofs.lean', shard_filename, candidates_dirpath,
                        results_dirpath], stdout=f, stderr=f)