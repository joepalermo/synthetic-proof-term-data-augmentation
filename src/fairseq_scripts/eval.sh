#!/bin/bash

DATASET_NAME=max-len-4096-2048_dist-filter-015_kernel-typecheck_backwards
MODEL_SIZE=big
MODEL_SUFFIX=first

CUDA_VISIBLE_DEVICES=0 fairseq-eval-lm /mnt/nfs/lean/preprocessed_data/${DATASET_NAME}_preprocessed \
--path /mnt/nfs/lean/checkpoints/${DATASET_NAME}_${MODEL_SIZE}_${MODEL_SUFFIX}_checkpoints/checkpoint_best.pt \
--seed 0 \
--model-parallel-size 1 \
--task language_modeling \
--bpe gpt2 \
--num-workers 64 \
--max-tokens 1536 \
--batch-size 512 \
--required-batch-size-multiple 64 \
--optimizer sgd \
--lr-scheduler fixed \
--sample-break-mode complete_doc \
--tensorboard-logdir /mnt/nfs/lean/logs/${DATASET_NAME}_${MODEL_SIZE}_${MODEL_SUFFIX}_logs \
--skip-invalid-size-inputs-valid-test \
--num-workers 2 \
--fp16 \
--ddp-backend no_c10d