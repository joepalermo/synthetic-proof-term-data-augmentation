#!/bin/bash

DATASET_NAME=max-len-4096-2048_dist-filter-015_kernel-typecheck_backwards
MODEL_SIZE=big
MODEL_SUFFIX=first

# if dataset has not been preprocessed, then preprocess it
if [ ! -d "/mnt/nfs/lean/preprocessed_data/${DATASET_NAME}_preprocessed" ]
then
    fairseq-preprocess --seed 0 --trainpref /mnt/nfs/lean/data/$DATASET_NAME/train.txt \
    --validpref /mnt/nfs/lean/data/$DATASET_NAME/val.txt \
    --testpref /mnt/nfs/lean/data/$DATASET_NAME/test.txt \
    --destdir /mnt/nfs/lean/preprocessed_data/${DATASET_NAME}_preprocessed \
    --task language_modeling \
    --bpe gpt2 \
    --workers 128 \
    --only-source
fi

# if data is not available locally, then copy data from NFS to local machine
if [ ! -d "/home/dgxuser/lean_data_preprocessed/${DATASET_NAME}_preprocessed" ]
then
  cp -r /mnt/nfs/lean/preprocessed_data/${DATASET_NAME}_preprocessed /home/dgxuser/lean_data_preprocessed/${DATASET_NAME}_preprocessed
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train /home/dgxuser/lean_data_preprocessed/${DATASET_NAME}_preprocessed \
--restore-file checkpoint_last.pt \
--seed 0 \
--model-parallel-size 1 \
--task language_modeling \
--bpe gpt2 \
--num-workers 64 \
--max-tokens 1536 \
--batch-size 512 \
--required-batch-size-multiple 64 \
--arch transformer_lm_gpt2_${MODEL_SIZE} \
--max-epoch 100000 \
--optimizer sgd \
--lr-scheduler fixed \
--lr 0.01 \
--dropout 0.5 \
--patience 100 \
--sample-break-mode complete_doc \
--decoder-embed-dim 2000 \
--tensorboard-logdir /mnt/nfs/lean/logs/${DATASET_NAME}_${MODEL_SIZE}_${MODEL_SUFFIX}_logs \
--save-dir /mnt/nfs/lean/checkpoints/${DATASET_NAME}_${MODEL_SIZE}_${MODEL_SUFFIX}_checkpoints \
--skip-invalid-size-inputs-valid-test \
--num-workers 2 \
--fp16 \
--keep-last-epochs 5 \
--ddp-backend no_c10d