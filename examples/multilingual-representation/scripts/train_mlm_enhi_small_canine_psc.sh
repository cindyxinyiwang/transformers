#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00


OUTDIR=/ocean/projects/dbs200003p/xinyiw1/rep/
DATADIR=/ocean/projects/dbs200003p/xinyiw1/data/enhi_small/
VOCABDIR=/ocean/projects/dbs200003p/xinyiw1/data/enhi_small/32k/

MODEL_DIR=${OUTDIR}/enhi_small_canine_mlm/
mkdir -p $OUTDIR
mkdir -p $MODEL_DIR

export HF_DATASETS_CACHE="${OUTDIR}/cache/datasets/"

#  --warmup_steps=1000 \
#  --num_train_epochs 50 \
#python -m torch.distributed.launch --nproc_per_node=2 --master_port=1200 code/run_mlm.py \
python code/run_mlm.py \
  --overwrite_output_dir \
  --model_type "canine-xlm-roberta" \
  --tokenizer_dir $VOCABDIR \
  --train_file ${DATADIR}/train.txt \
  --validation_file ${DATADIR}/en.valid.txt \
  --do_train \
  --do_eval \
  --save_steps=1000 \
  --save_total_limit=1 \
  --warmup_ratio 0.025 \
  --max_steps=50000 \
  --learning_rate=1e-4 \
  --max_seq_length 2048 \
  --max_position_embeddings 2050 \
  --hidden_size 512 \
  --intermediate_size 1024 \
  --num_attention_heads 8 \
  --num_hidden_layers 8 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --seed 42 \
  --remove_unused_columns False \
  --dataloader_num_workers 1 \
  --log_file ${MODEL_DIR}/train.log \
  --output_dir $MODEL_DIR
