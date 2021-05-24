#!/bin/bash
#SBATCH --partition=GPU-shared  
#SBATCH --nodes=1                                                                
#SBATCH --gres=gpu:1                                                             
#SBATCH --time=48:00:00


OUTDIR=/ocean/projects/dbs200003p/xinyiw1/rep/

MODEL_DIR=${OUTDIR}/enhi_small_mlm_bsz256_lr1e-4/
mkdir -p $OUTDIR
mkdir -p $MODEL_DIR

export HF_DATASETS_CACHE="${OUTDIR}/cache/datasets/"

#  --warmup_steps=1000 \
#  --num_train_epochs 50 \
#python -m torch.distributed.launch --nproc_per_node=2 code/run_mlm.py \
python  code/run_mlm.py \
  --overwrite_output_dir \
  --model_type "xlm-roberta" \
  --tokenizer_dir data/enhi_small_model/32k/ \
  --train_file data/enhi_small_model/train.txt \
  --validation_file data/enhi_small_model/en.valid.txt \
  --do_train \
  --do_eval \
  --save_steps=1000 \
  --save_total_limit=1 \
  --warmup_ratio 0.025 \
  --max_steps=50000 \
  --learning_rate=1e-4 \
  --max_seq_length 512 \
  --max_position_embeddings 520 \
  --hidden_size 512 \
  --intermediate_size 1024 \
  --num_attention_heads 8 \
  --num_hidden_layers 8 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 8 \
  --seed 42 \
  --log_file ${MODEL_DIR}/train.log \
  --output_dir $MODEL_DIR
