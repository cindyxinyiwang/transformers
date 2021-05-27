#!/bin/bash
#SBATCH --partition=GPU-shared
##SBATCH --nodes=1
##SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --gpus=v100-32:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

##SBATCH --partition=RM-shared
##SBATCH --ntasks-per-node=32


OUTDIR=/ocean/projects/dbs200003p/xinyiw1/rep/

OUTDIR=/ocean/projects/dbs200003p/xinyiw1/rep/
DATADIR=/ocean/projects/dbs200003p/xinyiw1/data/enhi/
VOCABDIR=/ocean/projects/dbs200003p/xinyiw1/data/enhi_small/32k/


MODEL_DIR=${OUTDIR}/enhi_mlm_bsz4096_lr1e-3/
mkdir -p $OUTDIR
mkdir -p $MODEL_DIR

export HF_DATASETS_CACHE="${OUTDIR}/cache/datasets/"

#  --warmup_steps=1000 \
#  --num_train_epochs 50 \
#python  code/run_mlm.py \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1221 code/run_mlm.py \
  --preprocessing_num_workers 32 \
  --overwrite_output_dir \
  --model_type "xlm-roberta" \
  --tokenizer_dir $VOCABDIR \
  --train_file ${DATADIR}/train.txt \
  --validation_file ${DATADIR}/valid_en.txt \
  --do_train \
  --do_eval \
  --save_steps=1000 \
  --save_total_limit=1 \
  --warmup_ratio 0.06 \
  --max_steps=50000 \
  --learning_rate=1e-3 \
  --max_seq_length 128 \
  --max_position_embeddings 520 \
  --hidden_size 512 \
  --intermediate_size 1024 \
  --num_attention_heads 8 \
  --num_hidden_layers 8 \
  --per_device_train_batch_size 256 \
  --gradient_accumulation_steps 4 \
  --dataloader_num_workers 4 \
  --remove_unused_columns False \
  --seed 42 \
  --log_file ${MODEL_DIR}/train.log \
  --output_dir $MODEL_DIR
