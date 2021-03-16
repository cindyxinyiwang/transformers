export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

OUTDIR=outputs/
MODEL_DIR=${OUTDIR}/mrhi_mlm_sde_word_test/
mkdir -p $OUTDIR
mkdir -p $MODEL_DIR

#  --train_file data/mrhi_model/mrhi-100k.txt \
#  --line_by_line \
#  --tokenizer_file data/mrhi_model/char4gram32k.vocab \
#python code/run_mlm.py \
#  --sde_ave True \
#  --max_steps=35000 \
#  --sde_selfnorm_w 0.1 \
#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 code/run_mlm.py \
python code/run_mlm.py \
  --SDE "full" \
  --sde_type "swap" \
  --overwrite_output_dir \
  --model_type "sde-xlm-roberta" \
  --tokenizer_dir data/mrhi_model/ \
  --train_file data/mrhi_model/mrhiwiki.train.txt \
  --validation_file data/mrhi_model/mrhiwiki.valid.txt \
  --do_train \
  --do_eval \
  --save_steps=1000 \
  --save_total_limit=2 \
  --warmup_steps=1000 \
  --num_train_epochs=10 \
  --learning_rate=5e-4 \
  --max_seq_length 256 \
  --max_position_embeddings 300 \
  --hidden_size 512 \
  --intermediate_size 1024 \
  --num_attention_heads 8 \
  --num_hidden_layers 8 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --seed 42 \
  --log_file ${MODEL_DIR}/train.log \
  --remove_unused_columns False \
  --output_dir $MODEL_DIR
