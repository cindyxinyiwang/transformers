

export CUDA_VISIBLE_DEVICES=0

python run_mlm.py \
  --model_type "xlm-roberta" \
  --tokenizer_dir data/sw_model/ \
  --train_file data/sw_model/swwiki.train.txt \
  --validation_file data/sw_model/swwiki.valid.txt \
  --do_train \
  --do_eval \
  --save_steps=500 \
  --save_total_limit=2 \
  --num_train_epochs=50 \
  --warmup_steps=1000 \
  --learning_rate=5e-4 \
  --max_seq_length 256 \
  --max_position_embeddings 300 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --output_dir outputs/sw_mlm
