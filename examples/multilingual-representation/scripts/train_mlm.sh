export CUDA_VISIBLE_DEVICES=1

python code/run_mlm.py \
  --model_type "xlm-roberta" \
  --tokenizer_dir data/sw_model/ \
  --SDE "precalc" \
  --train_file data/sw_model/swwiki.train.txt \
  --validation_file data/sw_model/swwiki.valid.txt \
  --do_train \
  --do_eval \
  --save_steps=1000 \
  --save_total_limit=2 \
  --num_train_epochs=50 \
  --warmup_steps=1000 \
  --learning_rate=5e-4 \
  --max_seq_length 256 \
  --max_position_embeddings 300 \
  --hidden_size 512 \
  --intermediate_size 1024 \
  --num_attention_heads 8 \
  --num_hidden_layers 8 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --seed 42 \
  --log_file outputs/sw_sde_mlm/train.log \
  --output_dir outputs/sw_sde_mlm
