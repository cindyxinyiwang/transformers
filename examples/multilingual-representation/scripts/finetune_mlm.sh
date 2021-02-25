export CUDA_VISIBLE_DEVICES=0

OUTDIR=outputs/
MODEL_DIR=${OUTDIR}/bert_mr_mlm/
mkdir -p $OUTDIR
mkdir -p $MODEL_DIR

#  --train_file data/mrhi_model/mrhi-test.txt \
#  --line_by_line \
#  --model_type "xlm-roberta" \
#  --SDE "precalc" \

#  --tau 0.1 \
#  --topk 10 \
#  --self_aug \
python code/run_mlm.py \
  --overwrite_output_dir \
  --model_name_or_path "bert-base-multilingual-cased" \
  --train_file data/mrhi_model/mr-10k.txt \
  --validation_file data/mrhi_model/mrhi-tail-valid.txt \
  --max_seq_length 256 \
  --do_train \
  --do_eval \
  --save_steps=1000 \
  --save_total_limit=2 \
  --num_train_epochs=5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --seed 42 \
  --log_file ${MODEL_DIR}/train.log \
  --output_dir $MODEL_DIR
