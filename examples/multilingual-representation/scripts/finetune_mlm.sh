export CUDA_VISIBLE_DEVICES=1

OUTDIR=outputs/
MODEL_DIR=${OUTDIR}/bert_jointfo0.1k_mlm_tau0_gmask/
mkdir -p $OUTDIR
mkdir -p $MODEL_DIR

#  --train_file data/mrhi_model/mrhi-test.txt \
#  --line_by_line \
#  --model_type "xlm-roberta" \
#  --SDE "precalc" \

#  --topk 10 \
#  --self_aug \

#  --tau 0.1 \
#  --augment_model_path outputs/bert_jointfo0.1k_mlm_tau0/ \
python code/finetune_mlm.py \
  --grad_mask \
  --overwrite_output_dir \
  --model_name_or_path "bert-base-multilingual-cased" \
  --train_file data/mono/is/iswiki.train.txt \
  --meta_train_file data/mono/fo/fo0.1k.train.txt \
  --validation_file data/mono/is/iswiki.valid.txt \
  --max_seq_length 256 \
  --do_train \
  --do_eval \
  --save_steps=1000 \
  --max_steps=1000 \
  --save_total_limit=2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --seed 42 \
  --log_file ${MODEL_DIR}/train.log \
  --output_dir $MODEL_DIR
