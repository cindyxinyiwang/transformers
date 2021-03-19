
INPUT=mrhi_model/mrhiwiki.train.txt
VSIZE=64000
MNAME=mrhi_model/64k/sentencepiece.bpe

python spm.py --input=$INPUT \
   --model_prefix=$MNAME \
   --vocab_size=$VSIZE --character_coverage=1.0 --model_type=unigram

#OUTPUT=data/mrhi_model/mrhi.train.txt
#TRAIN=raw/swwiki.train
#VALID=raw/swwiki.valid
#TRAIN_PROC=processed/swwiki-spm32k.train
#VALID_PROC=processed/swwiki-spm32k.valid
#touch $TRAIN
#touch $VALID
#cat $INPUT | awk -F '\n' -v train="$TRAIN" -v valid="$VALID" '{if(rand()<0.95) {print  > train} else {print  > valid}}'  
#
#spm_encode --model=$MNAME.model \
#  --output_format=piece < $TRAIN > $TRAIN_PROC
#
#spm_encode --model=$MNAME.model \
#  --output_format=piece < $VALID > $VALID_PROC
#
