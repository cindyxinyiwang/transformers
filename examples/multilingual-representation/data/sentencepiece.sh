
INPUT=data/raw/hiwiki-latest-pages-articles.txt,data/raw/mrwiki-latest-pages-articles.txt
VSIZE=32000
MNAME=data/mrhi_model/sentencepiece.bpe

spm_train --input=$INPUT \
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
