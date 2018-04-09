#!/usr/bin/env bash
lang1=$1
lang2=$2
lang3=$3

NMT=$(pwd)
export PATH=$PATH:$NMT/bin

NAME="run_${lang1}_${lang2}-${lang3}"
OUT="temp/$NAME"

DATA="/projects/tir1/users/dsachan/multilingual_nmt/data/${lang1}_${lang2}-${lang3}"
TRAIN_SRC=$DATA/train.${lang1}
TRAIN_TGT=$DATA/train.${lang2}-${lang3}
TEST_SRC=$DATA/test.${lang1}
TEST_TGT=$DATA/test.${lang2}-${lang3}
VALID_SRC=$DATA/dev.${lang1}
VALID_TGT=$DATA/dev.${lang2}-${lang3}

DATA_L1_L2="/projects/tir1/users/dsachan/multilingual_nmt/data/${lang1}_${lang2}"
DATA_L1_L3="/projects/tir1/users/dsachan/multilingual_nmt/data/${lang1}_${lang3}"

TEST_SRC_L1_L2=$DATA_L1_L2/test.${lang1}
TEST_TGT_L1_L2=$DATA_L1_L2/test.${lang2}

TEST_SRC_L1_L3=$DATA_L1_L3/test.${lang1}
TEST_TGT_L1_L3=$DATA_L1_L3/test.${lang3}

BPE_OPS=32000

echo "Output dir = $OUT"
[ -d $OUT ] || mkdir -p $OUT
[ -d $OUT/data ] || mkdir -p $OUT/data
[ -d $OUT/models ] || mkdir $OUT/models
[ -d $OUT/test ] || mkdir -p  $OUT/test

echo "Step 1a: Preprocess inputs"

echo "Learning BPE on source and target combined"
cat ${TRAIN_SRC} ${TRAIN_TGT} | learn_bpe -s ${BPE_OPS} > $OUT/data/bpe-codes.${BPE_OPS}

echo "Applying BPE on source"
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < $TRAIN_SRC > $OUT/data/train.${lang1}
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < $VALID_SRC > $OUT/data/valid.${lang1}
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < $TEST_SRC > $OUT/data/test.${lang1}

echo "Applying BPE on target"
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} <  $TRAIN_TGT > $OUT/data/train.${lang2}-${lang3}
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} <  $VALID_TGT > $OUT/data/valid.${lang2}-${lang3}
# We dont touch the test References, No BPE on them!
cp $TEST_TGT $OUT/data/test.${lang2}

# Apply BPE Coding to the languages
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < ${TEST_SRC_L1_L2} > ${OUT}/data/test_${lang2}.src
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < ${TEST_SRC_L1_L3} > ${OUT}/data/test_${lang3}.src


# Create vocabulary file for BPE
echo -e "<unk>\n<s>\n</s>" > "${OUT}/data/vocab.bpe.${BPE_OPS}"
cat "${OUT}/data/train.${lang1}" "${OUT}/data/train.${lang2}" | \
    get_vocab | cut -f1 -d ' ' >> "${OUT}/data/vocab.bpe.${BPE_OPS}"

# Duplicate vocab file with language suffix
cp "${OUT}/data/vocab.bpe.${BPE_OPS}" "${OUT}/data/vocab.bpe.${BPE_OPS}.${lang1}"
cp "${OUT}/data/vocab.bpe.${BPE_OPS}" "${OUT}/data/vocab.bpe.${BPE_OPS}.${lang2}-${lang3}"

python -m nmt.nmt \
    --src=${lang1} --tgt=${lang2}-${lang3} \
    --hparams_path=nmt/standard_hparams/ted17.json \
    --out_dir=$OUT/models \
    --vocab_prefix=${OUT}/data/vocab.bpe.${BPE_OPS} \
    --train_prefix=${OUT}/data/train \
    --dev_prefix=${OUT}/data/valid \
    --test_prefix=${OUT}/data/test

mv $OUT/models/output_dev{,.bpe}
mv $OUT/models/output_test{,.bpe}

cat $OUT/models/output_dev.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid.out
cat $OUT/models/output_test.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test.out

# Compute BLEU score
multi-bleu $OUT/data/test.${lang2}-${lang3} < $OUT/test/test.out > $OUT/test/test.tc.bleu
t2t-bleu --translation=$OUT/test/test.out --reference=$OUT/data/test.${lang2}-${lang3} > $OUT/test/test.t2t-bleu


# Do inference on lang2
python -m nmt.nmt \
    --src=${lang1} --tgt=${lang2}-${lang3} \
    --ckpt=$OUT/models/translate.ckpt \
    --hparams_path=nmt/standard_hparams/ted17.json \
    --out_dir=$OUT/models \
    --vocab_prefix=${OUT}/data/vocab.bpe.${BPE_OPS} \
    --inference_input_file=${OUT}/data/test_${lang2}.src \
    --inference_output_file=${OUT}/test/test.${lang2}.out \
    --inference_ref_file=${TEST_TGT_L1_L2}

mv ${OUT}/test/test.${lang2}{.out,.bpe}
cat ${OUT}/test/test.${lang2}.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > ${OUT}/test/test.${lang2}.out

# Compute BLEU score
multi-bleu ${TEST_TGT_L1_L2} < ${OUT}/test/test.${lang2}.out > $OUT/test/test_${lang2}.tc.bleu
t2t-bleu --translation=${OUT}/test/test.${lang2}.out --reference=${TEST_TGT_L1_L2} > $OUT/test/test_${lang2}.t2t-bleu


# Do inference on lang3
python -m nmt.nmt \
    --src=${lang1} --tgt=${lang2}-${lang3} \
    --ckpt=$OUT/models/translate.ckpt \
    --hparams_path=nmt/standard_hparams/ted17.json \
    --out_dir=$OUT/models \
    --vocab_prefix=${OUT}/data/vocab.bpe.${BPE_OPS} \
    --inference_input_file=${OUT}/data/test_${lang3}.src \
    --inference_output_file=${OUT}/test/test.${lang3}.out \
    --inference_ref_file=${TEST_TGT_L1_L3}

mv ${OUT}/test/test.${lang3}{.out,.bpe}
cat ${OUT}/test/test.${lang3}.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > ${OUT}/test/test.${lang3}.out

# Compute BLEU score
multi-bleu ${TEST_TGT_L1_L3} < ${OUT}/test/test.${lang3}.out > $OUT/test/test_${lang3}.tc.bleu
t2t-bleu --translation=${OUT}/test/test.${lang3}.out --reference=${TEST_TGT_L1_L3} > $OUT/test/test_${lang3}.t2t-bleu
