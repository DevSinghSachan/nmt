lang1=$1
lang2=$2

NMT=$(pwd)
export PATH=$PATH:$NMT/bin

NAME="run_${lang1}_${lang2}"
OUT="temp/$NAME"

DATA="/projects/tir1/users/dsachan/multilingual_nmt/data/${lang1}_${lang2}"
TRAIN_SRC=$DATA/train.${lang1}
TRAIN_TGT=$DATA/train.${lang2}
TEST_SRC=$DATA/test.${lang1}
TEST_TGT=$DATA/test.${lang2}
VALID_SRC=$DATA/dev.${lang1}
VALID_TGT=$DATA/dev.${lang2}

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
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} <  $TRAIN_TGT > $OUT/data/train.${lang2}
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} <  $VALID_TGT > $OUT/data/valid.${lang2}
# We dont touch the test References, No BPE on them!
cp $TEST_TGT $OUT/data/test.${lang2}

# Create vocabulary file for BPE
echo -e "<unk>\n<s>\n</s>" > "${OUT}/data/vocab.bpe.${BPE_OPS}"
cat "${OUT}/data/train.${lang1}" "${OUT}/data/train.${lang2}" | \
    get_vocab | cut -f1 -d ' ' >> "${OUT}/data/vocab.bpe.${BPE_OPS}"

# Duplicate vocab file with language suffix
cp "${OUT}/data/vocab.bpe.${BPE_OPS}" "${OUT}/data/vocab.bpe.${BPE_OPS}.${lang1}"
cp "${OUT}/data/vocab.bpe.${BPE_OPS}" "${OUT}/data/vocab.bpe.${BPE_OPS}.${lang2}"

python -m nmt.nmt \
    --src=${lang1} --tgt=${lang2} \
    --hparams_path=nmt/standard_hparams/ted17.json \
    --out_dir=$OUT/models \
    --vocab_prefix=${OUT}/data/vocab.bpe.${BPE_OPS} \
    --train_prefix=${OUT}/data/train \
    --dev_prefix=${OUT}/data/valid \
    --test_prefix=${OUT}/data/test

mv $OUT/test/output_dev{,.bpe}
mv $OUT/test/output_test{,.bpe}

cat $OUT/test/output_dev.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid.out
cat $OUT/test/output_test.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test.out

t2t-bleu --translation=$OUT/test/test.out --reference=$OUT/data/test.${lang2} > lang2.t2t-bleu
