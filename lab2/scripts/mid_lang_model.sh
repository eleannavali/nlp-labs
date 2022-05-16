## Preprocess language model
source /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5/path.sh || exit 1;

# Use kaldi command IRSTLM
# cd ../../kaldi/tools/irstlm/scripts
# IRSTLM="/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/tools/irstlm"
# bash build-lm.sh -i /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/dict/lm_train.text -n 2 -o /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/lm_tmp/mid_bigram_model.ilm.gz
# bash build-lm.sh -i /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/dict/lm_train.text -n 1 -o /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/lm_tmp/mid_unigram_model.ilm.gz

# # Use kaldi command ARPA
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/tools/irstlm/src
# ./compile-lm /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/lm_tmp/mid_bigram_model.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/nist_lm/lm_phone_bg.arpa.gz
# ./compile-lm /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/lm_tmp/mid_unigram_model.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/nist_lm/lm_phone_ug.arpa.gz

# # # Create fst with kaldi
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5/
# bash utils/prepare_lang.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/dict "<oov>" /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/lang /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/lang

# # # Sort files in kaldi data
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev
# sort -k 1 -o ./wav.scp ./wav.scp
# sort -k 1 -o ./text ./text
# sort -k 1 -o ./text_in_english.txt ./text_in_english.txt
# sort -k 1 -o utt2spk utt2spk
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train
# sort -k 1 -o ./wav.scp ./wav.scp
# sort -k 1 -o ./text ./text
# sort -k 1 -o ./text_in_english.txt ./text_in_english.txt
# sort -k 1 -o ./utt2spk ./utt2spk
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test
# sort -k 1 -o ./wav.scp ./wav.scp
# sort -k 1 -o ./text ./text
# sort -k 1 -o ./text_in_english.txt ./text_in_english.txt
# sort -k 1 -o ./utt2spk ./utt2spk

# # # Create spk2utt
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5/
# ./utils/utt2spk_to_spk2utt.pl /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev/utt2spk> /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev/spk2utt
# ./utils/utt2spk_to_spk2utt.pl /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train/utt2spk > /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train/spk2utt
# ./utils/utt2spk_to_spk2utt.pl /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test/utt2spk > /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test/spk2utt

# Create grammar fst
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/timit/s5/local/
# bash timit_format_data.sh

## Peplexity
cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/tools/irstlm/src
# ./compile-lm /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/lm_tmp/mid_unigram_model.ilm.gz -eval=/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/dict/lm_dev.text
# ./compile-lm /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/lm_tmp/mid_bigram_model.ilm.gz -eval=/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/dict/lm_dev.text
# ./compile-lm /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/lm_tmp/mid_unigram_model.ilm.gz -eval=/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/dict/lm_test.text
# ./compile-lm /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/lm_tmp/mid_bigram_model.ilm.gz -eval=/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/local/dict/lm_test.text