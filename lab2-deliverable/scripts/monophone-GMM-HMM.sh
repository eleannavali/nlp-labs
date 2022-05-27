source /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5/path.sh || exit 1;

# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5
# bash steps/train_mono.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/lang /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono

# # Bigram graph
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5
# bash utils/mkgraph.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/lang_phones_bg  /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono/graph_bg

# # Unigram graph
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5
# bash utils/mkgraph.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/lang_phones_ug  /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono/graph_ug

## Viterbi
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5
# bash steps/decode.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono/graph_bg /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono/decode_bg_test
# bash steps/decode.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono/graph_bg /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono/decode_bg_dev

# bash steps/decode.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono/graph_ug /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono/decode_ug_test
# bash steps/decode.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono/graph_ug /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono/decode_ug_dev

# Alignment
cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5
bash steps/align_si.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/lang /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono_ali