source /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5/path.sh || exit 1;

# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5
# bash steps/train_deltas.sh 2000 10000 /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/lang /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/mono_ali /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone

# # Bigram graph
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5
# bash utils/mkgraph.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/lang_phones_bg  /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone/graph_bg

# # # Unigram graph
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5
# bash utils/mkgraph.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/lang_phones_ug  /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone/graph_ug

# ## Viterbi
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5
# bash steps/decode.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone/graph_bg /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone/decode_bg_test
# bash steps/decode.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone/graph_bg /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone/decode_bg_dev

# bash steps/decode.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone/graph_ug /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone/decode_ug_test
# bash steps/decode.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone/graph_ug /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone/decode_ug_dev

## Print mfcc ark data 
# cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5
# cat /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train/mfcc/raw_mfcc_train.1.ark | copy-feats ark:- ark,t:- | less > /home/eleanna/Desktop/master/nlp/nlp-labs/show_ark_1.txt

# Alignment
cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5
bash steps/align_si.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/lang /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone_train_ali
bash steps/align_si.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/lang /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone_test_ali
bash steps/align_si.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/lang /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/exp/triphone_dev_ali