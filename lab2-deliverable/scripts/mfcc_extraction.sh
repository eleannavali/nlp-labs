source /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5/path.sh || exit 1;

cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5
# bash steps/make_mfcc.sh --mfcc-config  /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/conf/mfcc.conf /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train/log /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train/mfcc
# bash steps/make_mfcc.sh --mfcc-config  /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/conf/mfcc.conf /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test/log /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test/mfcc
# bash steps/make_mfcc.sh --mfcc-config  /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/conf/mfcc.conf /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev/log /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev/mfcc

bash steps/compute_cmvn_stats.sh  /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train/log /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/train/cmvn
bash steps/compute_cmvn_stats.sh  /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test/log /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/test/cmvn
bash steps/compute_cmvn_stats.sh  /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev/log /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data/dev/cmvn