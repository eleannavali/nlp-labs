# Create soft links to folders
ln -s /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5/steps /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/steps
ln -s /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5/utils /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/utils

# Better to use absolute paths in soft links
ln -s /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5/steps/score_kaldi.sh /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/local/score_kaldi.sh

# Create some directories
cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/data
mkdir lang local 
cd local
mkdir dict lm_tmp nist_lm
