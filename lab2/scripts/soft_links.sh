# Create soft links to folders
ln -s ../../kaldi/egs/wsj/s5/steps ../kaldi_process/steps
ln -s ../../kaldi/egs/wsj/s5/utils ../kaldi_process/utils

# Better to use absolute paths in soft links
ln -s /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5/steps/score_kaldi.sh /home/eleanna/Desktop/master/nlp/nlp-labs/lab2/kaldi_process/local/score_kaldi.sh

# Create some directories
cd ../kaldi_process
mkdir data
cd data
mkdir lang local 
cd local
mkdir dict lm_tmp nist_lm
