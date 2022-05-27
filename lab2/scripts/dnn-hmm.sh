source /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/wsj/s5/path.sh || exit 1;

cd /home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc
# bash run_dnn.sh

# python3 torch_dataset.py
## python3 torch_dnn.py

## Train model
# python3 timit_dnn.py

## Extract posteriors in test dataset
# python3 extract_posteriors.py

## Decode posteriors by using graph
bash run_dnn.sh