import os

spath='../../kaldi/egs/usc/filesets/'
dpath='../../kaldi/egs/usc/data/'
with open(spath+'testing.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'test/uttids','a') as fd:
            fd.write(utterance_id)

with open(spath+'training.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'train/uttids','a') as fd:
            fd.write(utterance_id)

with open(spath+'validation.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'dev/uttids','a') as fd:
            fd.write(utterance_id)
        
