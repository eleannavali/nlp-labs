import os

spath='../../kaldi/egs/usc/filesets/'
dpath='../../kaldi/egs/data/'
with open(spath+'testing.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'test/uttids.txt','a') as fd:
            fd.write(utterance_id)

with open(spath+'training.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'train/uttids.txt','a') as fd:
            fd.write(utterance_id)

with open(spath+'validation.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'dev/uttids.txt','a') as fd:
            fd.write(utterance_id)
        
