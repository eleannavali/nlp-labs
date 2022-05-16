import os

spath='../../kaldi/egs/usc/filesets/'
dpath='../../kaldi/egs/usc/data/'
wpath='/home/eleanna/Desktop/master/nlp/nlp-labs/kaldi/egs/usc/wav/'


open(dpath+'test/wav.scp', 'w').close()
with open(spath+'testing.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'test/wav.scp','a') as fd:
            fd.write(utterance_id.split('\n')[0]+' '+wpath+utterance_id.split('\n')[0]+'.wav'+'\n')

open(dpath+'train/wav.scp', 'w').close()
with open(spath+'training.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'train/wav.scp','a') as fd:
            fd.write(utterance_id.split('\n')[0]+' '+wpath+utterance_id.split('\n')[0]+'.wav'+'\n')

open(dpath+'dev/wav.scp', 'w').close()
with open(spath+'validation.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'dev/wav.scp','a') as fd:
            fd.write(utterance_id.split('\n')[0]+' '+wpath+utterance_id.split('\n')[0]+'.wav'+'\n')
        
