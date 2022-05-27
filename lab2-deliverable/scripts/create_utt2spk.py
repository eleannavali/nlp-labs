import os

spath='../../kaldi/egs/usc/filesets/'
dpath='../../kaldi/egs/usc/data/'

open(dpath+'test/utt2spk', 'w').close()
with open(spath+'testing.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'test/utt2spk','a') as fd:
            fd.write(utterance_id.split("\n")[0]+' '+utterance_id.split('_')[0]+'\n')

open(dpath+'train/utt2spk', 'w').close()
with open(spath+'training.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'train/utt2spk','a') as fd:
            fd.write(utterance_id.split("\n")[0]+' '+utterance_id.split('_')[0]+'\n')

open(dpath+'dev/utt2spk', 'w').close()
with open(spath+'validation.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'dev/utt2spk','a') as fd:
            fd.write(utterance_id.split("\n")[0]+' '+utterance_id.split('_')[0]+'\n')
        
