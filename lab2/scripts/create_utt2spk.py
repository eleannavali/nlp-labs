import os

spath='../../kaldi/egs/usc/filesets/'
dpath='../../kaldi/egs/data/'

open(dpath+'test/wav.scp', 'w').close()
with open(spath+'testing.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'test/utt2spk.txt','a') as fd:
            fd.write(utterance_id.split("\n")[0]+' '+utterance_id.split('_')[0]+'\n')

open(dpath+'train/wav.scp', 'w').close()
with open(spath+'training.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'train/utt2spk.txt','a') as fd:
            fd.write(utterance_id.split("\n")[0]+' '+utterance_id.split('_')[0]+'\n')

open(dpath+'dev/wav.scp', 'w').close()
with open(spath+'validation.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'dev/utt2spk.txt','a') as fd:
            fd.write(utterance_id.split("\n")[0]+' '+utterance_id.split('_')[0]+'\n')
        
