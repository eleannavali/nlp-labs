import os

spath='../../kaldi/egs/usc/filesets/'
dpath='../../kaldi/egs/data/'
transpath='../../kaldi/egs/usc/transcriptions.txt'

with open(transpath,'r') as ft:
    texts = ft.readlines()


open(dpath+'test/text.txt', 'w').close()
with open(spath+'testing.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'test/text.txt','a') as fd:
            id = int(utterance_id.split('_')[1].split('\n')[0])-1
            fd.write(utterance_id.split('\n')[0]+' '+texts[id][4:])

open(dpath+'train/text.txt', 'w').close()
with open(spath+'training.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'train/text.txt','a') as fd:
            id = int(utterance_id.split('_')[1].split('\n')[0])-1
            fd.write(utterance_id.split('\n')[0]+' '+texts[id][4:])

open(dpath+'dev/text.txt', 'w').close()
with open(spath+'validation.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'dev/text.txt','a') as fd:
            id = int(utterance_id.split('_')[1].split('\n')[0])-1
            fd.write(utterance_id.split('\n')[0]+' '+texts[id][4:])
        
