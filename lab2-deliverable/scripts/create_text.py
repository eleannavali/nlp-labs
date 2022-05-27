import os

spath='../../kaldi/egs/usc/filesets/'
dpath='../../kaldi/egs/usc/data/'
localpath='../kaldi_process/data/local/dict/'
transpath='../../kaldi/egs/usc/transcriptions.txt'

with open(transpath,'r') as ft:
    texts = ft.readlines()


open(dpath+'test/text_in_english.txt', 'w').close()
open(localpath+'lm_test.text', 'w').close()
with open(spath+'testing.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'test/text_in_english.txt','a') as fd1:
            id = int(utterance_id.split('_')[1].split('\n')[0])-1
            fd1.write(utterance_id.split('\n')[0]+' '+texts[id][4:])
        with open(localpath+'lm_test.text','a') as fd2:
            id = int(utterance_id.split('_')[1].split('\n')[0])-1
            fd2.write(utterance_id.split('\n')[0]+' <s> '+texts[id][4:].split('\n')[0] + ' </s>\n')

open(dpath+'train/text_in_english.txt', 'w').close()
open(localpath+'lm_train.text', 'w').close()
with open(spath+'training.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'train/text_in_english.txt','a') as fd1:
            id = int(utterance_id.split('_')[1].split('\n')[0])-1
            fd1.write(utterance_id.split('\n')[0]+' '+texts[id][4:])
        with open(localpath+'lm_train.text','a') as fd2:
            id = int(utterance_id.split('_')[1].split('\n')[0])-1
            fd2.write(utterance_id.split('\n')[0]+' <s> '+texts[id][4:].split('\n')[0] + ' </s>\n')

open(dpath+'dev/text_in_english.txt', 'w').close()
open(localpath+'lm_dev.text', 'w').close()
with open(spath+'validation.txt','r') as fs:
    utterance_ids = fs.readlines()
    for utterance_id in utterance_ids:
        with open(dpath+'dev/text_in_english.txt','a') as fd1:
            id = int(utterance_id.split('_')[1].split('\n')[0])-1
            fd1.write(utterance_id.split('\n')[0]+' '+texts[id][4:])
        with open(localpath+'lm_dev.text','a') as fd2:
            id = int(utterance_id.split('_')[1].split('\n')[0])-1
            fd2.write(utterance_id.split('\n')[0]+' <s> '+texts[id][4:].split('\n')[0] + ' </s>\n')

