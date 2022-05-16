import os

spath='../../kaldi/egs/usc/data/'
localpath='../../kaldi/egs/usc/data/local/dict/'


open(localpath+'lm_test.text', 'w').close()
with open(spath+'test/text','r') as fs:
    lines = fs.readlines()
    for line in lines:
        with open(localpath+'lm_test.text','a') as fd2:
            fd2.write(line.split(' ')[0]+' <s> '+ ' '.join(line.split(' ')[1:]).split('\n')[0] + ' </s>\n')

open(localpath+'lm_train.text', 'w').close()
with open(spath+'train/text','r') as fs:
    lines = fs.readlines()
    for line in lines:
        with open(localpath+'lm_train.text','a') as fd2:
            fd2.write(line.split(' ')[0]+' <s> '+ ' '.join(line.split(' ')[1:]).split('\n')[0] + ' </s>\n')

open(localpath+'lm_dev.text', 'w').close()
with open(spath+'dev/text','r') as fs:
    lines = fs.readlines()
    for line in lines:
        with open(localpath+'lm_dev.text','a') as fd2:
            fd2.write(line.split(' ')[0]+' <s> '+ ' '.join(line.split(' ')[1:]).split('\n')[0] + ' </s>\n')