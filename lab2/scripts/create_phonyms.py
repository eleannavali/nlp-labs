import os
import re

spath='../../kaldi/egs/usc/filesets/'
dpath='../../kaldi/egs/usc/data/'
lexpath='../../kaldi/egs/usc/lexicon.txt'

with open(lexpath,'r') as ft:
    lines = ft.readlines()
    phonems = {}
    for line in lines:
        line = line.split('\n')[0]
        phonems[line.split('\t')[0].lower()] = ' '.join(line.split(' ')[1:]).lower()


open(dpath+'test/text', 'w').close()
with open(dpath+'test/text_in_english.txt','r') as fs:
    lines = fs.readlines()
    for line in lines :
        line = line.split('\n')[0]
        line = re.sub(r"[^A-Za-z0-9'_ -]", '', line)
        line = re.sub(r"-", ' ', line)
        words = line.split(' ')
        with open(dpath+'test/text','a') as fd:
            fd.write(words[0] + ' sil')

            for word in words[1:] : 
                fd.write( ' ' + phonems[word.lower()])
            fd.write(' sil \n')


open(dpath+'train/text', 'w').close()
with open(dpath+'train/text_in_english.txt','r') as fs:
    lines = fs.readlines()
    for line in lines :
        line = line.split('\n')[0]
        line = re.sub(r"[^A-Za-z0-9'_ -]", '', line)
        line = re.sub(r"-", ' ', line)
        words = line.split(' ')
        with open(dpath+'train/text','a') as fd:
            fd.write(words[0] + ' sil')

            for word in words[1:] : 
                fd.write( ' ' + phonems[word.lower()])
            fd.write(' sil \n')


open(dpath+'dev/text', 'w').close()
with open(dpath+'dev/text_in_english.txt','r') as fs:
    lines = fs.readlines()
    for line in lines :
        line = line.split('\n')[0]
        line = re.sub(r"[^A-Za-z0-9'_ -]", '', line)
        line = re.sub(r"-", ' ', line)
        words = line.split(' ')
        with open(dpath+'dev/text','a') as fd:
            fd.write(words[0] + ' sil')

            for word in words[1:] : 
                fd.write( ' ' + phonems[word.lower()])
            fd.write(' sil \n')

# Keep only the different phonems
list_phonems = ' '.join(phonems.values()).split(' ')
phonems = set(list_phonems)
phonems = sorted(phonems)[1:]
# print(phonems,len(phonems))

open('../kaldi_process/data/local/dict/lexicon.txt', 'w').close()
with open('../kaldi_process/data/local/dict/lexicon.txt','a') as fd:
    for phonem in phonems:
        fd.write(phonem + ' ' + phonem + '\n')


phonems = [re.sub(r"sil", '', phonem) for phonem in list_phonems]
phonems = set(phonems)
phonems = sorted(phonems)[1:]

open('../kaldi_process/data/local/dict/nonsilence_phones.txt', 'w').close()
with open('../kaldi_process/data/local/dict/nonsilence_phones.txt','a') as fd:
    for phonem in phonems:
        fd.write(phonem + '\n')