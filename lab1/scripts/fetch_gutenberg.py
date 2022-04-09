import re
import sys

import contractions
import nltk
from copy import deepcopy
from util import dict_to_txt

def download_corpus(corpus="gutenberg"):
    """Download Project Gutenberg corpus, consisting of 18 classic books
    Book list:
       ['austen-emma.txt',
        'austen-persuasion.txt',
        'austen-sense.txt',
        'bible-kjv.txt',
        'blake-poems.txt',
        'bryant-stories.txt',
        'burgess-busterbrown.txt',
        'carroll-alice.txt',
        'chesterton-ball.txt',
        'chesterton-brown.txt',
        'chesterton-thursday.txt',
        'edgeworth-parents.txt',
        'melville-moby_dick.txt',
        'milton-paradise.txt',
        'shakespeare-caesar.txt',
        'shakespeare-hamlet.txt',
        'shakespeare-macbeth.txt',
        'whitman-leaves.txt']
    """
    # download corpus data
    nltk.download(corpus)
    corp = nltk.corpus.gutenberg  #load it 
    raw = corp.raw() # get the raw data (unprocessed)

    return raw


def identity_preprocess(s):
    return s


def clean_text(s):
    s = s.strip()  # strip leading / trailing spaces
    s = s.lower()  # convert to lowercase
    s = contractions.fix(s)  # e.g. don't -> do not, you're -> you are
    s = re.sub("\s+", " ", s)  # strip multiple whitespace (replace multiple whitespaces with just one)
    s = re.sub(r"[^a-z\s]", " ", s)  # keep only lowercase letters and spaces

    return s


def tokenize(s):
    tokenized = [w for w in s.split(" ") if len(w) > 0]  #split words by whitespaces, ignore empty string

    return tokenized


def preprocess(s):
    return tokenize(clean_text(s))


def process_file(corpus, preprocess=identity_preprocess, by='line'):
    if by == 'line':
      lines = [preprocess(ln) for ln in corpus.split("\n")]
      lines = [ln for ln in lines if len(ln) > 0]  # Ignore empty lines
      return lines
    elif by == 'word' :
      words = [preprocess(ln) for ln in corpus.split(" ")]
      words = [wrd for wrd in words if len(wrd) > 0]  # Ignore empty words
      return words
    
def create_dict(processed):
    tokens = [word for sentence in processed for word in sentence]
    print(f" {len(tokens)} total tokens in combined corpus")
    discrete_tokens = set(tokens)
    print(f" {len(discrete_tokens)} discrete tokens (words) in combined corpus")
    freq = nltk.FreqDist(tokens)
    freq_dict = dict(freq)
    return freq_dict

def clean_dict(freq_dict):

    print(" Words before deletion : ",len(freq_dict))
    temp_dict = deepcopy(freq_dict)
    for key in temp_dict.keys() :
        if freq_dict[key] < 5 :
            del freq_dict[key]

    print(" Words after deletion : ",len(freq_dict))
    return freq_dict


if __name__ == "__main__":
    CORPUS = []
    # combine 2 corpus  "gutenberg" and "webtext"
    if len(sys.argv) > 1:
        for i in range(1,len(sys.argv)):
            CORPUS.append(sys.argv[i])
    else:
        CORPUS.append('gutenberg')

    raw_combined = ''
    for corpus in CORPUS:
        raw_corpus = download_corpus(corpus=corpus)
        print(f"Corpus length : {len(raw_corpus)} tokens")
        raw_combined += raw_corpus
    preprocessed = process_file(raw_combined, preprocess=preprocess)
    print(f"Combined corpus length : {len(raw_combined)} tokens")
    dict_to_txt('../vocab/words.vocab.txt', clean_dict(create_dict(preprocessed)))
    
    # for words in preprocessed:
    #     sys.stdout.write(" ".join(words))
    #     sys.stdout.write("\n")
