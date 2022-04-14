import glob
import os
import re
from gensim.models import Word2Vec
import numpy as np
import sklearn

#SCRIPT_DIRECTORY = os.path.realpath(__file__)

data_dir = "../data/aclImdb/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
pos_train_dir = os.path.join(train_dir, "pos")
neg_train_dir = os.path.join(train_dir, "neg")
pos_test_dir = os.path.join(test_dir, "pos")
neg_test_dir = os.path.join(test_dir, "neg")

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 5000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000


SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(SEED)


def strip_punctuation(s):
    return re.sub(r"[^a-zA-Z\s]", " ", s)


def preprocess(s):
    return re.sub("\s+", " ", strip_punctuation(s).lower())


def tokenize(s):
    return s.split(" ")


def preproc_tok(s):
    return tokenize(preprocess(s))


def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, "*.txt"))
    data = []

    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        try :
            with open(sample, "r") as fd:
                x = [preprocess(l) for l in fd][0]
                data.append(x)
        except :
            print("Can't open file  ", sample)

    return data


def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)

    return list(corpus[indices]), list(y[indices])


def extract_nbow(corpus, model_path='../outputs/gutenberg_w2v.100d.10epochs.model'):
    """Extract neural bag of words representations"""
    # load w2v model 
    model = Word2Vec.load(model_path)
    model_wv = model.wv
    
    raise NotImplementedError("Implement nbow extractor")


def train_sentiment_analysis(train_corpus, train_labels):
    """Train a sentiment analysis classifier using NBOW + Logistic regression"""
    raise NotImplementedError("Implement sentiment analysis training")


def evaluate_sentiment_analysis(classifier, test_corpus, test_labels):
    """Evaluate classifier in the test corpus and report accuracy"""
    raise NotImplementedError("Implement sentiment analysis evaluation")


if __name__ == "__main__":
    # read review folders
    train_pos = read_samples(pos_train_dir) 
    train_neg = read_samples(neg_train_dir) 
    test_pos = read_samples(pos_test_dir) 
    test_neg = read_samples(neg_test_dir) 
    for t in [train_pos, train_neg, test_pos, test_neg] :
        print(type(t))
        print(len(t))


    # TODO: read Imdb corpus
    pos = train_pos + test_pos 
    neg = train_neg + test_neg 
    corpus, labels = create_corpus(pos, neg)

    nbow_corpus = extract_nbow(corpus)
    # (
    #     train_corpus,
    #     test_corpus,
    #     train_labels,
    #     test_labels,
    # ) = sklearn.model_selection.train_test_split(corpus, labels)

    # TODO: train / evaluate and report accuracy
