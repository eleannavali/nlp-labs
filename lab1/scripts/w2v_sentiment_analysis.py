from cgi import test
import glob
import os
import re
from gensim.models import Word2Vec
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt 
from gensim.models import KeyedVectors
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
NUM_W2V_TO_LOAD = 2000000


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

def load_model(model_path, google=False):
    if google == True:
        model_wv = KeyedVectors.load_word2vec_format("../data/GoogleNews-vectors-negative300.bin",
                                                  binary=True, limit= NUM_W2V_TO_LOAD)
    else:
        model = Word2Vec.load(model_path)
        model_wv = model.wv
    return model_wv

def extract_nbow(corpus, model_path='../outputs/gutenberg_w2v.100d.10epochs.model'):
    """Extract neural bag of words representations"""
    # load w2v model 
    model_wv = load_model(model_path, google=False)
    nbow = [] 
    for review in corpus :
        # average embedding
        wv_dim = model_wv.vector_size
        avg_embed = np.zeros(wv_dim)
        for word in review :
            if word in model_wv :
                embed = model_wv[word]
            else :
                 embed = np.zeros(wv_dim) 
            avg_embed += embed
        # normalize in [-1, +1] range
        nbow.append(avg_embed)
    nbow = np.array(nbow) 
    # NORMALIZATION DOES NOT GIVE BETTER RESULTS SO WE DONT USE IT .
    # Normalize in [-1, +1] range  
    #nbow =  2*((nbow - nbow.min(axis=0) )/ (nbow.max(axis=0) - nbow.min(axis=0))) - 1 
    # Z-normalization 
    #nbow = (nbow - nbow.mean(axis=0))/nbow.var(axis=0)
    return nbow 

def train_sentiment_analysis(train_corpus, train_labels):
    """Train a sentiment analysis classifier using NBOW + Logistic regression"""
     # Initiate logistic regr model 
    log_reg = LogisticRegression(solver='liblinear', penalty='l1')
    log_reg.fit(train_corpus, train_labels)
    return log_reg

def evaluate_sentiment_analysis(classifier, test_corpus, test_labels):
    """Evaluate classifier in the test corpus and report accuracy"""
    preds = classifier.predict(test_corpus)
    report = classification_report(test_labels, preds)
    print(report)
    confusion_matrix(test_labels, preds, labels=[0, 1])
    plot_confusion_matrix(classifier, test_corpus, test_labels)  
    plt.show()


if __name__ == "__main__":
    # read review folders
    train_pos = read_samples(pos_train_dir, preproc_tok) 
    train_neg = read_samples(neg_train_dir, preproc_tok) 
    test_pos = read_samples(pos_test_dir, preproc_tok) 
    test_neg = read_samples(neg_test_dir, preproc_tok) 
    for t in [train_pos, train_neg, test_pos, test_neg] :
        print('type :',type(t))
        print('length', len(t))

    pos = train_pos + test_pos 
    neg = train_neg + test_neg 
    corpus, labels = create_corpus(pos, neg)

    nbow_corpus = extract_nbow(corpus)
    (
        train_corpus,
        test_corpus,
        train_labels,
        test_labels,
    ) = sklearn.model_selection.train_test_split(nbow_corpus, labels)
    log_model = train_sentiment_analysis(train_corpus, train_labels)
    evaluate_sentiment_analysis(log_model, test_corpus, test_labels)

