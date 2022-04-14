from statistics import mode
from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import sys


def load_model(model_path, google=False):
    if google == True:
        model_wv = KeyedVectors.load_word2vec_format("../data/GoogleNews-vectors-negative300.bin",
                                                      binary=True, limit=2000000)
    else:
        model = Word2Vec.load(model_path)
        model_wv = model.wv
    return model_wv


if __name__ == "__main__":

    words = []
    if len(sys.argv) > 5:
        print('Too many parameters!')
        exit(0)
    for i in range(2, len(sys.argv)):
        words.append(sys.argv[i])

    model_path = "../outputs/" + str(sys.argv[1])  
    if sys.argv[1].startswith("Google"):
        google = True
    else : google = False

    model = load_model(model_path, google)
    v = model.most_similar(
        positive=[words[0], words[2]], negative=[words[1]])
    with open("../outputs/w2v_semantic_predictions.txt", 'a') as f:
        f.write(sys.argv[1] + '\n')
        f.write(str(v[0]) + '\n')
        f.write("**********"*10 + '\n')
    print(v)
