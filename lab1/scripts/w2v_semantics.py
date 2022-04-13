from statistics import mode
from numpy import dot 
from numpy.linalg import norm 
from gensim.models import Word2Vec
import sys 


def load_model(model_path) :
    return Word2Vec.load(model_path)


if __name__ == "__main__":

    words = []
    if len(sys.argv)>5:
        print('Too many parameters!')
        exit(0)
    for  i in range(2, len(sys.argv)) : 
        words.append(sys.argv[i])
    
    model_path = "../outputs/" + str(sys.argv[1])  #/gutenberg_w2v.100d.model"
    model = load_model(model_path)
    model_wv = model.wv 
    v = model_wv.most_similar(positive=[words[0], words[2]], negative=[words[1]])
    with open("../outputs/w2v_semantic_predictions.txt", 'a') as f :
        f.write(sys.argv[1] + '\n')
        f.write(str(v[0]) + '\n')
        f.write("**********"*10 + '\n')
    print(v)
