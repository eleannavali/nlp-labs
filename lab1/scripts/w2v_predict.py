from statistics import mode
from numpy import dot 
from numpy.linalg import norm 
from gensim.models import Word2Vec
import sys 


def load_model(model_path) :
    return Word2Vec.load(model_path)

def cosine_distance (model, words ) :
    cosine_dict = {}
    word_list = []
    # embedding of word[0]
    
    target_list = []
    for i in range(len(words) -1) :
        target_list.append((words[i],words[i+1 :] )) 
    for (item,targets) in target_list : 
        a = model[item]
        for target in targets :
            b = model[target]  
            cos_sim = dot(a, b)/(norm(a)*norm(b))
            cosine_dict[item + ' ' + target] = cos_sim
    return cosine_dict

if __name__ == "__main__":

    words = []
    for  i in range(2, len(sys.argv)) : 
        words.append(sys.argv[i])
    
    model_path = "../outputs/" + str(sys.argv[1])  #/gutenberg_w2v.100d.model"
    model = load_model(model_path)
    model_wv = model.wv 
    distances = cosine_distance(model_wv, words)
    with open("../outputs/w2v_predictions.txt", 'a') as f :
        f.write(sys.argv[1] + '\n')
        f.write(str(distances) + '\n')
        f.write("**********"*10 + '\n')
    print(distances)
