from gensim.models import Word2Vec
import nltk

def write_words(model, txt_path="../data/gutenberg.txt") : 

    with open(txt_path, 'r') as fd:
        vocab = []
        lines = [ln for ln in fd.readlines() if len(ln) > 0]
        for line in lines : 
            words = line.strip().split(' ')
            for word in words :
                vocab.append(word)       
    freq = nltk.FreqDist(vocab)
    freq_dict = dict(freq)  
    vocab = [] 
    for k, v in freq_dict.items() :
        if v >= 10 : 
            vocab.append(k)


    with open('../vocab/metadata.tsv', 'a') as fd : 
        for word in vocab :
            fd.write(word + '\n')
    with open('../vocab/embeddings.tsv', 'a') as fd : 
        for word in vocab :
            embedding = model[word]
            for value in embedding : 
                fd.write(str(value) + '\t')
            fd.write('\n')
    return 

if __name__ == "__main__" :
    model_path = '../outputs/gutenberg_w2v.100d.25win.10epochs.model'
    model = Word2Vec.load(model_path)
    model_wv = model.wv
    write_words(model = model_wv)