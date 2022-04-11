
from tqdm import tqdm

from helpers import run_cmd
from util import dict_to_txt


def read_test_set(fname):
    pairs = []
    with open(fname, "r") as fd:
        lines = [ln.strip().split("\t") for ln in fd.readlines()]

        for ln in lines:
            correct = ln[1]
            wrong = ln[0]

            pairs.append((wrong, correct))

    return pairs

def count_edits(fname):
    transitions = {}
    with open(fname, "r") as fd:
        lines = [ln.strip().split("\t") for ln in fd.readlines()]

        for ln in lines:
            if len(ln)>1:
                char1 = ln[0]
                char2 = ln[1]

                if char1 + ' '+ char2 in transitions.keys():
                    transitions[char1 + ' '+ char2] += 1
                else:
                    transitions[char1 + ' '+ char2] = 1             
    transitions_sorted = dict(sorted(transitions.items() ,reverse=True, key=lambda kv: kv[1]))
    N = sum(transitions_sorted.values())
    dict_to_txt('../outputs/edit_frequency.txt', transitions_sorted) 
    for key in transitions_sorted.keys():
        transitions_sorted[key]= round(transitions_sorted[key]/N,6)
    dict_to_txt('../outputs/edit_relative_frequency.txt', transitions_sorted) 
    return
    



if __name__ == "__main__":
    # pairs = read_test_set("../data/wiki.txt")
    # open('../outputs/wiki_edits.txt', 'w').close()
    # with open('../outputs/wiki_edits.txt', 'a') as f:
    #     for pair in tqdm(pairs):
    #         edits = run_cmd(f"bash word_edits.sh {pair[0]} {pair[1]}")
    #         f.write(edits + '\n')
    count_edits('../outputs/wiki_edits.txt')
