import string
from math import log

EPS = "<eps>"  # Define once. Use the same EPS everywhere

CHARS = list("abcdefghijklmnopqrstuvwxyz")

INFINITY = 1000000000

# Write dictionary as tab delimited columns
# make sure file is empty so we don't append values again.
def dict_to_txt(path, dict): 
  open(path, 'w').close()
  for k, v in dict.items():
    with open(path, 'a') as f:
        f.write(k + "\t" + str(v) + "\n")
  return

def create_words(path="../vocab/words.vocab.txt"):
    discrete_tokens = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        discrete_tokens.append(line.split('\t')[0])
    token_dict = {}
    token_dict['<eps>'] = 0
    for i, token in enumerate(discrete_tokens) : 
        token_dict[token] = i + 1 
    # sort dictionary by values
    token_dict = {k: v for k, v in sorted(token_dict.items(), key=lambda item: item[1])}
    dict_to_txt('../vocab/words.syms', token_dict)
    return 


def create_symbols():
    letters = {}
    letters[EPS] = 0 
    for i,letter in enumerate(CHARS) : 
        letters[letter] = i+1
    dict_to_txt('../vocab/chars.syms', letters)

def create_transducer(wins, wdel, wsub, name):
    ''' Create a txt description of transducer:
            ex. 0 0 a a 0
                 0 0 a b 1
    '''
    with open('../fsts/'+name+'.fst', 'w') as f:
        CHARS.append('<eps>')
        for let in CHARS : 
            for let_i in CHARS : 
                if let == let_i : 
                    f.write("0 0 " + let + " " + let_i + " " + str(0) + "\n")
                else:
                    if let=='<eps>':
                        f.write("0 0 " + let + " " + let_i + " " + str(wins) + "\n")
                    elif let_i=='<eps>':
                        f.write("0 0 " + let + " " + let_i + " " + str(wdel) + "\n")
                    else:
                        f.write("0 0 " + let + " " + let_i + " " + str(wsub) + "\n")
        f.write(str(0))

def create_weighted_transducer():
    ''' Create a txt description of transducer:
            ex. 0 0 a a 0
                 0 0 a b 1
    '''
    with open('../outputs/edit_relative_frequency.txt', 'r') as fd:
        lines = [ln.strip().split("\t") for ln in fd.readlines()]
        edits_dict ={}
        for ln in lines:
            edits_dict[ln[0]] = float(ln[1])

        with open('../fsts/E.fst', 'w') as f:
            CHARS.append('<eps>')
            for let in CHARS : 
                for let_i in CHARS : 
                    if let == let_i : 
                        f.write("0 0 " + let + " " + let_i + " " + str(0) + "\n")
                    else:
                        if let + ' ' + let_i in edits_dict.keys():
                            f.write("0 0 " + let + " " + let_i + " " + str(-log(edits_dict[let + ' ' + let_i])) + "\n")
                        else:
                            f.write("0 0 " + let + " " + let_i + " " + str(-log(0.00001)) + "\n")
            f.write(str(0))

def create_sub_transducer(wins, wdel, wsub, name):
    with open('../fsts/'+name+'.fst', 'w') as f:
        CHARS.append('<eps>')
        for let in CHARS : 
            for let_i in CHARS : 
                if let == let_i : 
                    f.write("0 0 " + let + " " + let_i + " " + str(0) + "\n")
                else:
                    if let=='<eps>':
                        f.write("0 0 " + let + " " + let_i + " " + str(wins) + "\n")
                    elif let_i=='<eps>':
                        f.write("0 0 " + let + " " + let_i + " " + str(wdel) + "\n")
                    else:
                        f.write("0 0 " + let + " " + let_i + " " + str(wsub) + "\n") 
            break
        f.write(str(0))


def calculate_arc_weight(frequency):
    """Function to calculate the weight of an arc based on a frequency count

    Args:
        frequency (float): Frequency count

    Returns:
        (float) negative log of frequency

    """
    # TODO: INSERT YOUR CODE HERE
    raise NotImplementedError(
        "You need to implement calculate_arc_weight function in scripts/util.py!!!"
    )


def format_arc(src, dst, ilabel, olabel, weight=0):
    """Create an Arc, i.e. a line of an openfst text format file

    Args:
        src (int): source state
        dst (int): destination state
        ilabel (str): input label
        olabel (str): output label
        weight (float): arc weight

    Returns:
        (str) The formatted line as a string
    http://www.openfst.org/twiki/bin/view/FST/FstQuickTour#CreatingShellFsts
    """
    # TODO: INSERT YOUR CODE HERE
    return str(src) + " " + str(dst) + " " + ilabel + " " + olabel + " " + str(weight)
    raise NotImplementedError(
        "You need to implement format_arc function in scripts/util.py!!!"
    )

