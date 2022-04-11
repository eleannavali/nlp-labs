"""
USAGE:
    python3 create_syms_and_transducer.py 

"""

from util import create_symbols, create_words, create_transducer, create_sub_transducer



if __name__ == "__main__":
    create_symbols()
    create_words()
    create_transducer(1,1,1,'L')
    create_sub_transducer(1,1,1,'sub_L')
    create_transducer(1,1,1.5,'L_weighted')
    create_sub_transducer(1,1,1.5,'sub_L_weighted')
    print("Chars and words syms are ready. Transducers L also ready!")