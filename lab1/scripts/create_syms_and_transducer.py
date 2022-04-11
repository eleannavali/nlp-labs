"""
USAGE:
    python3 create_syms_and_transducer.py 

"""

from util import create_symbols, create_words, create_transducer, create_sub_transducer



if __name__ == "__main__":
    create_symbols()
    create_words()
    create_transducer()
    create_sub_transducer()
    print("Chars and words syms are ready. Transducer L also ready!")