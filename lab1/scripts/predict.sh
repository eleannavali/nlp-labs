#!/usr/bin/env bash

# Run spell corrector for an input word

# Usage:
#   bash scripts/predict.sh MY_SPELL_CHECKER tst NAME_SPELL_CHECKER
# Output:
#   test

# Command line args
SPELL_CHECKER_COMPILED=${1}
WORD=${2}
TEXT_NAME=${3}

###
# Make sure these files exist
CHARSYMS=../vocab/chars.syms
WORDSYMS=../vocab/words.syms
###

# Make input fst for the misspelled word
python mkfstinput.py ${WORD} |
    # Compile and compose with the spell checker
    fstcompile --isymbols=${CHARSYMS} --osymbols=${CHARSYMS} |
    fstcompose - ${SPELL_CHECKER_COMPILED} |
    # Get shortest path and sort arcs
    fstshortestpath |
    fstrmepsilon |
    fsttopsort |
    # print output fst using words.syms
    fstprint -isymbols=${CHARSYMS} -osymbols=${WORDSYMS} |
    # Get destination word (corrected)
    cut -f4 |
    # Ignore epsilon outputs
    grep -v "<eps>" |
    # Ignore accepting state line
    head -n -1 |
    # Remove trailing new line
    tr -d '\n'  > ../outputs/${TEXT_NAME}_${WORD}.txt 
