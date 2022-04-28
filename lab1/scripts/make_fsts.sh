## Transducers:
# create L transducers
fstcompile --isymbols=../vocab/chars.syms --osymbols=../vocab/chars.syms ../fsts/L.fst ../fsts/L.binfst
fstcompile --isymbols=../vocab/chars.syms --osymbols=../vocab/chars.syms ../fsts/L_weighted.fst ../fsts/L_weighted.binfst
# sub transducer
fstcompile --isymbols=../vocab/chars.syms --osymbols=../vocab/chars.syms ../fsts/sub_L.fst ../fsts/sub_L.binfst
fstcompile --isymbols=../vocab/chars.syms --osymbols=../vocab/chars.syms ../fsts/sub_L_weighted.fst ../fsts/sub_L_weighted.binfst
# create E transducer
fstcompile --isymbols=../vocab/chars.syms --osymbols=../vocab/chars.syms ../fsts/E.fst ../fsts/E.binfst

# draw graph
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/chars.syms ../fsts/sub_L.binfst | dot -Tpng > ../img/sub_L.png
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/chars.syms ../fsts/sub_L_weighted.binfst | dot -Tpng > ../img/sub_L_weighted.png


## Acceptors:
# create V acceptor 
cat ../vocab/words.syms | python3 makelex.py > ../fsts/V.fst
fstcompile --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms ../fsts/V.fst ../fsts/V.binfst
# sub acceptor
head -20 ../vocab/words.syms | python3 makelex.py > ../fsts/sub_V.fst
fstcompile --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms ../fsts/sub_V.fst ../fsts/sub_V.binfst
# create W acceptor
fstcompile --isymbols=../vocab/words.syms --osymbols=../vocab/words.syms ../fsts/W.fst ../fsts/W.binfst
# create sub W acceptor
fstcompile --isymbols=../vocab/words.syms --osymbols=../vocab/words.syms ../fsts/sub_W.fst ../fsts/sub_W.binfst

# draw graph
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms ../fsts/sub_V.binfst | dot -Tpng > ../img/sub_V.png
fstdraw --isymbols=../vocab/words.syms --osymbols=../vocab/words.syms ../fsts/sub_W.binfst | dot -Tpng > ../img/sub_W.png

## General optimization 
fstrmepsilon ../fsts/V.binfst| fstdeterminize | fstminimize > ../fsts/V_opt.binfst
fstrmepsilon ../fsts/sub_V.binfst| fstdeterminize | fstminimize >../fsts/sub_V_opt.binfst
# rm epsilon opt
fstrmepsilon ../fsts/sub_V.binfst >../fsts/V_opt_rmepsilon.binfst
# determinize opt
fstdeterminize ../fsts/sub_V.binfst >../fsts/V_opt_determinize.binfst
# minimize opt
fstminimize ../fsts/sub_V.binfst >../fsts/V_opt_minimize.binfst

fstrmepsilon ../fsts/W.binfst| fstdeterminize | fstminimize > ../fsts/W_opt.binfst
fstrmepsilon ../fsts/sub_W.binfst| fstdeterminize | fstminimize >../fsts/sub_W_opt.binfst

# draw optimized graph
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms ../fsts/sub_V_opt.binfst | dot -Tpng > ../img/sub_V_opt.png
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms ../fsts/V_opt_rmepsilon.binfst | dot -Tpng > ../img/V_opt_rmepsilon.png
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms ../fsts/V_opt_determinize.binfst | dot -Tpng > ../img/V_opt_determinize.png
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms ../fsts/V_opt_minimize.binfst | dot -Tpng > ../img/V_opt_minimize.png

# sort L and V fst inputs
# fstarcsort --sort_type=olabel ../fsts/L.binfst ../fsts/L_sorted.binfst
# fstarcsort --sort_type=ilabel ../fsts/V_opt.binfst ../fsts/V_opt_sorted.binfst
fstarcsort ../fsts/V_opt.binfst ../fsts/V_opt_sorted.binfst
fstarcsort ../fsts/W_opt.binfst ../fsts/W_opt_sorted.binfst

fstarcsort ../fsts/sub_V_opt.binfst ../fsts/sub_V_opt_sorted.binfst
fstarcsort ../fsts/sub_W_opt.binfst ../fsts/sub_W_opt_sorted.binfst

## Spell Checkers:
#create S spell checker
fstcompose ../fsts/L.binfst ../fsts/V_opt_sorted.binfst ../fsts/S.binfst
fstcompose ../fsts/L_weighted.binfst ../fsts/V_opt_sorted.binfst ../fsts/S_weighted.binfst
# create EV spell checker 
fstcompose ../fsts/E.binfst ../fsts/V_opt_sorted.binfst ../fsts/EV.binfst

#create LVW spell checker -> SW
fstcompose ../fsts/S.binfst ../fsts/W_opt_sorted.binfst ../fsts/LVW.binfst
#create EVW spell checker 
fstcompose ../fsts/EV.binfst ../fsts/W_opt_sorted.binfst ../fsts/EVW.binfst

## VW composition
fstcompose ../fsts/sub_V_opt_sorted.binfst ../fsts/sub_W_opt_sorted.binfst ../fsts/sub_VW.binfst

# draw optimized VW graph
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms ../fsts/sub_VW.binfst | dot -Tpng > ../img/sub_VW.png
