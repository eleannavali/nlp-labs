# create transducer
fstcompile --isymbols=../vocab/chars.syms --osymbols=../vocab/chars.syms ../fsts/L.fst ../fsts/L.binfst
# sub transducer
fstcompile --isymbols=../vocab/chars.syms --osymbols=../vocab/chars.syms ../fsts/sub_L.fst ../fsts/sub_L.binfst

# draw graph
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/chars.syms ../fsts/sub_L.binfst | dot -Tpng > ../img/sub_L.png

# create acceptor 
cat ../vocab/words.syms | python3 makelex.py > ../fsts/V.fst
fstcompile --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms ../fsts/V.fst ../fsts/V.binfst

# sub acceptor
tail -10 ../vocab/words.syms | python3 makelex.py > ../fsts/sub_V.fst
fstcompile --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms ../fsts/sub_V.fst ../fsts/sub_V.binfst

# draw graph
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms ../fsts/sub_V.binfst | dot -Tpng > ../img/sub_V.png

# optimization 
fstrmepsilon ../fsts/V.binfst| fstdeterminize | fstminimize > ../fsts/V_opt.binfst
fstrmepsilon ../fsts/sub_V.binfst| fstdeterminize | fstminimize >../fsts/sub_V_opt.binfst

# draw optimized graph
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms ../fsts/sub_V_opt.binfst | dot -Tpng > ../img/sub_V_opt.png

# sort L and V fst inputs
fstarcsort --sort_type=olabel ../fsts/L.binfst ../fsts/L.binfst
fstarcsort --sort_type=ilabel ../fsts/V_opt.binfst ../fsts/V_opt.binfst

#create S spell checker
fstcompose ../fsts/L.binfst ../fsts/V_opt.binfst ../fsts/S.binfst
