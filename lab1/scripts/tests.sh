#!/usr/bin/env bash

bash predict.sh ../fsts/S_weighted.binfst tst Weighted_Speller
bash predict.sh ../fsts/S_weighted.binfst cwt Weighted_Speller
bash predict.sh ../fsts/S_weighted.binfst cit Weighted_Speller
bash predict.sh ../fsts/S_weighted.binfst posrpone Weighted_Speller


bash predict.sh ../fsts/S.binfst tst Simple_Speller
bash predict.sh ../fsts/S.binfst cwt Simple_Speller
bash predict.sh ../fsts/S.binfst cit Simple_Speller
bash predict.sh ../fsts/S.binfst posrpone Simple_Speller