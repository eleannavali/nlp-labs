#!/usr/bin/env bash

bash predict.sh ../fsts/S_weighted.binfst tst Weighted_Speller
bash predict.sh ../fsts/S_weighted.binfst cwt Weighted_Speller
bash predict.sh ../fsts/S_weighted.binfst cit Weighted_Speller
bash predict.sh ../fsts/S_weighted.binfst posrone Weighted_Speller
bash predict.sh ../fsts/S_weighted.binfst posrpone Weighted_Speller


bash predict.sh ../fsts/S.binfst tst Simple_Speller
bash predict.sh ../fsts/S.binfst cwt Simple_Speller
bash predict.sh ../fsts/S.binfst cit Simple_Speller
bash predict.sh ../fsts/S.binfst posrone Simple_Speller
bash predict.sh ../fsts/S.binfst posrpone Simple_Speller

bash predict.sh ../fsts/EV.binfst tst EV_Speller
bash predict.sh ../fsts/EV.binfst cwt EV_Speller
bash predict.sh ../fsts/EV.binfst cit EV_Speller
bash predict.sh ../fsts/EV.binfst posrone EV_Speller
bash predict.sh ../fsts/EV.binfst posrpone EV_Speller

bash predict.sh ../fsts/LVW.binfst tst LVW_Speller
bash predict.sh ../fsts/LVW.binfst cwt LVW_Speller
bash predict.sh ../fsts/LVW.binfst cit LVW_Speller
bash predict.sh ../fsts/LVW.binfst posrone LVW_Speller
bash predict.sh ../fsts/LVW.binfst posrpone LVW_Speller

bash predict.sh ../fsts/EVW.binfst tst EVW_Speller
bash predict.sh ../fsts/EVW.binfst cwt EVW_Speller
bash predict.sh ../fsts/EVW.binfst cit EVW_Speller
bash predict.sh ../fsts/EVW.binfst posrone EVW_Speller
bash predict.sh ../fsts/EVW.binfst posrpone EVW_Speller