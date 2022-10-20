import requests
import pandas as pd
import codecs
from bs4 import BeautifulSoup

from f1pm.betstrategiesevaluations.model_strategy import ModelStrategy

data_base_path = "../../data/bet365_odds/gp_eeuu_2022/"

coche_ganador_url = data_base_path + "gp_eeuu_coche_ganador.html"
# coche_ganador_url = data_base_path + "gp_eeuu_ganador_final.text"
coche_ganador_url = data_base_path + "Piloto mas rapido.html"

with open(coche_ganador_url, 'r', encoding='utf-8') as f:
    coche_ganador_html_str = f.read()

soup = BeautifulSoup(coche_ganador_html_str, 'html.parser')

page_bets_names = soup.find_all("span", {"class": "gl-ParticipantBorderless_Name"})
page_bets_odds = soup.find_all("span", {"class": "gl-ParticipantBorderless_Odds"})

odds_table_dict = {name.text: float(odds.text) for name, odds in zip(page_bets_names, page_bets_odds)}

import numpy as np
odds_values = np.array(list(odds_table_dict.values()))
total_probs = np.sum([1/odd for odd in odds_values])

odss_single_probs = 1/odds_table_dict['Lewis Hamilton'] + 1/odds_table_dict['George Russell']
odss_single_probs = odss_single_probs/total_probs
odds_car_probs = 1/11.0
print(ModelStrategy.eval_kelly_criterion(strategy_probability=odss_single_probs, bet_odds=11))


print(ModelStrategy.eval_kelly_criterion(strategy_probability=(1/15 + 1/17)/1.2557863851330238, bet_odds=11))

print(ModelStrategy.eval_kelly_criterion(strategy_probability=(1/2.2 + 1/15)/1.2557863851330238, bet_odds=2.1))

print(ModelStrategy.eval_kelly_criterion(strategy_probability=(1/8 + 1/13)/1.2557863851330238, bet_odds=5.5))

print(ModelStrategy.eval_kelly_criterion(strategy_probability=(1/8 + 1/13)/1.2557863851330238, bet_odds=1.33))
# TODO:
#      Create definitiosn to automatically read files ?
#       (this in order to identify which bets seems nice?
#       Also if a bet is prof. chck if same equivalent in car is a good idea too.
#       (Indentify if prob jointly (or split?) is profitable using current probabilities from odds?)