import requests
import pandas as pd
import codecs
from bs4 import BeautifulSoup

from f1pm.betstrategiesevaluations.model_strategy import ModelStrategy

data_base_path = "../../data/bet365_odds/2022_mexico/"


def process_bet365_boderless_odds_table(url):
    with open(url, 'r', encoding='utf-8') as f:
        html_str = f.read()

    soup = BeautifulSoup(html_str, 'html.parser')

    page_bets_names = soup.find_all("span", {"class": "gl-ParticipantBorderless_Name"})
    page_bets_odds = soup.find_all("span", {"class": "gl-ParticipantBorderless_Odds"})

    odds_table_dict = {name.text: float(odds.text) for name, odds in zip(page_bets_names, page_bets_odds)}

    return odds_table_dict


def process_bet365_columns_odds_table(url):
    with open(url, 'r', encoding='utf-8') as f:
        html_str = f.read()

    soup = BeautifulSoup(html_str, 'html.parser')

    page_bets_names = soup.find_all("div", {"class": "srb-ParticipantLabel_Name"})
    page_bets_odds = soup.find_all("span", {"class": "gl-ParticipantOddsOnly_Odds"})

    odds_table_dict = {name.text: float(odds.text) for name, odds in zip(page_bets_names, page_bets_odds)}

    return odds_table_dict


car_grid_win_dict = process_bet365_boderless_odds_table(data_base_path + "car_grid_win.html")
car_race_win_dict = process_bet365_boderless_odds_table(data_base_path + "car_race_win.html")
driver_grid_win_dict = process_bet365_boderless_odds_table(data_base_path + "grid_win.html")
driver_race_win_dict = process_bet365_boderless_odds_table(data_base_path + "race_win.html")

driver_race_top_6_dict = process_bet365_columns_odds_table(data_base_path + "race_top_6.html")
driver_race_top_3_dict = process_bet365_columns_odds_table(data_base_path + "race_top_3.html")


# TODO:
#      Create definitiosn to automatically read files ?
#       (this in order to identify which bets seems nice?
#       Also if a bet is prof. chck if same equivalent in car is a good idea too.
#       (Indentify if prob jointly (or split?) is profitable using current probabilities from odds?)