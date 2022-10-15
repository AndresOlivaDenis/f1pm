import requests
import pandas as pd
from bs4 import BeautifulSoup

DRIVER_STANDING_URL = "https://www.formula1.com/en/results.html/2022/drivers.html"
CONSTRUCTOR_STANDING_URL = 'https://www.formula1.com/en/results.html/2022/team.html'


def request_current_drivers_standing(drivers_standing_url=DRIVER_STANDING_URL):
    r = requests.get(drivers_standing_url)
    soup = BeautifulSoup(r.text, 'html.parser')

    tables = soup.find_all("table")
    driver_standing_table = tables[0]

    driver_standing_table_rows = driver_standing_table.find_all('tr')

    driver_standing_table_as_lst = [[row_col_val.text.replace("\n", " ") for row_col_val in row.find_all('td')[1:-1]]
                                    for row in driver_standing_table_rows[1:]]

    df_driver_standing_table = pd.DataFrame(driver_standing_table_as_lst,
                                            columns=['POS', 'DRIVER', 'NATIONALITY', 'CAR', 'PTS'])

    df_driver_standing_table['DRIVER'] = df_driver_standing_table['DRIVER'].str.strip()
    df_driver_standing_table['CAR'] = df_driver_standing_table['CAR'].str.strip()
    df_driver_standing_table['NATIONALITY'] = df_driver_standing_table['NATIONALITY'].str.strip()

    return df_driver_standing_table


def request_current_constructors_standing(constructors_standing_url=CONSTRUCTOR_STANDING_URL):
    r = requests.get(constructors_standing_url)
    soup = BeautifulSoup(r.text, 'html.parser')

    tables = soup.find_all("table")
    constructor_standing_table = tables[0]

    constructor_standing_table_rows = constructor_standing_table.find_all('tr')

    constructor_standing_table_as_lst = [[row_col_val.text.replace("\n", " ")
                                          for row_col_val in row.find_all('td')[1:-1]]
                                         for row in constructor_standing_table_rows[1:]]

    df_constructor_standing_table = pd.DataFrame(constructor_standing_table_as_lst,
                                                 columns=['POS', 'TEAM', 'PTS'])

    df_constructor_standing_table['TEAM'] = df_constructor_standing_table['TEAM'].str.strip()

    return df_constructor_standing_table


def request_quali_results(quali_results_url):
    r = requests.get(quali_results_url)
    soup = BeautifulSoup(r.text, 'html.parser')

    tables = soup.find_all("table")
    qualy_standing_table = tables[0]

    qualy_standing_table_rows = qualy_standing_table.find_all('tr')

    qualy_standing_table_as_lst = [[row_col_val.text.replace("\n", " ") for row_col_val in row.find_all('td')[1:-1]]
                                   for row in qualy_standing_table_rows[1:]]

    df_qualy_standing_table = pd.DataFrame(qualy_standing_table_as_lst,
                                           columns=['POS', 'NO', 'DRIVER', 'CAR', 'Q1', 'Q2', 'Q3', 'LAPS'])

    df_qualy_standing_table['DRIVER'] = df_qualy_standing_table['DRIVER'].str.strip()
    df_qualy_standing_table['CAR'] = df_qualy_standing_table['CAR'].str.strip()

    return df_qualy_standing_table


if __name__ == '__main__':
    current_driver_standing_table = request_current_drivers_standing(
        "https://www.formula1.com/en/results.html/2022/drivers.html")
    current_constructor_standing_table = request_current_constructors_standing(
        'https://www.formula1.com/en/results.html/2022/team.html')

    quali_results_url_ = "https://www.formula1.com/en/results.html/2022/races/1134/japan/qualifying.html"
    grid_results = request_quali_results(quali_results_url=quali_results_url_)
