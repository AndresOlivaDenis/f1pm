import unittest
import pandas as pd

from f1pm.webrequests.bets365_savedshtml_reads import process_bet365_boderless_odds_table, \
    process_bet365_columns_odds_table
from f1pm.webrequests.f1com_standing_requests import request_current_drivers_standing, \
    request_current_constructors_standing, request_quali_results

TEST_DRIVER_STANDING_URL = "https://www.formula1.com/en/results.html/2021/drivers.html"
TEST_CONSTRUCTOR_STANDING_URL = 'https://www.formula1.com/en/results.html/2021/team.html'
TEST_QUALY_RESULTS_URL = "https://www.formula1.com/en/results.html/2022/races/1134/japan/qualifying.html"

BET365_FILES_PATH = "data/bet365_odds/2022_mexico/"


class TestWebRequests(unittest.TestCase):
    def test_request_current_drivers_standing(self):
        expected_df = {
            'POS': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                    '19', '20', '21'],
            'DRIVER': ['Max Verstappen VER', 'Lewis Hamilton HAM', 'Valtteri Bottas BOT', 'Sergio Perez PER',
                       'Carlos Sainz SAI', 'Lando Norris NOR', 'Charles Leclerc LEC', 'Daniel Ricciardo RIC',
                       'Pierre Gasly GAS', 'Fernando Alonso ALO', 'Esteban Ocon OCO', 'Sebastian Vettel VET',
                       'Lance Stroll STR', 'Yuki Tsunoda TSU', 'George Russell RUS', 'Kimi RÃ¤ikkÃ¶nen RAI',
                       'Nicholas Latifi LAT', 'Antonio Giovinazzi GIO', 'Mick Schumacher MSC', 'Robert Kubica KUB',
                       'Nikita Mazepin MAZ'],
            'NATIONALITY': ['NED', 'GBR', 'FIN', 'MEX', 'ESP', 'GBR', 'MON', 'AUS', 'FRA', 'ESP', 'FRA', 'GER', 'CAN',
                            'JPN', 'GBR', 'FIN', 'CAN', 'ITA', 'GER', 'POL', 'RAF'],
            'CAR': ['Red Bull Racing Honda', 'Mercedes', 'Mercedes', 'Red Bull Racing Honda', 'Ferrari',
                    'McLaren Mercedes', 'Ferrari', 'McLaren Mercedes', 'AlphaTauri Honda', 'Alpine Renault',
                    'Alpine Renault', 'Aston Martin Mercedes', 'Aston Martin Mercedes', 'AlphaTauri Honda',
                    'Williams Mercedes', 'Alfa Romeo Racing Ferrari', 'Williams Mercedes', 'Alfa Romeo Racing Ferrari',
                    'Haas Ferrari', 'Alfa Romeo Racing Ferrari', 'Haas Ferrari'],
            'PTS': ['395.5', '387.5', '226', '190', '164.5', '160', '159', '115', '110', '81', '74', '43', '34', '32',
                    '16', '10', '7', '3', '0', '0', '0']
        }
        expected_df = pd.DataFrame(expected_df)
        current_driver_standing_table = request_current_drivers_standing(TEST_DRIVER_STANDING_URL)
        pd.testing.assert_frame_equal(current_driver_standing_table, expected_df)

    def test_request_current_constructors_standing(self):
        expected_df = {'POS': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                       'TEAM': ['Mercedes', 'Red Bull Racing Honda', 'Ferrari', 'McLaren Mercedes',
                                'Alpine Renault', 'AlphaTauri Honda', 'Aston Martin Mercedes', 'Williams Mercedes',
                                'Alfa Romeo Racing Ferrari', 'Haas Ferrari'],
                       'PTS': ['613.5', '585.5', '323.5', '275', '155', '142', '77', '23', '13', '0']}
        expected_df = pd.DataFrame(expected_df)
        current_constructors_standing = request_current_constructors_standing(TEST_CONSTRUCTOR_STANDING_URL)
        pd.testing.assert_frame_equal(current_constructors_standing, expected_df)

    def test_request_quali_results(self):
        expected_df = {
            'POS': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                    '18', '19', '20'],
            'NO': ['1', '16', '55', '11', '31', '44', '14', '63', '5', '4', '3', '77', '22', '24', '47', '23',
                   '10', '20', '18', '6'],
            'DRIVER': ['Max Verstappen VER', 'Charles Leclerc LEC', 'Carlos Sainz SAI', 'Sergio Perez PER',
                       'Esteban Ocon OCO', 'Lewis Hamilton HAM', 'Fernando Alonso ALO', 'George Russell RUS',
                       'Sebastian Vettel VET', 'Lando Norris NOR', 'Daniel Ricciardo RIC', 'Valtteri Bottas BOT',
                       'Yuki Tsunoda TSU', 'Zhou Guanyu ZHO', 'Mick Schumacher MSC', 'Alexander Albon ALB',
                       'Pierre Gasly GAS', 'Kevin Magnussen MAG', 'Lance Stroll STR', 'Nicholas Latifi LAT'],
            'CAR': ['Red Bull Racing RBPT', 'Ferrari', 'Ferrari', 'Red Bull Racing RBPT', 'Alpine Renault', 'Mercedes',
                    'Alpine Renault', 'Mercedes', 'Aston Martin Aramco Mercedes', 'McLaren Mercedes',
                    'McLaren Mercedes', 'Alfa Romeo Ferrari', 'AlphaTauri RBPT', 'Alfa Romeo Ferrari', 'Haas Ferrari',
                    'Williams Mercedes', 'AlphaTauri RBPT', 'Haas Ferrari', 'Aston Martin Aramco Mercedes',
                    'Williams Mercedes'],
            'Q1': ['1:30.224', '1:30.402', '1:30.336', '1:30.622', '1:30.696', '1:30.906', '1:30.603', '1:30.865',
                   '1:31.256', '1:30.881', '1:30.880', '1:31.226', '1:31.130', '1:30.894', '1:31.152', '1:31.311',
                   '1:31.322', '1:31.352', '1:31.419', '1:31.511'],
            'Q2': ['1:30.346', '1:30.486', '1:30.444',
                   '1:29.925', '1:30.357', '1:30.443', '1:30.343', '1:30.465', '1:30.656', '1:30.473', '1:30.659',
                   '1:30.709', '1:30.808', '1:30.953', '1:31.439', '', '', '', '', ''],
            'Q3': ['1:29.304', '1:29.314', '1:29.361', '1:29.709', '1:30.165', '1:30.261', '1:30.322', '1:30.389',
                   '1:30.554', '1:31.003', '', '', '', '', '', '', '', '', '', ''],
            'LAPS': ['13', '13', '13', '15', '18', '20', '15', '19', '15', '18', '11', '12', '15', '12', '12', '6', '9',
                     '6', '6', '8']}
        expected_df = pd.DataFrame(expected_df)
        grid_results = request_quali_results(quali_results_url=TEST_QUALY_RESULTS_URL)
        pd.testing.assert_frame_equal(grid_results, expected_df)

    def test_process_bet365_boderless_odds_table_car_grid_win(self):
        expected_dict = {'Ferrari': 1.83, 'Red Bull': 2.0, 'Mercedes': 15.0, 'Alpine': 51.0, 'Aston Martin': 51.0,
                         'McLaren': 67.0, 'AlphaTauri': 601.0, 'Alfa Romeo': 651.0, 'Williams': 1001.0, 'Haas': 1001.0}

        car_grid_win_dict = process_bet365_boderless_odds_table(BET365_FILES_PATH + "car_grid_win.html")
        self.assertDictEqual(car_grid_win_dict, expected_dict)

    def test_process_bet365_boderless_odds_table_car_race_win(self):
        expected_dict = {'Red Bull': 1.28, 'Ferrari': 4.33, 'Mercedes': 9.0, 'Alpine': 176.0, 'McLaren': 201.0,
                         'Aston Martin': 501.0, 'AlphaTauri': 701.0, 'Alfa Romeo': 801.0, 'Williams': 1001.0,
                         'Haas': 1001.0}

        car_race_win_dict = process_bet365_boderless_odds_table(BET365_FILES_PATH + "car_race_win.html")
        self.assertDictEqual(car_race_win_dict, expected_dict)

    def test_process_bet365_boderless_odds_table_driver_race_win(self):
        expected_dict = {'Max Verstappen': 1.57, 'Charles Leclerc': 5.5, 'Sergio Perez': 5.5, 'Lewis Hamilton': 13.0,
                         'Carlos Sainz': 13.0, 'George Russell': 26.0, 'Fernando Alonso': 251.0, 'Lando Norris': 251.0,
                         'Esteban Ocon': 501.0, 'Pierre Gasly': 1001.0, 'Daniel Ricciardo': 1001.0,
                         'Lance Stroll': 1001.0, 'Sebastian Vettel': 1001.0, 'Valtteri Bottas': 1251.0,
                         'Alex Albon': 2001.0, 'Kevin Magnussen': 2001.0, 'Mick Schumacher': 2501.0,
                         'Yuki Tsunoda': 2501.0, 'Guanyu Zhou': 2501.0, 'Nicholas Latifi': 3001.0}

        driver_race_win_dict = process_bet365_boderless_odds_table(BET365_FILES_PATH + "race_win.html")
        self.assertDictEqual(driver_race_win_dict, expected_dict)

    def test_process_bet365_boderless_odds_table_driver_grid_win(self):
        expected_dict = {'Charles Leclerc': 2.1, 'Max Verstappen': 2.25, 'Carlos Sainz': 6.5, 'Sergio Perez': 10.0,
                         'Lewis Hamilton': 21.0, 'George Russell': 34.0, 'Fernando Alonso': 101.0,
                         'Lance Stroll': 101.0, 'Sebastian Vettel': 101.0, 'Lando Norris': 126.0, 'Esteban Ocon': 151.0,
                         'Daniel Ricciardo': 151.0, 'Valtteri Bottas': 1001.0, 'Pierre Gasly': 1001.0,
                         'Alex Albon': 1501.0, 'Yuki Tsunoda': 1501.0, 'Kevin Magnussen': 2001.0,
                         'Mick Schumacher': 2001.0, 'Guanyu Zhou': 2001.0, 'Nicholas Latifi': 3001.0}

        driver_grid_win_dict = process_bet365_boderless_odds_table(BET365_FILES_PATH + "grid_win.html")
        self.assertDictEqual(driver_grid_win_dict, expected_dict)

    def test_process_bet365_boderless_odds_table_driver_race_top_3(self):
        expected_dict = {'Max Verstappen ': 1.25, 'Charles Leclerc ': 1.5, 'Sergio Perez ': 1.72, 'Carlos Sainz ': 2.37,
                         'Lewis Hamilton ': 2.2, 'George Russell ': 3.25, 'Fernando Alonso ': 13.0,
                         'Lando Norris ': 13.0, 'Esteban Ocon ': 34.0, 'Daniel Ricciardo ': 67.0,
                         'Sebastian Vettel ': 67.0, 'Pierre Gasly ': 67.0, 'Lance Stroll ': 67.0,
                         'Valtteri Bottas ': 151.0, 'Alex Albon ': 201.0, 'Kevin Magnussen ': 201.0,
                         'Guanyu Zhou ': 251.0, 'Yuki Tsunoda ': 251.0, 'Mick Schumacher ': 251.0,
                         'Nicholas Latifi ': 501.0}

        driver_race_top_3_dict = process_bet365_columns_odds_table(BET365_FILES_PATH + "race_top_3.html")
        self.assertDictEqual(driver_race_top_3_dict, expected_dict)

    def test_process_bet365_boderless_odds_table_driver_race_top_6(self):
        expected_dict = {'Max Verstappen ': 1.16, 'Charles Leclerc ': 1.2, 'Carlos Sainz ': 1.2, 'Sergio Perez ': 1.2,
                         'Lewis Hamilton ': 1.2, 'George Russell ': 1.25, 'Fernando Alonso ': 1.72,
                         'Lando Norris ': 2.1, 'Esteban Ocon ': 3.25, 'Daniel Ricciardo ': 7.0,
                         'Sebastian Vettel ': 11.0, 'Pierre Gasly ': 11.0, 'Lance Stroll ': 11.0,
                         'Valtteri Bottas ': 21.0, 'Alex Albon ': 21.0, 'Guanyu Zhou ': 26.0, 'Kevin Magnussen ': 26.0,
                         'Yuki Tsunoda ': 34.0, 'Mick Schumacher ': 34.0, 'Nicholas Latifi ': 101.0}

        driver_race_top_6_dict = process_bet365_columns_odds_table(BET365_FILES_PATH + "race_top_6.html")
        self.assertDictEqual(driver_race_top_6_dict, expected_dict)
