import unittest
import pandas as pd

from f1pm.webrequests.f1com_standing_requests import request_current_drivers_standing, \
    request_current_constructors_standing, request_quali_results

TEST_DRIVER_STANDING_URL = "https://www.formula1.com/en/results.html/2021/drivers.html"
TEST_CONSTRUCTOR_STANDING_URL = 'https://www.formula1.com/en/results.html/2021/team.html'
TEST_QUALY_RESULTS_URL = "https://www.formula1.com/en/results.html/2022/races/1134/japan/qualifying.html"


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

