import unittest
import pandas as pd
import numpy as np
from f1pm.betstrategiesevaluations.model_strategy_zero import ModelOneStrategyZero
from f1pm.historicaldataprocessing.historical_data_processing_m1 import process_historical_historical_data_m1
from f1pm.historicaldataprocessing.tools import compute_historical_sub_data_set
from f1pm.probabilityestimates.pe_historical_data import ProbabilityEstimateHistoricalData
from f1pm.webrequests.f1com_standing_requests import request_current_drivers_standing, request_quali_results, \
    request_current_constructors_standing

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

TESTS_PATH = "data/"

TESTS_HISTORICAL_RACES_RESULTS_PATH = TESTS_PATH + "historical_races_results/"
TESTS_DRIVER_STANDINGS_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "driver_standings.csv"
TESTS_RESULTS_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "results.csv"
TESTS_RACES_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "races.csv"
TESTS_QUALIFYING_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "qualifying.csv"
TESTS_DRIVERS_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "drivers.csv"
TESTS_CONSTRUCTOR_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "constructor_standings.csv"

TEST_DRIVER_STANDING_URL = "https://www.formula1.com/en/results.html/2021/drivers.html"
TEST_CONSTRUCTOR_STANDING_URL = 'https://www.formula1.com/en/results.html/2021/team.html'
TEST_QUALY_RESULTS_URL = "https://www.formula1.com/en/results.html/2021/races/1107/abu-dhabi/qualifying.html"


class TestModelOneStrategyZero(unittest.TestCase):
    df_data, df_data_all = process_historical_historical_data_m1(
        driver_standings_file_path=TESTS_DRIVER_STANDINGS_FILE_PATH,
        results_file_path=TESTS_RESULTS_FILE_PATH,
        races_file_path=TESTS_RACES_FILE_PATH,
        qualifying_file_path=TESTS_QUALIFYING_FILE_PATH,
        drivers_file_path=TESTS_DRIVERS_FILE_PATH,
        constructor_file_path=TESTS_CONSTRUCTOR_FILE_PATH
    )

    grid_results = request_quali_results(TEST_QUALY_RESULTS_URL)
    current_driver_standing_table = request_current_drivers_standing(TEST_DRIVER_STANDING_URL)
    current_constructors_standing_table = request_current_constructors_standing(TEST_CONSTRUCTOR_STANDING_URL)

    def test_model_one_strategy_zero_compute_grid_estimate_pole_position(self):
        # Parameters from class atributes.
        test_df_data = self.df_data.copy()
        current_driver_standing_table = self.current_driver_standing_table.copy()

        # Loading Current driver championship standings
        current_driver_standing_table = current_driver_standing_table.set_index('DRIVER')

        df_driver_standing = current_driver_standing_table[['POS', 'CAR']]
        df_driver_standing = df_driver_standing.astype({'POS': int})
        df_driver_standing = df_driver_standing.rename(columns={'POS': 'driver_championship_standing'})

        # Probability estimates object
        subdatset_params_dict = dict(year_lower_threshold=2000,
                                     year_upper_threshold=None,
                                     round_lower_threshold=5,
                                     round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(test_df_data, subdatset_params_dict)

        # Strategy zero evaluation
        mosz = ModelOneStrategyZero(pehd)
        sz_grid_prob_estimate = mosz.compute_grid_estimate(target_cumsum_position=1,
                                                           current_driver_standing_table=df_driver_standing,
                                                           ci=0.1,
                                                           subset_n_threshold=10)

        pd.testing.assert_series_equal(sz_grid_prob_estimate.fair_bet_value.loc['Max Verstappen VER'],
                                       pd.Series({'driver_championship_standing': 1,
                                                  'CAR': 'Red Bull Racing Honda',
                                                  'win_probability': 2.474074074074074,
                                                  'CI_lower': 2.7775902614317247,
                                                  'CI_upper': 2.230355934930123},
                                                 name='Max Verstappen VER'))

        pd.testing.assert_series_equal(sz_grid_prob_estimate.race_target_position_prob.loc['Valtteri Bottas BOT'],
                                       pd.Series({'driver_championship_standing': 3,
                                                  'CAR': 'Mercedes',
                                                  'win_probability': 0.16417910447761194,
                                                  'CI_lower': 0.13088859768210173,
                                                  'CI_upper': 0.19746961127312215},
                                                 name='Valtteri Bottas BOT'))

    def test_model_one_strategy_zero_compute_grid_estimate_top_3_position(self):
        # Parameters from class atributes.
        test_df_data = self.df_data.copy()
        current_driver_standing_table = self.current_driver_standing_table.copy()

        # Loading Current driver championship standings
        current_driver_standing_table = current_driver_standing_table.set_index('DRIVER')

        df_driver_standing = current_driver_standing_table[['POS', 'CAR']]
        df_driver_standing = df_driver_standing.astype({'POS': int})
        df_driver_standing = df_driver_standing.rename(columns={'POS': 'driver_championship_standing'})

        # Probability estimates object
        subdatset_params_dict = dict(year_lower_threshold=2008,
                                     year_upper_threshold=None,
                                     round_lower_threshold=5,
                                     round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(test_df_data, subdatset_params_dict)

        # Strategy zero evaluation
        mosz = ModelOneStrategyZero(pehd)
        sz_grid_prob_estimate = mosz.compute_grid_estimate(target_cumsum_position=3,
                                                           current_driver_standing_table=df_driver_standing,
                                                           ci=0.05,
                                                           subset_n_threshold=10)

        pd.testing.assert_series_equal(sz_grid_prob_estimate.race_target_position_prob.loc['Max Verstappen VER'],
                                       pd.Series({'driver_championship_standing': 1,
                                                  'CAR': 'Red Bull Racing Honda',
                                                  'Top 3 probabilities': 0.7929515418502203,
                                                  'CI_lower': 0.7402413633721723,
                                                  'CI_upper': 0.8456617203282684},
                                                 name='Max Verstappen VER'))

        pd.testing.assert_series_equal(sz_grid_prob_estimate.fair_bet_value.loc['Lewis Hamilton HAM'],
                                       pd.Series({'driver_championship_standing': 2,
                                                  'CAR': 'Mercedes',
                                                  'Top 3 probabilities': 1.6170212765957446,
                                                  'CI_lower': 1.8006120680375968,
                                                  'CI_upper': 1.4674044092828882},
                                                 name='Lewis Hamilton HAM'))

    def test_model_one_strategy_zero_compute_race_estimate_top_3_position(self):
        # Parameters from class atributes.
        test_df_data = self.df_data.copy()
        current_driver_standing_table = self.current_driver_standing_table.copy()

        # Loading Current driver championship standings
        current_driver_standing_table = current_driver_standing_table.set_index('DRIVER')

        df_driver_standing = current_driver_standing_table[['POS', 'CAR']]
        df_driver_standing = df_driver_standing.astype({'POS': int})
        df_driver_standing = df_driver_standing.rename(columns={'POS': 'driver_championship_standing'})

        # Probability estimates object
        subdatset_params_dict = dict(year_lower_threshold=2008,
                                     year_upper_threshold=None,
                                     round_lower_threshold=5,
                                     round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(test_df_data, subdatset_params_dict)

        # Strategy zero evaluation
        mosz = ModelOneStrategyZero(pehd)
        sz_race_prob_estimate = mosz.compute_race_estimate(target_cumsum_position=3,
                                                           current_driver_standing_table=df_driver_standing,
                                                           ci=0.05,
                                                           subset_n_threshold=10)

        pd.testing.assert_series_equal(sz_race_prob_estimate.race_target_position_prob.loc['Max Verstappen VER'],
                                       pd.Series({'driver_championship_standing': 1,
                                                  'CAR': 'Red Bull Racing Honda',
                                                  'Top 3 probabilities': 0.7929515418502202,
                                                  'CI_lower': 0.7402413633721721,
                                                  'CI_upper': 0.8456617203282684},
                                                 name='Max Verstappen VER'))

        pd.testing.assert_series_equal(sz_race_prob_estimate.fair_bet_value.loc['Lando Norris NOR'],
                                       pd.Series({'driver_championship_standing': 6,
                                                  'CAR': 'McLaren Mercedes',
                                                  'Top 3 probabilities': 6.162162162162162,
                                                  'CI_lower': 8.73959869756528,
                                                  'CI_upper': 4.7587396608725285},
                                                 name='Lando Norris NOR'))

        self.assertEqual(sz_race_prob_estimate.fair_bet_value_normalized, None)
        self.assertEqual(sz_race_prob_estimate.race_target_position_prob_normalized, None)

    def test_model_one_strategy_zero_compute_race_estimate_top_1_position(self):
        # Parameters from class atributes.
        test_df_data = self.df_data.copy()
        current_driver_standing_table = self.current_driver_standing_table.copy()

        # Loading Current driver championship standings
        current_driver_standing_table = current_driver_standing_table.set_index('DRIVER')

        df_driver_standing = current_driver_standing_table[['POS', 'CAR']]
        df_driver_standing = df_driver_standing.astype({'POS': int})
        df_driver_standing = df_driver_standing.rename(columns={'POS': 'driver_championship_standing'})

        # Probability estimates object
        subdatset_params_dict = dict(year_lower_threshold=2012,
                                     year_upper_threshold=None,
                                     round_lower_threshold=5,
                                     round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(test_df_data, subdatset_params_dict)

        # Strategy zero evaluation
        mosz = ModelOneStrategyZero(pehd)
        sz_race_prob_estimate = mosz.compute_race_estimate(target_cumsum_position=1,
                                                           current_driver_standing_table=df_driver_standing,
                                                           ci=0.05,
                                                           subset_n_threshold=10)

        pd.testing.assert_series_equal(sz_race_prob_estimate.race_target_position_prob.loc['Max Verstappen VER'],
                                       pd.Series({'driver_championship_standing': 1,
                                                  'CAR': 'Red Bull Racing Honda',
                                                  'win_probability': 0.5235294117647059,
                                                  'CI_lower': 0.4484514935743805,
                                                  'CI_upper': 0.5986073299550313},
                                                 name='Max Verstappen VER'))

        pd.testing.assert_series_equal(sz_race_prob_estimate.fair_bet_value.loc['Carlos Sainz SAI'],
                                       pd.Series({'driver_championship_standing': 5,
                                                  'CAR': 'Ferrari',
                                                  'win_probability': 34.0,
                                                  'CI_lower': 249.15051317056327,
                                                  'CI_upper': 18.24488202094849},
                                                 name='Carlos Sainz SAI'))

    def test_model_one_strategy_zero_compute_conditioning_on_grid_race_estimate_top_1_position(self):
        # Parameters from class atributes.
        test_df_data = self.df_data.copy()
        current_driver_standing_table = self.current_driver_standing_table.copy()
        grid_results = self.grid_results.copy()

        # Loading Current driver championship standings
        current_driver_standing_table = current_driver_standing_table.set_index('DRIVER')
        grid_results = grid_results.set_index('DRIVER')

        df_race_day_grid = ModelOneStrategyZero.compute_race_day_grid_df(current_driver_standing_table, grid_results)

        # Probability estimates object
        subdatset_params_dict = dict(year_lower_threshold=2012,
                                     year_upper_threshold=None,
                                     round_lower_threshold=5,
                                     round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(test_df_data, subdatset_params_dict)

        # Strategy zero evaluation
        mosz = ModelOneStrategyZero(pehd)
        sz_race_prob_estimate = mosz.compute_conditioning_on_grid_race_estimate(target_cumsum_position=1,
                                                                                race_day_grid=df_race_day_grid,
                                                                                ci=0.05)
        pd.testing.assert_series_equal(sz_race_prob_estimate.race_target_position_prob.loc['Max Verstappen VER'],
                                       pd.Series({'grid': 1.0,
                                                  'driver_championship_standing': 1.0,
                                                  'win_probability': 0.7,
                                                  'CI_lower': 0.5926483513769705,
                                                  'CI_upper': 0.8073516486230294},
                                                 name='Max Verstappen VER'))

        pd.testing.assert_series_equal(sz_race_prob_estimate.race_target_position_prob.loc['Lando Norris NOR'],
                                       pd.Series({'grid': 3.0,
                                                  'driver_championship_standing': 6.0,
                                                  'win_probability': 0.07692307692307693,
                                                  'CI_lower': -0.06792865278562824,
                                                  'CI_upper': 0.2217748066317821},
                                                 name='Lando Norris NOR'))

        pd.testing.assert_series_equal(
            sz_race_prob_estimate.race_target_position_prob_normalized.loc['Max Verstappen VER'],
            pd.Series({'grid': 1.0,
                       'driver_championship_standing': 1.0,
                       'win_probability': 0.6084798135859378,
                       'CI_lower': 0.5681844446627009,
                       'CI_upper': 0.6418966432794251},
                      name='Max Verstappen VER'))

    def test_model_one_strategy_zero_compute_conditioning_on_grid_race_estimate_top_3_position(self):
        # Parameters from class atributes.
        test_df_data = self.df_data.copy()
        current_driver_standing_table = self.current_driver_standing_table.copy()
        grid_results = self.grid_results.copy()

        # Loading Current driver championship standings
        current_driver_standing_table = current_driver_standing_table.set_index('DRIVER')
        grid_results = grid_results.set_index('DRIVER')

        df_race_day_grid = ModelOneStrategyZero.compute_race_day_grid_df(current_driver_standing_table, grid_results)

        # Probability estimates object
        subdatset_params_dict = dict(year_lower_threshold=2000,
                                     year_upper_threshold=None,
                                     round_lower_threshold=5,
                                     round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(test_df_data, subdatset_params_dict)

        # Strategy zero evaluation
        mosz = ModelOneStrategyZero(pehd)
        sz_race_prob_estimate = mosz.compute_conditioning_on_grid_race_estimate(target_cumsum_position=3,
                                                                                race_day_grid=df_race_day_grid,
                                                                                ci=0.05)
        pd.testing.assert_series_equal(sz_race_prob_estimate.race_target_position_prob.loc['Max Verstappen VER'],
                                       pd.Series({'grid': 1.0,
                                                  'driver_championship_standing': 1.0,
                                                  'Top 3 probabilities': 0.8962962962962963,
                                                  'CI_lower': 0.8448677626998752,
                                                  'CI_upper': 0.9477248298927173},
                                                 name='Max Verstappen VER'))

        pd.testing.assert_series_equal(sz_race_prob_estimate.fair_bet_value.loc['Lando Norris NOR'],
                                       pd.Series({'grid': 3.0,
                                                  'driver_championship_standing': 6.0,
                                                  'Top 3 probabilities': 2.875,
                                                  'CI_lower': 6.528291898130343,
                                                  'CI_upper': 1.8434105717440372},
                                                 name='Lando Norris NOR'))

    def test_model_one_strategy_zero_compute_race_day_grid_df(self):
        expected_df = {'grid': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                       'driver_championship_standing': [1, 2, 6, 4, 5, 3, 7, 14, 11, 8, 10, 9, 13, 18, 12, 17, 15, 16,
                                                        19, 21],
                       'constructor_championship_standing': [2, 1, 4, 2, 3, 1, 3, 6, 5, 4, 5, 6,
                                                             7, 9, 7, 8, 8, 9, 10, 10]}
        expected_df = pd.DataFrame(expected_df)
        expected_df.index = ['Max Verstappen VER', 'Lewis Hamilton HAM', 'Lando Norris NOR', 'Sergio Perez PER',
                             'Carlos Sainz SAI', 'Valtteri Bottas BOT', 'Charles Leclerc LEC', 'Yuki Tsunoda TSU',
                             'Esteban Ocon OCO', 'Daniel Ricciardo RIC', 'Fernando Alonso ALO', 'Pierre Gasly GAS',
                             'Lance Stroll STR', 'Antonio Giovinazzi GIO', 'Sebastian Vettel VET',
                             'Nicholas Latifi LAT', 'George Russell RUS', 'Kimi RÃ¤ikkÃ¶nen RAI', 'Mick Schumacher MSC',
                             'Nikita Mazepin MAZ']
        expected_df.index.name = 'DRIVER'

        current_driver_standing_table = self.current_driver_standing_table.copy()
        grid_results = self.grid_results.copy()
        current_constructors_standing_table = self.current_constructors_standing_table.copy()

        df_race_day_grid_construc = ModelOneStrategyZero.compute_race_day_grid_df(current_driver_standing_table,
                                                                                  grid_results,
                                                                                  current_constructors_standing_table)

        pd.testing.assert_frame_equal(df_race_day_grid_construc, expected_df, check_dtype=False)

    def test_model_one_strategy_zero_compute_mapping_drivers_to_car_race_estimate(self):
        # Parameters from class atributes.
        test_df_data = self.df_data.copy()
        current_driver_standing_table = self.current_driver_standing_table.copy()

        # Loading Current driver championship standings
        current_driver_standing_table = current_driver_standing_table.set_index('DRIVER')

        df_driver_standing = current_driver_standing_table[['POS', 'CAR']]
        df_driver_standing = df_driver_standing.astype({'POS': int})
        df_driver_standing = df_driver_standing.rename(columns={'POS': 'driver_championship_standing'})

        # Probability estimates object
        subdatset_params_dict = dict(year_lower_threshold=2012,
                                     year_upper_threshold=None,
                                     round_lower_threshold=5,
                                     round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(test_df_data, subdatset_params_dict)

        # Strategy zero evaluation
        mosz = ModelOneStrategyZero(pehd)
        sz_race_prob_estimate = mosz.compute_race_estimate(target_cumsum_position=1,
                                                           current_driver_standing_table=df_driver_standing,
                                                           ci=0.05,
                                                           subset_n_threshold=10)

        sz_car_race_prob_estimate = mosz.compute_mapping_drivers_to_car_race_estimate(sz_probability_estimate=sz_race_prob_estimate,
                                                          current_driver_standing_table=current_driver_standing_table)

        e_race_prob =sz_race_prob_estimate.race_target_position_prob.copy()

        expected_probs = e_race_prob.loc['Max Verstappen VER'][['win_probability', 'CI_lower', 'CI_upper']]
        expected_probs += e_race_prob.loc['Sergio Perez PER'][['win_probability', 'CI_lower', 'CI_upper']]
        actual_probs = sz_car_race_prob_estimate.race_target_position_prob.loc['Red Bull Racing Honda']
        np.testing.assert_almost_equal(expected_probs.values, actual_probs.values)

        expected_probs = e_race_prob.loc['Lewis Hamilton HAM'][['win_probability', 'CI_lower', 'CI_upper']]
        expected_probs += e_race_prob.loc['Valtteri Bottas BOT'][['win_probability', 'CI_lower', 'CI_upper']]
        actual_probs = sz_car_race_prob_estimate.race_target_position_prob.loc['Mercedes']
        np.testing.assert_almost_equal(expected_probs.values, actual_probs.values)