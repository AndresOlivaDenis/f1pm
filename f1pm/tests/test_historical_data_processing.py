import unittest
import pandas as pd
import numpy as np

from f1pm.historicaldataprocessing.historical_data_processing_m1 import process_historical_historical_data_m1, \
    MODEL_AA_DATA_COLUMNS
from f1pm.historicaldataprocessing.tools import compute_historical_sub_data_set, get_previuous_driver_races_results, \
    compute_latest_descritive_variables, add_latest_positions_and_grid_descriptive_variables

TESTS_PATH = "data/"

TESTS_HISTORICAL_RACES_RESULTS_PATH = TESTS_PATH + "historical_races_results/"
TESTS_DRIVER_STANDINGS_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "driver_standings.csv"
TESTS_RESULTS_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "results.csv"
TESTS_RACES_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "races.csv"
TESTS_QUALIFYING_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "qualifying.csv"
TESTS_DRIVERS_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "drivers.csv"
TESTS_CONSTRUCTOR_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "constructor_standings.csv"


class TestHistoricalDataProcessing(unittest.TestCase):
    df_data, df_data_all = process_historical_historical_data_m1(
        driver_standings_file_path=TESTS_DRIVER_STANDINGS_FILE_PATH,
        results_file_path=TESTS_RESULTS_FILE_PATH,
        races_file_path=TESTS_RACES_FILE_PATH,
        qualifying_file_path=TESTS_QUALIFYING_FILE_PATH,
        drivers_file_path=TESTS_DRIVERS_FILE_PATH,
        constructor_file_path=TESTS_CONSTRUCTOR_FILE_PATH
    )

    def test_process_historical_historical_data_m1(self):
        exoected_results_file = TESTS_PATH + "expected_results/process_historical_historical_data_m1.csv"
        expected_df = pd.read_csv(exoected_results_file)

        pd.testing.assert_frame_equal(expected_df, self.df_data.copy())
        # df_data.to_csv(exoected_results_file, index=False)

    def test_tools_compute_historical_sub_data_set(self):
        # sub_data_set_one
        exoected_results_file = TESTS_PATH + "expected_results/sub_data_set_one.csv"
        expected_data_set_one_df = pd.read_csv(exoected_results_file)

        df_sub_data_set_one = compute_historical_sub_data_set(self.df_data.copy(),
                                                              grid=[2, 3],
                                                              driver_championship_standing=[5, 6],
                                                              year_lower_threshold=2008,
                                                              round_lower_threshold=5)
        df_sub_data_set_one = df_sub_data_set_one.drop(columns=['constructor_standing_position'])
        df_sub_data_set_one = df_sub_data_set_one.reset_index(drop=True)

        pd.testing.assert_frame_equal(df_sub_data_set_one, expected_data_set_one_df)

        # sub_data_set_two
        exoected_results_file = TESTS_PATH + "expected_results/sub_data_set_two.csv"
        expected_data_set_two_df = pd.read_csv(exoected_results_file)

        df_sub_data_set_two = compute_historical_sub_data_set(self.df_data.copy(),
                                                              grid=3,
                                                              driver_championship_standing=5,
                                                              year_lower_threshold=2008,
                                                              round_lower_threshold=5)
        df_sub_data_set_two = df_sub_data_set_two.drop(columns=['constructor_standing_position'])
        df_sub_data_set_two = df_sub_data_set_two.reset_index(drop=True)

        pd.testing.assert_frame_equal(df_sub_data_set_two, expected_data_set_two_df)

        # sub_data_set_three
        exoected_results_file = TESTS_PATH + "expected_results/sub_data_set_three.csv"
        expected_data_set_three_df = pd.read_csv(exoected_results_file)

        df_sub_data_set_three = compute_historical_sub_data_set(self.df_data.copy(),
                                                                grid=None,
                                                                driver_championship_standing=[2, 3],
                                                                year_lower_threshold=None,
                                                                round_lower_threshold=5)
        df_sub_data_set_three = df_sub_data_set_three.drop(columns=['constructor_standing_position'])
        df_sub_data_set_three = df_sub_data_set_three.reset_index(drop=True)

        pd.testing.assert_frame_equal(df_sub_data_set_three, expected_data_set_three_df)

    def test_tools_compute_historical_sub_data_set_constructor_tandings(self):
        # sub_data_set_one
        exoected_results_file = TESTS_PATH + "expected_results/sub_data_set_one_constructor.csv"
        expected_data_set_one_df = pd.read_csv(exoected_results_file)

        df_sub_data_set_one = compute_historical_sub_data_set(self.df_data.copy(),
                                                              grid=None,
                                                              driver_championship_standing=None,
                                                              constructor_championship_standing=[1, 2],
                                                              year_lower_threshold=2008,
                                                              round_lower_threshold=5)

        # df_sub_data_set_one.to_csv(exoected_results_file, index=False)
        df_sub_data_set_one = df_sub_data_set_one.reset_index(drop=True)

        pd.testing.assert_frame_equal(df_sub_data_set_one, expected_data_set_one_df)

        # sub_data_set_two
        exoected_results_file = TESTS_PATH + "expected_results/sub_data_set_two_constructor.csv"
        expected_data_set_two_df = pd.read_csv(exoected_results_file)

        df_sub_data_set_two = compute_historical_sub_data_set(self.df_data.copy(),
                                                              grid=1,
                                                              driver_championship_standing=None,
                                                              constructor_championship_standing=[3, 4],
                                                              year_lower_threshold=2008,
                                                              round_lower_threshold=5)

        # df_sub_data_set_two.to_csv(exoected_results_file, index=False)
        df_sub_data_set_two = df_sub_data_set_two.reset_index(drop=True)

        pd.testing.assert_frame_equal(df_sub_data_set_two, expected_data_set_two_df)

        # sub_data_set_three
        exoected_results_file = TESTS_PATH + "expected_results/sub_data_set_three_constructor.csv"
        expected_data_set_three_df = pd.read_csv(exoected_results_file)

        df_sub_data_set_three = compute_historical_sub_data_set(self.df_data.copy(),
                                                                grid=[1, 2, 3],
                                                                driver_championship_standing=2,
                                                                constructor_championship_standing=[3, 4],
                                                                year_lower_threshold=2008,
                                                                round_lower_threshold=5)

        # df_sub_data_set_three.to_csv(exoected_results_file, index=False)
        df_sub_data_set_three = df_sub_data_set_three.reset_index(drop=True)

        pd.testing.assert_frame_equal(df_sub_data_set_three, expected_data_set_three_df)

    def test_tools_get_previuous_driver_races_results(self):
        expected_df = {'raceId': [949, 950, 951, 952],
                       'race_name': ['Bahrain Grand Prix - 2016', 'Chinese Grand Prix - 2016',
                                     'Russian Grand Prix - 2016', 'Spanish Grand Prix - 2016'],
                       'driverRef': ['max_verstappen', 'max_verstappen', 'max_verstappen', 'max_verstappen'],
                       'qualifying_position': [10.0, 9.0, 9.0, 4.0], 'grid': [10, 9, 9, 4],
                       'position': ['6', '8', '\\N', '1'], 'positionOrder': [6, 8, 19, 1],
                       'driver_standing_position': [8.0, 9.0, 10.0, 6.0],
                       'constructor_standing_position': [6.0, 6.0, 6.0, 3.0], 'round': [2, 3, 4, 5],
                       'year': [2016.0, 2016.0, 2016.0, 2016.0],
                       'url': ['http://en.wikipedia.org/wiki/2016_Bahrain_Grand_Prix',
                               'http://en.wikipedia.org/wiki/2016_Chinese_Grand_Prix',
                               'http://en.wikipedia.org/wiki/2016_Russian_Grand_Prix',
                               'http://en.wikipedia.org/wiki/2016_Spanish_Grand_Prix'],
                       'driverId': [830, 830, 830, 830]}
        expected_df = pd.DataFrame(expected_df)

        data = self.df_data_all.copy()
        data = data[MODEL_AA_DATA_COLUMNS]
        previuous_races_results = get_previuous_driver_races_results(data, race_id=953, driver_id=830, n_races=4)

        previuous_races_results = previuous_races_results.reset_index(drop=True)
        previuous_races_results = previuous_races_results.astype({'round': np.int64})
        pd.testing.assert_frame_equal(previuous_races_results, expected_df)

        latest_position_results = previuous_races_results['position']
        latest_grid_results = previuous_races_results['qualifying_position']

        latest_position_descritive_variables = compute_latest_descritive_variables(latest_position_results)
        self.assertEqual(latest_position_descritive_variables, (5.0, 1.0))

        latest_grid_descritive_variables = compute_latest_descritive_variables(latest_grid_results)
        self.assertEqual(latest_grid_descritive_variables, (8.0, 4.0))

        previuous_races_results = get_previuous_driver_races_results(data, race_id=1083, driver_id=847, n_races=6)

        latest_position_results = previuous_races_results['position']
        latest_grid_results = previuous_races_results['qualifying_position']

        latest_position_descritive_variables = compute_latest_descritive_variables(latest_position_results)
        self.assertEqual(latest_position_descritive_variables, (4.0, 3.0))

        latest_grid_descritive_variables = compute_latest_descritive_variables(latest_grid_results)
        self.assertEqual(latest_grid_descritive_variables, (7.666666666666667, 4.0))

    def test_tools_get_previuous_driver_races_results(self):
        exoected_results_file = TESTS_PATH + "expected_results/get_previuous_driver_races_results_data_m1.csv"
        expected_data_set_aa = pd.read_csv(exoected_results_file)

        df_sub_data_set_2017 = compute_historical_sub_data_set(self.df_data_all.copy(),
                                                               grid=None,
                                                               driver_championship_standing=None,
                                                               year_lower_threshold=2017,
                                                               year_upper_threshold=2017,
                                                               round_lower_threshold=None)

        df_data_aa = df_sub_data_set_2017[MODEL_AA_DATA_COLUMNS]
        df_data_aa = add_latest_positions_and_grid_descriptive_variables(df_data_aa, n_race=5)

        # df_data_aa.to_csv(exoected_results_file, index=False)
        df_data_aa = df_data_aa.reset_index(drop=True)
        df_data_aa = df_data_aa.astype({'round': np.int64})

        pd.testing.assert_frame_equal(df_data_aa, expected_data_set_aa)
