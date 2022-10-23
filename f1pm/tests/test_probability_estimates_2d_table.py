import unittest
import pandas as pd

from f1pm.historicaldataprocessing.historical_data_processing_m1 import process_historical_historical_data_m1
from f1pm.historicaldataprocessing.tools import compute_historical_sub_data_set
from f1pm.probabilityestimates.pe_2d_table import ProbabilityEstimate2DTable
from f1pm.probabilityestimates.pe_historical_data import ProbabilityEstimateHistoricalData
import numpy as np

TESTS_PATH = "data/"

TESTS_HISTORICAL_RACES_RESULTS_PATH = TESTS_PATH + "historical_races_results/"
TESTS_DRIVER_STANDINGS_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "driver_standings.csv"
TESTS_RESULTS_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "results.csv"
TESTS_RACES_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "races.csv"
TESTS_QUALIFYING_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "qualifying.csv"
TESTS_DRIVERS_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "drivers.csv"
TESTS_CONSTRUCTOR_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "constructor_standings.csv"


class TestProbabilityEstimates2DTable(unittest.TestCase):
    df_data, df_data_all = process_historical_historical_data_m1(
        driver_standings_file_path=TESTS_DRIVER_STANDINGS_FILE_PATH,
        results_file_path=TESTS_RESULTS_FILE_PATH,
        races_file_path=TESTS_RACES_FILE_PATH,
        qualifying_file_path=TESTS_QUALIFYING_FILE_PATH,
        drivers_file_path=TESTS_DRIVERS_FILE_PATH,
        constructor_file_path=TESTS_CONSTRUCTOR_FILE_PATH
    )

    def test_pe_2d_table_compute_race_positions_2d_table(self):
        subdatset_params_dict_ = dict(year_lower_threshold=2000,
                                      year_upper_threshold=None,
                                      round_lower_threshold=5,
                                      round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(self.df_data.copy(), subdatset_params_dict_)

        pe2dt = ProbabilityEstimate2DTable(pehd)
        race_positions_2d_table_output = pe2dt.compute_race_positions_2d_table(ci=0.05)

        exoected_results_file = TESTS_PATH + "expected_results/2d_tables/compute_race_positions_2d_table.csv"
        # race_positions_2d_table_output.Probability.to_csv(exoected_results_file, index=False)
        e_race_2d_table = pd.read_csv(exoected_results_file)
        np.testing.assert_almost_equal(e_race_2d_table.values, race_positions_2d_table_output.Probability.values)

        exoected_results_file = TESTS_PATH + "expected_results/2d_tables/compute_race_positions_2d_table_ci_lower.csv"
        # race_positions_2d_table_output.CI_lower.to_csv(exoected_results_file, index=False)
        e_race_2d_table = pd.read_csv(exoected_results_file)
        np.testing.assert_almost_equal(e_race_2d_table.values, race_positions_2d_table_output.CI_lower.values)

    def test_pe_2d_table_compute_grid_positions_2d_table(self):
        subdatset_params_dict_ = dict(year_lower_threshold=2000,
                                      year_upper_threshold=None,
                                      round_lower_threshold=5,
                                      round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(self.df_data.copy(), subdatset_params_dict_)

        pe2dt = ProbabilityEstimate2DTable(pehd)
        grid_positions_2d_table_output = pe2dt.compute_grid_positions_2d_table(ci=0.05)

        exoected_results_file = TESTS_PATH + "expected_results/2d_tables/compute_grid_positions_2d_table.csv"
        # grid_positions_2d_table_output.Probability.to_csv(exoected_results_file, index=False)
        e_race_2d_table = pd.read_csv(exoected_results_file)
        np.testing.assert_almost_equal(e_race_2d_table.values, grid_positions_2d_table_output.Probability.values)

        exoected_results_file = TESTS_PATH + "expected_results/2d_tables/compute_grid_positions_2d_table_ci_upper.csv"
        # grid_positions_2d_table_output.CI_upper.to_csv(exoected_results_file, index=False)
        e_race_2d_table = pd.read_csv(exoected_results_file)
        np.testing.assert_almost_equal(e_race_2d_table.values, grid_positions_2d_table_output.CI_upper.values)

    def test_pe_2d_table_adjust_probabilities_from_events(self):
        subdatset_params_dict_ = dict(year_lower_threshold=2000,
                                      year_upper_threshold=None,
                                      round_lower_threshold=5,
                                      round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(self.df_data.copy(), subdatset_params_dict_)

        pe2dt = ProbabilityEstimate2DTable(pehd)
        race_positions_2d_table_output = pe2dt.compute_race_positions_2d_table(ci=0.05)

        targets_drivers = [1, 2, 5, 6]
        sc_pe_2d_table_df = race_positions_2d_table_output.Probability.loc[:, targets_drivers]
        adjusted_probs_2d_true = pe2dt.adjust_probabilities_from_events_true(sc_pe_2d_table_df=sc_pe_2d_table_df,
                                                                             evidence_events_positions_lst=[1, 2, 3])

        exoected_results_file = TESTS_PATH + "expected_results/2d_tables/adjusted_probs_2d_true.csv"
        # adjusted_probs_2d_true.to_csv(exoected_results_file, index=False)
        e_race_2d_table = pd.read_csv(exoected_results_file)
        np.testing.assert_almost_equal(e_race_2d_table.values, adjusted_probs_2d_true)

        adjusted_probs_2d_false = pe2dt.adjust_probabilities_from_events_false(sc_pe_2d_table_df=sc_pe_2d_table_df,
                                                                               evidence_events_positions_lst=[2, 3])

        exoected_results_file = TESTS_PATH + "expected_results/2d_tables/adjusted_probs_2d_false.csv"
        # adjusted_probs_2d_false.to_csv(exoected_results_file, index=False)
        e_race_2d_table = pd.read_csv(exoected_results_file)
        np.testing.assert_almost_equal(e_race_2d_table.values, adjusted_probs_2d_false.values)
