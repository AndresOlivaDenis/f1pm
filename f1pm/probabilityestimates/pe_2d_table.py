from collections import namedtuple

import numpy as np
import pandas as pd

from f1pm.historicaldataprocessing.historical_data_processing_m1 import process_historical_historical_data_m1
from f1pm.probabilityestimates.pe_historical_data import ProbabilityEstimateHistoricalData

DEFAULT_TABLE_CHAMPIONSHIP_STANDING_SET = list(range(1, 21))
DEFAULT_TARGET_POSITIONS_SET = list(range(1, 21))

Compute2DTableOutput = namedtuple('Compute2DTableOutput', ["Probability", "CI_lower", "CI_upper"])


class ProbabilityEstimate2DTable:
    """
    PE 2D Table:
    Columns: Drivers.
    Rows: target positions.
    Values: probabilities.

    """

    def __init__(self, probability_estimate_obj):
        """
        probability_estimate_obj:
            Expected with methods:
                -   compute_grid_estimate
                -   compute_race_estimate
                -   compute_conditioning_on_grid_race_estimate
        """

        self.pe_obj = probability_estimate_obj

    def compute_race_positions_2d_table(self,
                                        ci=0.05,
                                        driver_championship_standing_set=DEFAULT_TABLE_CHAMPIONSHIP_STANDING_SET,
                                        target_positions_set=DEFAULT_TABLE_CHAMPIONSHIP_STANDING_SET):

        positions_2d_table = pd.DataFrame(index=target_positions_set + [np.NaN])
        ci_lower_2d_table = positions_2d_table.copy()
        ci_upper_2d_table = positions_2d_table.copy()

        for driver_cs in driver_championship_standing_set:
            race_prob_estimate = self.pe_obj.compute_race_estimate(driver_championship_standing=driver_cs,
                                                                   constructor_championship_standing=None,
                                                                   ci=ci)

            # positions_2d_table
            res = race_prob_estimate.ci_position_probabilities.loc[target_positions_set, 'Probability']
            res[np.NaN] = race_prob_estimate.dnf_prob
            positions_2d_table.loc[:, driver_cs] = res.copy()

            # ci_lower_2d_table
            res = race_prob_estimate.ci_position_probabilities.loc[target_positions_set, 'CI_lower']
            res[np.NaN] = race_prob_estimate.dnf_prob
            ci_lower_2d_table.loc[:, driver_cs] = res.copy()

            # ci_upper_2d_table
            res = race_prob_estimate.ci_position_probabilities.loc[target_positions_set, 'CI_upper']
            res[np.NaN] = race_prob_estimate.dnf_prob
            ci_upper_2d_table.loc[:, driver_cs] = res.copy()

        race_positions_2d_table_output = Compute2DTableOutput(Probability=positions_2d_table,
                                                              CI_lower=ci_lower_2d_table,
                                                              CI_upper=ci_upper_2d_table)

        return race_positions_2d_table_output

    def compute_grid_positions_2d_table(self,
                                        ci=0.05,
                                        driver_championship_standing_set=DEFAULT_TABLE_CHAMPIONSHIP_STANDING_SET,
                                        target_positions_set=DEFAULT_TABLE_CHAMPIONSHIP_STANDING_SET):

        positions_2d_table = pd.DataFrame(index=target_positions_set + [np.NaN])
        ci_lower_2d_table = positions_2d_table.copy()
        ci_upper_2d_table = positions_2d_table.copy()

        for driver_cs in driver_championship_standing_set:
            grid_prob_estimate = self.pe_obj.compute_grid_estimate(driver_championship_standing=driver_cs,
                                                                   constructor_championship_standing=None,
                                                                   ci=ci)

            # positions_2d_table
            res = grid_prob_estimate.ci_position_probabilities.loc[target_positions_set, 'Probability']
            res[np.NaN] = grid_prob_estimate.dnf_prob
            positions_2d_table.loc[:, driver_cs] = res.copy()

            # ci_lower_2d_table
            res = grid_prob_estimate.ci_position_probabilities.loc[target_positions_set, 'CI_lower']
            res[np.NaN] = grid_prob_estimate.dnf_prob
            ci_lower_2d_table.loc[:, driver_cs] = res.copy()

            # ci_upper_2d_table
            res = grid_prob_estimate.ci_position_probabilities.loc[target_positions_set, 'CI_upper']
            res[np.NaN] = grid_prob_estimate.dnf_prob
            ci_upper_2d_table.loc[:, driver_cs] = res.copy()

        grid_positions_2d_table_output = Compute2DTableOutput(Probability=positions_2d_table,
                                                              CI_lower=ci_lower_2d_table,
                                                              CI_upper=ci_upper_2d_table)

        return grid_positions_2d_table_output

    def normalize_2d_table_rows_targets(self, pe_2d_table_df, normalization_targets, skip_nan=True):
        normalized_table = pe_2d_table_df.copy()
        current_rows_probs = pe_2d_table_df.sum(axis=1)
        scaling_factor = normalization_targets / current_rows_probs

        if skip_nan:
            scaling_factor = scaling_factor.drop(np.NaN)

        for index, factor in scaling_factor.items():
            normalized_table.loc[index, :] = normalized_table.loc[index, :] * factor

        return normalized_table

    def normalize_2d_table_columns(self, pe_2d_table_df):
        return pe_2d_table_df / pe_2d_table_df.sum()

    def apply_evidence_into_2d_table(self, pe_2d_table_df, evidence_2d_table_df):
        """
        Recibes a pd.Dataframe of probabilities!

        # Method can be used for:
        #   evidence and new probabilities values
        #   Update probs with CI values

        """
        # TODO: update pe_2d_table with evidence_2d_table values, and apply normalization!.
        #   Note: allow posible drivers columns names (Not only numeric, (maybe table can alrady contain probs!)

        updated_2d_table_df = pe_2d_table_df.copy()
        updated_2d_table_df.loc[evidence_2d_table_df.index, evidence_2d_table_df.columns] = evidence_2d_table_df.values

        # Rows normalization (without touching new evidence probs).
        normalization_target = pe_2d_table_df.sum(axis=1)

        updated_2d_table_df = self.normalize_2d_table_rows_targets(
            pe_2d_table_df=updated_2d_table_df,
            normalization_targets=normalization_target,
            skip_nan=True)

        # Columns normalization
        # updated_2d_table_df = self.normalize_2d_table_columns(updated_2d_table_df)
        return updated_2d_table_df

        # TODO: First tests, compare first norm by rows,them by columns, and see... (validate first that probs of evidence are the same!)
        # Normalization:
        #   First Normalize by rows.
        #       To a subset of columns ¿?
        #       Keep original sum targets ¿? (some positions doesnto sum to one!)
        #       How to modify NaN ¿?
        #   Second normalize by columns ¿?
        #       Noramlize by columns but not all values! Only values that has not been modified.
        pass

    def _apply_evidence_into_2d_table(self, pe_2d_table_df, evidence_2d_table_df):
        """
        Recibes a pd.Dataframe of probabilities!

        # Method can be used for:
        #   evidence and new probabilities values
        #   Update probs with CI values

        """
        # TODO: update pe_2d_table with evidence_2d_table values, and apply normalization!.
        #   Note: allow posible drivers columns names (Not only numeric, (maybe table can alrady contain probs!)

        updated_2d_table_df = pe_2d_table_df.copy()
        updated_2d_table_df.loc[evidence_2d_table_df.index, evidence_2d_table_df.columns] = evidence_2d_table_df.values

        # Rows normalization (without touching new evidence probs).
        non_evidence_columns = pe_2d_table_df.columns.difference(evidence_2d_table_df.columns)
        normalization_target = pe_2d_table_df.sum(axis=1) - evidence_2d_table_df.sum(axis=1)

        updated_2d_table_df.loc[:, non_evidence_columns] = self.normalize_2d_table_rows_targets(
            pe_2d_table_df=pe_2d_table_df.loc[:, non_evidence_columns],
            normalization_targets=normalization_target,
            skip_nan=True)

        # Columns normalization
        updated_2d_table_df = self.normalize_2d_table_columns(updated_2d_table_df)
        return updated_2d_table_df

        # TODO: First tests, compare first norm by rows,them by columns, and see... (validate first that probs of evidence are the same!)
        # Normalization:
        #   First Normalize by rows.
        #       To a subset of columns ¿?
        #       Keep original sum targets ¿? (some positions doesnto sum to one!)
        #       How to modify NaN ¿?
        #   Second normalize by columns ¿?
        #       Noramlize by columns but not all values! Only values that has not been modified.
        pass

    def adjust_probabilities_from_events_true(self, sc_pe_2d_table_df, evidence_events_positions_lst):
        """
        Adjust probabilities, by columns, of a list of position results being true.
        """
        adjusted_2d_table_df = sc_pe_2d_table_df.copy()

        index_isin_true = adjusted_2d_table_df.index.isin(evidence_events_positions_lst)
        index_isin_false = ~index_isin_true

        adjusted_2d_table_df.loc[index_isin_false, :] = 0.0
        adjusted_2d_table_df = self.normalize_2d_table_columns(adjusted_2d_table_df)

        return adjusted_2d_table_df

    def adjust_probabilities_from_events_false(self, sc_pe_2d_table_df, evidence_events_positions_lst):
        """
        Adjust probabilities, by columns, of a list of position results being false.
        """
        adjusted_2d_table_df = sc_pe_2d_table_df.copy()

        index_isin_true = adjusted_2d_table_df.index.isin(evidence_events_positions_lst)
        # index_isin_false = ~index_isin_true

        adjusted_2d_table_df.loc[index_isin_true, :] = 0.0
        adjusted_2d_table_df = self.normalize_2d_table_columns(adjusted_2d_table_df)

        return adjusted_2d_table_df


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Drivers championship standings race day grid ----------------------------------------------------------
    # Historical data processing ----------------------------------------------
    df_data_, df_data_all = process_historical_historical_data_m1()
    # -------------------------------------------------------------------------

    # Probability estimates ---------------------------------------------
    subdatset_params_dict_ = dict(year_lower_threshold=2000,
                                  year_upper_threshold=None,
                                  round_lower_threshold=5,
                                  round_upper_threshold=None)

    pehd = ProbabilityEstimateHistoricalData(df_data_, subdatset_params_dict_)

    # Creationg of 2D Probability tables ------------------------------------
    pe2dt = ProbabilityEstimate2DTable(pehd)

    race_positions_2d_table_output_ = pe2dt.compute_race_positions_2d_table(ci=0.05)
    grid_positions_2d_table_output_ = pe2dt.compute_grid_positions_2d_table(ci=0.05)

    sc_pe_2d_table_df_ = race_positions_2d_table_output_.Probability.loc[:, [1, 2, 3]]
    evidence_events_positions_lst_ = [1, 2, 3, 4, 5, 6]

    adjusted_probs_2d_true = pe2dt.adjust_probabilities_from_events_true(sc_pe_2d_table_df=sc_pe_2d_table_df_,
                                                                         evidence_events_positions_lst=[1, 2, 3])

    adjusted_probs_2d_false = pe2dt.adjust_probabilities_from_events_false(sc_pe_2d_table_df=sc_pe_2d_table_df_,
                                                                           evidence_events_positions_lst=[1, 2, 3])

    # TO DELTE!
    evidence_updated_table = pe2dt.apply_evidence_into_2d_table(
        pe_2d_table_df=race_positions_2d_table_output_.Probability,
        evidence_2d_table_df=adjusted_probs_2d_true)

    # TODO: testings cases for: apply_evidence_into_2d_table
    #   Bet save factors ideas:
    #       - update of table with a low_prob value
    #   Bet ideas releated:
    #       - top 6 position evidence false
    #       - race win evidence true
    #       - race win evidence false
    #       - top 6 position evidence true
    #       - grid conditional probs
    #
    #   top 3 position evidence true ¿?
    # Review! Calling twice apply_evidence_into_2d_table is the same as calling one time with joint

    # PRE-TESTINGS... ------------------------------------------------
    if False:
        race_prob_estimate_ = pehd.compute_race_estimate(driver_championship_standing=1,
                                                         constructor_championship_standing=None, ci=0.05)

        target_positions_set_ = list(range(1, 5))
        probabilities_ser = race_prob_estimate_.ci_position_probabilities.loc[target_positions_set_, 'Probability']
        probabilities_ser[np.NaN] = race_prob_estimate_.dnf_prob
