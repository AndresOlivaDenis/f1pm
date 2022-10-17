from collections import namedtuple

import pandas as pd
from f1pm.betstrategiesevaluations.model_strategy import ModelStrategy
from f1pm.historicaldataprocessing.historical_data_processing_m1 import process_historical_historical_data_m1
from f1pm.probabilityestimates.pe_historical_data import ProbabilityEstimateHistoricalData
from f1pm.webrequests.f1com_standing_requests import request_current_drivers_standing, request_quali_results

StrategyZeroProbabilityEstimate = namedtuple('StrategyZeroProbabilityEstimate', ["race_target_position_prob",
                                                                                 "race_target_position_prob_normalized",
                                                                                 "fair_bet_value",
                                                                                 "fair_bet_value_normalized"])


class ModelOneStrategyZero(ModelStrategy):

    @staticmethod
    def compute_race_day_grid_df(current_driver_standing_table, grid_results):
        if 'DRIVER' in current_driver_standing_table.columns:
            current_driver_standing_table = current_driver_standing_table.set_index('DRIVER')

        if 'DRIVER' in grid_results.columns:
            grid_results = grid_results.set_index('DRIVER')

        df_race_day_grid = pd.DataFrame(index=grid_results.index)

        df_race_day_grid['grid'] = grid_results['POS']
        df_race_day_grid['driver_championship_standing'] = current_driver_standing_table['POS']

        df_race_day_grid = df_race_day_grid.dropna()
        df_race_day_grid = df_race_day_grid.astype({'grid': int, 'driver_championship_standing': int})
        return df_race_day_grid

    def compute_race_estimate(self,
                              target_cumsum_position,
                              current_driver_standing_table,
                              ci=0.32,
                              subset_n_threshold=10):

        if 'DRIVER' in current_driver_standing_table.columns:
            current_driver_standing_table = current_driver_standing_table.set_index('DRIVER')

        cum_position_probabilities, ci_lower, ci_upper = dict(), dict(), dict()

        for index, row in current_driver_standing_table.iterrows():
            race_cond_estimate = self.pe_obj.compute_race_estimate(driver_championship_standing=
                                                                   row['driver_championship_standing'],
                                                                   ci=ci)

            if race_cond_estimate.data_set_length >= subset_n_threshold:
                cum_position_probabilities[index] = race_cond_estimate.ci_cum_position_probabilities.loc[
                    target_cumsum_position, 'Probability']
                ci_lower[index] = race_cond_estimate.ci_cum_position_probabilities.loc[
                    target_cumsum_position, 'CI_lower']
                ci_upper[index] = race_cond_estimate.ci_cum_position_probabilities.loc[
                    target_cumsum_position, 'CI_upper']
            else:
                print(f"Warning. data set for: [driver_championship_standing:{row['driver_championship_standing']}] "
                      f"is too small. Setting prob to zero")
                cum_position_probabilities[index] = 0.0
                ci_lower[index] = 0.0
                ci_upper[index] = 0.0

        # Result named tuple
        sz_race_prob_estimate = self.compute_strategy_zero_output(cum_position_probabilities, ci_lower, ci_upper,
                                                                  target_cumsum_position, current_driver_standing_table)

        return sz_race_prob_estimate

    def compute_grid_estimate(self,
                              target_cumsum_position,
                              current_driver_standing_table,
                              ci=0.32,
                              subset_n_threshold=10):
        if 'DRIVER' in current_driver_standing_table.columns:
            current_driver_standing_table = current_driver_standing_table.set_index('DRIVER')

        cum_position_probabilities, ci_lower, ci_upper = dict(), dict(), dict()

        for index, row in current_driver_standing_table.iterrows():
            grid_cond_estimate = self.pe_obj.compute_grid_estimate(driver_championship_standing=
                                                                   row['driver_championship_standing'],
                                                                   ci=ci)

            if grid_cond_estimate.data_set_length >= subset_n_threshold:
                cum_position_probabilities[index] = grid_cond_estimate.ci_cum_position_probabilities.loc[
                    target_cumsum_position, 'Probability']
                ci_lower[index] = grid_cond_estimate.ci_cum_position_probabilities.loc[
                    target_cumsum_position, 'CI_lower']
                ci_upper[index] = grid_cond_estimate.ci_cum_position_probabilities.loc[
                    target_cumsum_position, 'CI_upper']
            else:
                print(f"Warning. data set for: [driver_championship_standing:{row['driver_championship_standing']}] "
                      f"is too small. Setting prob to zero")
                cum_position_probabilities[index] = 0.0
                ci_lower[index] = 0.0
                ci_upper[index] = 0.0

        # Result named tuple
        sz_grid_prob_estimate = self.compute_strategy_zero_output(cum_position_probabilities, ci_lower, ci_upper,
                                                                  target_cumsum_position, current_driver_standing_table)

        return sz_grid_prob_estimate

    def compute_conditioning_on_grid_race_estimate(self,
                                                   target_cumsum_position,
                                                   race_day_grid,
                                                   ci=0.32,
                                                   subset_n_threshold=10):

        cum_position_probabilities, ci_lower, ci_upper = dict(), dict(), dict()

        for index, row in race_day_grid.iterrows():
            race_cond_estimate = self.pe_obj.compute_conditioning_on_grid_race_estimate(grid=row['grid'],
                                                                                        driver_championship_standing=
                                                                                        row[
                                                                                            'driver_championship_standing'],
                                                                                        ci=ci)

            if race_cond_estimate.data_set_length >= subset_n_threshold:
                cum_position_probabilities[index] = race_cond_estimate.ci_cum_position_probabilities.loc[
                    target_cumsum_position, 'Probability']
                ci_lower[index] = race_cond_estimate.ci_cum_position_probabilities.loc[
                    target_cumsum_position, 'CI_lower']
                ci_upper[index] = race_cond_estimate.ci_cum_position_probabilities.loc[
                    target_cumsum_position, 'CI_upper']
            else:
                print(f"Warning. data set for: [grid: {row['grid']}, "
                      f"driver_championship_standing:{row['driver_championship_standing']}] is too small. Setting prob to zero")
                cum_position_probabilities[index] = 0.0
                ci_lower[index] = 0.0
                ci_upper[index] = 0.0

        sz_race_prob_estimate = self.compute_strategy_zero_output(cum_position_probabilities, ci_lower, ci_upper,
                                                                  target_cumsum_position, race_day_grid)

        return sz_race_prob_estimate

    def compute_strategy_zero_output(self, cum_position_probabilities, ci_lower, ci_upper, target_cumsum_position,
                                     race_day_grid):
        race_target_position_prob = race_day_grid.copy()
        label = f'Top {target_cumsum_position} probabilities' if (target_cumsum_position > 1) else 'win_probability'

        race_target_position_prob[label] = cum_position_probabilities.values()
        race_target_position_prob['CI_lower'] = ci_lower.values()
        race_target_position_prob['CI_upper'] = ci_upper.values()

        # Fair bet value.
        fair_bet_value = race_target_position_prob.copy()
        fair_bet_value[[label, 'CI_lower', 'CI_upper']] = 1.0 / fair_bet_value[[label, 'CI_lower', 'CI_upper']]

        # Normalization of win probabilities: Take into account other drivers winning probabilities
        race_target_position_prob_normalized = None
        fair_bet_value_normalized = None

        if target_cumsum_position == 1:
            race_target_position_prob_normalized = race_day_grid.copy()
            normalization_factor = race_target_position_prob[label].sum()
            race_target_position_prob_normalized[label] = race_target_position_prob[label] / normalization_factor

            ci_lower_normalization_values = []
            for ci_lower_probability, cum_position_probability in zip(ci_lower.values(),
                                                                      cum_position_probabilities.values()):
                ci_normalization_factor = normalization_factor - cum_position_probability + ci_lower_probability
                ci_lower_normalization_values.append(ci_lower_probability / ci_normalization_factor)

            ci_upper_normalization_values = []
            for ci_upper_probability, cum_position_probability in zip(ci_upper.values(),
                                                                      cum_position_probabilities.values()):
                ci_normalization_factor = normalization_factor - cum_position_probability + ci_upper_probability
                ci_upper_normalization_values.append(ci_upper_probability / ci_normalization_factor)

            race_target_position_prob_normalized['CI_lower'] = ci_lower_normalization_values
            race_target_position_prob_normalized['CI_upper'] = ci_upper_normalization_values

            fair_bet_value_normalized = race_target_position_prob_normalized.copy()
            fair_bet_value_normalized[[label, 'CI_lower', 'CI_upper']] = 1.0 / fair_bet_value_normalized[
                [label, 'CI_lower', 'CI_upper']]

        # Result named tuple
        sz_race_prob_estimate = StrategyZeroProbabilityEstimate(race_target_position_prob=race_target_position_prob,
                                                                race_target_position_prob_normalized=race_target_position_prob_normalized,
                                                                fair_bet_value=fair_bet_value,
                                                                fair_bet_value_normalized=fair_bet_value_normalized)

        return sz_race_prob_estimate


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Drivers championship standings race day grid ----------------------------------------------------------

    driver_standing_url = "https://www.formula1.com/en/results.html/2022/drivers.html"
    quali_results_url_ = "https://www.formula1.com/en/results.html/2022/races/1134/japan/qualifying.html"

    current_driver_standing_table_ = request_current_drivers_standing(driver_standing_url)
    grid_results_ = request_quali_results(quali_results_url_)

    df_race_day_grid_ = ModelOneStrategyZero.compute_race_day_grid_df(current_driver_standing_table_, grid_results_)

    # Historical data processing ----------------------------------------------
    df_data_, df_data_all = process_historical_historical_data_m1()
    # -------------------------------------------------------------------------

    # Probability estimates ---------------------------------------------
    subdatset_params_dict_ = dict(year_lower_threshold=2000,
                                  year_upper_threshold=None,
                                  round_lower_threshold=5,
                                  round_upper_threshold=None)

    pehd = ProbabilityEstimateHistoricalData(df_data_, subdatset_params_dict_)

    # Strategy zero evaluation ========================================================
    mosz = ModelOneStrategyZero(pehd)

    sz_race_prob_estimate_non_cond = mosz.compute_race_estimate(target_cumsum_position=1,
                                                                current_driver_standing_table=
                                                                df_race_day_grid_[
                                                                    [
                                                                        'driver_championship_standing']],
                                                                ci=0.05,
                                                                subset_n_threshold=10)

    sz_grid_prob_estimate_non_cond = mosz.compute_grid_estimate(target_cumsum_position=1,
                                                                current_driver_standing_table=
                                                                df_race_day_grid_[
                                                                    [
                                                                        'driver_championship_standing']],
                                                                ci=0.05,
                                                                subset_n_threshold=10)

    sz_race_prob_estimate_ = mosz.compute_conditioning_on_grid_race_estimate(
        target_cumsum_position=3,
        race_day_grid=df_race_day_grid_,
        ci=0.67,
        subset_n_threshold=10)

    sz_race_prob_estimate_win = mosz.compute_conditioning_on_grid_race_estimate(
        target_cumsum_position=1,
        race_day_grid=df_race_day_grid_,
        ci=0.1,
        subset_n_threshold=10)


    # TODO:
    #   Unittests!
    #
