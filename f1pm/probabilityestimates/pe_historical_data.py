from collections import namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from f1pm.historicaldataprocessing.historical_data_processing_m1 import process_historical_historical_data_m1
from f1pm.historicaldataprocessing.tools import compute_historical_sub_data_set

RaceProbabilityEstimate = namedtuple('RaceProbabilityEstimate', ["position_probabilities",
                                                                 "ci_position_probabilities",
                                                                 "cum_position_probabilities",
                                                                 "ci_cum_position_probabilities",
                                                                 "dnf_prob",
                                                                 "ci_dnf_prob",
                                                                 "data_set_length"])

GridProbabilityEstimate = namedtuple('GridProbabilityEstimate', ["position_probabilities",
                                                                 "ci_position_probabilities",
                                                                 "cum_position_probabilities",
                                                                 "ci_cum_position_probabilities",
                                                                 "dnf_prob",
                                                                 "ci_dnf_prob",
                                                                 "data_set_length"])


class ProbabilityEstimateHistoricalData:

    def __init__(self, df_data, subdatset_params_dict):
        self.df_data = df_data.copy()

        if subdatset_params_dict is None:
            subdatset_params_dict = dict(year_lower_threshold=2000,
                                         year_upper_threshold=None,
                                         round_lower_threshold=5,
                                         round_upper_threshold=None)
        self.subdatset_params_dict = subdatset_params_dict

    def set_subdatset_params_dict(self, subdatset_params_dict):
        self.subdatset_params_dict = subdatset_params_dict

    def _compute_counting_probabilities(self, y, dnf_label='\\N'):
        # counting position probabilities. ----------------------------------------------------
        s_counting_probs = y.value_counts(normalize=True)

        if dnf_label in s_counting_probs:
            dnf_prob = s_counting_probs[dnf_label]  # TO Return
            s_counting_probs = s_counting_probs.drop(dnf_label)
        else:
            dnf_prob = 0

        s_counting_probs.index = s_counting_probs.index.astype(int)
        # s_counting_probs = s_counting_probs.sort_index()

        s_position_probs = pd.Series(0, index=range(1, 25))
        s_position_probs.loc[s_counting_probs.index] = s_counting_probs  # TO Return

        # cummulative counting probabilities ----------------------------------------------------
        s_cum_position_probs = s_position_probs.cumsum()

        return s_position_probs, s_cum_position_probs, dnf_prob

    def _compute_normal_approx_to_binomial_confidence_intervals(self, p, n, ci=0.05):
        p_se = ((p * (1 - p)) / n) ** 0.5
        z = st.norm().ppf(1 - ci / 2)
        p_lower_ci = p - z * p_se
        p_upper_ci = p + z * p_se

        result_df = pd.DataFrame()
        result_df['Probability'] = p
        # result_df['se'] = p_se
        result_df['CI_lower'] = p_lower_ci
        result_df['CI_upper'] = p_upper_ci

        return result_df

    def compute_grid_estimate(self, driver_championship_standing, ci=0.05):
        """
        Parameters
        ----------

        driver_championship_standing : Union[int, lst]
            Driver current championship standing

        ci : float
            confidence interval level

        Returns
        -------
            RaceProbabilityEstimate

        Notes
        -------

        """
        df_sub_data_set = compute_historical_sub_data_set(self.df_data,
                                                          grid=None,
                                                          driver_championship_standing=driver_championship_standing,
                                                          **self.subdatset_params_dict)
        sub_data_set_size = len(df_sub_data_set)

        # Target var: grid
        y = df_sub_data_set['grid']

        # Estimate of probabilities by counting
        s_position_probs, s_cum_position_probs, dnf_prob = self._compute_counting_probabilities(y, dnf_label=0)

        # Confidence intervals
        ci_position_probabilities = self._compute_normal_approx_to_binomial_confidence_intervals(p=s_position_probs, n=sub_data_set_size, ci=ci)
        ci_cum_position_probabilities = self._compute_normal_approx_to_binomial_confidence_intervals(p=s_cum_position_probs, n=sub_data_set_size, ci=ci)

        # Result named tuple
        grid_prob_estimate = GridProbabilityEstimate(position_probabilities=s_position_probs,
                                                     ci_position_probabilities=ci_position_probabilities,
                                                     cum_position_probabilities=s_cum_position_probs,
                                                     ci_cum_position_probabilities=ci_cum_position_probabilities,
                                                     dnf_prob=dnf_prob,
                                                     ci_dnf_prob=None,
                                                     data_set_length=sub_data_set_size)

        return grid_prob_estimate

    def compute_race_estimate(self, driver_championship_standing, ci=0.05):
        """
        Parameters
        ----------

        driver_championship_standing : Union[int, lst]
            Driver current championship standing

        ci : float
            confidence interval level

        Returns
        -------
            RaceProbabilityEstimate

        Notes
        -------

        """
        df_sub_data_set = compute_historical_sub_data_set(self.df_data,
                                                          grid=None,
                                                          driver_championship_standing=driver_championship_standing,
                                                          **self.subdatset_params_dict)
        sub_data_set_size = len(df_sub_data_set)

        # Target var: grid
        y = df_sub_data_set['position']

        # Estimate of probabilities by counting
        s_position_probs, s_cum_position_probs, dnf_prob = self._compute_counting_probabilities(y)

        # Confidence intervals
        ci_position_probabilities = self._compute_normal_approx_to_binomial_confidence_intervals(p=s_position_probs, n=sub_data_set_size, ci=ci)
        ci_cum_position_probabilities = self._compute_normal_approx_to_binomial_confidence_intervals(p=s_cum_position_probs, n=sub_data_set_size, ci=ci)

        # Result named tuple
        grid_prob_estimate = RaceProbabilityEstimate(position_probabilities=s_position_probs,
                                                     ci_position_probabilities=ci_position_probabilities,
                                                     cum_position_probabilities=s_cum_position_probs,
                                                     ci_cum_position_probabilities=ci_cum_position_probabilities,
                                                     dnf_prob=dnf_prob,
                                                     ci_dnf_prob=None,
                                                     data_set_length=sub_data_set_size)

        return grid_prob_estimate

    def compute_conditioning_on_grid_race_estimate(self,
                                                   grid,
                                                   driver_championship_standing,
                                                   ci=0.10):
        """
        Parameters
        ----------
        grid : Union[int, lst]
            Driver starting grid

        driver_championship_standing : Union[int, lst]
            Driver current championship standing

        ci : float
            confidence interval level

        Returns
        -------
            RaceProbabilityEstimate

        Notes
        -------

        """

        df_sub_data_set = compute_historical_sub_data_set(self.df_data,
                                                          grid=grid,
                                                          driver_championship_standing=driver_championship_standing,
                                                          **self.subdatset_params_dict)
        sub_data_set_size = len(df_sub_data_set)

        # Target var: Positin
        y = df_sub_data_set['position']

        # Estimate of probabilities by counting
        s_position_probs, s_cum_position_probs, dnf_prob = self._compute_counting_probabilities(y)

        # Confidence intervals
        ci_position_probabilities = self._compute_normal_approx_to_binomial_confidence_intervals(p=s_position_probs, n=sub_data_set_size, ci=ci)
        ci_cum_position_probabilities = self._compute_normal_approx_to_binomial_confidence_intervals(p=s_cum_position_probs, n=sub_data_set_size, ci=ci)

        # Result named tuple
        race_prob_estimate = RaceProbabilityEstimate(position_probabilities=s_position_probs,
                                                     ci_position_probabilities=ci_position_probabilities,
                                                     cum_position_probabilities=s_cum_position_probs,
                                                     ci_cum_position_probabilities=ci_cum_position_probabilities,
                                                     dnf_prob=dnf_prob,
                                                     ci_dnf_prob=None,
                                                     data_set_length=sub_data_set_size)

        return race_prob_estimate


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Historical data processing ----------------------------------------------
    df_data_, df_data_all = process_historical_historical_data_m1()
    # -------------------------------------------------------------------------

    # Probability estimates ---------------------------------------------
    subdatset_params_dict_ = dict(year_lower_threshold=2000,
                                  year_upper_threshold=None,
                                  round_lower_threshold=5,
                                  round_upper_threshold=None)

    pehd = ProbabilityEstimateHistoricalData(df_data_, subdatset_params_dict_)

    race_cond_estimate = pehd.compute_conditioning_on_grid_race_estimate(grid=2, driver_championship_standing=2)

    grid_estimate = pehd.compute_grid_estimate(driver_championship_standing=1)
    race_estimate = pehd.compute_race_estimate(driver_championship_standing=1)

    # Sum out vals
    positions_range = range(1, 25)
    sum_out_probs = {i: 0 for i in positions_range}
    for grid_position, grid_probability in grid_estimate.position_probabilities.items():
        race_cond_estimate_ = pehd.compute_conditioning_on_grid_race_estimate(grid=grid_position, driver_championship_standing=1)
        cond_pos_probs = race_cond_estimate_.position_probabilities

        for position in positions_range:
            sum_out_probs[position] += grid_probability*cond_pos_probs.loc[position]
    sum_out_probs = pd.Series(sum_out_probs)
