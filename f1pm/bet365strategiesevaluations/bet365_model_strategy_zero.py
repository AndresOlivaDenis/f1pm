from collections import namedtuple
import re

import pandas as pd
from f1pm.betstrategiesevaluations.model_strategy_zero import ModelOneStrategyZero
from f1pm.historicaldataprocessing.historical_data_processing_m1 import process_historical_historical_data_m1
from f1pm.probabilityestimates.pe_historical_data import ProbabilityEstimateHistoricalData
from f1pm.webrequests.bets365_savedshtml_reads import process_bet365_boderless_odds_table, \
    process_bet365_columns_odds_table, compute_uniform_mapping_drivers_to_car_odds
from f1pm.webrequests.f1com_standing_requests import request_current_drivers_standing, request_quali_results, \
    request_current_constructors_standing

RaceBetOpportunities = namedtuple('RaceBetOpportunities', ["driver_race_win",
                                                           "driver_top_3",
                                                           "driver_top_6",
                                                           "driver_top_10",
                                                           "car_race_win",
                                                           "car_race_win_joint_drivers_bet"
                                                           ])

GridBetOpportunities = namedtuple('GridBetOpportunities', ["driver_grid_win",
                                                           "car_grid_win",
                                                           "car_grid_win_joint_drivers_bet"
                                                           ])


def get_ranked_oportunities(res_df):
    # TODO:
    #   Subselect only good oportunities, eval kelly_criterium and rank by r
    pass


class Bet365ModelOneStrategyZero(ModelOneStrategyZero):

    def __init__(self, probability_estimate_obj, html_files_directory_path, current_driver_standing_table):
        """
        probability_estimate_obj:
            Expected with methods:
                -   compute_grid_estimate
                -   compute_race_estimate
                -   compute_conditioning_on_grid_race_estimate
        """

        self.pe_obj = probability_estimate_obj
        self.html_file_path = html_files_directory_path

        self.car_grid_win_dict = process_bet365_boderless_odds_table(html_files_directory_path + "car_grid_win.html")
        self.car_race_win_dict = process_bet365_boderless_odds_table(html_files_directory_path + "car_race_win.html")
        self.driver_grid_win_dict = process_bet365_boderless_odds_table(html_files_directory_path + "grid_win.html")
        self.driver_race_win_dict = process_bet365_boderless_odds_table(html_files_directory_path + "race_win.html")
        self.driver_race_top_10_dict = process_bet365_columns_odds_table(html_files_directory_path + "race_top_10.html")
        self.driver_race_top_6_dict = process_bet365_columns_odds_table(html_files_directory_path + "race_top_6.html")
        self.driver_race_top_3_dict = process_bet365_columns_odds_table(html_files_directory_path + "race_top_3.html")

        self.car_grid_win_fdu_dict = compute_uniform_mapping_drivers_to_car_odds(self.driver_grid_win_dict)
        self.car_race_win_fdu_dict = compute_uniform_mapping_drivers_to_car_odds(self.driver_race_win_dict)

        self.current_driver_standing_table = current_driver_standing_table.copy()
        if current_driver_standing_table.index.name != 'DRIVER':
            self.current_driver_standing_table = self.current_driver_standing_table.set_index('DRIVER')

        df_driver_standing = self.current_driver_standing_table[['POS', 'CAR']]
        df_driver_standing = df_driver_standing.astype({'POS': int})
        self.df_driver_standing = df_driver_standing.rename(columns={'POS': 'driver_championship_standing'})

    def eval_current_race_odds_oportunities_estimate(self,
                                                     ci=0.2,
                                                     subset_n_threshold=10,
                                                     look_for_constructor_standing=False):
        # Driver race win bets oportunities
        df_drivers_standings = self.df_driver_standing[['driver_championship_standing']]

        race_estimate = self.compute_race_estimate(target_cumsum_position=1,
                                                   current_driver_standing_table=df_drivers_standings,
                                                   ci=ci,
                                                   subset_n_threshold=subset_n_threshold,
                                                   look_for_constructor_standing=look_for_constructor_standing)
        driver_race_win = self.eval_oportunities(race_estimate, self.driver_race_win_dict)

        # Driver race top 3 bets oportunities
        race_top_3_estimate = self.compute_race_estimate(target_cumsum_position=3,
                                                         current_driver_standing_table=df_drivers_standings,
                                                         ci=ci,
                                                         subset_n_threshold=subset_n_threshold,
                                                         look_for_constructor_standing=look_for_constructor_standing)
        driver_top_3 = self.eval_oportunities(race_top_3_estimate, self.driver_race_top_3_dict)

        # Driver race top 6 bets oportunities
        race_top_6_estimate = self.compute_race_estimate(target_cumsum_position=6,
                                                         current_driver_standing_table=df_drivers_standings,
                                                         ci=ci,
                                                         subset_n_threshold=subset_n_threshold,
                                                         look_for_constructor_standing=look_for_constructor_standing)
        driver_top_6 = self.eval_oportunities(race_top_6_estimate, self.driver_race_top_6_dict)

        # Driver race top 10 bets oportunities
        race_top_10_estimate = self.compute_race_estimate(target_cumsum_position=10,
                                                          current_driver_standing_table=df_drivers_standings,
                                                          ci=ci,
                                                          subset_n_threshold=subset_n_threshold,
                                                          look_for_constructor_standing=look_for_constructor_standing)
        driver_top_10 = self.eval_oportunities(race_top_10_estimate, self.driver_race_top_10_dict)

        # Car race win opporunities
        race_car_win_estimate = self.compute_mapping_drivers_to_car_race_estimate(
            sz_probability_estimate=race_estimate,
            current_driver_standing_table=self.current_driver_standing_table)

        car_race_win = self.eval_oportunities(race_car_win_estimate, self.car_race_win_dict)
        car_race_win_joint_drivers_bet = self.eval_oportunities(race_car_win_estimate, self.car_race_win_fdu_dict)

        return RaceBetOpportunities(driver_race_win=driver_race_win,
                                    driver_top_3=driver_top_3,
                                    driver_top_6=driver_top_6,
                                    driver_top_10=driver_top_10,
                                    car_race_win=car_race_win,
                                    car_race_win_joint_drivers_bet=car_race_win_joint_drivers_bet)

    def eval_current_grid_odds_oportunities_estimate(self,
                                                     ci=0.2,
                                                     subset_n_threshold=10,
                                                     look_for_constructor_standing=False):
        # Driver grid win bets oportunities
        df_drivers_standings = self.df_driver_standing[['driver_championship_standing']]

        grid_estimate = self.compute_grid_estimate(target_cumsum_position=1,
                                                   current_driver_standing_table=df_drivers_standings,
                                                   ci=ci,
                                                   subset_n_threshold=subset_n_threshold,
                                                   look_for_constructor_standing=look_for_constructor_standing)
        driver_grid_win = self.eval_oportunities(grid_estimate, self.driver_grid_win_dict)

        # Car race win opporunities
        grid_car_win_estimate = self.compute_mapping_drivers_to_car_race_estimate(
            sz_probability_estimate=grid_estimate,
            current_driver_standing_table=self.current_driver_standing_table)

        car_grid_win = self.eval_oportunities(grid_car_win_estimate, self.car_grid_win_dict)
        car_grid_win_joint_drivers_bet = self.eval_oportunities(grid_car_win_estimate, self.car_grid_win_fdu_dict)

        return GridBetOpportunities(driver_grid_win=driver_grid_win,
                                    car_grid_win=car_grid_win,
                                    car_grid_win_joint_drivers_bet=car_grid_win_joint_drivers_bet)

    @staticmethod
    def eval_oportunities(model_zero_estimate, bets_odds_dict):
        oportunities_df = pd.DataFrame()  # = race_estimate.fair_bet_value.copy()
        oportunities_df['Probability'] = model_zero_estimate.race_target_position_prob['CI_lower']
        oportunities_df['Fair_value'] = model_zero_estimate.fair_bet_value['CI_lower']
        bets_odds_map = dict()
        for driver in bets_odds_dict:
            for index_val in oportunities_df.index:
                if re.search(driver, index_val):
                    bets_odds_map[index_val] = bets_odds_dict[driver]

        oportunities_df['bet365_odds'] = pd.Series(bets_odds_map)
        oportunities_df['bet365_probs'] = 1.0 / oportunities_df['bet365_odds']
        oportunities_df['is_an_opportunity'] = (oportunities_df['bet365_odds'] > oportunities_df['Fair_value']) & (
                oportunities_df['Probability'] > 0.0)

        # Kelly criiterion evaluations.
        f_kce, geometric_mean_rate_kce = dict(), dict()

        for index, row in oportunities_df.iterrows():
            if row['is_an_opportunity']:
                kelly_criterion_eval = Bet365ModelOneStrategyZero.eval_kelly_criterion(
                    strategy_probability=row['Probability'],
                    bet_odds=row['bet365_odds'])
                f_kce[index] = kelly_criterion_eval.f
                geometric_mean_rate_kce[index] = kelly_criterion_eval.geometric_mean_rate

        oportunities_df["f"] = pd.Series(f_kce)
        oportunities_df["geometric_mean_rate"] = pd.Series(geometric_mean_rate_kce)

        balance = 100
        e_balance_22_races = balance * ((1 + oportunities_df["geometric_mean_rate"]) ** 22)
        oportunities_df["expected_value (22 races) [%]"] = 100*(e_balance_22_races - balance)/balance

        return oportunities_df


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Drivers championship standings race day grid ----------------------------------------------------------

    driver_standing_url = "https://www.formula1.com/en/results.html/2022/drivers.html"
    quali_results_url_ = "https://www.formula1.com/en/results.html/2022/races/1134/japan/qualifying.html"
    constructor_standing_url = "https://www.formula1.com/en/results.html/2022/team.html"

    current_driver_standing_table_ = request_current_drivers_standing(driver_standing_url)
    grid_results_ = request_quali_results(quali_results_url_)
    current_constructors_standing_table_ = request_current_constructors_standing(constructor_standing_url)

    df_race_day_grid_ = Bet365ModelOneStrategyZero.compute_race_day_grid_df(current_driver_standing_table_,
                                                                            grid_results_)

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
    mosz = Bet365ModelOneStrategyZero(pehd,
                                      html_files_directory_path="../../data/bet365_odds/2022_mexico/",
                                      current_driver_standing_table=current_driver_standing_table_)

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

    # Constructor standings -----------------------------------------------
    df_race_day_grid_construc = Bet365ModelOneStrategyZero.compute_race_day_grid_df(current_driver_standing_table_,
                                                                                    grid_results_,
                                                                                    current_constructors_standing_table_)

    sz_grid_prob_estimate_non_cond_cons = mosz.compute_grid_estimate(target_cumsum_position=1,
                                                                     current_driver_standing_table=
                                                                     df_race_day_grid_construc[
                                                                         ['driver_championship_standing',
                                                                          'constructor_championship_standing']],
                                                                     ci=0.05,
                                                                     subset_n_threshold=10,
                                                                     look_for_constructor_standing=True)

    sz_car_race_prob_estimate_non_cond = mosz.compute_mapping_drivers_to_car_race_estimate(
        sz_probability_estimate=sz_race_prob_estimate_non_cond,
        current_driver_standing_table=current_driver_standing_table_)

    sz_car_race_prob_estimate_cond = mosz.compute_mapping_drivers_to_car_race_estimate(
        sz_probability_estimate=sz_race_prob_estimate_win,
        current_driver_standing_table=current_driver_standing_table_)

    sz_car_grid_prob_estimate_non_cond = mosz.compute_mapping_drivers_to_car_race_estimate(
        sz_probability_estimate=sz_grid_prob_estimate_non_cond,
        current_driver_standing_table=current_driver_standing_table_)

    mosz.eval_current_race_odds_oportunities_estimate(ci=0.2,
                                                      subset_n_threshold=10,
                                                      look_for_constructor_standing=False)

    mosz.eval_current_grid_odds_oportunities_estimate(ci=0.2,
                                                      subset_n_threshold=10,
                                                      look_for_constructor_standing=False)
