from collections import namedtuple

import numpy as np
from scipy.stats import uniform

KellyCriterionEvaluation = namedtuple('KellyCriterionEvaluation', ["f",
                                                                   "strategy_probability",
                                                                   "bet_odds_probability",
                                                                   "bet_odds",
                                                                   "Strategy_odds_fair_value",
                                                                   "geometric_mean_rate",
                                                                   # "Optimal fraction to invest (f)",
                                                                   "is_good_bet"])
UniformBetCombination = namedtuple('UniformBetCombination', ["bets_composition_dict",
                                                             "bet_odds_value"])


class ModelStrategy:

    def __init__(self, probability_estimate_obj):
        """
        probability_estimate_obj:
            Expected with methods:
                -   compute_grid_estimate
                -   compute_race_estimate
                -   compute_conditioning_on_grid_race_estimate
        """

        self.pe_obj = probability_estimate_obj

    @staticmethod
    def eval_kelly_criterion(strategy_probability, bet_odds):
        """
        #       Bets_odds probs vs Strategy_probs.
        #       Bets_odds vs Strategy_odds.
        #       Optimal fraction to invest.
        #       Bool (if it is a good idea to invest or not!!)
        """
        if strategy_probability > 1.0:
            raise ValueError('Invalid input: Strategy probability is greater than one, please review')

        if bet_odds < 1.0:
            raise ValueError('Invalid input: bets odds are less than one please review')

        p = strategy_probability  # Probability of win
        q = 1 - p  # q is the probability of a loss
        # b = 4 - 1  # the proportion of the bet gained with a win
        b = bet_odds - 1  # the proportion of the bet gained with a win

        f = p - q / b

        if bet_odds < 1.0:
            print("WARNING!: bet_odds < 1.0, so bet_pro > 1.0 and criterioum is not ok")

        Strategy_odds = 1.0 / strategy_probability
        bet_odds_probability = 1.0 / bet_odds

        # Geometric mean.
        geom_mean_rate = ((1 + f * b) ** p) * ((1 - f * 1.0) ** (1 - p))
        geom_mean_rate = geom_mean_rate - 1

        criterion_eval = KellyCriterionEvaluation(f=f,
                                                  strategy_probability=strategy_probability,
                                                  bet_odds_probability=bet_odds_probability,
                                                  bet_odds=bet_odds,
                                                  Strategy_odds_fair_value=Strategy_odds,
                                                  geometric_mean_rate=geom_mean_rate,
                                                  is_good_bet=(f > 0)
                                                              and (strategy_probability > bet_odds_probability)
                                                              and (bet_odds > Strategy_odds))

        return criterion_eval

    @staticmethod
    def compute_uniform_bets_combination(bet_odds_list):
        """
        TODO:
        estimate bets combination such that each scenario return equal profit.

        return both fraction of bet & jointly_odd

        ie:
        2.25 & 4.00
        will return:
        f{
            4.00 -> 0.35, (approx! It was atually rounded)
            2.25 -> 0.65, (approx! It was atually rounded)
        }

        """
        a_bet_odds_list = np.array(bet_odds_list)

        bets_composition = 1.0 / a_bet_odds_list
        bets_composition = bets_composition / bets_composition.sum()

        bet_odds_value = bets_composition * a_bet_odds_list

        if len(np.unique(np.around(bet_odds_value, decimals=8))) > 1:
            raise ValueError(f"Error computing uniform combination, bet odds values are not unique: {bet_odds_value}")

        bet_odds_value = bet_odds_value[0]  # must be the same! (bet_composition*a_bet_odds_list).mean()

        bets_composition_dict = {bet_odd: bet_composition
                                 for bet_odd, bet_composition in zip(a_bet_odds_list, bets_composition)}

        bet_combination = UniformBetCombination(bets_composition_dict=bets_composition_dict,
                                                bet_odds_value=bet_odds_value)

        return bet_combination

    @staticmethod
    def _evaluate_geometric_mean_bet_long_run(strategy_prob, odds, f, n_steps=100, initial_invest=1, n_samples=5000):
        account_balance = initial_invest * np.ones(n_samples)
        for i in range(n_steps):
            is_a_suscessfull_bet = 1.0 * (uniform.rvs(size=n_samples) <= strategy_prob)
            step_invest = account_balance * f
            account_balance -= step_invest
            account_balance += step_invest * odds * is_a_suscessfull_bet

        account_median = np.median(account_balance)
        geometric_mean = (account_median / initial_invest) ** (1 / n_steps) - 1

        return geometric_mean


if __name__ == '__main__':
    import pandas as pd

    # TODO: add unittests for methods in base class!
    #   (also refactor returns to named tuple?)

    bet_combination_2p25_4p00 = ModelStrategy.compute_uniform_bets_combination(bet_odds_list=[2.25, 4.00])
    bet_combination_2p25_4p00_7p50 = ModelStrategy.compute_uniform_bets_combination(bet_odds_list=[2.25, 4.00, 7.5])

    print(ModelStrategy.eval_kelly_criterion(strategy_probability=0.4333, bet_odds=2.5))
    print(ModelStrategy.eval_kelly_criterion(strategy_probability=0.5666666666666667, bet_odds=2.5))
    print(ModelStrategy.eval_kelly_criterion(strategy_probability=0.5 * (0.4333 + 0.5666666666666667),
                                             bet_odds=2.5))

    print("GP us.")
    bets_composition_dict, bet_odds_value = ModelStrategy.compute_uniform_bets_combination(
        bet_odds_list=[6.00, 10.00])
    print(ModelStrategy.eval_kelly_criterion(strategy_probability=0.097812 + 0.182018, bet_odds=bet_odds_value))
    print(ModelStrategy.eval_kelly_criterion(strategy_probability=0.103683 + 0.189229, bet_odds=bet_odds_value))
    print(ModelStrategy.eval_kelly_criterion(strategy_probability=0.134328 + 0.226866, bet_odds=bet_odds_value))
    print(ModelStrategy.eval_kelly_criterion(strategy_probability=0.097812, bet_odds=11))
    print(ModelStrategy.eval_kelly_criterion(strategy_probability=0.182018, bet_odds=6))

    print(ModelStrategy.eval_kelly_criterion(strategy_probability=0.559441 + 0.320000, bet_odds=1.33))
    print(ModelStrategy.eval_kelly_criterion(strategy_probability=0.559441 + 0.320000, bet_odds=1.33))

    #
    #
    #
    print(" - ")
    p = 0.204111  # 0.197542
    bo = 5.5
    balance = 121 + 100
    kelly_criterion_eval = ModelStrategy.eval_kelly_criterion(strategy_probability=p, bet_odds=bo)
    geometric_mean_bet = ModelStrategy._evaluate_geometric_mean_bet_long_run(strategy_prob=p, odds=bo, f=kelly_criterion_eval.f)
    print(f"p: {p}, bo: {bo}, balance: {balance}")
    print(kelly_criterion_eval._asdict())
    print("Expected value (22 races):", balance*((1 + geometric_mean_bet)**22))
    print("Expected value (22 races):", balance*((1 + kelly_criterion_eval.geometric_mean_rate)**22))
    print("geometric_mean rate:  ", geometric_mean_bet)
    print(f"To bet: {kelly_criterion_eval.f*balance}  (based on: {balance})")
