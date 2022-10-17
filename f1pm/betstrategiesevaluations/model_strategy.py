import numpy as np


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

        p = strategy_probability  # Probability of win
        q = 1 - p  # q is the probability of a loss
        # b = 4 - 1  # the proportion of the bet gained with a win
        b = bet_odds - 1  # the proportion of the bet gained with a win

        f = p - q / b

        if bet_odds < 1.0:
            print("WARNING!: bet_odds < 1.0, so bet_pro > 1.0 and criterioum is not ok")

        Strategy_odds = 1.0/strategy_probability
        bet_odds_probability = 1.0/bet_odds

        diagnostics_dict = {'strategy_probability': strategy_probability,
                            'bet_odds_probability': bet_odds_probability,
                            'bet_odds': bet_odds,
                            'Strategy_odds (fair value)': Strategy_odds,
                            'Optimal fraction to invest (f)': f,
                            'is_good_bet': (f > 0)
                                           and (strategy_probability > bet_odds_probability)
                                           and (bet_odds > Strategy_odds)}

        return f, diagnostics_dict
        pass

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
        bet_odds_value = bet_odds_value[0]  # must be the same! (bet_composition*a_bet_odds_list).mean()

        bets_composition_dict = {bet_odd: bet_composition
                                 for bet_odd, bet_composition in zip(a_bet_odds_list, bets_composition)}

        return bets_composition_dict, bet_odds_value


if __name__ == '__main__':
    import pandas as pd

    # TODO: add unittests for methods in base class!
    #   (also refactor returns to named tuple?)

    bets_composition_dict, bet_odds_value = ModelStrategy.compute_uniform_bets_combination(
        bet_odds_list=[2.25, 4.00])

    bets_composition_dict_b, bet_odds_value_b = ModelStrategy.compute_uniform_bets_combination(
        bet_odds_list=[2.25, 4.00, 7.5])
    print([bet * com for bet, com in bets_composition_dict_b.items()])

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

    #

