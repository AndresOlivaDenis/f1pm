import unittest
from f1pm.betstrategiesevaluations.model_strategy import ModelStrategy


class TestModelStrategy(unittest.TestCase):

    def test_model_strategy_compute_uniform_bets_combination_case_one(self):
        # Parameters from class atributes.
        bet_combination_2p25_4p00 = ModelStrategy.compute_uniform_bets_combination(bet_odds_list=[2.25, 4.00])

        self.assertDictEqual(bet_combination_2p25_4p00.bets_composition_dict, {2.25: 0.64, 4.0: 0.36})
        self.assertEqual(bet_combination_2p25_4p00.bet_odds_value, 1.44)
        for bet, fraction in bet_combination_2p25_4p00.bets_composition_dict.items():
            print(f"bet: {bet}, fraction: {fraction}")
            print(f"bet * fraction: {bet * fraction}, (expected: {bet_combination_2p25_4p00.bet_odds_value})")
            self.assertEqual(bet_combination_2p25_4p00.bet_odds_value, bet * fraction)

    def test_model_strategy_compute_uniform_bets_combination_case_two(self):
        # Parameters from class atributes.
        bet_combination_2p25_4p00_7p50 = ModelStrategy.compute_uniform_bets_combination(bet_odds_list=[2.25, 4.00, 7.5])

        self.assertDictEqual(bet_combination_2p25_4p00_7p50.bets_composition_dict, {2.25: 0.5369127516778524,
                                                                                    4.0: 0.30201342281879195,
                                                                                    7.5: 0.16107382550335572})
        self.assertEqual(bet_combination_2p25_4p00_7p50.bet_odds_value, 1.2080536912751678)
        for bet, fraction in bet_combination_2p25_4p00_7p50.bets_composition_dict.items():
            print(f"bet: {bet}, fraction: {fraction}")
            print(f"bet * fraction: {bet * fraction}, (expected: {bet_combination_2p25_4p00_7p50.bet_odds_value})")
            self.assertEqual(bet_combination_2p25_4p00_7p50.bet_odds_value, bet * fraction)

    def test_model_strategy_compute_uniform_bets_combination_case_three(self):
        # Parameters from class atributes.
        bet_combination_2p25_4p00_5p50_7p2 = ModelStrategy.compute_uniform_bets_combination(
            bet_odds_list=[2.25, 4.00, 5.5, 7.2])

        self.assertDictEqual(bet_combination_2p25_4p00_5p50_7p2.bets_composition_dict, {2.25: 0.43781094527363185,
                                                                                        4.0: 0.24626865671641793,
                                                                                        5.5: 0.17910447761194032,
                                                                                        7.2: 0.13681592039800997})
        self.assertAlmostEqual(bet_combination_2p25_4p00_5p50_7p2.bet_odds_value, 0.985074626865672)
        for bet, fraction in bet_combination_2p25_4p00_5p50_7p2.bets_composition_dict.items():
            print(f"bet: {bet}, fraction: {fraction}")
            print(f"bet * fraction: {bet * fraction}, (expected: {bet_combination_2p25_4p00_5p50_7p2.bet_odds_value})")
            self.assertAlmostEqual(bet_combination_2p25_4p00_5p50_7p2.bet_odds_value, bet * fraction)

    def test_model_strategy_eval_kelly_criterion_case_one(self):
        kelly_criterion_eval = ModelStrategy.eval_kelly_criterion(strategy_probability=0.6, bet_odds=2)
        self.assertAlmostEqual(kelly_criterion_eval.f, 0.2)
        self.assertDictEqual(kelly_criterion_eval._asdict(), {'f': 0.19999999999999996,
                                                              'strategy_probability': 0.6,
                                                              'bet_odds_probability': 0.5,
                                                              'bet_odds': 2,
                                                              'Strategy_odds_fair_value': 1.6666666666666667,
                                                              'is_good_bet': True})

    def test_model_strategy_eval_kelly_criterion_case_zero(self):
        kelly_criterion_eval = ModelStrategy.eval_kelly_criterion(strategy_probability=0.5, bet_odds=2)
        self.assertAlmostEqual(kelly_criterion_eval.f, 0.0)
        self.assertDictEqual(kelly_criterion_eval._asdict(), {'f': 0.0,
                                                              'strategy_probability': 0.5,
                                                              'bet_odds_probability': 0.5,
                                                              'bet_odds': 2,
                                                              'Strategy_odds_fair_value': 2.0,
                                                              'is_good_bet': False})

    def test_model_strategy_eval_kelly_criterion_case_bad_bet(self):
        kelly_criterion_eval = ModelStrategy.eval_kelly_criterion(strategy_probability=0.4, bet_odds=2)
        self.assertAlmostEqual(kelly_criterion_eval.f, -0.2)
        self.assertDictEqual(kelly_criterion_eval._asdict(), {'f': -0.19999999999999996,
                                                              'strategy_probability': 0.4,
                                                              'bet_odds_probability': 0.5,
                                                              'bet_odds': 2,
                                                              'Strategy_odds_fair_value': 2.5,
                                                              'is_good_bet': False})
