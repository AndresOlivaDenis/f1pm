import unittest
import pandas as pd

from f1pm.historicaldataprocessing.historical_data_processing_m1 import process_historical_historical_data_m1
from f1pm.historicaldataprocessing.tools import compute_historical_sub_data_set
from f1pm.probabilityestimates.pe_historical_data import ProbabilityEstimateHistoricalData

TESTS_PATH = "data/"

TESTS_HISTORICAL_RACES_RESULTS_PATH = TESTS_PATH + "historical_races_results/"
TESTS_DRIVER_STANDINGS_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "driver_standings.csv"
TESTS_RESULTS_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "results.csv"
TESTS_RACES_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "races.csv"
TESTS_QUALIFYING_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "qualifying.csv"
TESTS_DRIVERS_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "drivers.csv"
TESTS_CONSTRUCTOR_FILE_PATH = TESTS_HISTORICAL_RACES_RESULTS_PATH + "constructor_standings.csv"


class TestProbabilityEstimates(unittest.TestCase):

    def test_pe_historical_data_compute_conditioning_on_grid_race_estimate_test_one(self):
        e_cum_position_probs = {1: 0.24050632911392406, 2: 0.6075949367088608, 3: 0.7088607594936709,
                                4: 0.7468354430379747, 5: 0.7594936708860759, 6: 0.7721518987341771,
                                7: 0.8101265822784809, 8: 0.8481012658227847, 9: 0.8607594936708859,
                                10: 0.8607594936708859, 11: 0.8734177215189871, 12: 0.8734177215189871,
                                13: 0.8734177215189871, 14: 0.8734177215189871, 15: 0.8860759493670883,
                                16: 0.8860759493670883, 17: 0.8860759493670883, 18: 0.8860759493670883,
                                19: 0.8860759493670883, 20: 0.8860759493670883, 21: 0.8860759493670883,
                                22: 0.8860759493670883, 23: 0.8860759493670883, 24: 0.8860759493670883}
        e_cum_position_probs = pd.Series(e_cum_position_probs)

        e_position_probs = {1: 0.24050632911392406, 2: 0.3670886075949367, 3: 0.10126582278481013,
                            4: 0.0379746835443038, 5: 0.012658227848101266, 6: 0.012658227848101266,
                            7: 0.0379746835443038, 8: 0.0379746835443038, 9: 0.012658227848101266, 10: 0.0,
                            11: 0.012658227848101266, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.012658227848101266, 16: 0.0,
                            17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0}
        e_position_probs = pd.Series(e_position_probs)

        df_data, df_data_all = process_historical_historical_data_m1(
            driver_standings_file_path=TESTS_DRIVER_STANDINGS_FILE_PATH,
            results_file_path=TESTS_RESULTS_FILE_PATH,
            races_file_path=TESTS_RACES_FILE_PATH,
            qualifying_file_path=TESTS_QUALIFYING_FILE_PATH,
            drivers_file_path=TESTS_DRIVERS_FILE_PATH,
            constructor_file_path=TESTS_CONSTRUCTOR_FILE_PATH
        )

        subdatset_params_dict_ = dict(year_lower_threshold=2000,
                                      year_upper_threshold=None,
                                      round_lower_threshold=5,
                                      round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(df_data, subdatset_params_dict_)
        race_prob_estimate = pehd.compute_conditioning_on_grid_race_estimate(grid=2, driver_championship_standing=2)

        pd.testing.assert_series_equal(race_prob_estimate.cum_position_probabilities, e_cum_position_probs)
        pd.testing.assert_series_equal(race_prob_estimate.position_probabilities, e_position_probs)

    def test_pe_historical_data_compute_conditioning_on_grid_race_estimate_test_two(self):
        e_cum_position_probs = {1: 0.6739130434782609, 2: 0.8260869565217391, 3: 0.8913043478260869,
                                4: 0.9021739130434783, 5: 0.9130434782608696, 6: 0.923913043478261,
                                7: 0.9347826086956523, 8: 0.9347826086956523, 9: 0.9456521739130437,
                                10: 0.9456521739130437, 11: 0.9456521739130437, 12: 0.956521739130435,
                                13: 0.956521739130435, 14: 0.956521739130435, 15: 0.956521739130435,
                                16: 0.956521739130435, 17: 0.956521739130435, 18: 0.956521739130435,
                                19: 0.956521739130435, 20: 0.956521739130435, 21: 0.956521739130435,
                                22: 0.956521739130435, 23: 0.956521739130435, 24: 0.956521739130435}
        e_cum_position_probs = pd.Series(e_cum_position_probs)

        e_position_probs = {1: 0.6739130434782609, 2: 0.15217391304347827, 3: 0.06521739130434782,
                            4: 0.010869565217391304, 5: 0.010869565217391304, 6: 0.010869565217391304,
                            7: 0.010869565217391304, 8: 0.0, 9: 0.010869565217391304, 10: 0.0, 11: 0.0,
                            12: 0.010869565217391304, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0,
                            20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0}
        e_position_probs = pd.Series(e_position_probs)

        e_ci_cum_position_probs = {
            'Probability': [0.6739130434782609, 0.8260869565217391, 0.8913043478260869, 0.9021739130434783,
                            0.9130434782608696, 0.923913043478261, 0.9347826086956523, 0.9347826086956523,
                            0.9456521739130437, 0.9456521739130437, 0.9456521739130437, 0.956521739130435,
                            0.956521739130435, 0.956521739130435, 0.956521739130435, 0.956521739130435,
                            0.956521739130435, 0.956521739130435, 0.956521739130435, 0.956521739130435,
                            0.956521739130435, 0.956521739130435, 0.956521739130435, 0.956521739130435],
            # 'se': [0.04887364512834941, 0.0395170982237559, 0.03245078336503383, 0.030972663905888674,
            #        0.029376692377920698, 0.027642463311432493, 0.025742048830616836, 0.025742048830616836,
            #        0.023635396237188286, 0.023635396237188286, 0.023635396237188286, 0.021261288996601062,
            #        0.021261288996601062, 0.021261288996601062, 0.021261288996601062, 0.021261288996601062,
            #        0.021261288996601062, 0.021261288996601062, 0.021261288996601062, 0.021261288996601062,
            #        0.021261288996601062, 0.021261288996601062, 0.021261288996601062, 0.021261288996601062],
            'CI_lower': [0.5935230510265562, 0.7610871141817966, 0.8379275591106945, 0.8512284144815283,
                         0.8647231192552091, 0.8784452374425782, 0.8924407063114503, 0.8924407063114503,
                         0.9067754066878694, 0.9067754066878694, 0.9067754066878694, 0.9215500308107123,
                         0.9215500308107123, 0.9215500308107123, 0.9215500308107123, 0.9215500308107123,
                         0.9215500308107123, 0.9215500308107123, 0.9215500308107123, 0.9215500308107123,
                         0.9215500308107123, 0.9215500308107123, 0.9215500308107123, 0.9215500308107123],
            'CI_upper': [0.7543030359299655, 0.8910867988616816, 0.9446811365414793, 0.9531194116054282,
                         0.9613638372665302, 0.9693808495139438, 0.9771245110798543, 0.9771245110798543,
                         0.984528941138218, 0.984528941138218, 0.984528941138218, 0.9914934474501578,
                         0.9914934474501578, 0.9914934474501578, 0.9914934474501578, 0.9914934474501578,
                         0.9914934474501578, 0.9914934474501578, 0.9914934474501578, 0.9914934474501578,
                         0.9914934474501578, 0.9914934474501578, 0.9914934474501578, 0.9914934474501578]}
        e_ci_cum_position_probs = pd.DataFrame(e_ci_cum_position_probs)
        e_ci_cum_position_probs.index = range(1, 25)

        df_data, df_data_all = process_historical_historical_data_m1(
            driver_standings_file_path=TESTS_DRIVER_STANDINGS_FILE_PATH,
            results_file_path=TESTS_RESULTS_FILE_PATH,
            races_file_path=TESTS_RACES_FILE_PATH,
            qualifying_file_path=TESTS_QUALIFYING_FILE_PATH,
            drivers_file_path=TESTS_DRIVERS_FILE_PATH,
            constructor_file_path=TESTS_CONSTRUCTOR_FILE_PATH
        )

        subdatset_params_dict_ = dict(year_lower_threshold=2008,
                                      year_upper_threshold=None,
                                      round_lower_threshold=5,
                                      round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(df_data, subdatset_params_dict_)
        race_prob_estimate = pehd.compute_conditioning_on_grid_race_estimate(grid=1, driver_championship_standing=1)

        pd.testing.assert_series_equal(race_prob_estimate.cum_position_probabilities, e_cum_position_probs)
        pd.testing.assert_series_equal(race_prob_estimate.position_probabilities, e_position_probs)
        pd.testing.assert_frame_equal(race_prob_estimate.ci_cum_position_probabilities, e_ci_cum_position_probs)

    def test_pe_historical_data_compute_conditioning_on_grid_race_estimate_test_three(self):
        e_cum_position_probs = {1: 0.03571428571428571, 2: 0.10714285714285714, 3: 0.39285714285714285,
                                4: 0.5357142857142857, 5: 0.6428571428571428, 6: 0.7142857142857142,
                                7: 0.7142857142857142, 8: 0.7499999999999999, 9: 0.7857142857142856,
                                10: 0.7857142857142856, 11: 0.7857142857142856, 12: 0.7857142857142856,
                                13: 0.7857142857142856, 14: 0.7857142857142856, 15: 0.7857142857142856,
                                16: 0.7857142857142856, 17: 0.7857142857142856, 18: 0.7857142857142856,
                                19: 0.7857142857142856, 20: 0.7857142857142856, 21: 0.7857142857142856,
                                22: 0.8214285714285713, 23: 0.8214285714285713, 24: 0.8214285714285713}
        e_cum_position_probs = pd.Series(e_cum_position_probs)

        e_position_probs = {1: 0.03571428571428571, 2: 0.07142857142857142, 3: 0.2857142857142857,
                            4: 0.14285714285714285, 5: 0.10714285714285714, 6: 0.07142857142857142, 7: 0.0,
                            8: 0.03571428571428571, 9: 0.03571428571428571, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0,
                            15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.03571428571428571,
                            23: 0.0, 24: 0.0}
        e_position_probs = pd.Series(e_position_probs)

        e_ci_cum_position_probs = {
            'Probability': [0.03571428571428571, 0.10714285714285714, 0.39285714285714285, 0.5357142857142857,
                            0.6428571428571428, 0.7142857142857142, 0.7142857142857142, 0.7499999999999999,
                            0.7857142857142856, 0.7857142857142856, 0.7857142857142856, 0.7857142857142856,
                            0.7857142857142856, 0.7857142857142856, 0.7857142857142856, 0.7857142857142856,
                            0.7857142857142856, 0.7857142857142856, 0.7857142857142856, 0.7857142857142856,
                            0.7857142857142856, 0.8214285714285713, 0.8214285714285713, 0.8214285714285713],
            # 'se': [0.03507073235935591, 0.05845122059892653, 0.09229618630166095, 0.09424976123424064,
            #        0.09055224157805536, 0.08537347209531385, 0.08537347209531385, 0.08183170883849715,
            #        0.07754430690597279, 0.07754430690597279, 0.07754430690597279, 0.07754430690597279,
            #        0.07754430690597279, 0.07754430690597279, 0.07754430690597279, 0.07754430690597279,
            #        0.07754430690597279, 0.07754430690597279, 0.07754430690597279, 0.07754430690597279,
            #        0.07754430690597279, 0.07237888244444445, 0.07237888244444445, 0.07237888244444445],
            'CI_lower': [-0.021971935606845225, 0.010999154940972228, 0.24104342606506704, 0.3806872241088347,
                         0.4939119598688925, 0.5738588490642968, 0.5738588490642968, 0.615398816917361,
                         0.6581652512505581, 0.6581652512505581, 0.6581652512505581, 0.6581652512505581,
                         0.6581652512505581, 0.6581652512505581, 0.6581652512505581, 0.6581652512505581,
                         0.6581652512505581, 0.6581652512505581, 0.6581652512505581, 0.6581652512505581,
                         0.6581652512505581, 0.7023759041251326, 0.7023759041251326, 0.7023759041251326],
            'CI_upper': [0.09340050703541665, 0.20328655934474205, 0.5446708596492187, 0.6907413473197367,
                         0.791802325845393, 0.8547125795071315, 0.8547125795071315, 0.8846011830826388,
                         0.913263320178013, 0.913263320178013, 0.913263320178013, 0.913263320178013, 0.913263320178013,
                         0.913263320178013, 0.913263320178013, 0.913263320178013, 0.913263320178013, 0.913263320178013,
                         0.913263320178013, 0.913263320178013, 0.913263320178013, 0.94048123873201, 0.94048123873201,
                         0.94048123873201]}
        e_ci_cum_position_probs = pd.DataFrame(e_ci_cum_position_probs)
        e_ci_cum_position_probs.index = range(1, 25)

        df_data, df_data_all = process_historical_historical_data_m1(
            driver_standings_file_path=TESTS_DRIVER_STANDINGS_FILE_PATH,
            results_file_path=TESTS_RESULTS_FILE_PATH,
            races_file_path=TESTS_RACES_FILE_PATH,
            qualifying_file_path=TESTS_QUALIFYING_FILE_PATH,
            drivers_file_path=TESTS_DRIVERS_FILE_PATH,
            constructor_file_path=TESTS_CONSTRUCTOR_FILE_PATH
        )

        subdatset_params_dict_ = dict(year_lower_threshold=2008,
                                      year_upper_threshold=None,
                                      round_lower_threshold=5,
                                      round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(df_data, subdatset_params_dict_)
        race_prob_estimate = pehd.compute_conditioning_on_grid_race_estimate(grid=5, driver_championship_standing=4)

        pd.testing.assert_series_equal(race_prob_estimate.cum_position_probabilities, e_cum_position_probs)
        pd.testing.assert_series_equal(race_prob_estimate.position_probabilities, e_position_probs)
        pd.testing.assert_frame_equal(race_prob_estimate.ci_cum_position_probabilities, e_ci_cum_position_probs)

    def test_pe_historical_data_compute_grid_estimate(self):
        e_grid = {1: 0.4041916167664671, 2: 0.24850299401197604, 3: 0.12574850299401197, 4: 0.05389221556886228,
                  5: 0.041916167664670656, 6: 0.041916167664670656, 7: 0.014970059880239521, 8: 0.005988023952095809,
                  9: 0.0, 10: 0.014970059880239521, 11: 0.005988023952095809, 12: 0.0, 13: 0.0,
                  14: 0.014970059880239521, 15: 0.005988023952095809, 16: 0.0029940119760479044, 17: 0.0,
                  18: 0.0029940119760479044, 19: 0.0, 20: 0.008982035928143712, 21: 0.0029940119760479044, 22: 0.0,
                  23: 0.0, 24: 0.0029940119760479044}
        e_grid = pd.Series(e_grid)

        e_ci_cum = {'Probability': [0.4041916167664671, 0.6526946107784432, 0.7784431137724551, 0.8323353293413174,
                                    0.874251497005988, 0.9161676646706587, 0.9311377245508982, 0.937125748502994,
                                    0.937125748502994, 0.9520958083832335, 0.9580838323353293, 0.9580838323353293,
                                    0.9580838323353293, 0.9730538922155688, 0.9790419161676647, 0.9820359281437125,
                                    0.9820359281437125, 0.9850299401197604, 0.9850299401197604, 0.9940119760479041,
                                    0.997005988023952, 0.997005988023952, 0.997005988023952, 0.9999999999999999],
                    # 'se': [0.026851819690017268, 0.026051788348697456, 0.02272389812699236, 0.020440744569063987,
                    #        0.018142468207523715, 0.015164218331733606, 0.013855570494921605, 0.013281958407885265,
                    #        0.013281958407885265, 0.011685676277682789, 0.010965269417216766, 0.010965269417216766,
                    #        0.010965269417216766, 0.00886019407806066, 0.007837962767361893, 0.007267630633558445,
                    #        0.007267630633558445, 0.006644514460623139, 0.006644514460623139, 0.004221476144910787,
                    #        0.0029895265623292346, 0.0029895265623292346, 0.0029895265623292346, 5.765432361767715e-10],
                    'CI_lower': [0.35156301725466976, 0.6016340438821359, 0.7339050918551929, 0.7922722061687693,
                                 0.8386929127285786, 0.8864463428867587, 0.903981305395596, 0.9110935883793799,
                                 0.9110935883793799, 0.9291923037439811, 0.936592299196806, 0.936592299196806,
                                 0.936592299196806, 0.9556882309265349, 0.9636797914314694, 0.967791633848998,
                                 0.967791633848998, 0.9720069310821835, 0.9720069310821835, 0.9857380348422841,
                                 0.9911466236309608, 0.9911466236309608, 0.9911466236309608, 0.9999999988699959],
                    'CI_upper': [0.4568202162782644, 0.7037551776747504, 0.8229811356897173, 0.8723984525138655,
                                 0.9098100812833975, 0.9458889864545587, 0.9582941437062004, 0.9631579086266081,
                                 0.9631579086266081, 0.9749993130224859, 0.9795753654738527, 0.9795753654738527,
                                 0.9795753654738527, 0.9904195535046028, 0.9944040409038599, 0.9962802224384271,
                                 0.9962802224384271, 0.9980529491573373, 0.9980529491573373, 1.0022859172535243,
                                 1.002865352416943, 1.002865352416943, 1.002865352416943, 1.0000000011300039]}

        e_ci_cum = pd.DataFrame(e_ci_cum)
        e_ci_cum.index = range(1, 25)

        df_data, df_data_all = process_historical_historical_data_m1(
            driver_standings_file_path=TESTS_DRIVER_STANDINGS_FILE_PATH,
            results_file_path=TESTS_RESULTS_FILE_PATH,
            races_file_path=TESTS_RACES_FILE_PATH,
            qualifying_file_path=TESTS_QUALIFYING_FILE_PATH,
            drivers_file_path=TESTS_DRIVERS_FILE_PATH,
            constructor_file_path=TESTS_CONSTRUCTOR_FILE_PATH
        )

        subdatset_params_dict_ = dict(year_lower_threshold=2000,
                                      year_upper_threshold=None,
                                      round_lower_threshold=5,
                                      round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(df_data, subdatset_params_dict_)
        grid_prob_estimate = pehd.compute_grid_estimate(driver_championship_standing=1, ci=0.05)

        pd.testing.assert_series_equal(grid_prob_estimate.position_probabilities, e_grid)
        pd.testing.assert_frame_equal(grid_prob_estimate.ci_cum_position_probabilities, e_ci_cum)

    def test_pe_historical_data_compute_race_estimate(self):
        df_data, df_data_all = process_historical_historical_data_m1(
            driver_standings_file_path=TESTS_DRIVER_STANDINGS_FILE_PATH,
            results_file_path=TESTS_RESULTS_FILE_PATH,
            races_file_path=TESTS_RACES_FILE_PATH,
            qualifying_file_path=TESTS_QUALIFYING_FILE_PATH,
            drivers_file_path=TESTS_DRIVERS_FILE_PATH,
            constructor_file_path=TESTS_CONSTRUCTOR_FILE_PATH
        )

        subdatset_params_dict_ = dict(year_lower_threshold=2000,
                                      year_upper_threshold=None,
                                      round_lower_threshold=5,
                                      round_upper_threshold=None)

        pehd = ProbabilityEstimateHistoricalData(df_data, subdatset_params_dict_)

        driver_championship_standing_testing_lst = [1, 3]
        for driver_standing in driver_championship_standing_testing_lst:

            grid_estimate = pehd.compute_grid_estimate(driver_championship_standing=driver_standing)
            race_estimate = pehd.compute_race_estimate(driver_championship_standing=driver_standing)

            positions_range = range(1, 25)
            race_sum_out_probs = {i: 0 for i in positions_range}
            for grid_position, grid_probability in grid_estimate.position_probabilities.items():
                race_cond_estimate_ = pehd.compute_conditioning_on_grid_race_estimate(grid=grid_position,
                                                                                      driver_championship_standing=driver_standing)
                cond_pos_probs = race_cond_estimate_.position_probabilities

                for position in positions_range:
                    race_sum_out_probs[position] += grid_probability * cond_pos_probs.loc[position]
            race_sum_out_probs = pd.Series(race_sum_out_probs)

            pd.testing.assert_series_equal(race_sum_out_probs, race_estimate.position_probabilities)
