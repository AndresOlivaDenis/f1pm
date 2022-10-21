import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


from f1pm.historicaldataprocessing.historical_data_processing_m1 import process_historical_historical_data_m1, \
    MODEL_AA_DATA_COLUMNS
from f1pm.historicaldataprocessing.tools import compute_historical_sub_data_set, \
    add_latest_positions_and_grid_descriptive_variables
from f1pm.probabilityestimates.pe_historical_data import ProbabilityEstimateHistoricalData

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    HISTORICAL_RACES_RESULTS_PATH = "../../../data/historical_races_results/"
    DRIVER_STANDINGS_FILE_PATH = HISTORICAL_RACES_RESULTS_PATH + "driver_standings.csv"
    RESULTS_FILE_PATH = HISTORICAL_RACES_RESULTS_PATH + "results.csv"
    RACES_FILE_PATH = HISTORICAL_RACES_RESULTS_PATH + "races.csv"
    QUALIFYING_FILE_PATH = HISTORICAL_RACES_RESULTS_PATH + "qualifying.csv"
    DRIVERS_FILE_PATH = HISTORICAL_RACES_RESULTS_PATH + "drivers.csv"
    CONSTRUCTOR_FILE_PATH = HISTORICAL_RACES_RESULTS_PATH + "constructor_standings.csv"

    # Historical data processing ----------------------------------------------
    df_data_, df_data_all = process_historical_historical_data_m1(driver_standings_file_path=DRIVER_STANDINGS_FILE_PATH,
                                          results_file_path=RESULTS_FILE_PATH,
                                          races_file_path=RACES_FILE_PATH,
                                          qualifying_file_path=QUALIFYING_FILE_PATH,
                                          drivers_file_path=DRIVERS_FILE_PATH,
                                          constructor_file_path=CONSTRUCTOR_FILE_PATH)

    df_data_aa = df_data_all[MODEL_AA_DATA_COLUMNS]
    df_data_aa = add_latest_positions_and_grid_descriptive_variables(df_data_aa, n_race=5)
    # -------------------------------------------------------------------------

    # Probability estimates ---------------------------------------------
    subdatset_params_dict_ = dict(year_lower_threshold=2000,
                                  year_upper_threshold=None,
                                  round_lower_threshold=5,
                                  round_upper_threshold=None)

    df_sub_data_set = compute_historical_sub_data_set(df_data_aa,
                                                      grid=None,
                                                      driver_championship_standing=None,
                                                      year_lower_threshold=2000,
                                                      year_upper_threshold=None,
                                                      round_lower_threshold=5,
                                                      round_upper_threshold=None)

    pehd = ProbabilityEstimateHistoricalData(df_data_, subdatset_params_dict_)
    race_cond_estimate = pehd.compute_conditioning_on_grid_race_estimate(grid=2, driver_championship_standing=2)

    from sklearn.naive_bayes import GaussianNB

    Xcolumns = ['grid', 'driver_standing_position', 'lastest_5_position_best', 'lastest_5_position_mean',
                'lastest_5_grid_best', 'lastest_5_grid_mean']
    y_label = 'position'
    XY = df_data_aa[Xcolumns + [y_label]]
    XY = XY.dropna()
    x = XY[Xcolumns]
    y = XY[y_label]
    y = (y == '1')*1.0

    # Init the Gaussian Classifier
    model = GaussianNB()
    # Train the model
    model.fit(x, y)
    # Predict Output
    probs_predict = model.predict_proba([(2, 2)])

    probs_series = pd.Series({classe: prob for classe, prob in zip(model.classes_, probs_predict[0])})
    probs_series = probs_series[probs_series.index!='\\N']
    probs_series.index = probs_series.index.astype(int)
    probs_series = probs_series.sort_index()

    # Plot Confusion Matrix
    preds = model.predict(x)
    mat = confusion_matrix(preds, y)

    names = np.unique(preds)
    sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=names, yticklabels=names)
    plt.xlabel('Truth')
    plt.ylabel('Predicted')

    from sklearn.metrics import classification_report

    print(classification_report(y, preds))
