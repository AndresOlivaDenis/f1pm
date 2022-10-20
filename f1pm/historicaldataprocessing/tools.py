import numpy as np
import pandas as pd

from f1pm.historicaldataprocessing.historical_data_processing_m1 import process_historical_historical_data_m1, \
    MODEL_AA_DATA_COLUMNS


def compute_historical_sub_data_set(data_df,
                                    grid,
                                    driver_championship_standing,
                                    constructor_championship_standing=None,
                                    year_lower_threshold=2000,
                                    year_upper_threshold=None,
                                    round_lower_threshold=5,
                                    round_upper_threshold=None):
    # Casting inputs)
    if isinstance(grid, tuple):
        grid_lst = list(grid)
    elif (not isinstance(grid, list)) and (grid is not None):
        grid_lst = [grid]
    else:
        grid_lst = grid

    if isinstance(driver_championship_standing, tuple):
        driver_championship_standing_lst = list(driver_championship_standing)
    elif (not isinstance(driver_championship_standing, list)) and (driver_championship_standing is not None):
        driver_championship_standing_lst = [driver_championship_standing]
    else:
        driver_championship_standing_lst = driver_championship_standing

    if isinstance(constructor_championship_standing, tuple):
        constructor_championship_standing_lst = list(constructor_championship_standing)
    elif (not isinstance(constructor_championship_standing, list)) and (constructor_championship_standing is not None):
        constructor_championship_standing_lst = [constructor_championship_standing]
    else:
        constructor_championship_standing_lst = constructor_championship_standing

    df_sub_data_set = data_df.copy()

    # round_threshold
    if round_lower_threshold is not None:
        df_sub_data_set = df_sub_data_set[df_sub_data_set['round'] >= round_lower_threshold]

    if round_upper_threshold is not None:
        df_sub_data_set = df_sub_data_set[df_sub_data_set['round'] <= round_upper_threshold]

    # year_threshold
    if year_lower_threshold is not None:
        df_sub_data_set = df_sub_data_set[df_sub_data_set['year'] >= year_lower_threshold]

    if year_upper_threshold is not None:
        df_sub_data_set = df_sub_data_set[df_sub_data_set['year'] <= year_upper_threshold]

    # grid lst selection
    if grid_lst is not None:
        df_is_in_grid_lst = pd.DataFrame()
        for grid in grid_lst:
            df_is_in_grid_lst[grid] = df_sub_data_set['grid'] == grid

        is_in_grid_lst = df_is_in_grid_lst.any(axis=1)
        df_sub_data_set = df_sub_data_set[is_in_grid_lst]

    # Driver championship standings
    if driver_championship_standing_lst is not None:
        df_is_in_championship_lst = pd.DataFrame()
        for driver_championship_standing in driver_championship_standing_lst:
            df_is_in_championship_lst[driver_championship_standing] = df_sub_data_set[
                                                                          'driver_standing_position'] == driver_championship_standing

        is_in_championship_lst = df_is_in_championship_lst.any(axis=1)
        df_sub_data_set = df_sub_data_set[is_in_championship_lst]

    # constructor championship standing
    if constructor_championship_standing_lst is not None:
        df_is_in_constructor_lst = pd.DataFrame()
        for constructor_championship in constructor_championship_standing_lst:
            df_is_in_constructor_lst[constructor_championship] = df_sub_data_set[
                                                                     'constructor_standing_position'] == constructor_championship

        is_in_constructor_lst = df_is_in_constructor_lst.any(axis=1)
        df_sub_data_set = df_sub_data_set[is_in_constructor_lst]

    return df_sub_data_set


def get_previuous_driver_races_results(df_merge, race_id, driver_id, n_races=5):
    # TODO: add unittest unique tests for this
    df_lastest_n_driver_results = df_merge.copy()

    # Firter driver races
    is_driver_target = df_lastest_n_driver_results['driverId'] == driver_id
    df_lastest_n_driver_results = df_lastest_n_driver_results[is_driver_target]

    # Filter target races.
    race_id_row = df_lastest_n_driver_results[df_lastest_n_driver_results['raceId'] == race_id].iloc[0]
    race_year = race_id_row['year']
    race_round = race_id_row['round']

    df_lastest_n_driver_results = df_lastest_n_driver_results[df_lastest_n_driver_results['year'] == race_year]

    races_targets_rounds = list(range(race_round - n_races, race_round))
    isin_target_races = df_lastest_n_driver_results['round'].isin(races_targets_rounds)
    df_lastest_n_driver_results = df_lastest_n_driver_results[isin_target_races]

    return df_lastest_n_driver_results


def compute_latest_descritive_variables(latest_position_results):
    latest_position_results = latest_position_results[latest_position_results != '\\N'].astype(float)
    latest_position_results = latest_position_results.sort_values(ascending=True)
    # latest_position_results = latest_position_results.iloc[:3]

    if len(latest_position_results) > 0:
        result_mean = latest_position_results.mean()
        result_best = latest_position_results.iloc[0]
    else:
        result_mean = np.NaN
        result_best = np.NaN

    return result_mean, result_best


def add_latest_positions_and_grid_descriptive_variables(data_df, n_race=5):
    # TODO: add unittest unique tests for this

    print("Adding latest positions and grid descriptive variables ...")
    position_label = f'lastest_{n_race}_position'
    grid_label = f'lastest_{n_race}_grid'
    dnf_label = f'lastest_{n_race}_dnf_results'

    e_data_df = data_df.copy()

    e_data_df[position_label + "_best"] = np.NaN
    e_data_df[position_label + "_mean"] = np.NaN
    e_data_df[grid_label + "_best"] = np.NaN
    e_data_df[grid_label + "_mean"] = np.NaN
    # e_data_df[dnf_label] = np.NaN

    for index, row in data_df.iterrows():
        df_lastest_n_driver_results = get_previuous_driver_races_results(
            data_df,
            race_id=row['raceId'],
            driver_id=row['driverId'],
            n_races=n_race)

        if len(df_lastest_n_driver_results) == n_race:
            latest_position_results = df_lastest_n_driver_results['position']
            latest_grid_results = df_lastest_n_driver_results['qualifying_position']

            is_dnf = latest_position_results == '\\N'
            # e_data_df.loc[index, dnf_label] = is_dnf.sum() / n_race

            result_mean, result_best = compute_latest_descritive_variables(latest_position_results)
            e_data_df.loc[index, position_label + "_mean"] = result_mean
            e_data_df.loc[index, position_label + "_best"] = result_best

            result_mean, result_best = compute_latest_descritive_variables(latest_grid_results)
            e_data_df.loc[index, grid_label + "_mean"] = result_mean
            e_data_df.loc[index, grid_label + "_best"] = result_best

    return e_data_df


if __name__ == '__main__':
    df_data, df_data_all = process_historical_historical_data_m1()

    df_sub_data_set_three = compute_historical_sub_data_set(df_data,
                                                            grid=None,
                                                            driver_championship_standing=[2, 3],
                                                            year_lower_threshold=None,
                                                            round_lower_threshold=5)

    df_sub_data_set_three_ = compute_historical_sub_data_set(df_data,
                                                             grid=None,
                                                             driver_championship_standing=(2, 3),
                                                             year_lower_threshold=None,
                                                             round_lower_threshold=5)

    df_sub_data_set_constructor = compute_historical_sub_data_set(df_data,
                                                                  grid=None,
                                                                  driver_championship_standing=[1],
                                                                  constructor_championship_standing=[2],
                                                                  year_lower_threshold=None,
                                                                  round_lower_threshold=5)

    df_sub_data_set_constructor_ = compute_historical_sub_data_set(df_data,
                                                                   grid=None,
                                                                   driver_championship_standing=[1],
                                                                   constructor_championship_standing=2,
                                                                   year_lower_threshold=None,
                                                                   round_lower_threshold=5)

    df_sub_data_set_constructor__ = compute_historical_sub_data_set(df_data,
                                                                    grid=None,
                                                                    driver_championship_standing=[1],
                                                                    constructor_championship_standing=(2,),
                                                                    year_lower_threshold=None,
                                                                    round_lower_threshold=5)

    df_sub_data_set_from_2015 = compute_historical_sub_data_set(df_data_all,
                                                                grid=None,
                                                                driver_championship_standing=None,
                                                                year_lower_threshold=2015,
                                                                round_lower_threshold=None)

    df_data_aa = df_sub_data_set_from_2015[MODEL_AA_DATA_COLUMNS]
    df_data_aa = add_latest_positions_and_grid_descriptive_variables(df_data_aa, n_race=5)

    previuous_races_results = get_previuous_driver_races_results(df_data_aa, race_id=953, driver_id=830, n_races=4)
