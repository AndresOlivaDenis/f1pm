import pandas as pd

from f1pm.historicaldataprocessing.historical_data_processing_m1 import process_historical_historical_data_m1


def compute_historical_sub_data_set(data_df,
                                    grid,
                                    driver_championship_standing,
                                    year_lower_threshold=2000,
                                    year_upper_threshold=None,
                                    round_lower_threshold=5,
                                    round_upper_threshold=None):
    # Casting inputs)
    if (not isinstance(grid, list)) and (grid is not None):
        grid_lst = [grid]
    else:
        grid_lst = grid

    if (not isinstance(driver_championship_standing, list)) and (driver_championship_standing is not None):
        driver_championship_standing_lst = [driver_championship_standing]
    else:
        driver_championship_standing_lst = driver_championship_standing

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

    # champion
    if driver_championship_standing_lst is not None:
        df_is_in_championship_lst = pd.DataFrame()
        for driver_championship_standing in driver_championship_standing_lst:
            df_is_in_championship_lst[driver_championship_standing] = df_sub_data_set[
                                                                          'driver_standing_position'] == driver_championship_standing

        is_in_championship_lst = df_is_in_championship_lst.any(axis=1)
        df_sub_data_set = df_sub_data_set[is_in_championship_lst]

    return df_sub_data_set


if __name__ == '__main__':
    df_data, df_data_all = process_historical_historical_data_m1()

    df_sub_data_set_three = compute_historical_sub_data_set(df_data,
                                                            grid=None,
                                                            driver_championship_standing=[2, 3],
                                                            year_lower_threshold=None,
                                                            round_lower_threshold=5)