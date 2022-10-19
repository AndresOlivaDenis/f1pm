import pandas as pd

from f1pm.historicaldataprocessing.historical_data_processing_m1 import process_historical_historical_data_m1


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
                                                                  constructor_championship_standing=(2, ),
                                                                  year_lower_threshold=None,
                                                                  round_lower_threshold=5)
