import pandas as pd

HISTORICAL_RACES_RESULTS_PATH = "../../data/historical_races_results/"
DRIVER_STANDINGS_FILE_PATH = HISTORICAL_RACES_RESULTS_PATH + "driver_standings.csv"
RESULTS_FILE_PATH = HISTORICAL_RACES_RESULTS_PATH + "results.csv"
RACES_FILE_PATH = HISTORICAL_RACES_RESULTS_PATH + "races.csv"
QUALIFYING_FILE_PATH = HISTORICAL_RACES_RESULTS_PATH + "qualifying.csv"
DRIVERS_FILE_PATH = HISTORICAL_RACES_RESULTS_PATH + "drivers.csv"
CONSTRUCTOR_FILE_PATH = HISTORICAL_RACES_RESULTS_PATH + "constructor_standings.csv"

REDUCED_DATA_COLUMNS = ['raceId', 'race_name', 'driverRef', 'qualifying_position', 'grid', 'position', 'positionOrder',
                        'driver_standing_position', 'constructor_standing_position', 'round', 'year', 'url']
MODEL_AA_DATA_COLUMNS = REDUCED_DATA_COLUMNS + ['driverId']


def process_historical_historical_data_m1(driver_standings_file_path=DRIVER_STANDINGS_FILE_PATH,
                                          results_file_path=RESULTS_FILE_PATH,
                                          races_file_path=RACES_FILE_PATH,
                                          qualifying_file_path=QUALIFYING_FILE_PATH,
                                          drivers_file_path=DRIVERS_FILE_PATH,
                                          constructor_file_path=CONSTRUCTOR_FILE_PATH
                                          ):
    # Paths ------------------------------------------------------------------------
    # Inputs.

    # Files reading ----------------------------------------------------------------
    df_driver_standings = pd.read_csv(driver_standings_file_path)
    df_results = pd.read_csv(results_file_path)
    df_qualifying = pd.read_csv(qualifying_file_path)
    df_races = pd.read_csv(races_file_path)
    df_drivers = pd.read_csv(drivers_file_path)
    df_constructor_standings = pd.read_csv(constructor_file_path)

    # Merge df ---------------------------------------------------------------------
    df_merge = df_results.copy()

    # races updates
    df_merge['race_name'] = None
    df_merge['url'] = None
    df_merge['round'] = None

    races_unique = pd.unique(df_merge['raceId'])
    df_races = df_races.set_index('raceId')
    for race_id in races_unique:
        race_index_bools = df_merge['raceId'] == race_id

        df_merge.loc[race_index_bools, "race_name"] = str(df_races.loc[race_id, 'name']) + " - " + str(
            df_races.loc[race_id, 'year'])
        df_merge.loc[race_index_bools, "url"] = df_races.loc[race_id, 'url']
        df_merge.loc[race_index_bools, "round"] = df_races.loc[race_id, 'round']
        df_merge.loc[race_index_bools, "year"] = df_races.loc[race_id, 'year']

    df_races = df_races.reset_index()

    # drivers updates
    df_merge['driverRef'] = None

    drivers_unique = pd.unique(df_merge['driverId'])
    df_drivers = df_drivers.set_index('driverId')
    for driver_id in drivers_unique:
        driver_index_bools = df_merge['driverId'] == driver_id
        df_merge.loc[driver_index_bools, "driverRef"] = df_drivers.loc[driver_id, 'driverRef']

    df_drivers = df_drivers.reset_index()

    df_merge = df_merge.set_index(['raceId', 'driverId'])

    #  driver_standing_position
    df_driver_standings = df_driver_standings.set_index(['raceId', 'driverId'])
    df_merge['driver_standing_position'] = df_driver_standings['position']
    df_driver_standings.reset_index()

    # qualification
    df_qualifying = df_qualifying.set_index(['raceId', 'driverId'])
    df_merge['qualifying_position'] = df_qualifying['position']
    df_qualifying = df_qualifying.reset_index()

    df_merge = df_merge.reset_index()

    # Constructor standing
    df_merge = df_merge.set_index(['raceId', 'constructorId'])

    df_constructor_standings = df_constructor_standings.set_index(['raceId', 'constructorId'])
    df_merge['constructor_standing_position'] = df_constructor_standings['position']
    df_constructor_standings.reset_index()

    df_merge = df_merge.reset_index()

    df_merge_all = df_merge.copy()
    df_merge = df_merge[REDUCED_DATA_COLUMNS]

    df_merge = df_merge.astype(
        {'raceId': 'int64', 'qualifying_position': 'float64', 'grid': 'int64', 'positionOrder': 'int64',
         'driver_standing_position': 'float64', 'constructor_standing_position': 'float64',
         'round': 'int64', 'year': 'float64'})

    return df_merge, df_merge_all


if __name__ == '__main__':
    df_data, df_data_all = process_historical_historical_data_m1()
