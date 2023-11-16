#  Helper functions
import pandas as pd

from utils.matsim_pipeline_setup import load_yaml_config, logger


def create_unique_leg_ids():
    """
    If the input leg data doesn't have unique IDs for each leg, create them.
    Adds a column with the name as specified in the settings by leg_id_column, writes back to csv
    Note: This does obviously not create unique leg ids in the expanded population, only in the input leg data for further processing.
    """
    settings = load_yaml_config('settings.yaml')
    LEGS_FILE = settings['mid_trips_file']
    LEG_ID_COLUMN = settings['id_columns']['leg_id_column']
    LEG_NON_UNIQUE_ID_COLUMN = settings['id_columns']['leg_non_unique_id_column']
    PERSON_ID_COLUMN = settings['id_columns']['person_id_column']

    logger.info(f"Creating unique leg ids in {LEGS_FILE}...")
    legs_file = pd.read_csv(LEGS_FILE)
    try:
        test = legs_file[LEG_NON_UNIQUE_ID_COLUMN]
    except (KeyError, ValueError):
        logger.warning(f"{LEG_NON_UNIQUE_ID_COLUMN} not found in {LEGS_FILE}, trying to read as ';' separated file...")
        legs_file = pd.read_csv(LEGS_FILE, sep=';')
        try:
            test = legs_file[LEG_NON_UNIQUE_ID_COLUMN]
        except (KeyError, ValueError):
            logger.warning(f"{LEG_NON_UNIQUE_ID_COLUMN} still not found in {LEGS_FILE}, verify column name and try again.")
            raise

    if LEG_ID_COLUMN in legs_file.columns:
        logger.info(f"Legs file already has unique leg ids, skipping.")
        return
    if not LEG_NON_UNIQUE_ID_COLUMN:
        raise ValueError(f"Please specify leg_non_unique_id_column in settings.yaml.")

    # Create unique leg ids
    legs_file[LEG_ID_COLUMN] = legs_file[PERSON_ID_COLUMN].astype(str) + "_" + legs_file[LEG_NON_UNIQUE_ID_COLUMN].astype(str)

    # Write back to file
    legs_file.to_csv(LEGS_FILE, index=False)
    logger.info(f"Created unique leg ids in {LEGS_FILE}.")


def read_csv(csv_path, testcol):
    """
    Read a csv file with unknown separator and return a dataframe.
    :param csv_path: Path to csv file.
    :param testcol: Column name that should be present in the file.
    """
    try:
        df = pd.read_csv(csv_path)
        test = df[testcol]
    except (KeyError, ValueError):
        logger.info(f"ID column '{testcol}' not found in {csv_path}, trying to read as ';' separated file...")
        df = pd.read_csv(csv_path, sep=';')
        try:
            test = df[testcol]
            logger.info("Success.")
        except (KeyError, ValueError):
            logger.error(f"ID column '{testcol}' still not found in {csv_path}, verify column name and try again.")
            raise
    return df
