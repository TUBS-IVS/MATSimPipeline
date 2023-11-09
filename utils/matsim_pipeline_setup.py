import os
from datetime import datetime

import pandas as pd
import yaml

from utils.logger import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))  # Assuming matsim_pipeline_setup.py is one level down from the project root

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', current_time)


def create_output_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")
    return OUTPUT_DIR


def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        logger.info(f"Loaded config from {file_path}")
    return config


def create_unique_leg_ids():
    """
    If the input leg data doesn't have unique IDs for each leg, create them.
    Adds a column with the name as specified in the settings by leg_id_column, writes back to csv
    """
    settings = load_yaml_config('settings.yaml')
    LEGS_FILE = settings['mid_trips_file']
    LEG_ID_COLUMN = settings['id_columns']['leg_id_column']
    LEG_NON_UNIQUE_ID_COLUMN = settings['id_columns']['leg_non_unique_id_column']
    PERSON_ID_COLUMN = settings['id_columns']['person_id_column']

    logger.info(f"Creating unique leg ids in {LEGS_FILE}...")
    try:
        legs_file = pd.read_csv(LEGS_FILE)
        test = legs_file[LEG_NON_UNIQUE_ID_COLUMN]
    except (KeyError, ValueError):
        logger.warning(f"Failed to load CSV data from {LEGS_FILE} with default separator. Trying ';'.")
        legs_file = pd.read_csv(LEGS_FILE, sep=';')

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
