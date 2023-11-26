#  Helper functions
import gzip
import os
import re
import shutil

import pandas as pd
from shapely import Point

from utils import settings_values as s
from utils.logger import logging

logger = logging.getLogger(__name__)


def open_text_file(file_path, mode):
    """
    Open a text file, also works for gzipped files.
    """
    is_gzip = False
    with open(file_path, 'rb') as f:
        # Read the first two bytes for the magic number
        magic_number = f.read(2)
        is_gzip = magic_number == b'\x1f\x8b'

    if is_gzip:
        return gzip.open(file_path, mode)
    else:
        return open(file_path, mode, encoding='utf-8')


def modify_text_file(input_file, output_file, replace, replace_with):
    """
    Replace text in a text file.
    Also works for gzipped files.
    """
    logger.info(f"Replacing '{replace}' with '{replace_with}' in {input_file}...")
    with open_text_file(input_file, 'rt') as f:
        file_content = f.read()

    modified_content = file_content.replace(replace, replace_with)

    with open_text_file(output_file, 'wt') as f:
        f.write(modified_content)
    logger.info(f"Wrote modified file to {output_file}.")


def create_unique_leg_ids():
    """
    If the input leg data doesn't have unique IDs for each leg, create them.
    Adds a column with the name as specified in the settings by leg_id_column, writes back to csv
    Note: This does obviously not create unique leg ids in the expanded population, only in the input leg data for further processing.
    """
    logger.info(f"Creating unique leg ids in {s.MiD_TRIPS_FILE}...")
    legs_file = read_csv(s.MiD_TRIPS_FILE, s.PERSON_ID_COL)

    if s.LEG_ID_COL in legs_file.columns:
        logger.info(f"Legs file already has unique leg ids, skipping.")
        return
    if not s.LEG_NON_UNIQUE_ID_COL:
        raise ValueError(f"Please specify leg_non_unique_id_column in settings.yaml.")

    # Create unique leg ids
    legs_file[s.LEG_ID_COL] = legs_file[s.PERSON_ID_COL].astype(str) + "_" + legs_file[s.LEG_NON_UNIQUE_ID_COL].astype(str)

    # Write back to file
    legs_file.to_csv(s.MiD_TRIPS_FILE, index=False)
    logger.info(f"Created unique leg ids in {s.MiD_TRIPS_FILE}.")


def read_csv(csv_path, test_col):
    """
    Read a csv file with unknown separator and return a dataframe.
    :param csv_path: Path to csv file.
    :param test_col: Column name that should be present in the file.
    """
    try:
        df = pd.read_csv(csv_path)
        test = df[test_col]
    except (KeyError, ValueError):
        logger.info(f"ID column '{test_col}' not found in {csv_path}, trying to read as ';' separated file...")
        df = pd.read_csv(csv_path, sep=';')
        try:
            test = df[test_col]
            logger.info("Success.")
        except (KeyError, ValueError):
            logger.error(f"ID column '{test_col}' still not found in {csv_path}, verify column name and try again.")
            raise
    return df


def string_to_shapely_point(point_string):
    # Use a regular expression to extract numbers
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", point_string)

    # Convert the extracted strings to float and create a Shapely Point
    if len(matches) == 2:
        x, y = map(float, matches)
        return Point(x, y)
    else:
        raise ValueError("Invalid point string format")


def seconds_from_datetime(datetime):
    """
    Convert a datetime object to seconds since midnight of the referenced day.
    :param datetime: A datetime object.
    """
    return (datetime - pd.Timestamp(s.BASE_DATE)).total_seconds()


def compress_to_gz(input_file, delete_original=True):
    logger.info(f"Compressing {input_file} to .gz...")
    with open(input_file, 'rb') as f_in:
        with gzip.open(f"{input_file}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if delete_original:
        os.remove(input_file)
    logger.info(f"Compressed to {input_file}.gz.")


def find_outer_boundary(gdf, method='convex_hull'):
    combined = gdf.geometry.unary_union

    # Calculate the convex hull or envelope
    if method == 'convex_hull':
        outer_boundary = combined.convex_hull
    elif method == 'envelope':
        outer_boundary = combined.envelope
    else:
        raise ValueError("Method must be 'convex_hull' or 'envelope'")

    return outer_boundary
