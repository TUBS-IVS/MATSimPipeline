#  Helper functions
import gzip
import json
import os
import random
import re
import shutil
from typing import List, Dict, Union

import geopandas as gpd
import numpy as np
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


def read_csv(csv_path: str, test_col=None, use_cols=None):
    """
    Read a csv file with unknown separator and return a dataframe.
    :param csv_path: Path to csv file.
    :param test_col: Column name that should be present in the file.
    :param use_cols: List of columns to use from the file. Defaults to all columns.
    """
    try:
        df = pd.read_csv(csv_path, sep=',', usecols=use_cols)
        if test_col is not None:
            test = df[test_col]
    except (KeyError, ValueError):  # Sometimes also throws without test_col, when the file is not comma-separated. This is good.
        logger.info(f"ID column '{test_col}' not found in {csv_path}, trying to read as ';' separated file...")
        df = pd.read_csv(csv_path, sep=';', usecols=use_cols)
        try:
            if test_col is not None:
                test = df[test_col]
        except (KeyError, ValueError):
            logger.error(f"ID column '{test_col}' still not found in {csv_path}, verify column name and try again.")
            raise
        logger.info("Success.")
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


def distribute_by_weights(data_to_distribute: pd.DataFrame, weighted_points_in_cells: pd.DataFrame, external_id_column,
                          cut_missing_ids=False):
    """
    Distribute data points from `weights_df` across the population dataframe based on weights (e.g. assign buildings to households).

    The function modifies the internal population dataframe by appending the point IDs from the weights dataframe
    based on their weights and the count of each ID in the population dataframe.

    Args:
        data_to_distribute (pd.DataFrame): DataFrame containing the data on cell level to distribute.
        weighted_points_in_cells (pd.DataFrame): DataFrame containing the ID of the geography, point IDs, and their weights.
        Must contain all geography IDs in the population dataframe. Is allowed to contain more (they will be skipped).
        external_id_column (str): The column name of the ID in the weights dataframe (e.g. 'BLOCK_NR').
        cut_missing_ids (bool): If True, IDs in the population dataframe that are not in the weights dataframe are cut from the population dataframe.
    """
    logger.info("Starting distribution by weights...")

    if not data_to_distribute[external_id_column].isin(weighted_points_in_cells[external_id_column]).all():
        if cut_missing_ids:
            logger.warning(f"Not all geography IDs in the population dataframe are in the weights dataframe. "
                           f"Cutting missing IDs: {set(data_to_distribute[external_id_column]) - set(weighted_points_in_cells[external_id_column])}")
            data_to_distribute = data_to_distribute[
                data_to_distribute[external_id_column].isin(weighted_points_in_cells[external_id_column])].copy()
        else:
            raise ValueError(f"Not all geography IDs in the population dataframe are in the weights dataframe. "
                             f"Missing IDs: {set(data_to_distribute[external_id_column]) - set(weighted_points_in_cells[external_id_column])}")

    # Count of each ID in population_df
    id_counts = data_to_distribute[external_id_column].value_counts().reset_index()
    id_counts.columns = [external_id_column, '_processing_count']
    logger.info(f"Computed ID counts for {len(id_counts)} unique IDs.")

    # Merge with weights_df
    weighted_points_in_cells = pd.merge(weighted_points_in_cells, id_counts, on=external_id_column, how='left')

    def distribute_rows(group):
        total_count = group['_processing_count'].iloc[0]
        if total_count == 0 or pd.isna(total_count):
            logger.debug(f"Geography ID {group[external_id_column].iloc[0]} is not in the given dataframe, "
                         f"likely because no person/activity etc. exists there. Skipping distribution for this ID.")
            return []
        # Compute distribution
        group['_processing_repeat_count'] = (group['ewzahl'] / group['ewzahl'].sum()) * total_count
        group['_processing_int_part'] = group['_processing_repeat_count'].astype(int)
        group['_processing_frac_part'] = group['_processing_repeat_count'] - group['_processing_int_part']

        # Distribute remainder
        remainder = total_count - group['_processing_int_part'].sum()
        assert remainder >= 0 and remainder % 1 == 0, f"Remainder is {remainder}, should be a positive integer."
        remainder = int(remainder)
        top_indices = group['_processing_frac_part'].nlargest(remainder).index
        group.loc[top_indices, '_processing_int_part'] += 1

        # Expand rows based on int_part
        expanded = []
        for _, row in group.iterrows():
            expanded.extend([row.to_dict()] * int(row['_processing_int_part']))
        return expanded

    expanded_rows = []
    for _, group in weighted_points_in_cells.groupby(external_id_column):
        expanded_rows.extend(distribute_rows(group))

    expanded_weights_df = pd.DataFrame(expanded_rows).drop(
        columns=['_processing_count', '_processing_repeat_count', '_processing_int_part', '_processing_frac_part'])
    logger.info(f"Generated expanded weights DataFrame with {len(expanded_weights_df)} rows.")
    if len(expanded_weights_df) != data_to_distribute.shape[0]:
        raise ValueError(f"Expanded weights DataFrame has {len(expanded_weights_df)} rows, "
                         f"but the population DataFrame has {data_to_distribute.shape[0]} rows.")

    # Add a sequence column to both dataframes to prevent cartesian product on merge
    data_to_distribute['_processing_seq'] = data_to_distribute.groupby(external_id_column).cumcount()
    expanded_weights_df['_processing_seq'] = expanded_weights_df.groupby(external_id_column).cumcount()

    # Merge using the ID column and the sequence
    data_to_distribute = pd.merge(data_to_distribute, expanded_weights_df, on=[external_id_column, '_processing_seq'],
                                  how='left').drop(columns='_processing_seq')

    logger.info("Completed distribution by weights.")
    return data_to_distribute


def random_point_in_polygon(polygon):
    if not polygon.is_valid or polygon.is_empty:
        raise ValueError("Invalid polygon")

    min_x, min_y, max_x, max_y = polygon.bounds

    while True:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            return random_point


def calculate_condition_likelihoods(df, filter_col, target_col) -> dict:
    """
    Calculate the likelihood of a target condition being true under different conditions.
    :param df: DataFrame containing the relevant data.
    :param filter_col: Given the unique values in this column,
    :param target_col: calculate the likelihood of each target condition.
    :return: Dictionary with likelihoods for each unique value in the filter column.
    """
    likelihoods = {}

    for condition in df[filter_col].unique():
        likelihood = df[df[filter_col] == condition][target_col].mean()
        likelihoods[condition] = likelihood

    logger.info(f"Calculated likelihoods for {len(likelihoods)} unique values in {filter_col}.")
    logger.info(f"{likelihoods}")
    return likelihoods


def calculate_value_frequencies_df(df, filter_col, target_col) -> pd.DataFrame:
    """
    Calculate the normalized frequency of each target value for each unique value in the filter column.
    :param df:
    :param filter_col: Given all unique values in this column,
    :param target_col: calculate the frequency of each target value.
    :return:
    """
    # Create a grouped DataFrame
    grouped = df.groupby([filter_col, target_col]).size().unstack(fill_value=0)

    # Normalize the counts to get frequencies
    frequencies_df = grouped.div(grouped.sum(axis=1), axis=0)

    logger.info(f"Calculated frequencies for {len(frequencies_df)} unique values in {filter_col}.")
    logger.info(f"{frequencies_df}")

    return frequencies_df


def summarize_slack_factors(slack_df):
    """
    Take the slack_factors df and summarize them by activities, for
    use in the activity_placer.
    :param slack_df:
    :return:
    """
    logger.info(f"Summarizing slack factors for {len(slack_df)} rows...")
    slack_df = slack_df[(slack_df['slack_factor'] > 1) & (slack_df['slack_factor'] < 50)]
    logger.info(f"Dropped outliers and false positives, {len(slack_df)} rows remaining.")

    grouped = slack_df.groupby(['start_activity', 'via_activity', 'end_activity'])

    summary_df = grouped['slack_factor'].agg(['median', 'mean', 'std', 'count']).reset_index()

    # Rename columns for clarity
    summary_df.columns = ['start_activity', 'via_activity', 'end_activity',
                          'median_slack_factor', 'mean_slack_factor',
                          'std_slack_factor', 'count_observations']
    logger.info(f"Summarized slack factors for {len(summary_df)} unique activity combinations.")
    return summary_df


def calculate_travel_time_matrix(cells_gdf, speed):
    """
    Constructs a travel time matrix for teleported modes (e.g. walk, bike).

    :param cells_gdf: GeoDataFrame with cells.
    :param speed: Movement speed in units per second.
    :return: DataFrame with columns FROM, TO, VALUE (travel time in seconds).
    """
    # Calculate the center of each cell
    centers = cells_gdf.geometry.centroid

    # Prepare data for the DataFrame
    from_list = []
    to_list = []
    value_list = []

    # Calculate distances and travel times
    for i, from_point in enumerate(centers):
        for j, to_point in enumerate(centers):
            distance = from_point.distance(to_point)  # Distance between centers
            travel_time = distance / speed  # Convert distance to time
            from_list.append(cells_gdf.iloc[i].name)  # TODO: Check identifier
            to_list.append(cells_gdf.iloc[j].name)
            value_list.append(travel_time)

    # Create the DataFrame
    travel_time_df = pd.DataFrame({'FROM': from_list, 'TO': to_list, 'VALUE': value_list})

    return travel_time_df


class SlackFactors:
    """
    Manages slack factors for different activities.
    """

    def __init__(self, slack_factors_csv_path: str):
        self.slack_factors_df = read_csv(slack_factors_csv_path)

    def get_slack_factor(self, activity_from: str, activity_via: str, activity_to: str) -> float:
        """
        Retrieve the slack factor for a given activity combination.
        """
        slack_factor_row = self.slack_factors_df.loc[
            (self.slack_factors_df['start_activity'] == activity_from) &
            (self.slack_factors_df['via_activity'] == activity_via) &
            (self.slack_factors_df['end_activity'] == activity_to)
            ]

        if not slack_factor_row.empty:
            slack_factor: float = slack_factor_row['median_slack_factor'].iloc[0]
        else:
            # Fallback to a default slack factor if not found
            logger.debug(f"No slack factor found for activities: {activity_from}, {activity_via}, {activity_to}. "
                         f"Using default slack factor of {s.DEFAULT_SLACK_FACTOR}")
            slack_factor = s.DEFAULT_SLACK_FACTOR

        return slack_factor

    def calculate_expected_time_with_slack(self, time_from_start_to_via: float, time_from_via_to_end: float, activity_from: str,
                                           activity_via: str, activity_to: str) -> float:
        """
        Calculates the expected time with slack for a given activity combination. Makes sure the returned time is plausible,

        """
        expected_time: float = ((time_from_start_to_via + time_from_via_to_end) /
                                self.get_slack_factor(activity_from, activity_via, activity_to))
        time_diff = np.abs(time_from_start_to_via - time_from_via_to_end)
        if expected_time < time_diff:  # this makes the trip impossible. We find a reasonable alternative:
            logger.debug(f"Expected time is too low. Returning time difference plus half of the smaller time.")
            expected_time = time_diff + (min(time_from_start_to_via, time_from_via_to_end) / 2)
        return expected_time

    def calculate_expected_distance_with_slack(self, distance_from_start_to_via: float, distance_from_via_to_end: float,
                                               activity_from: str, activity_via: str, activity_to: str) -> float:
        """
        Identical to calculate_expected_time_with_slack, but different parameter names for clarity.
        """
        expected_distance: float = ((distance_from_start_to_via + distance_from_via_to_end) /
                                    self.get_slack_factor(activity_from, activity_via, activity_to))
        distance_diff = np.abs(distance_from_start_to_via - distance_from_via_to_end)
        if expected_distance < distance_diff:
            logger.debug(f"Expected distance is too low. Returning distance difference plus half of the smaller distance.")
            expected_distance = distance_diff + (min(distance_from_start_to_via, distance_from_via_to_end) / 2)
        return expected_distance

    def get_all_estimated_times_with_slack(self, leg_chain, level=0):
        """
        Recursive function that adds columns for each level of slack factor calculation until all needed levels
        have been processed. The columns are named level_0, level_1, etc. and contain the slack-estimated direct
        time of all legs up to and including the entry (i.e., how long it would take without the detour of the
        in-between activities). The last column contains the estimated direct time from chain start to chain end.
        :param leg_chain: df
        :param level: int, highest level reached
        :return:
        """
        if level == 0:
            times_col = s.LEG_DURATION_MINUTES_COL
        else:
            times_col = f"level_{level}"

        # Base cases
        len_times_col = leg_chain[times_col].notna().sum()
        if len_times_col == 0:
            logger.warning(f"Received empty DataFrame, returning unchanged.")
            return leg_chain, level
        elif len_times_col == 1:
            logger.debug(f"Received single leg, returning unchanged.")
            return leg_chain, level
        elif len_times_col == 2:
            logger.debug(f"Two legs remain to estimate, solving last level.")
            return self.solve_level(leg_chain, level), level + 1
        # Recursive case
        else:
            logger.debug(f"More than two legs remain to estimate, solving level {level}.")
            updated_leg_chain = self.solve_level(leg_chain, level)
            return self.get_all_estimated_times_with_slack(updated_leg_chain, level + 1)

    def get_all_adjusted_times_with_slack(self, leg_chain, real_total_time):  # TODO: finish!!!
        """
        When the real total time is known, this function can be used to adjust the estimated times with slack.
        This guarantees that a valid leg chain can be built.
        This method ties it all together.
        """
        df, highest_level = self.get_all_estimated_times_with_slack(leg_chain)

        # Highest level to including level 1
        for level in range(highest_level, 0, -1):
            # Adjust for every higher-level leg
            for i, row in df.iterrows():
                # Check if there are non-NaN values to the right of the current leg in the higher-level columns
                # This means the leg already belongs to another higher-level leg ;(
                if df.loc[i, f'level_{level + 1}':].notna().any():
                    continue

                # Find the first non-NaN values below the current row in the lower-level column
                lower_level_legs = df.loc[i:, f'level_{level}'].dropna()

                # If there are two legs below the current leg, apply the formula
                if len(lower_level_legs) == 2:
                    leg1 = lower_level_legs.iloc[0]
                    leg2 = lower_level_legs.iloc[1]
                    delta_L_high = real_total_time - df.loc[i, f'level_{level + 1}']

                    L_bounds1 = df.loc[i, f'level_{level}_lower_bound']
                    L_bounds2 = df.loc[i, f'level_{level}_lower_bound']

                    L_high = df.loc[i, f'level_{level + 1}']

                    delta_L1 = (L_bounds1 * delta_L_high * (leg1 + leg2) ** 2) / (L_high * (leg1 * L_bounds1 + leg2 * L_bounds2))
                    delta_L2 = (L_bounds2 * delta_L_high * (leg1 + leg2) ** 2) / (L_high * (leg1 * L_bounds1 + leg2 * L_bounds2))

                    df.loc[i, f'level_{level}'] += delta_L1
                    df.loc[i + 1, f'level_{level}'] += delta_L2

        return df

    def solve_level(self, leg_chain, level):  # TODO: finish
        """
        Adds a column for given level of slack factor calculation, containing the time with slack of all legs up to and
        including the entry. For better performance, keep the number of columns to a minimum.
        :param leg_chain: df
        :param level: Next level to calculate
        :return: copy of leg_chain with added column
        """
        leg_chain = leg_chain.copy()
        leg_chain[f"level_{level + 1}"] = np.nan

        if level == 0:
            times_col = s.LEG_DURATION_MINUTES_COL
            if leg_chain[times_col].notna().sum() != len(leg_chain):
                logger.warning(f"Found NaN values in {times_col}, may produce incorrect results.")
        else:
            times_col = f"level_{level}"  # Here we expect some NaN values

        legs_to_process = leg_chain[leg_chain[times_col].notna()].copy()
        legs_to_process['original_index'] = legs_to_process.index

        # Reset index for reliable pairing
        legs_to_process.reset_index(drop=True, inplace=True)
        legs_to_process['pair_id'] = legs_to_process.index // 2

        for pair_id, group in legs_to_process.groupby('pair_id'):
            if len(group) == 1:
                time = group.iloc[0][times_col]
            else:
                time = self.calculate_expected_time_with_slack(group.iloc[0][times_col],
                                                               group.iloc[1][times_col],
                                                               group.iloc[0][s.LEG_FROM_ACTIVITY_COL],
                                                               group.iloc[0][s.LEG_TO_ACTIVITY_COL],
                                                               group.iloc[1][s.LEG_TO_ACTIVITY_COL])

            leg_chain.loc[group['original_index'].iloc[-1], f"level_{level + 1}"] = time

        return leg_chain


class TTMatrices:
    """
    Manages travel time matrices for various modes of transportation.
    """

    def __init__(self, car_tt_matrices_csv_paths: List[str], pt_tt_matrices_csv_paths: List[str],
                 bike_tt_matrix_csv_path: str, walk_tt_matrix_csv_path: str):
        self.tt_matrices: Dict[str, Union[Dict[str, pd.DataFrame], pd.DataFrame, None]] = {'car': {}, 'pt': {}, 'bike': None,
                                                                                           'walk': None}

        # Read car and pt matrices for each hour
        for mode, csv_paths in zip(['car', 'pt'], [car_tt_matrices_csv_paths, pt_tt_matrices_csv_paths]):
            for hour, path in enumerate(csv_paths):
                try:
                    self.tt_matrices[mode][str(hour)] = read_csv(path)
                except Exception as e:
                    logger.error(f"Error reading {mode} matrix for hour {hour}: {e}")
                    raise e

        # Read bike and walk matrices
        try:
            self.tt_matrices['bike'] = read_csv(bike_tt_matrix_csv_path)
            self.tt_matrices['walk'] = read_csv(walk_tt_matrix_csv_path)
        except Exception as e:
            logger.error(f"Error reading bike/walk matrices: {e}")
            raise e

        # Validation
        tt_rows_num: int = len(self.tt_matrices['car']['0'])

        for mode, matrices in self.tt_matrices.items():
            if mode in ['bike', 'walk']:
                if len(matrices) != 1:
                    logger.warning(f"Expected 1 {mode} matrix, found {len(matrices)}")
                for df in matrices.values():
                    if not {'FROM', 'TO', 'VALUE'}.issubset(df.columns):
                        raise ValueError(f"Invalid {mode} matrix. Columns must include FROM, TO, VALUE.")
                    if len(df) != tt_rows_num:
                        raise ValueError(f"Invalid {mode} matrix. Number of rows must be {tt_rows_num}.")
            else:
                if len(matrices) != 24:
                    logger.warning(f"Expected 24 {mode} matrices, found {len(matrices)}")
                for df in matrices.values():
                    if not {'FROM', 'TO', 'VALUE'}.issubset(df.columns):
                        raise ValueError(f"Invalid {mode} matrix. Columns must include FROM, TO, VALUE.")
                    if len(df) != tt_rows_num:
                        raise ValueError(f"Invalid {mode} matrix. Number of rows must be {tt_rows_num}.")

        logger.info(f"Loaded travel time matrices for {len(self.tt_matrices['car'])} hours.")

    def get_tt_matrix(self, mode: str, hour: int = None):
        """
        Retrieve the travel time matrix for a given mode and hour.
        :param mode: car, pt, bike, walk
        :param hour: 0 - 23
        :return:
        """
        if mode not in ['car', 'pt', 'bike', 'walk']:
            raise ValueError("Invalid mode. Choose from 'car', 'pt', 'bike', 'walk'.")

        if mode in ['car', 'pt']:
            if hour is None:
                raise ValueError(f"Hour must be specified for mode {mode}.")
            return self.tt_matrices[mode].get(str(hour))

        return self.tt_matrices[mode]

    def get_weighted_tt_matrix_two_modes(self, mode1, weight1, mode2, weight2, hour=None):

        tt_matrix1 = self.get_tt_matrix(mode1, hour)
        tt_matrix2 = self.get_tt_matrix(mode2, hour)

        if tt_matrix1 is None or tt_matrix2 is None:
            raise ValueError("One or both of the travel time matrices could not be retrieved.")

        weighted_tt_matrix = tt_matrix1.copy()
        weighted_tt_matrix['VALUE'] = tt_matrix1['VALUE'].multiply(weight1).add(tt_matrix2['VALUE'].multiply(weight2))

        return weighted_tt_matrix

    def get_weighted_tt_matrix_n_modes(self, mode_weights: Dict[str, float], hour: int = None) -> pd.DataFrame:
        """
        Get a weighted travel time matrix for multiple modes.
        :param mode_weights: Dictionary with mode names as keys and weights as values.
        :param hour: Hour of the day for modes with time-dependent matrices (car, pt). 0-23.
        """
        weighted_tt_matrix = None
        total_weight = sum(mode_weights.values())

        for mode, weight in mode_weights.items():
            tt_matrix = self.get_tt_matrix(mode, hour)
            weighted_matrix = tt_matrix * (weight / total_weight)

            if weighted_tt_matrix is None:
                weighted_tt_matrix = weighted_matrix
            else:
                weighted_tt_matrix += weighted_matrix

        return weighted_tt_matrix

    def get_travel_time(self, cell_from, cell_to, mode, hour=None):
        """
        Get the travel time between two cells for a specified mode and time of day.

        :param cell_from: Starting cell id.
        :param cell_to: Destination cell id.
        :param mode: Mode of transportation (car, pt, bike, walk).
        :param hour: Hour of the day for modes with time-dependent matrices (car, pt). 0-23.
        :return: Travel time between the two cells.
        """
        if mode not in ['car', 'pt', 'bike', 'walk']:
            raise ValueError("Invalid mode. Choose from 'car', 'pt', 'bike', 'walk'.")

        tt_matrix = self.get_tt_matrix(mode, hour)

        filtered_matrix = tt_matrix[(tt_matrix['FROM'] == cell_from) & (tt_matrix['TO'] == cell_to)]

        # Extract travel time from the filtered row
        if not filtered_matrix.empty:
            travel_time = filtered_matrix['VALUE'].iloc[0]
        else:
            logger.error(f"No travel time found for cells {cell_from} and {cell_to}. Using default value of 20 minutes.")
            travel_time = 20

        return travel_time


def sigmoid(x, beta, delta_T):
    """
    Sigmoid function for likelihood calculation.

    :param x: The input value (time differential).
    :param beta: Controls the steepness of the sigmoid's transition.
    :param delta_T: The midpoint of the sigmoid's transition.
    :return: Sigmoid function value.
    """
    return 1 / (1 + np.exp(-beta * (x - delta_T)))


def check_distance(leg_to_find, leg_to_compare):
    distance_to_find = leg_to_find[s.LEG_DISTANCE_COL]
    distance_to_compare = leg_to_compare[s.LEG_DISTANCE_COL]

    if pd.isnull(distance_to_find) or pd.isnull(distance_to_compare):
        return False

    difference = abs(distance_to_find - distance_to_compare)
    range_tolerance = distance_to_find * 0.05

    return difference <= range_tolerance


def check_time(leg_to_find, leg_to_compare):
    # Using constant variables instead of strings
    leg_begin_to_find = leg_to_find[s.LEG_START_TIME_COL]
    leg_end_to_find = leg_to_find[s.LEG_END_TIME_COL]
    leg_begin_to_compare = leg_to_compare[s.LEG_START_TIME_COL]
    leg_end_to_compare = leg_to_compare[s.LEG_END_TIME_COL]

    # Reduce the time range for short legs to avoid false positives (NaN evaluates to False)
    time_range = pd.Timedelta(minutes=5) if leg_to_find[s.LEG_DURATION_MINUTES_COL] > 5 and leg_to_compare[
        s.LEG_DURATION_MINUTES_COL] > 5 else pd.Timedelta(minutes=2)

    if pd.isnull([leg_begin_to_find, leg_end_to_find, leg_begin_to_compare, leg_end_to_compare]).any():
        return False

    begin_difference = abs(leg_begin_to_find - leg_begin_to_compare)
    end_difference = abs(leg_end_to_find - leg_end_to_compare)

    return (begin_difference <= time_range) and (end_difference <= time_range)


def check_mode(leg_to_find, leg_to_compare):
    """
    Check if the modes of two legs are compatible.
    Note: Adjusting the mode "car" to "ride" based on age is now its own function.
    :param leg_to_find:
    :param leg_to_compare:
    :return:
    """
    mode_to_find = leg_to_find[s.LEG_MAIN_MODE_COL]
    mode_to_compare = leg_to_compare[s.LEG_MAIN_MODE_COL]

    if mode_to_find == mode_to_compare and mode_to_find != s.MODE_UNDEFINED:  # Make sure we don't pair undefined modes
        return True

    mode_pairs = {(s.MODE_CAR, s.MODE_RIDE), (s.MODE_RIDE, s.MODE_CAR),
                  (s.MODE_WALK, s.MODE_BIKE), (s.MODE_BIKE, s.MODE_WALK)}
    if (mode_to_find, mode_to_compare) in mode_pairs:
        return True

    if s.MODE_UNDEFINED in [mode_to_find, mode_to_compare]:
        # Assuming if one mode is undefined and the other is car, they pair as ride
        # The mode is not updated here (in contrast to prev. work), because we don't know yet if the leg is connected.
        return s.MODE_CAR in [mode_to_find, mode_to_compare]

    return False


def check_activity(leg_to_find, leg_to_compare):  # TODO: Possibly create a matrix of compatible activities
    compatible_activities = {
        s.ACTIVITY_SHOPPING: [s.ACTIVITY_ERRANDS],
        s.ACTIVITY_ERRANDS: [s.ACTIVITY_SHOPPING, s.ACTIVITY_LEISURE],
        s.ACTIVITY_LEISURE: [s.ACTIVITY_ERRANDS, s.ACTIVITY_SHOPPING, s.ACTIVITY_MEETUP],
        s.ACTIVITY_MEETUP: [s.ACTIVITY_LEISURE]}

    type_to_find = leg_to_find[s.LEG_TO_ACTIVITY_COL]
    type_to_compare = leg_to_compare[s.LEG_TO_ACTIVITY_COL]

    if (type_to_find == type_to_compare or
            s.ACTIVITY_ACCOMPANY_ADULT in [type_to_find, type_to_compare] or
            s.ACTIVITY_PICK_UP_DROP_OFF in [type_to_find, type_to_compare]):
        return True
    elif s.ACTIVITY_UNSPECIFIED in [type_to_find, type_to_compare] or pd.isnull([type_to_find, type_to_compare]).any():
        logger.debug("Activity Type Undefined or Null (which usually means person has no legs).")
        return False
    # Assuming trip home (works, but not really plausible, thus commented out for now)
    # elif (type_to_find == s.ACTIVITY_HOME and type_to_compare != s.ACTIVITY_WORK) or \
    #         (type_to_compare == s.ACTIVITY_HOME and type_to_find != s.ACTIVITY_WORK):
    #     return True

    return type_to_compare in compatible_activities.get(type_to_find, [])


class Capacities:
    """
    Turns given capacities into point capacities that the locator can handle.
    - Loads data as either shp or csv (either points or cells shp must be given)
    - Translates between internal activity types and given capacities.
    - If cell capacities and shp point capacities are given, weighted-distributes the cell capacities to the points.
    - If cell capacities and shp points with possible activities are given, distributes the cell capacities to the points accordingly.
    - If cell capacities and shp raw points are given, distributes the cell capacities to the points evenly.
    - If cell capacities and csv point capacities are given, weighted-distributes; and creates random points.
    - If shp point capacities are given, uses those directly.

    The result are always located point capacities with the best possible information, that can be used by the activity placer.
    """

    def __init__(self, capa_cells_shp_path: str = None, capa_points_shp_path: str = None, capa_cells_csv_path: str = None,
                 capa_points_csv_path: str = None):

        logger.info("Initializing capacities...")
        if capa_points_shp_path is not None:
            if capa_cells_shp_path is not None:
                self.capa_points_gdf = gpd.read_file(capa_points_shp_path)
                self.capa_cells_gdf = gpd.read_file(capa_cells_shp_path)
                self.capa_points_gdf = distribute_by_weights(self.capa_points_gdf, self.capa_cells_gdf, 'cell_id')
            elif capa_cells_csv_path is not None:
                self.capa_points_gdf = gpd.read_file(capa_points_shp_path)
                self.capa_cells_df = read_csv(capa_cells_csv_path)
                self.capa_points_gdf = distribute_by_weights(self.capa_points_gdf, self.capa_cells_df, 'cell_id')
            else:
                self.capa_points_gdf = gpd.read_file(capa_points_shp_path)

        elif capa_cells_shp_path is not None:
            if capa_points_csv_path is not None:
                self.capa_cells_gdf = gpd.read_file(capa_cells_shp_path)
                self.capa_points_df = read_csv(capa_points_csv_path)
                self.capa_points_df['geometry'] = self.capa_points_df['geometry'].apply(shapely.wkt.loads)
                self.capa_points_gdf = gpd.GeoDataFrame(self.capa_points_df, geometry='geometry')
                self.capa_points_gdf.crs = self.capa_cells_gdf.crs
                self.capa_points_gdf = distribute_by_weights(self.capa_points_gdf, self.capa_cells_gdf, 'cell_id')
            else:
                self.capa_cells_gdf = gpd.read_file(capa_cells_shp_path)
                self.capa_points_gdf = self.capa_cells_gdf.copy()
                self.capa_points_gdf['geometry'] = self.capa_points_gdf['geometry'].apply(random_point_in_polygon)

        else:
            raise ValueError("Either capa_points_shp_path or capa_cells_shp_path must be given.")
        logger.info(f"Created capacity_gdf for {len(self.capa_points_gdf)} points.")

        self.translate_and_split_potentials()
        self.round_capacities()  # Round point capacities to integers

        logger.info("Initialized capacities.")

    def translate_and_split_potentials(self, translation_dict=None):
        logger.info("Translating capacities...")
        if translation_dict is None:
            translation_dict = {
                "Jobs Teilzeit": {"work": 1},
                "Jobs Vollzeit hohe Widerstandsempfindlichkeit": {"work": 1},
                "Jobs Vollzeit (AV+AF)": {"work": 1},
                "Einwohner": {"home": 0.99, "meetup": 0.01},
                "Zielpotenzial periodischer Bedarf, Einwohner": {"shopping": 0.8, "errands": 0.2},
                "Kitaplätze+Schulplätze+Ärzte+Einwohner": {"early_education": 0.5, "education": 0.5},
                "Berufsschulplätze": {"education": 1},
                "Zielpotenzial aperiodischer zentrenrelevanter Bedarf": {"business": 0.1, "shopping": 0.9},
                "Zielpotenzial aperiodischer Bedarf (EA+EB+EM)": {"shopping": 1},
                "Zielpotenzial Baumarkt, Kfz-Handel etc.": {"shopping": 1},
                "Zielpotenzial Möbelhäuser": {"shopping": 1},
                "Zielpotenzial periodischer Bedarf": {"shopping": 0.8, "errands": 0.2},
                "Zielpotenzial Natur": {"leisure": 1},
                "Zielpotenzial Gastronomie": {"leisure": 0.7, "meetup": 0.3},
                "Grundschulplätze": {"early_education": 1},
                "Studienplätze": {"education": 1},
                "Kitaplätze": {"daycare": 1},
                "Zielpotenzial Theater/Kino/Stadion": {"leisure": 1},
                "Ärzte, Krankenhäuser": {"errands": 1},
                "Zielpotenzial Post/Bank/Behörde/VHS": {"errands": 0.8, "business": 0.1, "lessons": 0.1},
                "Schulplätze SEK I": {"education": 1},
                "Schulplätze SEK II": {"education": 1},
                "Beschäftigte DL + Zielpotenzial Private Erledigungen": {"work": 0.5, "errands": 0.5},
                "Zielpotenzial Sport": {"sports": 1},
                "Schulplätze SEK I+II": {"education": 1},
                "SG_WSC": {"unspecified": 1}
            }

        for original_col, translations in translation_dict.items():
            for new_col, weight in translations.items():
                if new_col not in self.capa_points_gdf.columns:
                    self.capa_points_gdf[new_col] = np.nan
                self.capa_points_gdf[new_col] += self.capa_points_gdf[original_col].fillna(0) * weight
            self.capa_points_gdf.drop(columns=[original_col], inplace=True)

        logger.info("Translated capacities.")

    def round_capacities(self):
        """
        Round the capacities while preserving the sum.
        """
        logger.info("Rounding capacities to integers while preserving the sum...")
        for col in self.capa_points_gdf.columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(self.capa_points_gdf[col]):
                continue

            # Calculate the sum before rounding
            sum_before_rounding = self.capa_points_gdf[col].sum()

            # Round down the values in the column and keep track of the decimal part
            self.capa_points_gdf[col], decimal_part = divmod(self.capa_points_gdf[col], 1)

            # Calculate the difference between the sum before rounding and the sum after rounding
            diff = sum_before_rounding - self.capa_points_gdf[col].sum()

            # Adjust the rounded values to preserve the sum
            while diff > 0:
                # Find the index of the maximum decimal part
                idx_max_decimal = decimal_part.idxmax()
                # Add 1 to the corresponding value in the DataFrame
                self.capa_points_gdf.loc[idx_max_decimal, col] += 1
                # Subtract 1 from the corresponding value in the decimal part Series
                decimal_part.loc[idx_max_decimal] -= 1
                # Subtract 1 from the difference
                diff -= 1
        logger.info("Rounded capacities.")


def convert_to_list(s):
    """
    This is weirdly and annoyingly necessary because of the way the lists are stored.
    This is needed to correctly convert the string representation of a list to an actual list.
    """
    if pd.isna(s):
        return s
    try:
        # Replace unwanted characters
        cleaned_string = s.replace("\'", "\"")
        # Use json.loads to correctly interpret the string as a list of strings
        return json.loads(cleaned_string)
    except ValueError:
        return s
