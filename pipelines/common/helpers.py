#  Helper functions
import gzip
import os
import random
import re
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from shapely import Point

from utils import matsim_pipeline_setup as m
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


def read_csv(csv_path: str, test_col: str = None, use_cols: list = None):
    """
    Read a csv file with unknown separator and return a dataframe.
    :param csv_path: Path to csv file.
    :param test_col: Column name that should be present in the file.
    :param use_cols: List of columns to use from the file. Defaults to all columns.
    """
    try:
        df = pd.read_csv(csv_path, sep=',', usecols=use_cols)
        if test_col:
            test = df[test_col]
    except (KeyError, ValueError):
        logger.info(f"ID column '{test_col}' not found in {csv_path}, trying to read as ';' separated file...")
        df = pd.read_csv(csv_path, sep=';', usecols=use_cols)
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


def plot_column(df, column, title=None, xlabel=None, ylabel='Frequency', plot_type=None, figsize=(10, 6), save_name=None):
    """
    Plots a column from a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column (str): Column name to plot.
    title (str, optional): Title of the plot. Defaults to None, which will use the column name.
    xlabel (str, optional): Label for the x-axis. Defaults to None, which will use the column name.
    ylabel (str, optional): Label for the y-axis. Defaults to 'Frequency'.
    plot_type (str, optional): Type of plot (hist, bar, box, violin, strip, swarm, point). If None, the plot type is inferred from the column type. Defaults to None.
    figsize (tuple, optional): Size of the figure (width, height). Defaults to (10, 6).
    save_name (str, optional): Name with file extension to save the figure. If None, the figure is not saved. Defaults to None.
    """
    # Infer plot type if not specified
    if plot_type is None:
        if pd.api.types.is_numeric_dtype(df[column]):
            plot_type = 'hist'
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            plot_type = 'hist'
        else:
            plot_type = 'bar'

    # Set plot title and labels
    if title is None:
        title = f'Distribution of {column}'
    if xlabel is None:
        xlabel = column

    # Create the plot
    plt.figure(figsize=figsize)
    if plot_type == 'hist':
        sns.histplot(df[column].dropna(), kde=True)  # KDE for numeric and datetime
    elif plot_type == 'bar':
        sns.countplot(x=column, data=df)
    elif plot_type == 'box':
        sns.boxplot(x=column, data=df)
    elif plot_type == 'violin':
        sns.violinplot(x=column, data=df)
    elif plot_type == 'strip':
        sns.stripplot(x=column, data=df)
    elif plot_type == 'swarm':
        sns.swarmplot(x=column, data=df)
    elif plot_type == 'point':
        sns.pointplot(x=column, data=df)
    else:
        raise ValueError("Unsupported plot type. Use 'hist' or 'bar'.")

    # Set title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Save the plot if a save path is provided
    if save_name:
        save_name = os.path.join(m.OUTPUT_DIR, save_name)
        plt.savefig(save_name, bbox_inches='tight')

    else:
        plt.show()


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
            (self.slack_factors_df['activity_from'] == activity_from) &
            (self.slack_factors_df['activity_via'] == activity_via) &
            (self.slack_factors_df['activity_to'] == activity_to)
            ]

        if not slack_factor_row.empty:
            slack_factor: float = slack_factor_row['slack_factor'].iloc[0]
        else:
            # Fallback to a default slack factor if not found
            logger.debug(f"No slack factor found for activities: {activity_from}, {activity_via}, {activity_to}. "
                         f"Using default slack factor of {s.DEFAULT_SLACK_FACTOR}")
            slack_factor = s.DEFAULT_SLACK_FACTOR

        return slack_factor

    def calculate_expected_time_with_slack(self, time_from_start_to_via: float, time_from_via_to_end: float, activity_from: str,
                                           activity_via: str, activity_to: str) -> float:
        expected_time: float = ((time_from_start_to_via + time_from_via_to_end) *
                                self.get_slack_factor(activity_from, activity_via, activity_to))
        return expected_time

    def calculate_expected_distance_with_slack(self, distance_from_start_to_via: float, distance_from_via_to_end: float,
                                               activity_from: str, activity_via: str, activity_to: str) -> float:
        """
        Identical to calculate_expected_time_with_slack, but different parameter names for clarity.
        """
        expected_distance: float = ((distance_from_start_to_via + distance_from_via_to_end) *
                                    self.get_slack_factor(activity_from, activity_via, activity_to))
        return expected_distance

    def get_all_times_with_slack(self, leg_chain, level=0):
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
        len_times_col = len(leg_chain[times_col].notna())
        if len_times_col == 0:
            logger.warning(f"Received empty DataFrame, returning unchanged.")
            return leg_chain, level
        elif len_times_col == 1:
            logger.debug(f"Received single leg, returning unchanged.")
            return leg_chain, level
        elif len_times_col == 2:
            logger.debug(f"Two legs remain to estimate, solving last level.")
            return self.solve_level(leg_chain, level), level
        # Recursive case
        else:
            level += 1
            leg_chain = self.solve_level(leg_chain, level)
            return self.get_all_times_with_slack(leg_chain, level)

    def solve_level(self, leg_chain, level):  # TODO: finish
        """
        Adds a column for given level of slack factor calculation, containing the time with slack of all legs up to and
        including the entry. For better performance, keep the number of columns to a minimum.
        :param leg_chain:
        :param level:
        :return:
        """
        if level == 0:
            times_col = s.LEG_DURATION_MINUTES_COL
            if len(leg_chain[times_col].notna()) != len(leg_chain):
                logger.warning(f"Found NaN values in {times_col}, may produce incorrect results.")
        else:
            times_col = f"level_{level - 1}"  # Here we expect some NaN values

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

            leg_chain.loc[group['original_index'].iloc[-1], f"level_{level}"] = time
        # Remove the temporary pair_id column
        leg_chain.drop(columns='pair_id', inplace=True)

        return leg_chain


class TTMatrices:
    """
    Manages travel time matrices for various modes of transportation.
    """

    def __init__(self, car_tt_matrices_csv_paths: list, pt_tt_matrices_csv_paths: list, bike_tt_matrix_csv_path: str,
                 walk_tt_matrix_csv_path: str):
        self.tt_matrices = {'car': {}, 'pt': {}, 'bike': None, 'walk': None}

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
