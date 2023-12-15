import random

import geopandas as gpd
import numpy as np
import pandas as pd

from pipelines.common import helpers as h
from utils import settings_values as s
from utils.logger import logging

logger = logging.getLogger(__name__)


class ActivityLocator:
    """

    Normalizing the potentials according to the total main activity demand means that persons will be assigned to the activity
    locations exactly proportional to the location potentials. This also means that secondary activities will have to make do
    with the remaining capacity.
    :param persons: GeoDataFrame or DataFrame with persons, their home location point, target travel times and activities (LEGS = ROWS!!)
    :param capacities: GeoDataFrame, DataFrame or Path to .shp with activity location points and their capacities
    :param cells_shp_path: Path to a shapefile with cells
    :param tt_matrix_csv_path: Path to a csv file with travel times between cells
    :param persons_crs: CRS of the persons data (if given as a DataFrame)
    :param capacities_crs: CRS of the capacities data (if given as a DataFrame)
    :param persons_geometry_col: Name of the geometry column in the persons data (if given as a DataFrame)
    :param capacities_geometry_col: Name of the geometry column in the capacities data (if given as a DataFrame)
    """

    def __init__(self, persons, capacities, cells_shp_path, tt_matrix_csv_path, slack_factors_csv_path, target_crs="EPSG:25832",
                 persons_crs=None, capacities_crs=None,
                 persons_geometry_col=None, capacities_geometry_col=None):

        self.persons_gdf = self.load_data_into_gdf(persons, persons_geometry_col, persons_crs)
        self.capacity_points_gdf = self.load_data_into_gdf(capacities, capacities_geometry_col, capacities_crs)
        self.cells_gdf: gpd.GeoDataFrame = self.load_data_into_gdf(cells_shp_path)
        self.tt_matrix_df = pd.read_csv(tt_matrix_csv_path)
        self.slack_factors_df = pd.read_csv(slack_factors_csv_path)

        self.target_crs = target_crs
        self.capacity_cells_df = None
        self.located_main_activities_for_current_population = False

        self.perform_integrity_checks()
        self.match_crs(self.target_crs)

    @staticmethod
    def create_geodf_from_df(df, geo_col, crs):
        gdf = gpd.GeoDataFrame(df.drop(columns=[geo_col]), geometry=geo_col)
        gdf.crs = crs
        return gdf

    def load_data_into_gdf(self, data, geometry_col=None, crs=None):
        if isinstance(data, gpd.GeoDataFrame):
            logger.info(f"Loading GeoDataFrame with {len(data)} rows...")
            return data
        elif isinstance(data, pd.DataFrame):
            logger.info(f"Loading DataFrame with {len(data)} rows into GeoDataFrame...")
            return self.create_geodf_from_df(data, geometry_col, crs)
        elif isinstance(data, str):  # Assuming it's a file path to a shapefile
            logger.info(f"Loading shapefile from {data} into GeoDataFrame...")
            assert data.endswith(".shp"), "File path must point to a shapefile"
            return gpd.read_file(data)
        else:
            raise ValueError("Data must be a GeoDataFrame, DataFrame or a file path to a shapefile")

    def perform_integrity_checks(self):
        # Check for null geometries
        if self.persons_gdf.geometry.isnull().any():
            raise ValueError("Null geometries found in persons data")
        if self.capacity_points_gdf.geometry.isnull().any():
            raise ValueError("Null geometries found in capacities data")
        if self.cells_gdf.geometry.isnull().any():
            raise ValueError("Null geometries found in cells data")

    def match_crs(self, common_crs):
        if self.persons_gdf.crs != common_crs:
            logger.info(f"Matching CRS of persons data from {self.persons_gdf.crs} to {common_crs}...")
            self.persons_gdf = self.persons_gdf.to_crs(common_crs)
        if self.capacity_points_gdf.crs != common_crs:
            logger.info(f"Matching CRS of capacities data from {self.capacity_points_gdf.crs} to {common_crs}...")
            self.capacity_points_gdf = self.capacity_points_gdf.to_crs(common_crs)
        if self.cells_gdf.crs != common_crs:
            logger.info(f"Matching CRS of cells data from {self.cells_gdf.crs} to {common_crs}...")
            self.cells_gdf = self.cells_gdf.to_crs(common_crs)

    def assign_cells_to_persons(self):
        # Perform spatial join to find the cell each person is in
        persons_with_cells = gpd.sjoin(self.persons_gdf, self.cells_gdf, how="left", op="within").dropna(
            subset=[s.TT_MATRIX_CELL_ID_COL])

        # Check if there are persons without a cell
        missing_cells_count = len(self.persons_gdf) - len(persons_with_cells)
        if missing_cells_count > 0:
            logger.warning(f"{missing_cells_count} persons without a cell. They will be ignored.")

        self.persons_gdf = persons_with_cells

    def assign_cells_to_capacities(self):
        # Perform spatial join to find the cell each capacity point is in
        capacities_with_cells = gpd.sjoin(self.capacity_points_gdf, self.cells_gdf, how="left", op="within").dropna(
            subset=[s.TT_MATRIX_CELL_ID_COL])

        # Check if there are capacities without a cell
        missing_cells_count = len(self.capacity_points_gdf) - len(capacities_with_cells)
        if missing_cells_count > 0:
            logger.warning(f"{missing_cells_count} capacities without a cell. They will be ignored.")

        self.capacity_points_gdf = capacities_with_cells

    def aggregate_supply_to_cells(self):
        # Aggregate capacities per cell and type
        self.capacity_cells_df = self.capacity_points_gdf.groupby([s.TT_MATRIX_CELL_ID_COL, "activity_type"]).agg(
            capacities=('capacity_column_name', 'sum')).reset_index()

    def get_global_supply(self) -> pd.Series:

        # Global aggregation
        global_supply = self.capacity_cells_df.groupby("activity_type").sum()["capacities"]

        return global_supply

    def get_global_demand(self) -> pd.Series:
        # Aggregate persons per cell and activity type
        persons_per_cell_type = self.persons_gdf.groupby([s.TT_MATRIX_CELL_ID_COL, s.LEG_TO_ACTIVITY_COL]).size().reset_index(
            name="persons")

        # Global aggregation
        global_demand = persons_per_cell_type.groupby("activity_type").sum()["persons"]

        return global_demand

    def normalize_capacities(self):
        # Calculate total supply for each activity type
        total_supply_by_activity: pd.Series = self.get_global_supply()
        total_demand_by_activity: pd.Series = self.get_global_demand()

        # Calculate normalization factors as a Series
        normalization_factors = total_demand_by_activity / total_supply_by_activity
        normalization_factors = normalization_factors.fillna(0).replace([np.inf, -np.inf], 0)

        # Apply normalization
        self.capacity_cells_df['normalized_capacities'] = self.capacity_cells_df.apply(
            lambda row: row['capacities'] * normalization_factors.get(row['activity_type'], 0), axis=1)

        return self.capacity_cells_df

    def assign_persons_to_cells(self):

        n = s.N_CLOSEST_CELLS
        for _, cell in self.cells_gdf.iterrows():
            capacity_updates = {}
            persons_in_cell = self.persons_gdf[self.persons_gdf["cell_id"] == cell['cell_id']].groupby(
                s.LEG_DURATION_MINUTES_COL)
            cell_travel_times = self.tt_matrix_df[self.tt_matrix_df['FROM'] == cell['cell_id']]

            for target_time, group in persons_in_cell:
                candidates = self.get_n_closest_cells(cell_travel_times, target_time, n)

                for _, person in group.iterrows():
                    target_activity = person[s.LEG_TO_ACTIVITY_COL]
                    # Evaluate and assign - randomly but weighted by remaining capacity
                    target_cell = self.weighted_random_choice(candidates, target_activity)
                    # Accumulate changes instead of updating immediately
                    if (target_cell, target_activity) not in capacity_updates:
                        capacity_updates[(target_cell, target_activity)] = 1
                    else:
                        capacity_updates[(target_cell, target_activity)] += 1
            # Updating once per cell is faster than updating once per person and should be sufficient
            for (cell_id, activity_type), count in capacity_updates.items():
                self.update_capacity(cell_id, activity_type, count)

    # TODO: assignment function multiprocessed, using weighted by overall capacity, not remaining capacity

    def weighted_random_choice(self, candidates, activity_type):
        weights = [self.get_remaining_capacity(cell, activity_type) for cell in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]

    def get_remaining_capacity(self, cell, activity_type):
        return self.capacity_points_gdf[(self.capacity_points_gdf['cell_id'] == cell) &
                                        (self.capacity_points_gdf['activity_type'] == activity_type)]['total_capacity'].iloc[0]

    def get_n_closest_cells(self, cell_travel_times, target_time, n):
        time_diffs = np.abs(cell_travel_times['travel_time'].values - target_time)
        closest_indices = np.argpartition(time_diffs, n)[:n]

        return cell_travel_times.iloc[closest_indices]

    def update_capacity(self, cell_id, activity_type, count):
        self.capacity_points_gdf.loc[(self.capacity_points_gdf['cell_id'] == cell_id) &
                                     (self.capacity_points_gdf['activity_type'] == activity_type), 'total_capacity'] -= count

    def distribute_persons_to_capacity_points(self):
        h.distribute_by_weights(self.persons_gdf, self.capacity_points_gdf, "cell_ids")
        """
        Sub-distributes persons to individual capacity points within the same cell.
        :return: 
        """
        pass

    def locate_main_activities(self):
        self.assign_cells_to_persons()
        self.assign_cells_to_capacities()
        self.aggregate_supply_to_cells()
        self.normalize_capacities()

        self.assign_persons_to_cells()
        self.distribute_persons_to_capacity_points()

        return self.persons_gdf

    def locate_secondary_activities(self):
        """
        Locate activity chains between two known activity locations.

        :return:
        """

    def replace_population(self, replace_with, replace_with_crs=None, replace_with_geometry_col=None):
        """
        Replaces the population with a new one while keeping all other current data.
        Useful to stepwise assign different population groups to the same activity locations.
        :param replace_with:
        :param replace_with_crs:
        :param replace_with_geometry_col:
        :return:
        """
        self.persons_gdf = self.load_data_into_gdf(replace_with, replace_with_geometry_col, replace_with_crs)
        self.assign_cells_to_persons()
        self.match_crs(self.target_crs)

    # TODO: keep track of which persons have gotten either primary or secondary activities assigned
    # TODO:

    def calculate_expected_time_with_slack(self, time_from_start_to_via, time_from_via_to_end, activity_from, activity_via,
                                           activity_to):
        """
        Calculate the expected travel time for two legs of a trip, adjusted by a slack factor based on activities.

        :param time_from_start_to_via: Travel time from the start to the via activity.
        :param time_from_via_to_end: Travel time from the via to the end activity.
        :param activity_from: The starting activity.
        :param activity_via: The intermediate (via) activity.
        :param activity_to: The ending activity.
        :return: Total adjusted travel time as a float.
        """

        # Retrieve the slack factor based on activities
        slack_factor_row = self.slack_factors_df.loc[
            (self.slack_factors_df['activity_from'] == activity_from) &
            (self.slack_factors_df['activity_via'] == activity_via) &
            (self.slack_factors_df['activity_to'] == activity_to)
            ]

        if not slack_factor_row.empty:
            slack_factor = slack_factor_row['slack_factor'].iloc[0]
        else:
            # Fall back to a default slack factor if not found
            logger.debug(f"No slack factor found for activities: {activity_from}, {activity_via}, {activity_to}. "
                         f"Using default slack factor of {s.DEFAULT_SLACK_FACTOR}")
            slack_factor = s.DEFAULT_SLACK_FACTOR

        # Apply the slack factor to the sum of both leg times
        expected_time = (time_from_start_to_via + time_from_via_to_end) * slack_factor
        return expected_time

    def locate_single_activity(self, legs_df, min_tolerance, max_tolerance):
        """
        Locate a single activity between two known places using travel time matrix and capacity data.
        :param legs_df: DataFrame with the legs of a trip, including the activity to be located.
        :param min_tolerance: Minimum tolerance in minutes.
        :param max_tolerance: Maximum tolerance in minutes.
        :return: Cell ID of the best-suited location for the activity, or None if not found.
        """
        start_cell = legs_df.iloc[0]['cell_id']
        end_cell = legs_df.iloc[-1]['cell_id']
        activity_type = legs_df.iloc[1]['activity_type']

        time_start_to_act = legs_df.iloc[1]['desired_time_from_start']
        time_act_to_end = legs_df.iloc[1]['desired_time_to_end']

        potential_cells = None
        step_size = (max_tolerance-min_tolerance)/5
        tolerance = min_tolerance
        while potential_cells is None and tolerance <= max_tolerance:

            # Filter cells based on travel time criteria (times in minutes)
            potential_cells_start = self.tt_matrix_df[(self.tt_matrix_df['from_cell'] == start_cell) &
                                                      (self.tt_matrix_df['time'] >= time_start_to_act - tolerance) &
                                                      (self.tt_matrix_df['time'] <= time_start_to_act + tolerance)]

            potential_cells_end = self.tt_matrix_df[(self.tt_matrix_df['to_cell'] == end_cell) &
                                                    (self.tt_matrix_df['time'] >= time_act_to_end - tolerance) &
                                                    (self.tt_matrix_df['time'] <= time_act_to_end + tolerance)]

            # Find intersecting cells from both sets
            potential_cells = set(potential_cells_start['to_cell']).intersection(set(potential_cells_end['from_cell']))

            tolerance += step_size

        # Choose the cell with the highest capacity for the activity type
        best_cell = self.capacity_points_gdf[self.capacity_points_gdf['cell_id'].isin(potential_cells) &
                                             (self.capacity_points_gdf['activity_type'] == activity_type)].nlargest(1,
                                                                                                                    'capacity')

        return best_cell['cell_id'].iloc[0] if not best_cell.empty else None

    def locate_sec_activities(self, legs_to_locate, tolerance):  # TODO: this is unfinished
        """
        Locate a series of activities given the known start and end positions.

        :param legs_to_locate: DataFrame with legs of a trip, including known start and end positions.
        :param tolerance: Tolerance for travel time deviation.
        :return: Dictionary with cell IDs for each located activity.
        """
        located_activities = {}
        total_legs = len(legs_to_locate)

        if total_legs == 2:  # Directly use locate_single_activity for one intermediate leg
            activity_cell_id = self.locate_single_activity(legs_to_locate, tolerance)
            activity_type = legs_to_locate.iloc[1]['activity_type']
            located_activities[activity_type] = activity_cell_id
        else:
            # Break down into overlapping two-leg pairs and locate activities iteratively
            for i in range(1, total_legs - 1):
                two_leg_pair = legs_to_locate.iloc[i - 1: i + 2]  # Get two-leg pair
                expected_time_with_slack = self.calculate_expected_time_with_slack(two_leg_pair)

                # Locate the activity for the current two-leg pair
                activity_cell_id = self.locate_single_activity(two_leg_pair, expected_time_with_slack, tolerance)
                activity_type = two_leg_pair.iloc[1]['activity_type']

                if activity_cell_id:
                    located_activities[activity_type] = activity_cell_id

        return located_activities


