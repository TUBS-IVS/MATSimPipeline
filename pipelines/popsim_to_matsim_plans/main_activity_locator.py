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
    :param persons_crs: CRS of the persons data (if given as a DataFrame)
    :param capacities_crs: CRS of the capacities data (if given as a DataFrame)
    :param persons_geometry_col: Name of the geometry column in the persons data (if given as a DataFrame)
    :param capacities_geometry_col: Name of the geometry column in the capacities data (if given as a DataFrame)
    """

    def __init__(self, persons, capacities, cells_shp_path, target_crs="EPSG:25832",
                 persons_crs=None, capacities_crs=None,
                 persons_geometry_col=None, capacities_geometry_col=None):

        self.persons_gdf = self.load_data_into_gdf(persons, persons_geometry_col, persons_crs)
        self.capacity_points_gdf = self.load_data_into_gdf(capacities, capacities_geometry_col, capacities_crs)
        self.cells_gdf: gpd.GeoDataFrame = self.load_data_into_gdf(cells_shp_path)

        self.tt = h.TTMatrices(s.TT_MATRIX_CAR_FILES, s.TT_MATRIX_PT_FILES, s.TT_MATRIX_BIKE_FILE, s.TT_MATRIX_WALK_FILE)
        self.sf = h.SlackFactors(s.SLACK_FACTORS_FILE)

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

    # def combine_tt_matrices(self, df_car, df_pt):
    #     df_car = df_car.rename(columns={'VALUE': 'time_car'})
    #     df_pt = df_pt.rename(columns={'VALUE': 'time_pt'})
    #     combined_df = pd.merge(df_car, df_pt, on=['FROM', 'TO'], how='outer')
    #     return combined_df

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

    def locate_main_activity_cell(self):  # TODO: finish (3+h!)
        """
        Locates the main activity cell for each person, based on the travel time matrices,
        the normalized capacities, the desired activity type, and the desired travel time.
        :return:
        """
        n = s.N_CLOSEST_CELLS
        # Prepare a list to collect the target cells for each person
        person_target_cells = []

        # Even if it's annoying, itertuples is much faster than iterrows
        for _, cell in self.cells_gdf.itertuples(index=False):
            capacity_updates = {}

            # Filter for only the columns we need
            filtered_persons = self.persons_gdf[[s.LEG_MAIN_MODE_COL, s.HOUR_COL, s.LEG_DURATION_MINUTES_COL,]]

            # For each person, we need to find candidates based on time and mode (for the right tt matrix)
            # Find time deviations and potentials of the specific activity type for each candidate
            # Calculate the likelihoods for each candidate and weighted choose
            # Assign that cell to both main act and mirrored main. Distributor later also has to remember mirrored main.
            # At some point (end of loop?), update the capacities of chosen cells.

            # Group by mode, hour, and duration
            persons_in_cell = filtered_persons.groupby([s.LEG_MAIN_MODE_COL, s.HOUR_COL, s.LEG_DURATION_MINUTES_COL])

            for (mode, hour, target_time), group in persons_in_cell:
                tt_matrix = self.tt.get_tt_matrix(mode, hour)
                cell_travel_times = tt_matrix[tt_matrix['from_cell'] == getattr(cell, 'cell_id')]

                candidates = self.get_n_closest_cells(cell_travel_times, target_time, n)
                candidate_potentials = self.capacity_cells_df[self.capacity_cells_df['cell_id'].isin(candidates['to_cell'])]  # TODO: check
                time_deviations = np.abs(candidates['travel_time'].values - target_time)  # TODO check

                if candidate_potentials['normalized_capacities'].min() < 1:
                    pass
                for person in group.itertuples(index=False):
                    target_activity = getattr(person, s.LEG_TO_ACTIVITY_COL)
                    target_cell = self.weighted_random_choice(candidates, target_activity)

                    # Collect the target cells for each person
                    person_target_cells.append((getattr(person, 'person_id'), target_cell))

                    if (target_cell, target_activity) not in capacity_updates:
                        capacity_updates[(target_cell, target_activity)] = 1
                    else:
                        capacity_updates[(target_cell, target_activity)] += 1

            for (cell_id, activity_type), count in capacity_updates.items():
                self.update_capacity(cell_id, activity_type, count)

        # Create a DataFrame from the collected target cells and write to main df
        target_cells_df = pd.DataFrame(person_target_cells, columns=['person_id', 'target_cell'])
        self.persons_gdf = self.persons_gdf.merge(target_cells_df, on='person_id', how='left')

    # TODO: assignment function multiprocessed, using weighted by overall capacity, not remaining capacity
    @staticmethod
    def calculate_cell_likelihoods(potentials, time_diffs=None):
        """
        Calculates cell likelihoods, linear on potentials and time differentials using a sigmoid function.

        :param potentials: List or array of potential values for each candidate.
        :param time_diffs: List or array of time differential values for each candidate.
        :return: Array of likelihoods for each candidate.
        """
        if time_diffs is not None:
            sigmoid_values = h.sigmoid(time_diffs, s.SIGMOID_BETA, s.SIGMOID_DELTA_T)
            combined_factors = np.multiply(potentials, sigmoid_values)
        else:
            combined_factors = potentials

        # Normalize to ensure the sum of likelihoods is 1
        likelihoods = combined_factors / np.sum(combined_factors)
        return likelihoods

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

        self.locate_main_activity_cell()
        self.distribute_persons_to_capacity_points()

        return self.persons_gdf

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

    def locate_single_activity(self, start_cell, end_cell, activity_type, time_start_to_act, time_act_to_end,
                               mode, start_hour,
                               min_tolerance=None, max_tolerance=None):
        """
        Locate a single activity between two known places using travel time matrix and capacity data.
        When the maximum tolerance is exceeded, the tolerance is increased rapidly to find any viable location,
        with a logged warning. If this still fails, a random cell is returned to keep the pipeline running,
        with a logged error.
        :param start_cell: Cell ID of the starting location.
        :param end_cell: Cell ID of the ending location.
        :param activity_type: Activity type of the activity to locate.
        :param time_start_to_act: Travel time from the start to the activity in minutes.
        :param time_act_to_end: Travel time from the activity to the end in minutes.
        :param mode: Mode of travel.
        :param start_hour: Hour of the day when the leg starts.
        :param min_tolerance: Minimum tolerance in minutes.
        :param max_tolerance: Maximum tolerance in minutes.
        :return: Cell ID of the best-suited location for the activity.
        """
        if min_tolerance is None:
            min_tolerance = 2
        if max_tolerance is None:
            max_tolerance = 15

        step_size = max((max_tolerance - min_tolerance) / 5, 2)
        tolerance = min_tolerance
        exceed_max_tolerance_turns = 5
        while True:
            tt_matrix = self.tt.get_tt_matrix(mode, start_hour)

            # Filter cells based on travel time criteria (times in minutes)
            potential_cells_start = tt_matrix[(tt_matrix['from_cell'] == start_cell) &
                                              (tt_matrix['time'] >= time_start_to_act - tolerance) &
                                              (tt_matrix['time'] <= time_start_to_act + tolerance)]

            potential_cells_end = tt_matrix[(tt_matrix['to_cell'] == end_cell) &
                                            (tt_matrix['time'] >= time_act_to_end - tolerance) &
                                            (tt_matrix['time'] <= time_act_to_end + tolerance)]

            # Find intersecting cells from both sets
            potential_cells = set(potential_cells_start['to_cell']).intersection(set(potential_cells_end['from_cell']))
            if potential_cells:
                break
            if tolerance <= max_tolerance:
                tolerance += step_size
            else:
                # If max tolerance is exceeded, go mad to find a cell
                tolerance += (step_size * 5)
                exceed_max_tolerance_turns -= 1
                logger.warning(
                    f"Exceeding maximum tolerance of {max_tolerance} minutes to find viable location. Tolerance is now {tolerance} minutes.")
                if exceed_max_tolerance_turns == 0:
                    logger.error(
                        f"No cell found for activity type {activity_type} between cells {start_cell} and {end_cell}."
                        f"Returning random cell to keep the pipeline running.")
                    return random.choice(self.capacity_points_gdf['cell_id'].unique())

        # Choose the cell with the highest capacity for the activity type
        best_cell = self.capacity_points_gdf[self.capacity_points_gdf['cell_id'].isin(potential_cells) &
                                             (self.capacity_points_gdf['activity_type'] == activity_type)].nlargest(1,
                                                                                                                    'capacity')
        if best_cell.empty:  # For whatever reason
            logger.error(f"No cell found for activity type {activity_type} between cells {start_cell} and {end_cell}."
                         f"Returning random cell to keep the pipeline running.")
            return random.choice(self.capacity_points_gdf['cell_id'].unique())
        return best_cell['cell_id'].iloc[0] if not best_cell.empty else None

    def locate_sec_chain_long(self, legs_to_locate):  # TODO: refactor as recursive functions
        num_legs = len(legs_to_locate)
        if num_legs == 2:
            # Locate the activity between the two known locations
            self.locate_single_activity(legs_to_locate.iloc[0]['cell_from'],
                                        legs_to_locate.iloc[1]['cell_to'],
                                        legs_to_locate.iloc[0][s.LEG_TO_ACTIVITY_COL],
                                        legs_to_locate.iloc[0][s.LEG_DURATION_MINUTES_COL],
                                        legs_to_locate.iloc[1][s.LEG_DURATION_MINUTES_COL],
                                        1, 10)
        elif num_legs == 3:
            # Calc time from start to second activity
            time_Start_B = self.sf.calculate_expected_time_with_slack(legs_to_locate.iloc[0][s.LEG_DURATION_MINUTES_COL],
                                                                   legs_to_locate.iloc[1][s.LEG_DURATION_MINUTES_COL],
                                                                   legs_to_locate.iloc[0][s.LEG_FROM_ACTIVITY_COL],
                                                                   legs_to_locate.iloc[0][s.LEG_TO_ACTIVITY_COL],
                                                                   legs_to_locate.iloc[1][s.LEG_TO_ACTIVITY_COL])

            # Locate second activity
            cell_B = self.locate_single_activity(legs_to_locate.iloc[0]['cell_from'],
                                                 legs_to_locate.iloc[2]['cell_to'],
                                                 legs_to_locate.iloc[1][s.LEG_TO_ACTIVITY_COL],
                                                 time_Start_B,
                                                 legs_to_locate.iloc[2][s.LEG_DURATION_MINUTES_COL])

            # Locate first activity
            cell_A = self.locate_single_activity(legs_to_locate.iloc[0]['cell_from'],
                                                 cell_B,
                                                 legs_to_locate.iloc[0][s.LEG_TO_ACTIVITY_COL],
                                                 legs_to_locate.iloc[0][s.LEG_DURATION_MINUTES_COL],
                                                 legs_to_locate.iloc[1][s.LEG_DURATION_MINUTES_COL])

            # Save cells
            legs_to_locate.iloc[0]['cell_to'] = cell_A
            legs_to_locate.iloc[1]['cell_from'] = cell_A
            legs_to_locate.iloc[1]['cell_to'] = cell_B
            legs_to_locate.iloc[2]['cell_from'] = cell_B

        elif num_legs == 4:
            # Calc time from start to second activity
            time_Start_B = self.sf.calculate_expected_time_with_slack(legs_to_locate.iloc[0][s.LEG_DURATION_MINUTES_COL],
                                                                   legs_to_locate.iloc[1][s.LEG_DURATION_MINUTES_COL],
                                                                   legs_to_locate.iloc[0][s.LEG_FROM_ACTIVITY_COL],
                                                                   legs_to_locate.iloc[0][s.LEG_TO_ACTIVITY_COL],
                                                                   legs_to_locate.iloc[1][s.LEG_TO_ACTIVITY_COL])
            # Calc time from second activity to end
            time_B_End = self.sf.calculate_expected_time_with_slack(legs_to_locate.iloc[2][s.LEG_DURATION_MINUTES_COL],
                                                                 legs_to_locate.iloc[3][s.LEG_DURATION_MINUTES_COL],
                                                                 legs_to_locate.iloc[2][s.LEG_FROM_ACTIVITY_COL],
                                                                 legs_to_locate.iloc[2][s.LEG_TO_ACTIVITY_COL],
                                                                 legs_to_locate.iloc[3][s.LEG_TO_ACTIVITY_COL])

            # Locate second activity
            cell_B = self.locate_single_activity(legs_to_locate.iloc[0]['cell_from'],
                                                 legs_to_locate.iloc[2]['cell_to'],
                                                 legs_to_locate.iloc[1][s.LEG_TO_ACTIVITY_COL],
                                                 time_Start_B,
                                                 time_B_End)

            # Locate first activity
            cell_A = self.locate_single_activity(legs_to_locate.iloc[0]['cell_from'],
                                                 cell_B,
                                                 legs_to_locate.iloc[0][s.LEG_TO_ACTIVITY_COL],
                                                 legs_to_locate.iloc[0][s.LEG_DURATION_MINUTES_COL],
                                                 legs_to_locate.iloc[1][s.LEG_DURATION_MINUTES_COL])

            # Locate third activity
            cell_C = self.locate_single_activity(cell_B,
                                                 legs_to_locate.iloc[3]['cell_to'],
                                                 legs_to_locate.iloc[2][s.LEG_TO_ACTIVITY_COL],
                                                 legs_to_locate.iloc[2][s.LEG_DURATION_MINUTES_COL],
                                                 legs_to_locate.iloc[3][s.LEG_DURATION_MINUTES_COL])

            # Save cells
            legs_to_locate.iloc[0]['cell_to'] = cell_A
            legs_to_locate.iloc[1]['cell_from'] = cell_A
            legs_to_locate.iloc[1]['cell_to'] = cell_B
            legs_to_locate.iloc[2]['cell_from'] = cell_B
            legs_to_locate.iloc[2]['cell_to'] = cell_C
            legs_to_locate.iloc[3]['cell_from'] = cell_C

        elif num_legs == 5:
            # Calc time from start to second activity
            time_Start_B = self.sf.calculate_expected_time_with_slack(legs_to_locate.iloc[0][s.LEG_DURATION_MINUTES_COL],
                                                                   legs_to_locate.iloc[1][s.LEG_DURATION_MINUTES_COL],
                                                                   legs_to_locate.iloc[0][s.LEG_FROM_ACTIVITY_COL],
                                                                   legs_to_locate.iloc[0][s.LEG_TO_ACTIVITY_COL],
                                                                   legs_to_locate.iloc[1][s.LEG_TO_ACTIVITY_COL])
            # Calc time from second activity to fourth activity
            time_B_D = self.sf.calculate_expected_time_with_slack(legs_to_locate.iloc[2][s.LEG_DURATION_MINUTES_COL],
                                                               legs_to_locate.iloc[3][s.LEG_DURATION_MINUTES_COL],
                                                               legs_to_locate.iloc[2][s.LEG_FROM_ACTIVITY_COL],
                                                               legs_to_locate.iloc[2][s.LEG_TO_ACTIVITY_COL],
                                                               legs_to_locate.iloc[3][s.LEG_TO_ACTIVITY_COL])

            # Calc time from start to fourth activity
            time_Start_D = self.sf.calculate_expected_time_with_slack(time_Start_B,
                                                                   time_B_D,
                                                                   legs_to_locate.iloc[0][s.LEG_FROM_ACTIVITY_COL],
                                                                   legs_to_locate.iloc[1][s.LEG_TO_ACTIVITY_COL],
                                                                   legs_to_locate.iloc[3][s.LEG_TO_ACTIVITY_COL])

            # Locate fourth activity
            cell_D = self.locate_single_activity(legs_to_locate.iloc[0]['cell_from'],
                                                 legs_to_locate.iloc[4]['cell_to'],
                                                 legs_to_locate.iloc[3][s.LEG_TO_ACTIVITY_COL],
                                                 time_Start_D,
                                                 legs_to_locate.iloc[4][s.LEG_DURATION_MINUTES_COL])

            # Locate second activity
            cell_B = self.locate_single_activity(legs_to_locate.iloc[0]['cell_from'],
                                                 cell_D,
                                                 legs_to_locate.iloc[1][s.LEG_TO_ACTIVITY_COL],
                                                 time_Start_B,
                                                 time_B_D)

            # Locate first activity
            cell_A = self.locate_single_activity(legs_to_locate.iloc[0]['cell_from'],
                                                 cell_B,
                                                 legs_to_locate.iloc[0][s.LEG_TO_ACTIVITY_COL],
                                                 legs_to_locate.iloc[0][s.LEG_DURATION_MINUTES_COL],
                                                 legs_to_locate.iloc[1][s.LEG_DURATION_MINUTES_COL])

            # Locate third activity
            cell_C = self.locate_single_activity(cell_B,
                                                 cell_D,
                                                 legs_to_locate.iloc[2][s.LEG_TO_ACTIVITY_COL],
                                                 legs_to_locate.iloc[2][s.LEG_DURATION_MINUTES_COL],
                                                 legs_to_locate.iloc[3][s.LEG_DURATION_MINUTES_COL])

            # Save cells
            legs_to_locate.iloc[0]['cell_to'] = cell_A
            legs_to_locate.iloc[1]['cell_from'] = cell_A
            legs_to_locate.iloc[1]['cell_to'] = cell_B
            legs_to_locate.iloc[2]['cell_from'] = cell_B
            legs_to_locate.iloc[2]['cell_to'] = cell_C
            legs_to_locate.iloc[3]['cell_from'] = cell_C
            legs_to_locate.iloc[3]['cell_to'] = cell_D
            legs_to_locate.iloc[4]['cell_from'] = cell_D


        elif num_legs == 6:
            # Calc time from start to second activity
            time_Start_B = self.sf.calculate_expected_time_with_slack(legs_to_locate.iloc[0][s.LEG_DURATION_MINUTES_COL],
                                                                   legs_to_locate.iloc[1][s.LEG_DURATION_MINUTES_COL],
                                                                   legs_to_locate.iloc[0][s.LEG_FROM_ACTIVITY_COL],
                                                                   legs_to_locate.iloc[0][s.LEG_TO_ACTIVITY_COL],
                                                                   legs_to_locate.iloc[1][s.LEG_TO_ACTIVITY_COL])
            # Calc time from second activity to fourth activity
            time_B_D = self.sf.calculate_expected_time_with_slack(legs_to_locate.iloc[2][s.LEG_DURATION_MINUTES_COL],
                                                               legs_to_locate.iloc[3][s.LEG_DURATION_MINUTES_COL],
                                                               legs_to_locate.iloc[2][s.LEG_FROM_ACTIVITY_COL],
                                                               legs_to_locate.iloc[2][s.LEG_TO_ACTIVITY_COL],
                                                               legs_to_locate.iloc[3][s.LEG_TO_ACTIVITY_COL])
            # Calc time from fourth activity to end activity
            time_D_End = self.sf.calculate_expected_time_with_slack(legs_to_locate.iloc[4][s.LEG_DURATION_MINUTES_COL],
                                                                 legs_to_locate.iloc[5][s.LEG_DURATION_MINUTES_COL],
                                                                 legs_to_locate.iloc[4][s.LEG_FROM_ACTIVITY_COL],
                                                                 legs_to_locate.iloc[4][s.LEG_TO_ACTIVITY_COL],
                                                                 legs_to_locate.iloc[5][s.LEG_TO_ACTIVITY_COL])

            # Calc time from start to fourth activity
            time_Start_D = self.sf.calculate_expected_time_with_slack(time_Start_B,
                                                                   time_B_D,
                                                                   legs_to_locate.iloc[0][s.LEG_FROM_ACTIVITY_COL],
                                                                   legs_to_locate.iloc[1][s.LEG_TO_ACTIVITY_COL],
                                                                   legs_to_locate.iloc[3][s.LEG_TO_ACTIVITY_COL])

            # Locate fourth activity
            cell_D = self.locate_single_activity(legs_to_locate.iloc[0]['cell_from'],
                                                 legs_to_locate.iloc[5]['cell_to'],
                                                 legs_to_locate.iloc[3][s.LEG_TO_ACTIVITY_COL],
                                                 time_Start_D,
                                                 time_D_End)

            # Locate second activity
            cell_B = self.locate_single_activity(legs_to_locate.iloc[0]['cell_from'],
                                                 cell_D,
                                                 legs_to_locate.iloc[1][s.LEG_TO_ACTIVITY_COL],
                                                 time_Start_B,
                                                 time_B_D)

            # Locate first activity
            cell_A = self.locate_single_activity(legs_to_locate.iloc[0]['cell_from'],
                                                 cell_B,
                                                 legs_to_locate.iloc[0][s.LEG_TO_ACTIVITY_COL],
                                                 legs_to_locate.iloc[0][s.LEG_DURATION_MINUTES_COL],
                                                 legs_to_locate.iloc[1][s.LEG_DURATION_MINUTES_COL])

            # Locate third activity
            cell_C = self.locate_single_activity(cell_B,
                                                 cell_D,
                                                 legs_to_locate.iloc[2][s.LEG_TO_ACTIVITY_COL],
                                                 legs_to_locate.iloc[2][s.LEG_DURATION_MINUTES_COL],
                                                 legs_to_locate.iloc[3][s.LEG_DURATION_MINUTES_COL])

            # Locate fifth activity
            cell_E = self.locate_single_activity(cell_D,
                                                 legs_to_locate.iloc[5]['cell_to'],
                                                 legs_to_locate.iloc[4][s.LEG_TO_ACTIVITY_COL],
                                                 legs_to_locate.iloc[4][s.LEG_DURATION_MINUTES_COL],
                                                 legs_to_locate.iloc[5][s.LEG_DURATION_MINUTES_COL])

            # Save cells
            legs_to_locate.iloc[0]['cell_to'] = cell_A
            legs_to_locate.iloc[1]['cell_from'] = cell_A
            legs_to_locate.iloc[1]['cell_to'] = cell_B
            legs_to_locate.iloc[2]['cell_from'] = cell_B
            legs_to_locate.iloc[2]['cell_to'] = cell_C
            legs_to_locate.iloc[3]['cell_from'] = cell_C
            legs_to_locate.iloc[3]['cell_to'] = cell_D
            legs_to_locate.iloc[4]['cell_from'] = cell_D
            legs_to_locate.iloc[4]['cell_to'] = cell_E
            legs_to_locate.iloc[5]['cell_from'] = cell_E


    def locate_sec_chain(self, legs_to_locate):
        """
        Locates any leg chain between two known locations using travel time matrix and capacity data.
        :param legs_to_locate: DataFrame with the legs to locate. Must have the following columns:
        cell_from, cell_to, activity_type, duration, mode, hour
        For explanation of the algorithm, see the thesis.
        :return: DataFrame with a new column with cells assigned to each leg.
        """
        legs_with_estimated_direct_times, highest_level = self.sf.get_all_estimated_times_with_slack(legs_to_locate)

        # Normalize the estimated times given the known direct time between start and end
        # Get the direct time
        # Get all modes and their

        # Get the direct time between those cells
        direct_time = self.tt.get_travel_time(legs_to_locate.iloc[0]['cell_from'], legs_to_locate.iloc[-1]['cell_to'], XXX)
        # Given that we are using cells, only adjust once the difference passes the tolerance

        # Locate activities top-down, starting with the second-highest level
        for level in range(highest_level - 1, 0, -1):
            if level == 0:
                times_col = s.LEG_DURATION_MINUTES_COL
                if len(legs_to_locate[times_col].notna()) != len(legs_to_locate):
                    logger.warning(f"Found NaN values in {times_col}, may produce incorrect results.")
            else:
                times_col = f"level_{level}"  # Here we expect some NaN values

            # Get "meta-legs" for this level
            legs_to_process = legs_to_locate[legs_to_locate[times_col].notna()].copy()
            legs_to_process['original_index'] = legs_to_process.index

            # Reset index for reliable pairing
            legs_to_process.reset_index(drop=True, inplace=True)
            legs_to_process['pair_id'] = legs_to_process.index // 2

            self.locate_sec_chain_long(legs_to_locate_level)
            # Get leg pairs where the middle activity needs to be located and send to the single activity locator

