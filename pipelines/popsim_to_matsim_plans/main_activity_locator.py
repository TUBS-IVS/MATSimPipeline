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
    :param persons_crs: CRS of the persons data (if given as a DataFrame)
    :param capacities_crs: CRS of the capacities data (if given as a DataFrame)
    :param persons_geometry_col: Name of the geometry column in the persons data (if given as a DataFrame)
    :param capacities_geometry_col: Name of the geometry column in the capacities data (if given as a DataFrame)
    """

    def __init__(self, persons: pd.DataFrame, target_crs="EPSG:25832",
                 persons_crs=None, capacities_crs=None,
                 persons_geometry_col=None, capacities_geometry_col=None):

        self.persons_df = persons.sort_values(by=s.UNIQUE_LEG_ID_COL, ignore_index=True)
        # self.capacity_points_gdf = self.load_data_into_gdf(capacities, capacities_geometry_col, capacities_crs)

        self.cells_gdf: gpd.GeoDataFrame = gpd.read_file(s.CAPA_CELLS_SHP_PATH)
        self.tt = h.TTMatrices(s.TT_MATRIX_CAR_FILES, s.TT_MATRIX_PT_FILES, s.TT_MATRIX_BIKE_FILE, s.TT_MATRIX_WALK_FILE)
        self.sf = h.SlackFactors(s.SLACK_FACTORS_FILE)
        self.capa = h.Capacities()

        self.target_crs = target_crs
        self.located_main_activities_for_current_population = False

        # self.match_crs(self.target_crs)

    def match_crs(self, common_crs):
        if self.persons_df.crs != common_crs:
            logger.info(f"Matching CRS of persons data from {self.persons_df.crs} to {common_crs}...")
            self.persons_df = self.persons_df.to_crs(common_crs)
        if self.cells_gdf.crs != common_crs:
            logger.info(f"Matching CRS of cells data from {self.cells_gdf.crs} to {common_crs}...")
            self.cells_gdf = self.cells_gdf.to_crs(common_crs)

    def locate_main_activity_cells(self, person):
        """
        Locates the main activity cell for each person, based on the travel time matrices,
        the normalized capacities, the desired activity type, and the desired travel time.
        :param person: Df with all rows for one person
        :return:
        """
        logger.info(f"Locating main activity cells for person {person[s.UNIQUE_P_ID_COL].iloc[0]}...")
        if person[s.LEG_NON_UNIQUE_ID_COL].iloc[0] != 1:
            logger.error(f"Person {person[s.UNIQUE_P_ID_COL].iloc[0]} has no leg 1 and thus no start. Locating randomly.")
            person[s.CELL_FROM_COL] = random.choice(self.capa.capa_csv_df['NAME'].unique())
            return person
        n = s.N_CLOSEST_CELLS
        try:
            main_activity_index = person[person[s.IS_MAIN_ACTIVITY_COL] == 1].index[0]
        except IndexError:
            logger.error(f"Person doesn't have a main activity. Not locating")
            return person
        # Select all rows from the beginning to the main activity row (inclusive)
        legs_to_place = person.loc[:main_activity_index]
        # legs_to_place = person.iloc[0:main_activity_index + 1]
        target_activity = legs_to_place.iloc[-1][s.TO_ACTIVITY_WITH_CONNECTED_COL]
        target_time = legs_to_place.iloc[0][s.HOME_TO_MAIN_TIME_COL] * 60  # in seconds

        # Debugging
        # if person[s.UNIQUE_P_ID_COL].iloc[0] == "88628810_5859_88628812.0":
        #     print("debug")

        try:
            hour = legs_to_place.iloc[-1][s.LEG_START_TIME_COL].hour
        except Exception:
            logger.error(f"Could not get hour.")
            hour = 12
        weights: dict = legs_to_place.set_index(s.MODE_TRANSLATED_COL)[s.LEG_DURATION_MINUTES_COL].to_dict()
        candidates = self.get_n_closest_cells(legs_to_place[s.CELL_FROM_COL].iloc[0],
                                              weights,
                                              target_time, n, hour)
        if candidates.empty:
            logger.error(f"No candidates found for person {person[s.UNIQUE_P_ID_COL].iloc[0]}")
            person[s.CELL_FROM_COL] = random.choice(self.capa.capa_csv_df['NAME'].unique())
            return person
        if candidates["Time Difference"].isna().all():
            logger.warning(f"Candidates for person {person[s.UNIQUE_P_ID_COL].iloc[0]} have NaN time differences."
                           f"Modes:{weights}. Hour:{hour}. Target time:{target_time}.")
        candidates['Capacity'] = candidates['TO'].apply(lambda cell_name: self.capa.get_capacity(target_activity, cell_name))
        # If any capacities are below 0, shift all capacities up so the lowest is 1
        if candidates['Capacity'].min() < 0:
            candidates['Capacity'] += (1 - candidates['Capacity'].min())
            logger.debug(f"Shifted capacities up to 1 for person {person[s.UNIQUE_P_ID_COL].iloc[0]}")
        try:
            candidates['Attractiveness'] = self.calculate_cell_likelihoods(candidates['Capacity'],
                                                                           candidates['Time Difference'])
        except Exception as e:
            logger.error(f"Could not calculate cell likelihoods.")
            logger.error(e)
            candidates['Attractiveness'] = 0
        # Select a cell based on the attractiveness
        total_weight = candidates['Attractiveness'].sum()
        if total_weight > 0:
            selected_cell = candidates.sample(weights='Attractiveness', n=1)
        else:
            selected_cell = candidates.sample(n=1)

        person.loc[main_activity_index, s.CELL_TO_COL] = selected_cell['TO'].iloc[0]

        # Also update the cell of the mirrored main activity
        mirroring = person[person[s.MIRRORS_MAIN_ACTIVITY_COL] == 1]
        if not mirroring.empty:
            mirroring_main_activity_index = mirroring.index[0]
            person.loc[mirroring_main_activity_index, s.CELL_TO_COL] = selected_cell['TO'].iloc[0]
            # Update capacity for mirroring, yes it also takes up capacity
            self.capa.decrement_capacity(target_activity, selected_cell['TO'].iloc[0])
        # Update capacity
        self.capa.decrement_capacity(target_activity, selected_cell['TO'].iloc[0])
        return person

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

    def get_n_closest_cells(self, cell, mode_weights, target_time, n, hour=None):
        matrix = self.tt.get_weighted_tt_matrix_n_modes(mode_weights, hour)

        filtered_matrix = matrix.loc[matrix['FROM'] == cell]

        time_diffs = np.abs(filtered_matrix['VALUE'] - target_time)

        df_with_diffs = pd.DataFrame({
            'TO': filtered_matrix['TO'],
            'Time Difference': time_diffs
        })

        # Sort by time differences and select top N rows
        closest_cells = df_with_diffs.sort_values(by='Time Difference').head(n)

        return closest_cells.reset_index(drop=True)

    def get_total_activity_counts(self):
        """
        Get the total counts for each activity type and store them in a dictionary.
        """
        activity_counts = self.persons_df[s.LEG_TO_ACTIVITY_COL].value_counts()

        activity_counts_dict = activity_counts.to_dict()

        return activity_counts_dict

    def locate_activities(self):
        """
        Locates all activities for all persons in the population.
        :return:
        """
        # Normalize capacities (make supply == demand)
        total_activity_counts = self.get_total_activity_counts()
        self.capa.normalize_capacities(total_activity_counts)

        # Split persons into connected and unconnected
        connected_persons = self.persons_df[self.persons_df[s.P_HAS_CONNECTIONS_COL] == 1]
        unconnected_persons = self.persons_df[self.persons_df[s.P_HAS_CONNECTIONS_COL] == 0]

        unconnected_persons = self.locate_unconnected_legs(unconnected_persons)
        connected_persons = self.locate_connected_legs(connected_persons)

        # Concatenate results
        self.persons_df = pd.concat([connected_persons, unconnected_persons], ignore_index=True)
        self.persons_df.sort_values(by=s.UNIQUE_LEG_ID_COL, inplace=True, ignore_index=True)

        return self.persons_df

    def locate_unconnected_legs(self, unconnected_persons):
        logger.info("Locating unconnected legs...")
        results = []

        for _, person in unconnected_persons.groupby(s.UNIQUE_P_ID_COL):
            person_with_main = self.locate_main_activity_cells(person)  # Adds to_cells
            person_with_main = h.add_from_cell_fast(person_with_main)  # Also adds those to from_cells
            person_with_all = self.locate_sec_chains(person_with_main)
            results.append(person_with_all)
        logger.info("Concatenating results from unconnected legs.")
        combined_df = pd.concat(results, ignore_index=True)
        logger.info("Unconnected legs located.")
        return combined_df

    def locate_connected_legs(self, connected_persons):
        logger.info("Locating connected legs...")
        results = []

        for _, household in connected_persons.groupby(s.UNIQUE_HH_ID_COL):
            # Get the person with the most connected legs in the household
            main_person_idx = household[s.NUM_CONNECTED_LEGS_COL].idxmax()
            main_person_id = household.loc[main_person_idx, s.UNIQUE_P_ID_COL]
            main_person = household[household[s.UNIQUE_P_ID_COL] == main_person_id]

            # Locate that person
            person_with_main = self.locate_main_activity_cells(main_person)  # Adds to_cells
            person_with_main = h.add_from_cell_fast(person_with_main)  # Also adds those to from_cells
            person_with_all = self.locate_sec_chains(person_with_main)

            # Update the original household df with the located person
            columns_to_update = [s.CELL_TO_COL, s.CELL_FROM_COL]
            located_person_subset = person_with_all[columns_to_update]
            household.update(located_person_subset)

            # Get all connected legs
            connected_legs = household[household[s.CONNECTED_LEGS_COL] == 1]
            #

            results.append(person_with_all)

        logger.info("Concatenating results from connected legs.")
        combined_df = pd.concat(results, ignore_index=True)
        logger.info("Connected legs located.")
        return combined_df

    def locate_single_activity(self, start_cell, end_cell, activity_type, time_start_to_act, time_act_to_end,
                               mode_start, mode_end, start_hour,
                               min_tolerance=None, max_tolerance=None):
        """
        Locate a single activity between two known places using travel time matrix and capacity data.
        When the maximum tolerance is exceeded, the tolerance is increased rapidly to find any viable location,
        with a logged warning. If this still fails, a random cell is returned to keep the pipeline running,
        with a logged error.
        :param start_cell: Cell ID of the starting location.
        :param end_cell: Cell ID of the ending location.
        :param activity_type: Activity type of the activity to locate.
        :param time_start_to_act: Travel time from the start to the activity in seconds.
        :param time_act_to_end: Travel time from the activity to the end in seconds.
        :param mode_start: Mode of travel from the start to the activity.
        :param mode_end: Mode of travel from the activity to the end.
        :param start_hour: Hour of the day when the leg starts.
        :param min_tolerance: Minimum tolerance in minutes.
        :param max_tolerance: Maximum tolerance in minutes.
        :return: Cell ID of the best-suited location for the activity.
        """
        if min_tolerance is None:
            min_tolerance = 10 * 60
        if max_tolerance is None:
            max_tolerance = 90 * 60

        step_size = max((max_tolerance - min_tolerance) / 5, 200)
        tolerance = min_tolerance
        exceed_max_tolerance_turns = 2

        tt_matrix_start = self.tt.get_tt_matrix(mode_start, start_hour)
        tt_matrix_end = self.tt.get_tt_matrix(mode_end, start_hour)

        while True:

            # Filter cells based on travel time criteria (times in seconds!)
            potential_cells_start = tt_matrix_start[(tt_matrix_start['FROM'] == start_cell) &
                                                    (tt_matrix_start['VALUE'] >= time_start_to_act - tolerance) &
                                                    (tt_matrix_start['VALUE'] <= time_start_to_act + tolerance)]

            potential_cells_end = tt_matrix_end[(tt_matrix_end['FROM'] == end_cell) &
                                                (tt_matrix_end['VALUE'] >= time_act_to_end - tolerance) &
                                                (tt_matrix_end['VALUE'] <= time_act_to_end + tolerance)]

            # Find intersecting cells from both sets
            potential_cells = set(potential_cells_start['TO']).intersection(set(potential_cells_end['TO']))
            if potential_cells:
                break
            if tolerance <= max_tolerance:
                tolerance += step_size
            else:
                # If max tolerance is exceeded, go mad to find a cell
                tolerance += (step_size * 10)
                exceed_max_tolerance_turns -= 1
                logger.warning(
                    f"Exceeding maximum tolerance of {max_tolerance} seconds to find viable location. Tolerance is now {tolerance} seconds.")
                if exceed_max_tolerance_turns == 0:
                    logger.error(
                        f"No cell found for activity type {activity_type} between cells {start_cell} and {end_cell}."
                        f"Returning random cell to keep the pipeline running.")
                    return random.choice(self.capa.capa_csv_df['NAME'].unique())

        # Choose the cell with the highest capacity for the activity type
        candidate_capas = self.capa.capa_csv_df[self.capa.capa_csv_df['NAME'].isin(potential_cells)]
        if candidate_capas.empty:
            logger.error(f"Cells {potential_cells} not found in capacities. Returning random cell to keep the pipeline running.")
            return random.choice(self.capa.capa_csv_df['NAME'].unique())

        try:
            activity_type_str = str(int(activity_type))
            best_cell = candidate_capas.nlargest(1, activity_type_str)
        except KeyError:
            logger.debug(f"Could not find activity type {activity_type} in capacities.")
            best_cell = candidate_capas.sample(n=1)

        if best_cell.empty:
            logger.error(f"No cell found for activity type {activity_type} between cells {start_cell} and {end_cell}."
                         f"Returning random cell to keep the pipeline running.")
            return random.choice(self.capa.capa_csv_df['NAME'].unique())

        return best_cell['NAME'].iloc[0] if not best_cell.empty else None

    def locate_sec_chains(self, person):
        """
        Locates all secondary activity chains for a person.
        Gets all individual unknown chains and sends them to the solver.
        :param person:
        :return:
        """
        logger.info(f"Locating secondary activity chains for person {person[s.UNIQUE_P_ID_COL].iloc[0]}...")
        # Get all unknown chains
        sec_chains = h.find_nan_chains(person, s.CELL_TO_COL)
        for chain in sec_chains:
            # Solve each chain
            if chain.empty:
                logger.warning(f"Found empty chain. Skipping.")
                continue
            located_chain = self.locate_sec_chain_solver(chain)

            # Update the original person df with the located chain
            columns_to_update = [s.CELL_TO_COL, s.CELL_FROM_COL]
            located_chain_subset = located_chain[columns_to_update]
            person.update(located_chain_subset)
        return person

    def locate_sec_chain_solver(self, legs_to_locate):
        """
        Locates any leg chain between two known locations using travel time matrix and capacity data.
        :param legs_to_locate: DataFrame with the legs to locate. Must have the following columns:
        cell_from, cell_to, activity_type, duration, mode, hour
        For explanation of the algorithm, see the thesis.
        :return: DataFrame with a new column with cells assigned to each leg.
        """
        legs_to_locate = legs_to_locate.copy()
        # if len(legs_to_locate) > 2:
        #     print("debug")
        try:
            hour = legs_to_locate.iloc[0][s.LEG_START_TIME_COL].hour
        except Exception:
            logger.error(f"Could not get hour. Using 8.")  # Never had this happen, but to be sure (e.g. if conversion failed)
            hour = 8

        if legs_to_locate[s.LEG_MAIN_MODE_COL].nunique() == 1:

            direct_time = self.tt.get_travel_time(legs_to_locate.iloc[0][s.CELL_FROM_COL],
                                                  legs_to_locate.iloc[-1][s.CELL_TO_COL],
                                                  legs_to_locate.iloc[0][s.MODE_TRANSLATED_COL],
                                                  hour)
        else:
            mode_weights: dict = legs_to_locate.set_index(s.MODE_TRANSLATED_COL)[s.LEG_DURATION_MINUTES_COL].to_dict()
            direct_time = self.tt.get_mode_weighted_travel_time(legs_to_locate.iloc[0][s.CELL_FROM_COL],
                                                                legs_to_locate.iloc[-1][s.CELL_TO_COL],
                                                                mode_weights,
                                                                hour)

        # Expects and returns minutes as in MiD. Thus, they must later be converted to seconds.
        legs_to_locate, highest_level = self.sf.get_all_adjusted_times_with_slack(legs_to_locate, direct_time / 60)

        def split_sec_legs_dataframe(df):
            segments = []
            start_idx = None

            for i, row in df.iterrows():
                if start_idx is None and pd.notna(row[s.CELL_FROM_COL]):
                    start_idx = i
                elif start_idx is not None and pd.notna(row[s.CELL_TO_COL]):
                    segments.append(df.loc[start_idx:i])
                    if pd.notna(row[s.CELL_FROM_COL]):
                        start_idx = i
                    else:
                        start_idx = None

            # Handle last segment if it ends with the DataFrame
            if start_idx is not None:
                segments.append(df.loc[start_idx:])

            return segments

        for level in range(highest_level - 1, -1, -1):  # to include 0
            times_col = f"level_{level}" if level != 0 else s.LEG_DURATION_MINUTES_COL

            if level == 0 and len(legs_to_locate[times_col].notna()) != len(legs_to_locate):
                logger.warning(f"Found NaN values in {times_col}, may produce incorrect results.")

            segments = split_sec_legs_dataframe(legs_to_locate)

            for segment in segments:
                if len(segment) == 1:
                    continue  # If there is only one leg in the group, the cell is already known
                times = segment.loc[segment[times_col].notna(), times_col]
                if len(times) != 2:
                    if len(segment) == 2:  # If the segment is two long, use the known times
                        times_col = s.LEG_DURATION_MINUTES_COL
                        times = segment.loc[segment[times_col].notna(), times_col]
                        if len(times) != 2:
                            logger.error(f"Found {len(times)} times in segment {segment}. Expected 2.")
                    else:
                        logger.error(f"Found {len(times)} times in segment {segment}. Expected 2.")
                    continue
                cell = self.locate_single_activity(segment[s.CELL_FROM_COL].iloc[0],
                                                   segment[s.CELL_TO_COL].iloc[-1],
                                                   segment[s.TO_ACTIVITY_WITH_CONNECTED_COL].iloc[0],
                                                   times.iloc[0] * 60,  # expects s, TTmatrices are in s
                                                   times.iloc[1] * 60,
                                                   segment[s.MODE_TRANSLATED_COL].iloc[0],
                                                   segment[s.MODE_TRANSLATED_COL].iloc[1],
                                                   hour)

                located_leg_index = segment[times_col].first_valid_index()  # The time is placed at the to-locate leg

                legs_to_locate.loc[located_leg_index, s.CELL_TO_COL] = cell
                legs_to_locate.loc[located_leg_index + 1, s.CELL_FROM_COL] = cell

        return legs_to_locate

