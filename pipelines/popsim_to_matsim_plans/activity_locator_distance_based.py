import random

import geopandas as gpd
import numpy as np
import pandas as pd

from pipelines.common import helpers as h
from utils import settings_values as s
from utils.logger import logging

logger = logging.getLogger(__name__)    

from typing import Tuple, List, Dict, Any
import numpy as np
from sklearn.neighbors import KDTree
import random as rnd

# TODO: Zum lAufen kriegen 11.06.24

class TargetLocations:
    """
    Spatial index of activity locations split by purpose.
    This class is used to quickly find the nearest activity locations for a given location.
    """

    def __init__(self, data: Dict[str, Dict[str, np.ndarray]], initial_capacities: Dict[str, np.ndarray]):
        self.data: Dict[str, Dict[str, np.ndarray]] = data
        self.initial_capacities: Dict[str, np.ndarray] = initial_capacities
        self.capacities: Dict[str, np.ndarray] = {purpose: capacities.copy() for purpose, capacities in initial_capacities.items()}
        self.indices: Dict[str, KDTree] = {}

        for purpose, pdata in self.data.items():
            print(f"Constructing spatial index for {purpose} ...")
            self.indices[purpose] = KDTree(pdata["locations"])

    def query(self, purpose: str, location: np.ndarray, num_candidates: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the nearest activity locations for a given location and purpose.
        :param purpose: The purpose category to query.
        :param location: A 1D numpy array representing the location to query (coordinates [1.5, 2.5]).
        :param num_candidates: The number of nearest candidates to return.
        :return: A tuple containing four numpy arrays: identifiers, locations, distances, and remaining capacities of the nearest candidates.
        """
        # Ensure location is a 2D array with a single location
        location = location.reshape(1, -1)

        # Query the KDTree for the nearest locations
        distances, indices = self.indices[purpose].query(location, k=num_candidates)
        print(f"Distances: {distances}")
        print(f"Indices: {indices}")

        # Get the identifiers, locations, and distances for the nearest neighbors
        identifiers = np.array(self.data[purpose]["identifiers"])[indices[0]]
        nearest_locations = np.array(self.data[purpose]["locations"])[indices[0]]
        
        # Get the remaining capacities for the nearest neighbors
        remaining_capacities = self.capacities[purpose][indices[0]]

        return identifiers, nearest_locations, distances, remaining_capacities

    def sample(self, purpose: str, random: rnd.Random) -> Tuple[Any, np.ndarray]:
        """
        Sample a random activity location for a given purpose.
        :param purpose: The purpose category to sample from.
        :param random: A random number generator.
        :return: A tuple containing the identifier and location of the sampled activity.
        """
        index = random.randint(0, len(self.data[purpose]["locations"]) - 1)
        identifier = self.data[purpose]["identifiers"][index]
        location = self.data[purpose]["locations"][index]
        return identifier, location

class LocationScoringFunction:
    def __init__(self, sigmoid_beta: float, sigmoid_delta_t: float):
        """
        Initialize the LocationScoringFunction with sigmoid parameters.
        
        :param sigmoid_beta: Controls the steepness of the sigmoid's transition.
        :param sigmoid_delta_t: The midpoint of the sigmoid's transition.
        """
        self.sigmoid_beta = sigmoid_beta
        self.sigmoid_delta_t = sigmoid_delta_t

    def sigmoid(self, x):
        """
        Sigmoid function for likelihood calculation.

        :param x: The input value (time differential) - can be a number, list, or numpy array.
        :return: Sigmoid function value.
        """
        x = np.array(x)  # Ensure x is a numpy array
        z = -self.sigmoid_beta * (x - self.sigmoid_delta_t)
        # Use np.clip to limit the values in z to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(z))

    def score_locations(self, identifiers: np.ndarray, locations: np.ndarray, distances: np.ndarray, capacities: np.ndarray, time_diffs=None) -> np.ndarray:
        """
        Evaluate the returned locations by distance and capacity and return a score.
        
        :param identifiers: Numpy array of identifiers for the returned locations.
        :param locations: Numpy array of locations for the returned locations.
        :param distances: Numpy array of distances for the returned locations.
        :param capacities: Numpy array of remaining capacities for the returned locations.
        :param time_diffs: List or array of time differential values for each candidate.
        :return: Numpy array of scores for the returned locations.
        """
        # Calculate the base score for each location
        base_scores = capacities / distances  # TODO: Improve scoring function

        # If time_diffs is provided, adjust scores using the sigmoid function
        if time_diffs is not None:
            sigmoid_values = self.sigmoid(time_diffs)
            adjusted_scores = np.multiply(base_scores, sigmoid_values)
        else:
            adjusted_scores = base_scores

        # Normalize the scores to ensure they sum to 1
        scores = adjusted_scores / np.sum(adjusted_scores)
        return scores

# # Usage
# location_scoring_function = LocationScoringFunction(sigmoid_beta=1.0, sigmoid_delta_t=0.0)
# identifiers = np.array([1, 2, 3])
# locations = np.array([[0, 0], [1, 1], [2, 2]])
# distances = np.array([10, 20, 30])
# capacities = np.array([100, 200, 300])
# time_diffs = np.array([1, 2, 3])

# scores = location_scoring_function.score_locations(identifiers, locations, distances, capacities, time_diffs)
# print("Scores:", scores)




    # def locate_single_activity(self, start_cell, end_cell, activity_type, time_start_to_act, time_act_to_end,
    #                            mode_start, mode_end, start_hour,
    #                            min_tolerance=None, max_tolerance=None):
    #     """
    #     Locate a single activity between two known places using travel time matrix and capacity data.
    #     When the maximum tolerance is exceeded, the tolerance is increased rapidly to find any viable location,
    #     with a logged warning. If this still fails, a random cell is returned to keep the pipeline running,
    #     with a logged error.
    #     :param start_cell: Cell ID of the starting location.
    #     :param end_cell: Cell ID of the ending location.
    #     :param activity_type: Activity type of the activity to locate.
    #     :param time_start_to_act: Travel time from the start to the activity in seconds.
    #     :param time_act_to_end: Travel time from the activity to the end in seconds.
    #     :param mode_start: Mode of travel from the start to the activity.
    #     :param mode_end: Mode of travel from the activity to the end.
    #     :param start_hour: Hour of the day when the leg starts.
    #     :param min_tolerance: Minimum tolerance in minutes.
    #     :param max_tolerance: Maximum tolerance in minutes.
    #     :return: Cell ID of the best-suited location for the activity.
    #     """
    #     if min_tolerance is None:
    #         min_tolerance = 10 * 60
    #     if max_tolerance is None:
    #         max_tolerance = 90 * 60

    #     step_size = max((max_tolerance - min_tolerance) / 5, 200)
    #     tolerance = min_tolerance
    #     exceed_max_tolerance_turns = 2

    #     tt_matrix_start = self.tt.get_tt_matrix(mode_start, start_hour)
    #     tt_matrix_end = self.tt.get_tt_matrix(mode_end, start_hour)

    #     while True:

    #         # Filter cells based on travel time criteria (times in seconds!)
    #         potential_cells_start = tt_matrix_start[(tt_matrix_start['FROM'] == start_cell) &
    #                                                 (tt_matrix_start['VALUE'] >= time_start_to_act - tolerance) &
    #                                                 (tt_matrix_start['VALUE'] <= time_start_to_act + tolerance)]

    #         potential_cells_end = tt_matrix_end[(tt_matrix_end['FROM'] == end_cell) &
    #                                             (tt_matrix_end['VALUE'] >= time_act_to_end - tolerance) &
    #                                             (tt_matrix_end['VALUE'] <= time_act_to_end + tolerance)]

    #         # Find intersecting cells from both sets
    #         potential_cells = set(potential_cells_start['TO']).intersection(set(potential_cells_end['TO']))
    #         if potential_cells:
    #             break
    #         if tolerance <= max_tolerance:
    #             tolerance += step_size
    #         else:
    #             # If max tolerance is exceeded, go mad to find a cell
    #             tolerance += (step_size * 10)
    #             exceed_max_tolerance_turns -= 1
    #             logger.warning(
    #                 f"Exceeding maximum tolerance of {max_tolerance} seconds to find viable location. Tolerance is now {tolerance} seconds.")
    #             if exceed_max_tolerance_turns == 0:
    #                 logger.error(
    #                     f"No cell found for activity type {activity_type} between cells {start_cell} and {end_cell}."
    #                     f"Returning random cell to keep the pipeline running.")
    #                 return random.choice(self.capa.capa_csv_df['NAME'].unique())

    #     # Choose the cell with the highest capacity for the activity type
    #     candidate_capas = self.capa.capa_csv_df[self.capa.capa_csv_df['NAME'].isin(potential_cells)]
    #     if candidate_capas.empty:
    #         logger.error(f"Cells {potential_cells} not found in capacities. Returning random cell to keep the pipeline running.")
    #         return random.choice(self.capa.capa_csv_df['NAME'].unique())

    #     try:
    #         activity_type_str = str(int(activity_type))
    #         best_cell = candidate_capas.nlargest(1, activity_type_str)
    #     except KeyError:
    #         logger.debug(f"Could not find activity type {activity_type} in capacities.")
    #         best_cell = candidate_capas.sample(n=1)

    #     if best_cell.empty:
    #         logger.error(f"No cell found for activity type {activity_type} between cells {start_cell} and {end_cell}."
    #                      f"Returning random cell to keep the pipeline running.")
    #         return random.choice(self.capa.capa_csv_df['NAME'].unique())

    #     return best_cell['NAME'].iloc[0] if not best_cell.empty else None

    # def locate_sec_chains(self, person):
    #     """
    #     Locates all secondary activity chains for a person.
    #     Gets all individual unknown chains and sends them to the solver.
    #     :param person:
    #     :return:
    #     """
    #     logger.info(f"Locating secondary activity chains for person {person[s.UNIQUE_P_ID_COL].iloc[0]}...")
    #     # Get all unknown chains
    #     sec_chains = h.find_nan_chains(person, s.CELL_TO_COL)
    #     for chain in sec_chains:
    #         # Solve each chain
    #         if chain.empty:
    #             logger.warning(f"Found empty chain. Skipping.")
    #             continue
    #         located_chain = self.locate_sec_chain_solver(chain)

    #         # Update the original person df with the located chain
    #         columns_to_update = [s.CELL_TO_COL, s.CELL_FROM_COL]
    #         located_chain_subset = located_chain[columns_to_update]
    #         person.update(located_chain_subset)
    #     return person

    # def locate_sec_chain_solver(self, legs_to_locate):
    #     """
    #     Locates any leg chain between two known locations using travel time matrix and capacity data.
    #     :param legs_to_locate: DataFrame with the legs to locate. Must have the following columns:
    #     cell_from, cell_to, activity_type, duration, mode, hour
    #     For explanation of the algorithm, see the thesis.
    #     :return: DataFrame with a new column with cells assigned to each leg.
    #     """
    #     legs_to_locate = legs_to_locate.copy()
    #     # if len(legs_to_locate) > 2:
    #     #     print("debug")
    #     try:
    #         hour = legs_to_locate.iloc[0][s.LEG_START_TIME_COL].hour
    #     except Exception:
    #         logger.error("Could not get hour. Using 8.")  # Never had this happen, but to be sure (e.g. if conversion failed)
    #         hour = 8

    #     if legs_to_locate[s.LEG_MAIN_MODE_COL].nunique() == 1:

    #         direct_time = self.tt.get_travel_time(legs_to_locate.iloc[0][s.CELL_FROM_COL],
    #                                               legs_to_locate.iloc[-1][s.CELL_TO_COL],
    #                                               legs_to_locate.iloc[0][s.MODE_TRANSLATED_COL],
    #                                               hour)
    #     else:
    #         mode_weights: dict = legs_to_locate.set_index(s.MODE_TRANSLATED_COL)[s.LEG_DURATION_MINUTES_COL].to_dict()
    #         direct_time = self.tt.get_mode_weighted_travel_time(legs_to_locate.iloc[0][s.CELL_FROM_COL],
    #                                                             legs_to_locate.iloc[-1][s.CELL_TO_COL],
    #                                                             mode_weights,
    #                                                             hour)

    #     # Expects and returns minutes as in MiD. Thus, they must later be converted to seconds.
    #     legs_to_locate, highest_level = self.sf.get_all_adjusted_times_with_slack(legs_to_locate, direct_time / 60)

    #     def split_sec_legs_dataframe(df):
    #         segments = []
    #         start_idx = None

    #         for i, row in df.iterrows():
    #             if start_idx is None and pd.notna(row[s.CELL_FROM_COL]):
    #                 start_idx = i
    #             elif start_idx is not None and pd.notna(row[s.CELL_TO_COL]):
    #                 segments.append(df.loc[start_idx:i])
    #                 if pd.notna(row[s.CELL_FROM_COL]):
    #                     start_idx = i
    #                 else:
    #                     start_idx = None

    #         # Handle last segment if it ends with the DataFrame
    #         if start_idx is not None:
    #             segments.append(df.loc[start_idx:])

    #         return segments

    #     for level in range(highest_level - 1, -1, -1):  # to include 0
    #         times_col = f"level_{level}" if level != 0 else s.LEG_DURATION_MINUTES_COL

    #         if level == 0 and len(legs_to_locate[times_col].notna()) != len(legs_to_locate):
    #             logger.warning(f"Found NaN values in {times_col}, may produce incorrect results.")

    #         segments = split_sec_legs_dataframe(legs_to_locate)

    #         for segment in segments:
    #             if len(segment) == 1:
    #                 continue  # If there is only one leg in the group, the cell is already known
    #             times = segment.loc[segment[times_col].notna(), times_col]
    #             if len(times) != 2:
    #                 if len(segment) == 2:  # If the segment is two long, use the known times
    #                     times_col = s.LEG_DURATION_MINUTES_COL
    #                     times = segment.loc[segment[times_col].notna(), times_col]
    #                     if len(times) != 2:
    #                         logger.error(f"Found {len(times)} times in segment {segment}. Expected 2.")
    #                 else:
    #                     logger.error(f"Found {len(times)} times in segment {segment}. Expected 2.")
    #                 continue
    #             cell = self.locate_single_activity(segment[s.CELL_FROM_COL].iloc[0],
    #                                                segment[s.CELL_TO_COL].iloc[-1],
    #                                                segment[s.TO_ACTIVITY_WITH_CONNECTED_COL].iloc[0],
    #                                                times.iloc[0] * 60,  # expects s, TTmatrices are in s
    #                                                times.iloc[1] * 60,
    #                                                segment[s.MODE_TRANSLATED_COL].iloc[0],
    #                                                segment[s.MODE_TRANSLATED_COL].iloc[1],
    #                                                hour)

    #             located_leg_index = segment[times_col].first_valid_index()  # The time is placed at the to-locate leg

    #             legs_to_locate.loc[located_leg_index, s.CELL_TO_COL] = cell
    #             legs_to_locate.loc[located_leg_index + 1, s.CELL_FROM_COL] = cell

    #     return legs_to_locate
    

