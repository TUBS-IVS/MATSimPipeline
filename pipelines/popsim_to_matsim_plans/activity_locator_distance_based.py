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
import pickle

# TODO: Zum lAufen kriegen 11.06.24


def reformat_locations(locations_data: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Reformat locations data from a nested dictionary to a dictionary of numpy arrays."""
    reformatted_data = {}

    for purpose, locations in locations_data.items():
        identifiers = []
        names = []
        coordinates = []
        capacities = []

        for location_id, location_details in locations.items():
            identifiers.append(location_id)
            names.append(location_details['name'])
            coordinates.append(location_details['coordinates'])
            capacities.append(location_details['capacity'])

        reformatted_data[purpose] = {
            'identifiers': np.array(identifiers, dtype=object),
            'names': np.array(names, dtype=str),
            'coordinates': np.array(coordinates, dtype=float),
            'capacities': np.array(capacities, dtype=float)
        }
    
    return reformatted_data


from typing import Dict, Tuple, Any
import numpy as np
from sklearn.neighbors import KDTree
import random as rnd

class TargetLocations:
    """
    Spatial index of activity locations split by purpose.
    This class is used to quickly find the nearest activity locations for a given location.
    """

    def __init__(self, data: Dict[str, Dict[str, np.ndarray]]):
        self.data: Dict[str, Dict[str, np.ndarray]] = data
        self.indices: Dict[str, KDTree] = {}

        for purpose, pdata in self.data.items():
            print(f"Constructing spatial index for {purpose} ...")
            self.indices[purpose] = KDTree(pdata["coordinates"])

    def query(self, purpose: str, location: np.ndarray, num_candidates: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the nearest activity locations for a given location and purpose.
        :param purpose: The purpose category to query.
        :param location: A 1D numpy array representing the location to query (coordinates [1.5, 2.5]).
        :param num_candidates: The number of nearest candidates to return.
        :return: A tuple containing four numpy arrays: identifiers, coordinates, distances, and remaining capacities of the nearest candidates.
        """
        # Ensure location is a 2D array with a single location
        location = location.reshape(1, -1)

        # Query the KDTree for the nearest locations
        candidate_distances, indices = self.indices[purpose].query(location, k=num_candidates)
        print(f"Distances: {candidate_distances}")
        print(f"Indices: {indices}")

        # Get the identifiers, coordinates, and distances for the nearest neighbors
        candidate_identifiers = np.array(self.data[purpose]["identifiers"])[indices[0]]
        candidate_names = np.array(self.data[purpose]["names"])[indices[0]]
        candidate_coordinates = np.array(self.data[purpose]["coordinates"])[indices[0]]
        candidate_capacities = np.array(self.data[purpose]["capacities"])[indices[0]]

        return candidate_identifiers, candidate_names, candidate_coordinates, candidate_capacities, candidate_distances

    def sample(self, purpose: str, random: rnd.Random) -> Tuple[Any, np.ndarray]:
        """
        Sample a random activity location for a given purpose.
        :param purpose: The purpose category to sample from.
        :param random: A random number generator.
        :return: A tuple containing the identifier and coordinates of the sampled activity.
        """
        index = random.randint(0, len(self.data[purpose]["coordinates"]) - 1)
        identifier = self.data[purpose]["identifiers"][index]
        coordinates = self.data[purpose]["coordinates"][index]
        return identifier, coordinates


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

with open('locations_data_with_capacities.pkl', 'rb') as file:
    locations_data = pickle.load(file)

logger.debug(f"Locations data with potentials: {locations_data}")

reformatted_data = reformat_locations(locations_data)

logger.debug(f"Reformatted locations data: {reformatted_data}")

MyTargetLocations = TargetLocations(reformatted_data)
#test with epsg:25832
test_candidates = MyTargetLocations.query("shop", np.array([549637.87573102, 5796618.40418383]), 5)
logger.debug(f"Test candidates for shop at 52.432047, 9.687902: Identifiers: {test_candidates[0]}, Names: {test_candidates[1]}, Coordinates: {test_candidates[2]}, Capacities: {test_candidates[3]}, Distances: {test_candidates[4]}")
             
# # Usage
# location_scoring_function = LocationScoringFunction(sigmoid_beta=1.0, sigmoid_delta_t=0.0)
# identifiers = np.array([1, 2, 3])
# locations = np.array([[0, 0], [1, 1], [2, 2]])
# distances = np.array([10, 20, 30])
# capacities = np.array([100, 200, 300])
# time_diffs = np.array([1, 2, 3])

# scores = location_scoring_function.score_locations(identifiers, locations, distances, capacities, time_diffs)
# print("Scores:", scores)


def find_circle_intersections(center1: np.ndarray, radius1: float, center2: np.ndarray, radius2: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the intersection points of two circles.
    
    :param center1: The center of the first circle (e.g., np.array([x1, y1])).
    :param radius1: The radius of the first circle.
    :param center2: The center of the second circle (e.g., np.array([x2, y2])).
    :param radius2: The radius of the second circle.
    :return: A tuple containing two intersection points (each as a np.ndarray).
             If no direct intersections, returns the closest points on each circle's circumference.
    """
    x1, y1 = center1
    x2, y2 = center2
    r1 = radius1
    r2 = radius2

    # Calculate the distance between the two centers
    d = np.linalg.norm(center1 - center2)

    # Handle non-intersection conditions:
    if d > (r1 + r2):
        logger.info("No direct intersection: The circles are too far apart.")
        logger.info("Finding point on the line with distances proportional to radii as fallback.")

        # Find the point on the line connecting the centers with proportional distances to the radii
        proportional_distance = r1 / (r1 + r2)
        point_on_line = center1 + proportional_distance * (center2 - center1)
        
        return point_on_line, None

    if d < abs(r1 - r2):
        logger.info("No direct intersection: One circle is contained within the other.")
        logger.info("Returning closest point on the circumference of the inner circle.")

        # Find the point on the larger circle nearest to the center of the smaller circle
        if r1 > r2:
            closest_point = center2 + r2 * (center1 - center2) / d
            return closest_point, None
        else:
            closest_point = center1 + r1 * (center2 - center1) / d
            return closest_point, None

    if d == 0 and r1 == r2:
        logger.info("Infinite intersections: The start and end points and radii are identical.")
        logger.info("Choosing a point on the perimeter of the circles.")
        # Choose a point on the perimeter of the circles
        intersect = np.array([x1 + r1, y1])
        return intersect, None
    
    if d == (r1 + r2) or d == abs(r1 - r2):
        logger.info("Whaaat? Tangential circles: The circles touch at exactly one point.")

        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = 0  # Tangential circles will have h = 0 as h = sqrt(r1^2 - a^2)

        x3 = x1 + a * (x2 - x1) / d
        y3 = y1 + a * (y2 - y1) / d

        intersection = np.array([x3, y3])

        return intersection, None

    # Calculate points of intersection
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(r1**2 - a**2)

    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d

    intersect1 = np.array([x3 + h * (y2 - y1) / d, y3 - h * (x2 - x1) / d])
    intersect2 = np.array([x3 - h * (y2 - y1) / d, y3 + h * (x2 - x1) / d])

    return intersect1, intersect2

def locate_single_activity(start_coord: np.ndarray, end_coord: np.ndarray, purpose: str, distance_start_to_act: float, distance_act_to_end: float) -> Tuple[str, np.ndarray]:
    """
    Locate a single activity that is at a specified distance from both start and end points.
    The activity should be located at the intersection of the two circles defined by the distances.

    :param start_coord:  Coordinates of the start point (e.g., np.array([x1, y1])).
    :param end_coord: Coordinates of the end point (e.g., np.array([x2, y2])).
    :param purpose: The purpose identifier for the activity location (placeholder in this context).
    :param distance_start_to_act: Distance from the start point to the activity location.
    :param distance_act_to_end: Distance from the activity location to the end point.
    :return: A tuple containing the activity identifier (as a placeholder) and its coordinates.
    """
    intersect1, intersect2 = find_circle_intersections(start_coord, distance_start_to_act, end_coord, distance_act_to_end)
    
    candidate_identifiers, candidate_names, candidate_coordinates, candidate_capacities, candidate_distances = MyTargetLocations.query(purpose, intersect1, 5)
    
    if intersect2 is not None:
        candidate_identifiers2, candidate_names2, candidate_coordinates2, candidate_capacities2, candidate_distances2 = MyTargetLocations.query(purpose, intersect2, 5)

        candidate_identifiers = np.concatenate((candidate_identifiers, candidate_identifiers2), axis=0)
        candidate_names = np.concatenate((candidate_names, candidate_names2), axis=0)
        candidate_coordinates = np.concatenate((candidate_coordinates, candidate_coordinates2), axis=0)
        candidate_capacities = np.concatenate((candidate_capacities, candidate_capacities2), axis=0)
        candidate_distances = np.concatenate((candidate_distances, candidate_distances2), axis=0)
        
    


    # Placeholder logic to choose one intersection point; here we choose the first one.
    chosen_intersection = intersect1

    return ("activity", chosen_intersection)

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
    

