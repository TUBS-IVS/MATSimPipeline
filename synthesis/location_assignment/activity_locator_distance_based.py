import math
import os
import pickle
import random as rnd
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import pprint

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from utils import settings as s, helpers as h, pipeline_setup
from utils.logger import logging
from utils.stats_tracker import stats_tracker

import time

logger = logging.getLogger(__name__)


# class DistanceBasedMainActivityLocator:
#     """
#     DISTANCE-based localizer.
#     Normalizing the potentials according to the total main activity demand means that persons will be assigned to the activity
#     locations exactly proportional to the location potentials. This also means that secondary activities will have to make do
#     with the remaining potential.
#     :param legs_dict: Dict of persons with their legs data
#     :param target_locations: TargetLocations object that provides query_within_radius method
#     :param radius: Search radius for locating activities
#     """
# 
#     def __init__(self, legs_dict: Dict[str, List[Dict[str, Any]]], target_locations, radius: float):
#         self.legs_dict = legs_dict
#         self.target_locations = target_locations  # TargetLocations object
#         self.radius = radius  # Radius for the search
#         self.located_main_activities_for_current_population = False
# 
#     def locate_main_activity(self, person_id: str):
#         """
#         Locates the main activity location for each person based on Euclidean distances,
#         normalized potentials, and the desired activity type. Uses the unsegmented legs dict.
#         :param person_id: Identifier for the person to locate main activity for
#         :return:
#         """
#         person_legs = self.legs_dict[person_id]
# 
#         if not person_legs:
#             # Person doesn't seem to have legs (no mobility). This is fine.
#             return
# 
#         home_location = person_legs[0]['from_location']
# 
#         main_activity_leg = None
#         for leg in person_legs:
#             if leg['to_act_type'] == 'main_activity':
#                 main_activity_leg = leg
#                 break
# 
#         if not main_activity_leg:
#             return
# 
#         target_activity = main_activity_leg['to_act_type']
# 
#         identifiers, names, coordinates, potentials, distances = find_ring_candidates(
#             home_location, target_activity, self.radius, self.target_locations)
# 
#         if len(identifiers) == 0:
#             # If no candidates are found within the radius, assign a random location
#             all_coords = self.target_locations.data[target_activity]['coordinates']
#             main_activity_leg['to_location'] = rnd.choice(all_coords)
#             return
# 
#         # Calculate attractiveness using the provided score_locations function
#         attractiveness = score_locations(potentials, distances)
# 
#         total_weight = np.sum(attractiveness)
#         if total_weight > 0:
#             selected_index = np.random.choice(len(attractiveness), p=attractiveness / total_weight)
#         else:
#             selected_index = np.random.choice(len(attractiveness))
# 
#         main_activity_leg['to_location'] = coordinates[selected_index]
# 
#     def locate_activities(self):
#         for person_id in self.legs_dict.keys():
#             self.locate_main_activity(person_id)
#         return self.legs_dict

def locate_main(person_legs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gets a person's main activity and locates it.
    Currently just uses the Euclidean distance and potentials.
    Planned to also use O-D matrices.
    :return: Updated list of legs with located main activities.
    """

    main_activity_leg = None
    main_activity_index = None
    for i, leg in enumerate(person_legs):
        if leg['is_main_activity']:
            main_activity_leg = leg
            main_activity_index = i
            break

    if not main_activity_leg:
        assert len(person_legs) <= 1, "Person has no main activity but has multiple legs."
        return person_legs

    target_activity = main_activity_leg['to_act_type']
    home_location = person_legs[0]['from_location']
    estimated_distance_home_to_main = person_legs[0]['home_to_main_distance']

    # Radii are iteratively spread by find_ring_candidates until a candidate is found
    radius1, radius2 = spread_distances(estimated_distance_home_to_main, estimated_distance_home_to_main)  # Initial
    candidates = find_ring_candidates(target_activity, home_location, radius1=radius1, radius2=radius2)
    act_identifier, act_name, act_coord, act_cap, act_dist, act_score = monte_carlo_select_candidate(candidates)

    # Update the main activity leg and the subsequent leg
    person_legs[main_activity_index]['to_location'] = act_coord
    person_legs[main_activity_index]['to_act_identifier'] = act_identifier
    person_legs[main_activity_index]['to_act_name'] = act_name
    person_legs[main_activity_index]['to_act_cap'] = act_cap
    person_legs[main_activity_index]['to_act_score'] = act_score

    if main_activity_index + 1 < len(person_legs):
        person_legs[main_activity_index + 1]['from_location'] = act_coord

    return person_legs


def reformat_locations(locations_data: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Reformat locations data from a nested dictionary to a dictionary of numpy arrays."""
    reformatted_data = {}

    for type, locations in locations_data.items():
        identifiers = []
        names = []
        coordinates = []
        potentials = []

        for location_id, location_details in locations.items():
            identifiers.append(location_id)
            names.append(location_details['name'])
            coordinates.append(location_details['coordinates'])
            try:
                potentials.append(location_details['potential'])
            except KeyError:
                logger.warning("Using old capacity name instead of potential name")
                potentials.append(location_details['capacity'])  # Old name

        reformatted_data[type] = {
            'identifiers': np.array(identifiers, dtype=object),
            'names': np.array(names, dtype=str),
            'coordinates': np.array(coordinates, dtype=float),
            'potentials': np.array(potentials, dtype=float)
        }

    return reformatted_data


class TargetLocations:
    """
    Spatial index of activity locations split by type.
    This class is used to quickly find the nearest activity locations for a given location.
    """

    def __init__(self, data: Dict[str, Dict[str, np.ndarray]]):
        self.data: Dict[str, Dict[str, np.ndarray]] = data
        self.indices: Dict[str, KDTree] = {}

        for type, pdata in self.data.items():
            logger.debug(f"Constructing spatial index for {type} ...")
            self.indices[type] = KDTree(pdata["coordinates"])

    def query_closest(self, type: str, location: np.ndarray, num_candidates: int = 1) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the nearest activity locations for a given location and type.
        :param type: The type category to query.
        :param location: A 1D numpy array representing the location to query (coordinates [1.5, 2.5]).
        :param num_candidates: The number of nearest candidates to return.
        :return: A tuple containing four numpy arrays: identifiers, coordinates, distances, and remaining potentials of the nearest candidates.
        """
        # Ensure location is a 2D array with a single location
        location = location.reshape(1, -1)

        # Query the KDTree for the nearest locations
        candidate_distances, indices = self.indices[type].query(location, k=num_candidates)
        logger.debug(f"Query Distances: {candidate_distances}")
        logger.debug(f"Query Indices: {indices}")

        # Get the identifiers, coordinates, and distances for the nearest neighbors
        candidate_identifiers = np.array(self.data[type]["identifiers"])[indices[0]]
        candidate_names = np.array(self.data[type]["names"])[indices[0]]
        candidate_coordinates = np.array(self.data[type]["coordinates"])[indices[0]]
        candidate_potentials = np.array(self.data[type]["potentials"])[indices[0]]

        return candidate_identifiers, candidate_names, candidate_coordinates, candidate_potentials, candidate_distances[
            0]

    def query_within_radius(self, act_type: str, location: np.ndarray, radius: float):
        """
        Find the activity locations within a given radius of a location and type.
        :param act_type: The activity category to query.
        :param location: A 1D numpy array representing the location to query (coordinates [1.5, 2.5]).
        :param radius: The maximum distance from the location to search for candidates.
        :return: A tuple containing four numpy arrays: identifiers, coordinates, distances, and remaining potentials of the nearest candidates.
        """
        # Ensure location is a 2D array with a single location
        location = location.reshape(1, -1)

        # Query the KDTree for locations within radius
        candidate_indices = self.indices[act_type].query_radius(location, radius)
        logger.debug(f"Query Indices: {candidate_indices}")

        # Get the identifiers, coordinates, and distances for locations within the radius
        candidate_identifiers = np.array(self.data[act_type]["identifiers"])[candidate_indices[0]]
        candidate_names = np.array(self.data[act_type]["names"])[candidate_indices[0]]
        candidate_coordinates = np.array(self.data[act_type]["coordinates"])[candidate_indices[0]]
        candidate_potentials = np.array(self.data[act_type]["potentials"])[candidate_indices[0]]
        # candidate_distances = np.linalg.norm(candidate_coordinates - location, axis=1)
        candidate_distances = None

        return candidate_identifiers, candidate_names, candidate_coordinates, candidate_potentials, candidate_distances

    def query_within_ring(self, act_type: str, location: np.ndarray, radius1: float, radius2: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the activity locations within a ring defined by two radii around a location and type.
        :param act_type: The activity category to query.
        :param location: A 1D numpy array representing the location to query (coordinates [1.5, 2.5]).
        :param radius1: Any of the two radii defining the ring.
        :param radius2: The other one.
        :return: A tuple containing four numpy arrays: identifiers, coordinates, distances, and remaining potentials of the nearest candidates.
        """
        # Ensure location is a 2D array with a single location
        location = location.reshape(1, -1)

        outer_radius = max(radius1, radius2)
        inner_radius = min(radius1, radius2)

        outer_indices = self.indices[act_type].query_radius(location, outer_radius)
        if outer_indices is None:
            return None
        if len(outer_indices[0]) == 0:
            return None

        inner_indices = self.indices[act_type].query_radius(location, inner_radius)

        outer_indices_set = set(outer_indices[0])
        inner_indices_set = set(inner_indices[0])

        annulus_indices = list(outer_indices_set - inner_indices_set)
        if not annulus_indices:
            return None

        # Get the identifiers, coordinates, and distances for locations within the annulus
        candidate_identifiers = np.array(self.data[act_type]["identifiers"])[annulus_indices]
        candidate_names = np.array(self.data[act_type]["names"])[annulus_indices]
        candidate_coordinates = np.array(self.data[act_type]["coordinates"])[annulus_indices]
        candidate_potentials = np.array(self.data[act_type]["potentials"])[annulus_indices]
        candidate_distances = None

        return candidate_identifiers, candidate_names, candidate_coordinates, candidate_potentials, candidate_distances

    def sample(self, act_type: str, random: rnd.Random) -> Tuple[Any, np.ndarray]:
        """
        Sample a random activity location for a given act_type.
        :param act_type: The act_type category to sample from.
        :param random: A random number generator.
        :return: A tuple containing the identifier and coordinates of the sampled activity.
        """
        index = random.randint(0, len(self.data[act_type]["coordinates"]) - 1)
        identifier = self.data[act_type]["identifiers"][index]
        coordinates = self.data[act_type]["coordinates"][index]
        return identifier, coordinates


# def sigmoid(x):
#     """
#     Sigmoid function for likelihood calculation.
#
#     :param x: The input value (e.g. distance from desired point) - can be a number, list, or numpy array.
#     :return: Sigmoid function value.
#     """
#     x = np.array(x)  # Ensure x is a numpy array
#     z = -self.sigmoid_beta * (x - self.sigmoid_delta_t)
#     # Use np.clip to limit the values in z to avoid overflow
#     z = np.clip(z, -500, 500)
#     return 1 / (1 + np.exp(z))


def score_locations(potentials: np.ndarray, distances: np.ndarray = None) -> np.ndarray:
    """
    Evaluate the returned locations by distance and potential and return a score.

    :param distances: Numpy array of distances from desired point for the returned locations.
    :param potentials: Numpy array of remaining potentials for the returned locations.
    :return: Numpy array of scores for the returned locations.
    """
    if distances is not None:
        base_scores = potentials / distances  # TODO: Improve scoring function

    else:
        base_scores = potentials

    # Normalize the scores to ensure they sum to 1
    scores = base_scores / np.sum(base_scores)
    return scores


def euclidean_distance(start: np.ndarray, end: np.ndarray) -> float:
    """Compute the Euclidean distance between two points."""
    return np.linalg.norm(end - start)


def find_circle_intersections(center1: np.ndarray, radius1: float, center2: np.ndarray, radius2: float) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Find the intersection points of two circles.
    
    :param center1: The center of the first circle (e.g., np.array([x1, y1])).
    :param radius1: The radius of the first circle.
    :param center2: The center of the second circle (e.g., np.array([x2, y2])).
    :param radius2: The radius of the second circle.
    :return: A tuple containing one or two intersection points (each as a np.ndarray).
    """
    x1, y1 = center1
    x2, y2 = center2
    r1 = radius1
    r2 = radius2

    # Calculate the distance between the two centers
    d = euclidean_distance(center1, center2)

    logger.debug(f"Center 1: {center1}, Radius 1: {radius1}, Center 2: {center2}, Radius 2: {radius2}")

    # Handle non-intersection conditions:
    if d == 0:
        if abs(r1 - r2) < 1e-4:
            logger.info("Infinite intersections: The start and end points and radii are identical.")
            logger.info("Choosing a point on the perimeter of the circles.")
            intersect = np.array([x1 + r1, y1])
            return intersect, None
        else:
            logger.info("No intersection: The circles are identical but have different radii.")
            logger.info("Choosing a point on the perimeter of the circles.")
            intersect = np.array([x1 + r1, y1])
            return intersect, None

    if d > (r1 + r2):
        logger.info("No direct intersection: The circles are too far apart.")
        logger.info("Finding point on the line with distances proportional to radii as fallback.")

        proportional_distance = r1 / (r1 + r2)
        point_on_line = center1 + proportional_distance * (center2 - center1)

        return point_on_line, None

    if d < abs(r1 - r2):
        logger.info("No direct intersection: One circle is contained within the other.")
        logger.info("Returning closest point on the circumference of the inner circle.")

        if r1 > r2:
            closest_point = center2 + r2 * (center1 - center2) / d
            return closest_point, None
        else:
            closest_point = center1 + r1 * (center2 - center1) / d
            return closest_point, None

    if d == (r1 + r2) or d == abs(r1 - r2):
        logger.info("Whaaat? Tangential circles: The circles touch at exactly one point.")

        a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
        h = 0  # Tangential circles will have h = 0 as h = sqrt(r1^2 - a^2)

        x3 = x1 + a * (x2 - x1) / d
        y3 = y1 + a * (y2 - y1) / d

        intersection = np.array([x3, y3])

        return intersection, None

    # Calculate points of intersection
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(r1 ** 2 - a ** 2)

    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d

    intersect1 = np.array([x3 + h * (y2 - y1) / d, y3 - h * (x2 - x1) / d])
    intersect2 = np.array([x3 - h * (y2 - y1) / d, y3 + h * (x2 - x1) / d])

    return intersect1, intersect2


def find_location_candidates(start_coord: np.ndarray, end_coord: np.ndarray, type: str,
                             distance_start_to_act: float, distance_act_to_end: float,
                             num_candidates: int) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Find n location candidates for a given activity type between two known locations.
    Returns two sets of candidates if two intersection points are found, otherwise only one set.
    """
    intersect1, intersect2 = find_circle_intersections(start_coord, distance_start_to_act, end_coord,
                                                       distance_act_to_end)

    candidate_identifiers, candidate_names, candidate_coordinates, candidate_potentials, candidate_distances = MyTargetLocations.query_closest(
        type, intersect1, num_candidates)

    if intersect2 is not None:
        candidate_identifiers2, candidate_names2, candidate_coordinates2, candidate_potentials2, candidate_distances2 = MyTargetLocations.query_closest(
            type, intersect2, num_candidates)

        return (
            candidate_identifiers, candidate_names, candidate_coordinates, candidate_potentials, candidate_distances), \
            (candidate_identifiers2, candidate_names2, candidate_coordinates2, candidate_potentials2,
             candidate_distances2)

    return (
        candidate_identifiers, candidate_names, candidate_coordinates, candidate_potentials, candidate_distances), None


from typing import Tuple
import numpy as np


def greedy_select_single_activity(start_coord: np.ndarray, end_coord: np.ndarray, type: str,
                                  distance_start_to_act: float, distance_act_to_end: float,
                                  num_candidates: int):
    """Place a single activity at the most likely location."""
    logger.debug(f"Greedy selecting activity for type {type} between {start_coord} and {end_coord}.")

    # Home locations aren't among the targets and are for now replaced by the start location
    if type == s.ACT_HOME:
        logger.info("Home activity detected. Secondary locator shouldn't be used for that. Returning start location.")
        return None, "home", start_coord, None, None, None

    candidates1, candidates2 = find_location_candidates(start_coord, end_coord, type, distance_start_to_act,
                                                        distance_act_to_end, num_candidates)

    if candidates2 is not None:
        combined_candidates = tuple(np.concatenate((arr1, arr2)) for arr1, arr2 in zip(candidates1, candidates2))
    else:
        combined_candidates = candidates1

    return monte_carlo_select_candidate(combined_candidates)


def monte_carlo_select_candidate(candidates, use_distance=True):
    if use_distance:
        scores = score_locations(candidates[-2], candidates[-1])
    else:
        scores = score_locations(candidates[-2])

    # Verify that scores are normalized to 1
    logger.debug(f"Scores: {scores}")
    assert np.isclose(np.sum(scores), 1.0), "Scores are not normalized to 1."  # TODO: Remove this line in production

    chosen_index = np.random.choice(len(scores), p=scores)

    # Return the chosen candidate with its score
    chosen_candidate = tuple((arr[chosen_index] if arr is not None else None) for arr in candidates) + (
        scores[chosen_index],)

    return chosen_candidate


def populate_legs_dict_from_df(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Uses the MiD df to populate a nested dictionary with leg information for each person.
    :param df: DataFrame containing MiD data (BUT units are always meters, seconds)
    :return: Nested dictionary with leg information for each person
    Example output:
    data = {
        '10000290_11563_10000291': [
            {
                'unique_leg_id': '10000290_11563_10000291_1.0',
                'to_act_type': 'shopping',
                'distance': 950.0,
                'from_location': array([552452.11071084, 5807493.538159]),
                'to_location': array([], dtype=float64),
                'mode': 'bike',
                'is_main_activity': 1,
                'home_to_main_distance': 120.0
            },
            {
                'unique_leg_id': '10000290_11563_10000291_2.0',
                'to_act_type': 'home',
                'distance': 1430.0,
                'from_location': array([], dtype=float64),
                'to_location': array([552452.11071084, 5807493.538159]),
                'mode': 'bike',
                'is_main_activity': 0,
                'home_to_main_distance': 120.0
            }
        ],
        '10000370_11564_10000371': [
            {
                'unique_leg_id': '10000370_11564_10000371_1.0',
                'to_act_type': 'leisure',
                'distance': 10450.0,
                'from_location': array([554098.49165674, 5802930.10530201]),
                'to_location': array([], dtype=float64),
                'mode': 'car',
                'is_main_activity': 1,
                'home_to_main_distance': 1500.0
            },
            {
                'unique_leg_id': '10000370_11564_10000371_2.0',
                'to_act_type': 'home',
                'distance': 7600.0,
                'from_location': array([], dtype=float64),
                'to_location': array([], dtype=float64),
                'mode': 'car',
                'is_main_activity': 0,
                'home_to_main_distance': 1500.0
            },
            {
                'unique_leg_id': '10000370_11564_10000371_3.0',
                'to_act_type': 'shopping',
                'distance': 13300.0,
                'from_location': array([], dtype=float64),
                'to_location': array([], dtype=float64),
                'mode': 'car',
                'is_main_activity': 0,
                'home_to_main_distance': 1500.0
            },
            {
                'unique_leg_id': '10000370_11564_10000371_4.0',
                'to_act_type': 'home',
                'distance': 13300.0,
                'from_location': array([], dtype=float64),
                'to_location': array([554098.49165674, 5802930.10530201]),
                'mode': 'walk',
                'is_main_activity': 0,
                'home_to_main_distance': 1500.0
            }
        ]
    }


    """

    # Extract relevant information into a new DataFrame
    legs_info_df = pd.DataFrame({
        s.UNIQUE_P_ID_COL: df[s.UNIQUE_P_ID_COL],
        'leg_info': list(zip(
            df[s.UNIQUE_LEG_ID_COL],
            df[s.ACT_TO_INTERNAL_COL],
            df[s.LEG_DISTANCE_METERS_COL],
            [np.array(loc) if loc is not None else np.array([]) for loc in df['from_location']],
            [np.array(loc) if loc is not None else np.array([]) for loc in df['to_location']],
            df[s.MODE_INTERNAL_COL],
            df[s.IS_MAIN_ACTIVITY_COL],
            df[s.HOME_TO_MAIN_METERS_COL]
        ))
    })

    # Transform each tuple into a dictionary
    def to_leg_dict(leg_tuple):
        return {
            s.UNIQUE_LEG_ID_COL: leg_tuple[0],
            'to_act_type': leg_tuple[1],
            'distance': leg_tuple[2],
            'from_location': leg_tuple[3],
            'to_location': leg_tuple[4],
            'mode': leg_tuple[5],
            'is_main_activity': leg_tuple[6],
            'home_to_main_distance': leg_tuple[7]
        }

    legs_info_df['leg_info'] = legs_info_df['leg_info'].map(to_leg_dict)

    # Group by unique person identifier and aggregate leg information
    grouped = legs_info_df.groupby(s.UNIQUE_P_ID_COL)['leg_info'].apply(list)

    # Convert the grouped Series to a dictionary
    nested_dict = grouped.to_dict()

    return nested_dict


def generate_random_location_within_hanover():
    """Generate a random coordinate within Hanover, Germany, in EPSG:25832."""
    xmin, xmax = 546000, 556000
    ymin, ymax = 5800000, 5810000
    x = rnd.uniform(xmin, xmax)
    y = rnd.uniform(ymin, ymax)
    return np.array([x, y])


def prepare_mid_df_for_legs_dict(filter_max_distance=None, number_of_persons=None) -> pd.DataFrame:
    """Temporarily prepare the MiD DataFrame for the leg dictionary function."""
    df = h.read_csv(h.get_files(s.ENHANCED_MID_FOLDER))

    # Initialize columns with empty objects to ensure compatibility
    df["from_location"] = None
    df["to_location"] = None

    # Throw out rows with missing values in the distance column
    row_count_before = df.shape[0]
    # Count rows with no leg
    no_leg_count = df[df[s.LEG_ID_COL].isna()].shape[0]
    df = df.dropna(subset=[s.LEG_DISTANCE_METERS_COL])
    logger.debug(f"People with no legs: {no_leg_count}")
    logger.debug(f"Dropped {row_count_before - df.shape[0]} rows with missing distance values.")

    # Identify and remove records of persons with any trip exceeding the max distance if filter_max_distance is specified
    if filter_max_distance is not None:
        person_ids_to_exclude = df[df[s.LEG_DISTANCE_METERS_COL] > filter_max_distance][s.PERSON_ID_COL].unique()
        row_count_before = df.shape[0]
        df = df[~df[s.PERSON_ID_COL].isin(person_ids_to_exclude)]
        logger.debug(
            f"Dropped {row_count_before - df.shape[0]} rows from persons with trips exceeding the max distance of {filter_max_distance} km.")

    # Ensure these columns are treated as object type to store arrays
    df["from_location"] = df["from_location"].astype(object)
    df["to_location"] = df["to_location"].astype(object)

    # Limit to the specified number of persons and keep all rows for these persons
    if number_of_persons is not None:
        person_ids = df[s.PERSON_ID_COL].unique()[:number_of_persons]
        df = df[df[s.PERSON_ID_COL].isin(person_ids)]

    # Add random home locations for each person
    for person_id, group in df.groupby(s.PERSON_ID_COL):
        home_location = generate_random_location_within_hanover()
        df.at[group.index[0], "from_location"] = home_location
        df.at[group.index[-1], "to_location"] = home_location

    logger.debug(df.head())
    return df


def segment_legs(nested_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[List[Dict[str, Any]]]]:
    """
    Segment the legs of each person into separate trips where only the start and end locations are known.
    :param nested_dict:
    :return:
    Example output:
    data = {
    '10000290_11563_10000291': [
        [
            {
                'leg_id': '10000290_11563_10000291_1.0',
                'to_act_type': 'shopping',
                'distance': 950.0,
                'from_location': np.array([552452.11071084, 5807493.538159]),
                'to_location': np.array([], dtype=float64),
                'mode': 'car',
                'is_main_activity': 1,
                'home_to_main_distance': 120.0
            },
            {
                'leg_id': '10000290_11563_10000291_2.0',
                'to_act_type': 'home',
                'distance': 1430.0,
                'from_location': np.array([], dtype=float64),
                'to_location': np.array([552452.11071084, 5807493.538159]),
                'mode': 'car',
                'is_main_activity': 0,
                'home_to_main_distance': 120.0
            }
        ],
        [
            {
                'leg_id': '10000290_11563_10000291_3.0',
                'to_act_type': 'work',
                'distance': 500.0,
                'from_location': np.array([552452.11071084, 5807493.538159]),
                'to_location': np.array([], dtype=float64),
                'mode': 'walk',
                'is_main_activity': 1,
                'home_to_main_distance': 100.0
            },
            {
                'leg_id': '10000290_11563_10000291_4.0',
                'to_act_type': 'home',
                'distance': 1000.0,
                'from_location': np.array([], dtype=float64),
                'to_location': np.array([552452.11071084, 5807493.538159]),
                'mode': 'bike',
                'is_main_activity': 0,
                'home_to_main_distance': 100.0
            }
        ]
    ],
    '10000370_11564_10000371': [
        [
            {
                'leg_id': '10000370_11564_10000371_1.0',
                'to_act_type': 'leisure',
                'distance': 10450.0,
                'from_location': np.array([554098.49165674, 5802930.10530201]),
                'to_location': np.array([], dtype=float64),
                'mode': 'car',
                'is_main_activity': 1,
                'home_to_main_distance': 1500.0
            },
            {
                'leg_id': '10000370_11564_10000371_2.0',
                'to_act_type': 'home',
                'distance': 7600.0,
                'from_location': np.array([], dtype=float64),
                'to_location': np.array([], dtype=float64),
                'mode': 'car',
                'is_main_activity': 0,
                'home_to_main_distance': 1500.0
            },
            {
                'leg_id': '10000370_11564_10000371_3.0',
                'to_act_type': 'shopping',
                'distance': 13300.0,
                'from_location': np.array([], dtype=float64),
                'to_location': np.array([], dtype=float64),
                'mode': 'car',
                'is_main_activity': 0,
                'home_to_main_distance': 1500.0
            },
            {
                'leg_id': '10000370_11564_10000371_4.0',
                'to_act_type': 'home',
                'distance': 13300.0,
                'from_location': np.array([], dtype=float64),
                'to_location': np.array([554098.49165674, 5802930.10530201]),
                'mode': 'walk',
                'is_main_activity': 0,
                'home_to_main_distance': 1500.0
            }
        ]
    ]
}

    """
    logger.debug(f"Segmenting legs for {len(nested_dict)} persons.")
    segmented_dict = defaultdict(list)

    for person_id, legs in nested_dict.items():
        segment = []
        for leg in legs:
            segment.append(leg)
            if leg['to_location'].size > 0:
                segmented_dict[person_id].append(segment)
                segment = []
        # If there are remaining legs in the segment after the loop, add them as well
        if segment:
            segmented_dict[person_id].append(segment)

    return dict(segmented_dict)
    #
    # logger.debug(f"Segmenting legs for {len(nested_dict)} persons.")
    # segmented_dict = defaultdict(list)
    #
    # for person_id, legs in nested_dict.items():
    #     segment = []
    #     for leg in legs:
    #         if not segment:
    #             # Start a new segment with the first leg
    #             segment.append(leg)
    #         else:
    #             segment.append(leg)
    #             if leg['to_location'].size > 0:
    #                 segmented_dict[person_id].append(segment)
    #                 segment = []
    #     # If there are remaining legs in the segment after the loop, add them as well
    #     if segment:
    #         segmented_dict[person_id].append(segment)
    #
    # return dict(segmented_dict)

def adjust_estimation_tree(tree: List[List[List[float]]], real_distance: float, strong_adjust: bool = True) -> List[
    List[List[float]]]:  # Tree level, Leg, Lengths
    """If there is strong overshooting or undershooting in the estimation tree, adjust the distances. 
    It tries to keep given slack constraints, unless it is not possible.
    
    TEMP: Erstmal nur strong adjust, weak adjust ist vlt grundsätzlich nicht sinnvoll.
    
    Parameters:
    tree: The estimation tree to adjust.
    real_distance: The real total distance to adjust the tree to.
    strong_adjust: If True, real_bounds are used for adjustment, if False, wanted_bounds are used.
    """

    logger.debug(tree)

    # Enter real distance as basis for possible adjustments
    tree[-1][0][2] = real_distance

    # If highest level is 1, we're done (we can't adjust anything, but this is never needed with level 1 - trees anyway)
    if len(tree) == 1:
        return tree

    if strong_adjust:
        l_bound_idx = 0
        u_bound_idx = 4
    else:
        l_bound_idx = 1
        u_bound_idx = 3

    if tree[-1][0][1] < real_distance < tree[-1][0][
        3]:  # Checking for wanted maximum and minimum, not real maximum and minimum
        logger.debug("Real total distance is within bounds of highest level, no adjustment needed.")
        stats_tracker.increment("adjustment_total_runs")
        stats_tracker.increment("adjustment_no_adjustment")
        return tree
    else:
        # Traverse from one below the highest level down to including level 1 (which has index 0) - and then down to -1 just for the last check
        for level in range(len(tree) - 1, -1, -1):
            for i in range(0, len(tree[level]), 2):  # Traverse in pairs, skipping the last one if it's an odd number
                if i == len(tree[level]) - 1:
                    continue

                higher_leg_value = tree[level + 1][i // 2][2]
                higher_leg_lower_bound = tree[level + 1][i // 2][l_bound_idx]
                higher_leg_upper_bound = tree[level + 1][i // 2][u_bound_idx]

                leg1_value = tree[level][i][2]
                leg2_value = tree[level][i + 1][2]

                if higher_leg_value < higher_leg_lower_bound:  # (Real) higher_leg_value can be zero if start and end location are the same
                    logger.debug(
                        f"Strong overshot. Level: {level}, i: {i}, higher_leg_value: {higher_leg_value}, higher_leg_lower_bound: {higher_leg_lower_bound}")
                    stats_tracker.increment("adjustment_overshot_cases")
                    # Yes, overshot: The "ground truth" is lower than the lower bound of the estimation
                    if leg1_value > leg2_value:
                        # Make longer leg shorter, shorter leg longer
                        L_bounds1 = abs(tree[level][i][0] - leg1_value)
                        L_bounds2 = abs(tree[level][i + 1][4] - leg2_value)
                        delta_L_high = abs(higher_leg_value - higher_leg_lower_bound)

                        delta_L1 = (L_bounds1 * delta_L_high) / (L_bounds1 + L_bounds2)
                        delta_L2 = (L_bounds2 * delta_L_high) / (L_bounds1 + L_bounds2)

                        tree[level][i][2] -= delta_L1
                        tree[level][i + 1][2] += delta_L2
                    else:
                        L_bounds1 = abs(tree[level][i][4] - leg1_value)
                        L_bounds2 = abs(tree[level][i + 1][0] - leg2_value)
                        delta_L_high = abs(higher_leg_value - higher_leg_lower_bound)

                        delta_L1 = (L_bounds1 * delta_L_high) / (L_bounds1 + L_bounds2)
                        delta_L2 = (L_bounds2 * delta_L_high) / (L_bounds1 + L_bounds2)

                        tree[level][i][2] += delta_L1
                        tree[level][i + 1][2] -= delta_L2

                elif higher_leg_value > higher_leg_upper_bound:

                    logger.debug(
                        f"Strong undershot. Level: {level}, i: {i}, higher_leg_value: {higher_leg_value}, higher_leg_upper_bound: {higher_leg_upper_bound}")
                    stats_tracker.increment("adjustment_undershot_cases")
                    # Yes, undershot: The "ground truth" is higher than the upper bound of the estimation
                    # Make both legs longer (both move to upper bound)

                    L_bounds1 = abs(tree[level][i][4] - leg1_value)
                    L_bounds2 = abs(tree[level][i + 1][4] - leg2_value)
                    delta_L_high = abs(higher_leg_value - higher_leg_upper_bound)

                    delta_L1 = (L_bounds1 * delta_L_high) / (L_bounds1 + L_bounds2)
                    delta_L2 = (L_bounds2 * delta_L_high) / (L_bounds1 + L_bounds2)

                    tree[level][i][2] += delta_L1
                    tree[level][i + 1][2] += delta_L2

        # Check plausibilities and return
        plausible = True
        for i in range(0, len(tree[0]), 1):
            leg_value = tree[0][i][2]
            leg_lower_bound = tree[0][i][0]
            leg_upper_bound = tree[0][i][4]
            if not leg_lower_bound <= leg_value <= leg_upper_bound:
                plausible = False
                break

        if strong_adjust:
            if plausible:
                logger.debug("Strong adjustment succeeded.")
                logger.debug(tree)
                stats_tracker.increment("adjustment_total_runs")
                stats_tracker.increment("adjustment_strong_adjustment_success")
                return tree
            else:
                logger.debug("Strong adjustment failed.")
                logger.debug(tree)
                stats_tracker.increment("adjustment_total_runs")
                stats_tracker.increment("adjustment_strong_adjustment_failure")
                return tree
        else:
            if plausible:
                # These are the bounds of the first level estimation, which are set in stone by the known real distances
                logger.debug("Adjustment succeeded with wanted bounds.")
                return tree
            else:
                logger.debug("Adjustment failed with wanted bounds, trying strong adjustment with real bounds.")
                return adjust_estimation_tree(tree, real_distance, strong_adjust=True)


def greedy_locate_segment(segment):
    if len(segment) == 0:
        raise ValueError("No legs in segment.")
    elif len(segment) == 1:
        assert segment[0]['from_location'].size > 0 and segment[0][
            'to_location'].size > 0, "Both start and end locations must be known for a single leg."
        return segment
    # if there are only two legs, we can find the loc immediately
    elif len(segment) == 2:
        logger.debug("Greedy locating. Only two legs in segment.")
        act_identifier, act_name, act_coord, act_cap, act_dist, act_score = greedy_select_single_activity(
            segment[0]['from_location'], segment[-1]['to_location'], segment[0]['to_act_type'],
            segment[0]['distance'], segment[-1]['distance'], 5)
        segment[0]['to_location'] = act_coord
        segment[-1]['from_location'] = act_coord
        segment[0]['to_act_identifier'] = act_identifier
        segment[0]['to_act_name'] = act_name
        segment[0]['to_act_cap'] = act_cap
        segment[0]['to_act_score'] = act_score
        return segment
    else:
        logger.debug(f"Greedy locating. Segment has {len(segment)} legs.")
        distances = [leg['distance'] for leg in segment]

        real_distance = euclidean_distance(segment[0]['from_location'], segment[-1]['to_location'])

        tree = h.build_estimation_tree(distances)
        tree = adjust_estimation_tree(tree, real_distance, strong_adjust=True)
        position_on_segment_info = build_position_on_segment_info(
            len(distances))  # tells us at each level which legs to look at
        assert len(tree) == len(position_on_segment_info), "Tree and position info must have the same length."

        for level in range(len(tree) - 1, -1, -1):
            for i, leg_idx in enumerate(position_on_segment_info[level]):
                logger.debug(f"Level: {level}, i: {i}, leg_idx: {leg_idx}")
                segment_step = 2 ** level
                from_location_idx = leg_idx - segment_step + 1  # + 1 because we get the from_location
                assert from_location_idx >= 0, "From location index must be greater or equal to 0."
                to_location_idx = min(len(segment) - 1, leg_idx + segment_step)

                if level == 0:
                    dist_start_to_act = segment[leg_idx]['distance']
                    dist_act_to_end = segment[to_location_idx]['distance']
                else:
                    dist_start_to_act = tree[level - 1][2 * i][2]
                    dist_act_to_end = tree[level - 1][2 * i + 1][2]

                act_identifier, act_name, act_coord, act_cap, act_dist, act_score = \
                    greedy_select_single_activity(segment[from_location_idx]['from_location'],
                                                  segment[to_location_idx]['to_location'],
                                                  segment[leg_idx]['to_act_type'], dist_start_to_act,
                                                  dist_act_to_end, 5)
                segment[leg_idx]['to_location'] = act_coord
                if leg_idx + 1 < len(segment) + 1:
                    segment[leg_idx + 1]['from_location'] = act_coord
                segment[leg_idx]['to_act_identifier'] = act_identifier
                segment[leg_idx]['to_act_name'] = act_name
                segment[leg_idx]['to_act_cap'] = act_cap
                segment[leg_idx]['to_act_score'] = act_score
        return segment


def add_from_locations(segment):
    """Add the from_location to each leg in the segment."""
    for i, leg in enumerate(segment):
        if i != 0:
            leg['from_location'] = segment[i - 1]['to_location']
    return segment


def insert_placed_distances(segment):
    """Inserts info on the actual distances between placed activities for a fully located segment.
    Optional; for debugging and evaluation."""
    for leg in segment:
        leg['placed_distance'] = euclidean_distance(leg['from_location'], leg['to_location'])
        leg['placed_distance_absolute_diff'] = abs(leg['distance'] - leg['placed_distance'])
        leg['placed_distance_relative_diff'] = leg['placed_distance_absolute_diff'] / leg['distance']
    return segment


def summarize_placement_results(flattened_segmented_dict):
    """Summarizes the placement results of a fully located segment.
    Optional; for debugging and evaluation."""
    discretization_errors = []
    relative_errors = []
    total_number_of_legs = sum([len(segment) for segment in flattened_segmented_dict.values()])
    for person_id, segment in flattened_segmented_dict.items():
        for leg in segment:
            discretization_errors.append(leg['placed_distance_absolute_diff'])
            relative_errors.append(leg['placed_distance_relative_diff'])
    mean_discretization_error = sum(discretization_errors) / total_number_of_legs
    mean_relative_error = sum(relative_errors) / total_number_of_legs
    median_discretization_error = np.median(discretization_errors)
    median_relative_error = np.median(relative_errors)

    logger.info(f"Total number of legs: {total_number_of_legs}")
    logger.info(f"Average discretization error: {mean_discretization_error}")
    logger.info(f"Average relative error: {mean_relative_error}")
    logger.info(f"Median discretization error: {median_discretization_error}")
    logger.info(f"Median relative error: {median_relative_error}")

    return total_number_of_legs, mean_discretization_error, mean_relative_error, median_discretization_error, median_relative_error


def select_interesting_trips(flattened_segmented_dict, n: int):  # TODO: Implement
    """Selects a few interesting agent trips for debugging and evaluation."""
    interesting_trips = []
    for person_id, segment in flattened_segmented_dict.items():
        if len(segment) > 2:
            interesting_trips.append(segment)
    return interesting_trips


def build_position_on_segment_info(n: int) -> list[list[int]]:
    """Based on the number of legs in a segment, returns a list of lists that tells us at each level which legs to process."""
    # The first list contains numbers from 0 to n-2
    original_list = list(range(n - 1))
    result = []
    seen_elements = set()

    while original_list:
        result.append(original_list)
        original_list = original_list[1::2]

    result.reverse()

    # Remove elements from all subsequent lists once they've appeared in an earlier list
    cleaned_result = []
    for lst in result:
        cleaned_list = [x for x in lst if x not in seen_elements]
        seen_elements.update(cleaned_list)
        cleaned_result.append(cleaned_list)

    cleaned_result.reverse()

    return cleaned_result


def flatten_segmented_dict(segmented_dict: Dict[str, List[List[Dict[str, Any]]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Recombine the segments of each person into a single list of legs."""
    for person_id, segments in segmented_dict.items():
        segmented_dict[person_id] = [leg for segment in segments for leg in segment]
    return segmented_dict


# def build_candidate_tree(segment, tree):
#     """Work in progress. """

#     # At the second highest level, get n candidates with score for the location
#     candidates = find_location_candidates(segment[0]['from_location'], segment[-1]['to_location'], segment[0]['to_act_type'], tree[0][0][2], tree[0][1][2], 5)
#     logger.debug(candidates)

#     # At the next levels, get n candidates with score for each connected location
#     for level in range(1, len(tree)):
#         for i in range(0, len(tree[level]), 2):
#             candidates = find_location_candidates(segment[i]['from_location'], segment[i+1]['to_location'], segment[i]['to_act_type'], tree[level][i][2], tree[level][i+1][2], 5)
#             logger.debug(candidates)


def spread_distances(distance1, distance2, iteration=0, first_step=20):
    """Increases the difference between two distances, keeping them positive."""
    step = first_step * 2 ** (iteration + 1)
    if distance1 > distance2:
        distance1 += step
        distance2 -= step
    else:
        distance1 -= step
        distance2 += step
    return max(0, distance1), max(0, distance2)


def simple_locate_segment(person_legs):
    """Assumes start and end locations of segment are identical."""
    if len(person_legs) == 0:
        raise ValueError("No legs in segment.")

    elif len(person_legs) == 1:
        assert person_legs[0]['from_location'].size > 0 and person_legs[0][
            'to_location'].size > 0, "Both start and end locations must be known for a single leg."
        return person_legs

    # if there are only two legs, get a location within the two circles around home
    elif len(person_legs) == 2:
        logger.debug("Simple locating. Only two legs in segment.")
        distance1 = person_legs[0]['distance']
        distance2 = person_legs[1]['distance']
        if abs(distance1 - distance2) < 30:  # always meters!
            distance1, distance2 = spread_distances(distance1, distance2, first_step=20)
        candidates = find_ring_candidates(person_legs[0]['to_act_type'], person_legs[0]['from_location'], distance1,
                                          distance2)
        act_identifier, act_name, act_coord, act_cap, act_dist, act_score = monte_carlo_select_candidate(candidates)
        person_legs[0]['to_location'] = act_coord
        person_legs[1]['from_location'] = act_coord
        person_legs[0]['to_act_identifier'] = act_identifier
        person_legs[0]['to_act_name'] = act_name
        person_legs[0]['to_act_cap'] = act_cap
        person_legs[0]['to_act_score'] = act_score
        return person_legs

    else:
        total_distance = sum([leg['distance'] for leg in person_legs])
        traveled_distance = 0
        for i, leg in enumerate(person_legs):
            traveled_distance += leg['distance']
            remaining_legs = len(person_legs) - i

            if remaining_legs == 1:
                break  # done because no processing on last leg needed

            elif person_legs[i]['to_act_type'] == s.ACT_HOME:
                logger.debug("Home activity. Placing at, you guessed it, home (start location).")
                act_identifier, act_name, act_coord, act_cap, act_dist, act_score = None, "home", person_legs[i][
                    'from_location'], None, None, None

            elif traveled_distance >= total_distance / 2:

                if remaining_legs == 2:
                    logger.debug("Selecting location using simple two-leg method.")
                    assert person_legs[-1] == person_legs[
                        i + 1], "Last leg must be the last leg."  # TODO: Remove this line in production
                    act_identifier, act_name, act_coord, act_cap, act_dist, act_score = \
                        greedy_select_single_activity(person_legs[i]['from_location'], person_legs[-1]['to_location'],
                                                      person_legs[i]['to_act_type'], person_legs[i]['distance'],
                                                      person_legs[-1]['distance'], 5)

                else:
                    logger.debug("Selecting location using ring with angle restriction.")
                    distance = person_legs[i]['distance']
                    radius1, radius2 = spread_distances(distance, distance, iteration=0, first_step=20)
                    candidates = find_ring_candidates(person_legs[i]['to_act_type'], person_legs[i]['from_location'],
                                                      radius1, radius2, restrict_angle=True,
                                                      direction_point=person_legs[-1]['to_location'])

                    act_identifier, act_name, act_coord, act_cap, act_dist, act_score = monte_carlo_select_candidate(
                        candidates)
            else:
                logger.debug("Selecting location using ring.")
                distance = person_legs[i]['distance']
                radius1, radius2 = spread_distances(distance, distance, iteration=0, first_step=20)
                candidates = find_ring_candidates(person_legs[i]['to_act_type'], person_legs[i]['from_location'],
                                                  radius1,
                                                  radius2)
                act_identifier, act_name, act_coord, act_cap, act_dist, act_score = monte_carlo_select_candidate(
                    candidates)

            person_legs[i]['to_location'] = act_coord
            person_legs[i + 1]['from_location'] = act_coord
            person_legs[i]['to_act_identifier'] = act_identifier
            person_legs[i]['to_act_name'] = act_name
            person_legs[i]['to_act_cap'] = act_cap
            person_legs[i]['to_act_score'] = act_score

        return person_legs


def find_ring_candidates(type: str, center: np.ndarray, radius1: float, radius2: float, max_iterations=15,
                         min_candidates=10, restrict_angle=False, direction_point=None, angle_range=math.pi / 2) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find candidates within a ring around a center point.
    Iteratively increase the radii until a sufficient number of candidates is found."""
    i = 0
    logger.debug(
        f"Finding candidates for type {type} within a ring around {center} with radii {radius1} and {radius2}.")
    while True:
        candidates = MyTargetLocations.query_within_ring(type, center, radius1, radius2)
        if candidates is not None:
            # Filter candidates by angle
            if restrict_angle:
                angle_candidates = []
                for j, candidate_location in enumerate(candidates[2]):
                    if is_within_angle(candidate_location, center, direction_point, angle_range):
                        angle_candidates.append(j)
                candidates = tuple(
                    [arr[angle] if arr is not None else None for angle in angle_candidates]
                    if arr is not None
                    else None
                    for arr in candidates
                )
            if len(candidates[0]) >= min_candidates:
                logger.debug(f"Found {len(candidates[0])} candidates.")
                stats_tracker.log(f"Find_ring_candidates: Iterations for {type}", i)
                return candidates
        radius1, radius2 = spread_distances(radius1, radius2, iteration=i, first_step=20)
        i += 1
        logger.debug(f"Iteration {i}. Increasing radii to {radius1} and {radius2}.")
        if i > max_iterations:
            raise ValueError(f"Not enough candidates found after {max_iterations} iterations.")


def angle_between(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle = math.atan2(delta_y, delta_x)
    return angle


def is_within_angle(point, center, direction_point, angle_range):
    angle_to_point = angle_between(center, point)
    angle_to_direction = angle_between(center, direction_point)
    lower_bound = angle_to_direction - angle_range / 2
    upper_bound = angle_to_direction + angle_range / 2

    # Normalize angles between -pi and pi
    angle_to_point = (angle_to_point + 2 * math.pi) % (2 * math.pi)
    lower_bound = (lower_bound + 2 * math.pi) % (2 * math.pi)
    upper_bound = (upper_bound + 2 * math.pi) % (2 * math.pi)

    if lower_bound < upper_bound:
        return lower_bound <= angle_to_point <= upper_bound
    else:  # Covers the case where the angle range crosses the -pi/pi boundary
        return angle_to_point >= lower_bound or angle_to_point <= upper_bound


def update_dataframe(df: pd.DataFrame, placed_dict: Dict[str, Any]) -> pd.DataFrame:
    # Initialize new columns with NaN values
    df['from_location'] = np.nan
    df['to_location'] = np.nan
    df['placed_distance'] = np.nan
    df['placed_distance_absolute_diff'] = np.nan
    df['placed_distance_relative_diff'] = np.nan

    # Flatten the dictionary and create a DataFrame
    records = []
    for key, value in placed_dict.items():
        for entry in value:
            records.extend(entry)

    data_df = pd.DataFrame(records)

    # Merge the DataFrame on leg_id
    df = df.merge(data_df[[s.UNIQUE_LEG_ID_COL, 'from_location', 'to_location', 'placed_distance',
                           'placed_distance_absolute_diff',
                           'placed_distance_relative_diff']],
                  on=s.UNIQUE_LEG_ID_COL, how='left')

    return df


def locate_main_activities(persons_legs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Processes each person's legs to locate main activities for all persons.
    :return: Updated dictionary with located main activities for all persons.
    """
    for person_id, legs in persons_legs.items():
        persons_legs[person_id] = locate_main(legs)
    return persons_legs


start_time = time.time()
os.chdir(pipeline_setup.PROJECT_ROOT)
with open('locations_data_with_potentials.pkl', 'rb') as file:
    locations_data = pickle.load(file)
reformatted_locations_data = reformat_locations(locations_data)
MyTargetLocations = TargetLocations(reformatted_locations_data)

df = prepare_mid_df_for_legs_dict(number_of_persons=1000)
logger.debug("df prepared.")
dictu = populate_legs_dict_from_df(df)
logger.debug("dict populated.")
with_main_dict = locate_main_activities(dictu)
segmented_dict = segment_legs(with_main_dict)
logger.debug("dict segmented.")
pprint.pprint(segmented_dict)

# Advanced petre locator


# Greedy petre locator
for person_id, segments in segmented_dict.items():
    for segment in segments:
        segment = greedy_locate_segment(segment)
        segment = insert_placed_distances(segment)  # Just for analysis

# Simple locator
# for person_id, person_legs in dictu.items():
#     person_legs = simple_locate_segment(person_legs)
#     person_legs = insert_placed_distances(person_legs)

# (slightly modified) Hörl locator
# result = myhoerl.process(reformatted_locations_data, segmented_dict)

df = update_dataframe(df, segmented_dict)

segmented_dict = flatten_segmented_dict(segmented_dict)
for person_id, segment in segmented_dict.items():
    logger.debug(f"Person ID: {person_id}")
    for leg in segment:
        logger.debug(leg)

df.to_csv('mid_with_locations.csv', index=False)

summarize_placement_results(segmented_dict)

stats_tracker.print_stats()

end_time = time.time()
logger.debug(f"Execution time: {end_time - start_time} seconds.")

logger.debug("done")
