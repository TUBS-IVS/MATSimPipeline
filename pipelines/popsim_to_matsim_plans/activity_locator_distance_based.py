import random

import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any

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

        :param x: The input value (e.g. distance from desired point) - can be a number, list, or numpy array.
        :return: Sigmoid function value.
        """
        x = np.array(x)  # Ensure x is a numpy array
        z = -self.sigmoid_beta * (x - self.sigmoid_delta_t)
        # Use np.clip to limit the values in z to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(z))

    def score_locations(self, distances: np.ndarray, capacities: np.ndarray) -> np.ndarray:
        """
        Evaluate the returned locations by distance and capacity and return a score.

        :param distances: Numpy array of distances from desired point for the returned locations.
        :param capacities: Numpy array of remaining capacities for the returned locations.
        :return: Numpy array of scores for the returned locations.
        """
        # Calculate the base score for each location
        base_scores = capacities / distances  # TODO: Improve scoring function

        # Normalize the scores to ensure they sum to 1
        scores = base_scores / np.sum(base_scores)
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
    :return: A tuple containing one or two intersection points (each as a np.ndarray).
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

    if d == 0 and r1 == r2:
        logger.info("Infinite intersections: The start and end points and radii are identical.")
        logger.info("Choosing a point on the perimeter of the circles.")
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

def find_location_candidates(start_coord: np.ndarray, end_coord: np.ndarray, purpose: str, distance_start_to_act: float, distance_act_to_end: float, num_candidates: int) -> Tuple[str, np.ndarray]:
    """
    Find n location candidates for a given activity purpose between two known locations.
    """
    intersect1, intersect2 = find_circle_intersections(start_coord, distance_start_to_act, end_coord, distance_act_to_end)
    
    candidate_identifiers, candidate_names, candidate_coordinates, candidate_capacities, candidate_distances = MyTargetLocations.query(purpose, intersect1, num_candidates)
    
    if intersect2 is not None:
        candidate_identifiers2, candidate_names2, candidate_coordinates2, candidate_capacities2, candidate_distances2 = MyTargetLocations.query(purpose, intersect2, num_candidates)

        candidate_identifiers = np.concatenate((candidate_identifiers, candidate_identifiers2), axis=0)
        candidate_names = np.concatenate((candidate_names, candidate_names2), axis=0)
        candidate_coordinates = np.concatenate((candidate_coordinates, candidate_coordinates2), axis=0)
        candidate_capacities = np.concatenate((candidate_capacities, candidate_capacities2), axis=0)
        candidate_distances = np.concatenate((candidate_distances, candidate_distances2), axis=0)
        
    candidate_scores = location_scoring_function.score_locations(candidate_distances, candidate_capacities)
    
    return (candidate_identifiers, candidate_names, candidate_coordinates, candidate_capacities, candidate_distances, candidate_scores)
    
    
def populate_legs_dict_from_df(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    nested_dict = defaultdict(list)
    
    # Populate the defaultdict with data from the DataFrame
    for row in df.itertuples(index=False):
        identifier: str = row.person_id

        from_location: np.ndarray = np.array(row.from_location) if row.from_location is not None else np.array([])
        to_location: np.ndarray = np.array(row.to_location) if row.to_location is not None else np.array([])

        leg_info: Dict[str, Any] = {
            'leg_id': row.leg_id,
            'to_activity': row.to_activity,
            'distance': row.distance,
            'from_location': from_location,
            'to_location': to_location
        }
        nested_dict[identifier].append(leg_info)

    # Convert defaultdict to dict for cleaner output and better compatibility
    return dict(nested_dict)

def segment_legs(nested_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[List[Dict[str, Any]]]]:
    segmented_dict = defaultdict(list)
    
    for person_id, legs in nested_dict.items():
        segment = []
        for leg in legs:
            if not segment:
                # Start a new segment with the first leg
                segment.append(leg)
            else:
                segment.append(leg)
                if leg['to_location'].size > 0:
                    segmented_dict[person_id].append(segment)
                    segment = []
        # If there are remaining legs in the segment after the loop, add them as well
        if segment:
            segmented_dict[person_id].append(segment)
        
    return dict(segmented_dict)




data = {
    'person_id': [
        'A', 'A', 'A', 'A', 'A', # Person A
        'B', 'B', 'B', 'B', 'B', 'B',  # Person B
        'C', 'C', 'C', 'C'  # Person C
    ],
    'leg_id': list(range(1, 16)),
    'to_activity': [
        'activity_1', 'activity_2', 'activity_3', 'activity_4', 'activity_5',  # Person A
        'activity_6', 'activity_7', 'activity_8', 'activity_9', 'activity_10', 'activity_11',  # Person B
        'activity_12', 'activity_13', 'activity_14', 'activity_15'  # Person C
    ],
    'distance': [
        100, 200, 300, 400, 500,  # Person A
        600, 700, 800, 900, 1000, 1100,  # Person B
        1200, 1300, 1400, 1500  # Person C
    ],
    'from_location': [
        (1, 2), None, (3, 4), None, None,  # Person A
        (5, 6), None, (7, 8), None, None, (9, 10),  # Person B
        (11, 12), None, None, (13, 14)  # Person C
    ],
    'to_location': [
        None, (3, 4), None, None, (5, 6),  # Person A
        None, (7, 8), None, None, (9, 10), None,  # Person B
        None, None, (13, 14), None  # Person C
    ]
}


def calculate_length_with_slack(length1, length2, slack_factor, min_slack_lower = 0.2, min_slack_upper = 0.2):
    """min_slacks must be between 0 and 0.49"""
    
    length_sum = length1 + length2
    length_diff = abs(length1 - length2)
    shorter_leg = min(length1, length2)
    
    result = length_sum / slack_factor
    
    wanted_minimum = length_diff + shorter_leg * min_slack_lower
    wanted_maximum = length_sum - shorter_leg * min_slack_upper
    
    if result < wanted_minimum:
        return wanted_minimum
    elif result > wanted_maximum:
        return wanted_maximum

    return result  # Within bounds
    




diccy = populate_legs_dict_from_df(pd.DataFrame(data))
segmented_dict = segment_legs(diccy)

for person_id, segments in segmented_dict.items():
    print(f"Person {person_id}:")
    for i, segment in enumerate(segments, 1):
        print(f"  Segment {i}:")
        for leg in segment:
            print(f"    {leg}")
    print()