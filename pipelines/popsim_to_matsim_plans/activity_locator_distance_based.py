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


from typing import Dict, Tuple, Any
import numpy as np
from sklearn.neighbors import KDTree
import random as rnd

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

with open('locations_data_with_capacities.pkl', 'rb') as file:
    locations_data = pickle.load(file)
reformatted_data = reformat_locations(locations_data)
MyTargetLocations = TargetLocations(reformatted_data)


def sigmoid(x):
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

def score_locations(distances: np.ndarray, capacities: np.ndarray) -> np.ndarray:
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

# with open('locations_data_with_capacities.pkl', 'rb') as file:
#     locations_data = pickle.load(file)

# logger.debug(f"Locations data with potentials: {locations_data}")

# reformatted_data = reformat_locations(locations_data)

# logger.debug(f"Reformatted locations data: {reformatted_data}")

# MyTargetLocations = TargetLocations(reformatted_data)
# #test with epsg:25832
# test_candidates = MyTargetLocations.query("shop", np.array([549637.87573102, 5796618.40418383]), 5)
# logger.debug(f"Test candidates for shop at 52.432047, 9.687902: Identifiers: {test_candidates[0]}, Names: {test_candidates[1]}, Coordinates: {test_candidates[2]}, Capacities: {test_candidates[3]}, Distances: {test_candidates[4]}")
             
# # # Usage
# # location_scoring_function = LocationScoringFunction(sigmoid_beta=1.0, sigmoid_delta_t=0.0)
# # identifiers = np.array([1, 2, 3])
# # locations = np.array([[0, 0], [1, 1], [2, 2]])
# # distances = np.array([10, 20, 30])
# # capacities = np.array([100, 200, 300])
# # time_diffs = np.array([1, 2, 3])

# # scores = location_scoring_function.score_locations(identifiers, locations, distances, capacities, time_diffs)
# # print("Scores:", scores)

def euclidean_distance(start: np.ndarray, end: np.ndarray) -> float:
    """Compute the Euclidean distance between two points."""
    return np.linalg.norm(end - start)

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
    d = euclidean_distance(center1, center2)

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
    candidate_scores = score_locations(candidate_distances, candidate_capacities)
    
    if intersect2 is not None:
        candidate_identifiers2, candidate_names2, candidate_coordinates2, candidate_capacities2, candidate_distances2 = MyTargetLocations.query(purpose, intersect2, num_candidates)
        candidate_scores2 = score_locations(candidate_distances2, candidate_capacities2)
    
        return [candidate_identifiers, candidate_names, candidate_coordinates, candidate_capacities, candidate_distances, candidate_scores],\
            [candidate_identifiers2, candidate_names2, candidate_coordinates2, candidate_capacities2, candidate_distances2, candidate_scores2]

    return [candidate_identifiers, candidate_names, candidate_coordinates, candidate_capacities, candidate_distances, candidate_scores], None
    
def populate_legs_dict_from_df(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Uses the MiD df to populate a nested dictionary with leg information for each person."""
    nested_dict = defaultdict(list)
    
    # Populate the defaultdict with data from the DataFrame
    for row in df.itertuples(index=False):
        identifier: str = getattr(row, s.UNIQUE_P_ID_COL)

        from_location: np.ndarray = np.array(row.from_location) if row.from_location is not None else np.array([])
        to_location: np.ndarray = np.array(row.to_location) if row.to_location is not None else np.array([])

        leg_info: Dict[str, Any] = {
            'leg_id': getattr(row, s.UNIQUE_LEG_ID_COL),
            'to_activity': getattr(row, s.LEG_TO_ACTIVITY_COL),
            'distance': getattr(row, s.LEG_DISTANCE_COL),
            'from_location': from_location,
            'to_location': to_location
        }
        nested_dict[identifier].append(leg_info)

    # Convert defaultdict to dict for cleaner output and better compatibility
    return dict(nested_dict)


def generate_random_location_within_hanover():
    """Generate a random coordinate within Hanover, Germany, in EPSG:25832."""
    xmin, xmax = 546000, 556000
    ymin, ymax = 5800000, 5810000
    x = random.uniform(xmin, xmax)
    y = random.uniform(ymin, ymax)
    return np.array([x, y])

def prepare_mid_df_for_legs_dict() -> pd.DataFrame:
    """Temporarily prepare the MiD DataFrame for the leg dictionary function."""
    df = h.read_csv(s.ENHANCED_MID_FILE)
    
    # Initialize columns with empty objects to ensure dtype compatibility
    df["from_location"] = None
    df["to_location"] = None
    
    # Throw out rows with missing values in the distance column
    n_cols = df.shape[1]
    df = df.dropna(subset=[s.LEG_DISTANCE_COL])
    logger.debug(f"Dropped {n_cols - df.shape[1]} rows with missing distance values.")

    # Ensure these columns are treated as object type to store arrays
    df["from_location"] = df["from_location"].astype(object)
    df["to_location"] = df["to_location"].astype(object)

    # Process each person
    for person_id, group in df.groupby(s.PERSON_ID_COL):
        random_location = generate_random_location_within_hanover()
        df.at[group.index[0], "from_location"] = random_location
        df.at[group.index[-1], "to_location"] = random_location
    
    print(df.head())
    return df

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

def calculate_length_with_slack(length1, length2, slack_factor = 2, min_slack_lower = 0.2, min_slack_upper = 0.2) -> List[float]:
    """min_slacks must be between 0 and 0.49"""
    
    length_sum = length1 + length2 # is also real maximum length
    length_diff = abs(length1 - length2) # is also real minimum length
    shorter_leg = min(length1, length2)
    
    result = length_sum / slack_factor
    
    wanted_minimum = length_diff + shorter_leg * min_slack_lower
    wanted_maximum = length_sum - shorter_leg * min_slack_upper
    
    if result <= wanted_minimum:
        result = wanted_minimum
    elif result > wanted_maximum:
        result = wanted_maximum
        
    # assert result is a number
    assert not np.isnan(result), f"Result is NaN. Lengths: {length1}, {length2}, Slack factor: {slack_factor}, Min slack lower: {min_slack_lower}, Min slack upper: {min_slack_upper}"

    return [length_diff, wanted_minimum, result, wanted_maximum, length_sum]
    
def build_estimation_tree(distances: List[float]) -> List[List[List[float]]]: # Tree level, Leg, Lengths
    tree: List[List[List[float]]] = []  
    
    while len(distances) > 1:
        new_distances: List[float] = []
        combined_pairs: List[List[float]] = []
        for i in range(0, len(distances) - 1, 2):
            combined_list: List[float] = calculate_length_with_slack(distances[i], distances[i + 1])
            new_distances.append(combined_list[2])
            combined_pairs.append(combined_list)
        
        if len(distances) % 2 != 0:  
            # Carry over the estimation from the so-far built tree
            last_pair = tree[-1][-1] if tree else [distances[-1], distances[-1], distances[-1], distances[-1], distances[-1]]
            combined_pairs.append(last_pair)
            new_distances.append(last_pair[2])  # Append only the center value for next level processing
        
        distances = new_distances
        tree.append(combined_pairs)

    return tree

def adjust_estimation_tree(tree: List[List[List[float]]], real_distance: float, strong_adjust: bool = True) -> List[List[List[float]]]: # Tree level, Leg, Lengths
    """If there is strong overshooting or undershooting in the estimation tree, adjust the distances. 
    It tries to keep given slack constraints, unless it is not possible.
    
    TEMP: Erstmal nur strong adjust, weak adjust ist vlt grundsätzlich nicht sinnvoll.
    
    Parameters:
    tree: The estimation tree to adjust.
    real_distance: The real total distance to adjust the tree to.
    strong_adjust: If True, real_bounds are used for adjustment, if False, wanted_bounds are used.
    """
    
    logger.debug(tree)
            
    # Adjust the highest level
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

    if tree[-1][0][1] < real_distance < tree[-1][0][3]: # Checking for wanted maximum and minimum, not real maximum and minimum
        logger.debug("Real total distance is within bounds of highest level.")
        return tree
    else:
        # Traverse from one below the highest level down to including level 1 (which has index 0) - and then down to -1 just for the last check
        for level in range(len(tree) - 1, -2, -1):
            
            # At the lowest level, check plausibilities and return
            # TODO: Move to end??
            if level == -1:
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
                        return tree
                    else:
                        logger.debug("Strong adjustment failed.")
                        logger.debug(tree)
                        return tree
                else:
                    if plausible:
                        # These are the bounds of the first level estimation, which are set in stone by the known real distances
                        logger.debug("Adjustment succeeded with wanted bounds.")
                        return tree
                    else:
                        logger.debug("Adjustment failed with wanted bounds, trying strong adjustment with real bounds.")
                        return adjust_estimation_tree(tree, real_distance, strong_adjust = True)
                        
                        
            for i in range(0, len(tree[level]), 2): # Traverse in pairs, skipping the last one if it's an odd number
                if i == len(tree[level]) - 1:
                    continue
                
                higher_leg_value = tree[level+1][i//2][2]
                higher_leg_lower_bound = tree[level+1][i//2][l_bound_idx]
                higher_leg_upper_bound = tree[level+1][i//2][u_bound_idx]

                leg1_value = tree[level][i][2]
                leg2_value = tree[level][i+1][2]

                if higher_leg_value < higher_leg_lower_bound: # (Real) higher_leg_value can be zero if start and end location are the same
                    logger.debug(f"Strong overshot. Level: {level}, i: {i}, higher_leg_value: {higher_leg_value}, higher_leg_lower_bound: {higher_leg_lower_bound}")
                    # Yes, overshot: The "ground truth" is lower than the lower bound of the estimation
                    if leg1_value > leg2_value:
                        # Make longer leg shorter, shorter leg longer
                        L_bounds1 = abs(tree[level][i][0] - leg1_value)
                        L_bounds2 = abs(tree[level][i+1][4] - leg2_value)
                        delta_L_high = abs(higher_leg_value - higher_leg_lower_bound)

                        delta_L1 = (L_bounds1 * delta_L_high) / (L_bounds1 + L_bounds2)
                        delta_L2 = (L_bounds2 * delta_L_high)/ (L_bounds1 + L_bounds2)

                        tree[level][i][2] -= delta_L1
                        tree[level][i+1][2] += delta_L2
                    else:
                        L_bounds1 = abs(tree[level][i][4] - leg1_value)
                        L_bounds2 = abs(tree[level][i+1][0] - leg2_value)
                        delta_L_high = abs(higher_leg_value - higher_leg_lower_bound)

                        delta_L1 = (L_bounds1 * delta_L_high) / (L_bounds1 + L_bounds2)
                        delta_L2 = (L_bounds2 * delta_L_high)/ (L_bounds1 + L_bounds2)

                        tree[level][i][2] += delta_L1
                        tree[level][i+1][2] -= delta_L2

                elif higher_leg_value > higher_leg_upper_bound: 
                    
                    logger.debug(f"Strong undershot. Level: {level}, i: {i}, higher_leg_value: {higher_leg_value}, higher_leg_upper_bound: {higher_leg_upper_bound}")
                    # Yes, undershot: The "ground truth" is higher than the upper bound of the estimation
                    # Make both legs longer (both move to upper bound)
                    L_bounds1 = abs(tree[level][i][4] - leg1_value)
                    L_bounds2 = abs(tree[level][i+1][4] - leg2_value)
                    delta_L_high = abs(higher_leg_value - higher_leg_upper_bound)

                    delta_L1 = (L_bounds1 * delta_L_high) / (L_bounds1 + L_bounds2)
                    delta_L2 = (L_bounds2 * delta_L_high)/ (L_bounds1 + L_bounds2)

                    tree[level][i][2] += delta_L1
                    tree[level][i+1][2] += delta_L2

    

def locate_segment(segment):
    
    if len(segment) == 0:
        raise ValueError("No legs in segment.")
    elif len(segment) == 1:
        assert segment[0]['from_location'].size > 0 and segment[0]['to_location'].size > 0, "Both start and end locations must be known for a single leg."
        return segment
    # if there are only two legs, we can find the loc immediately
    elif len(segment) == 2:
        return segment #TODO: Implement this
    else:
    
        
        # get distance entries of all legs
        distances = [leg['distance'] for leg in segment]
        for distance in distances: # All distances must be numbers
            assert isinstance(distance, (int, float)), f"Distance is not a number: {distance}"
            
        real_distance = euclidean_distance(segment[0]['from_location'], segment[-1]['to_location'])
        
        tree = build_estimation_tree(distances)
        tree = adjust_estimation_tree(tree, real_distance, strong_adjust = True)
        
        print (tree)
        
        
def greedy_locate_segment(segment):
    
    if len(segment) == 0:
        raise ValueError("No legs in segment.")
    elif len(segment) == 1:
        assert segment[0]['from_location'].size > 0 and segment[0]['to_location'].size > 0, "Both start and end locations must be known for a single leg."
        return segment
    # if there are only two legs, we can find the loc immediately
    elif len(segment) == 2:
        greedy_place_single_acitivity(segment)
        return segment #TODO: Implement this
    else:
        distances = [leg['distance'] for leg in segment]

        real_distance = euclidean_distance(segment[0]['from_location'], segment[-1]['to_location'])
        
        tree = build_estimation_tree(distances)
        tree = adjust_estimation_tree(tree, real_distance, strong_adjust = True)
        position_on_segment_tree = build_position_on_segment_tree(len(distances)) # same structure as tree
        
        for level in range(len(tree), 0, -1): # TODO: overhaul this placeholder
            for i in range(0, len(tree[level]), 2):
                if level == len(tree):
                    # At the highest level, get n candidates with score for the location
                    candidates = find_location_candidates(segment[i]['from_location'], segment[i+1]['to_location'], segment[i]['to_activity'], tree[level][i][2], tree[level][i+1][2], 5)
                    print(candidates)
                else:
                    # At the next levels, get n candidates with score for each connected location
                    for j in range(0, len(tree[level]), 2):
                        candidates = find_location_candidates(segment[j]['from_location'], segment[j+1]['to_location'], segment[j]['to_activity'], tree[level][j][2], tree[level][j+1][2], 5)
                        print(candidates)

        
def build_position_on_segment_tree(n: int) -> list[list[int]]:
    original_list = list(range(1, n + 1))
    result = []
    current_list = original_list

    while len(current_list) > 0:
        result.append(current_list)
        # Create the new list by taking every second element
        next_list = current_list[1::2]
        # If the original list has an odd length and we removed the last element, we add it back
        if len(current_list) % 2 == 1:
            next_list.append(current_list[-1])
        current_list = next_list
            
    return result
        

def plant_tree(segment: List[Dict[str, Any]], tree: List[List[List[float]]]):
    """Adds the estimation tree distances to the segment at the right places."""
    segment_step = 2**level
    segment_pos = segment_step -1
    
    location_from_pos = (n * segment_step) - 1
    location_to_pos = (n+2 * segment_step) + 1
    
    

def build_candidate_tree(segment, tree):
    
    # At the second highest level, get n candidates with score for the location
    candidates = find_location_candidates(segment[0]['from_location'], segment[-1]['to_location'], segment[0]['to_activity'], tree[0][0][2], tree[0][1][2], 5)
    print(candidates)
    
    # At the next levels, get n candidates with score for each connected location
    for level in range(1, len(tree)):
        for i in range(0, len(tree[level]), 2):
            candidates = find_location_candidates(segment[i]['from_location'], segment[i+1]['to_location'], segment[i]['to_activity'], tree[level][i][2], tree[level][i+1][2], 5)
            print(candidates)

df = prepare_mid_df_for_legs_dict()
print("df prepared.")
dictu = populate_legs_dict_from_df(df)
print("dict populated.")
segmented_dict = segment_legs(dictu)
print("dict segmented.")
print (segmented_dict)
# for person_id, segments in segmented_dict.items():
#     print(f"Person {person_id}:")
#     for i, segment in enumerate(segments, 1):
#         print(f"  Segment {i}:")
#         for leg in segment:
#             print(f"    {leg}")
#     print()

for person_id, segments in segmented_dict.items():
    for segment in segments:
        locate_segment(segment)
        
print("done")

