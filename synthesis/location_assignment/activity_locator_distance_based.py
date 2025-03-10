import copy
import math
import os
import pickle
import pprint
import random
import time
import json

from build.lib.ivs_helpers import stats_tracker
from openpyxl.pivot.fields import Boolean
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Literal
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
# from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree

from utils import settings as s, helpers as h, pipeline_setup
# from utils.types import PlanLeg, PlanSegment, SegmentedPlan, SegmentedPlans, UnSegmentedPlan, UnSegmentedPlans
from utils.logger import logging

from typing import NamedTuple, Tuple
from frozendict import frozendict

logger = logging.getLogger(__name__)


# @dataclass
# class Leg:
#     __slots__ = (
#         'leg_id', 'to_act_type', 'distance', 'from_location',
#         'to_location', 'mode', 'is_main_activity', 'home_to_main_distance'
#     )
#     leg_id: str
#     to_act_type: str
#     distance: float
#     from_location: np.ndarray
#     to_location: np.ndarray
#     mode: str
#     is_main_activity: int
#     home_to_main_distance: float

class Leg(NamedTuple):
    unique_leg_id: str
    from_location: np.ndarray
    to_location: np.ndarray
    distance: float
    to_act_type: str
    to_act_identifier: str


# Details are later also available in the unified df. This is just for algos that need it.
class DetailedLeg(NamedTuple):
    unique_leg_id: str
    from_location: np.ndarray
    to_location: np.ndarray
    distance: float
    to_act_type: str
    to_act_identifier: str
    mode: str
    is_main_activity: bool
    mirrors_main_activity: bool
    home_to_main_distance: float


Segment = Tuple[Leg, ...]  # A segment of a plan (immutable tuple of legs)
SegmentedPlan = Tuple[Segment, ...]  # A full plan split into segments
SegmentedPlans = frozendict[str, SegmentedPlan]  # All agents' plans (person_id -> SegmentedPlan)

DetailedSegment = Tuple[DetailedLeg, ...]  # A segment of a plan (immutable tuple of legs)
DetailedSegmentedPlan = Tuple[DetailedSegment, ...]  # A full plan split into segments
DetailedSegmentedPlans = frozendict[str, DetailedSegmentedPlan]  # All agents' plans (person_id -> SegmentedPlan)



class GermanPopulationDensity:
    pass


class GermanTrainStations:
    pass


# either near Bahnhof (pt) or according to landuse (car)
class CommuterPlacer:
    """
    Place commuters (defined by a commuter matrix and the home-main distance) near a train station or according to land use.
    Use before the main activity locator.

    Currently:
    - Sample n (given by matrix) commuters from whole region to another cell using distances
        - Get all distances of all persons to all cell centroids (takes 2 seconds)
        - Place all
    - Place them near a train station or randomly

    Possibilities:
    - Search for commuters more carefully using boundary intersection test
    - Apply commuters internally (within region)
    """

    # Values are 1.: in meters, 2.: guessed
    p_hat_zweitwohnung = {0: 0, 50000: 0.1, 100000: 0.4, 150000: 0.7, 200000: 0.8, 250000: 0.9, 300000: 0.9,
                          350000: 0.9, 400000: 1}

    # TODO: get station data from all of germany
    # TODO: same for land use(?)
    # Build KDTrees from stations and from land use centroids

    # def __init__(self, target_locations: TargetLocations, legs_dict: Dict[str, list[dict[str, Any]]]):
    #     self.target_locations = target_locations
    #     self.legs_dict = legs_dict

    def place_commuter(self):
        """
        Place commuters (defined by a matrix and the home-main distance) near a train station or according to land use.
        Run this before the main activity locator (so likely before anything else).
        """
        # for unsegmented_plan in unsegmented_plans.values():
        #     unsegmented_plan: UnSegmentedPlan

    def place_commuters_at_stations(self, target_cell, commuters: SegmentedPlans):
        """
        We don't care about the exact traffic in outside cells. But we do want commuters that use pt to keep using it.
        """

        # TODO: DO we really?
        raise NotImplementedError

    def place_commuter_by_landuse(self, target_cell, commuters):
        raise NotImplementedError

    def run(self):
        for person_id, person_legs in tqdm(self.legs_dict.items(), desc="Processing persons"):
            self.locate_main(person_legs)  # In-place
        return self.legs_dict

    def is_commuter(self, person_legs: List[Dict[str, Any]]) -> bool:
        """Check if a person is a commuter based on their legs."""
        raise NotImplementedError
        # PreFilter by existing work or education or business legs
        # Sort first up to wanted number+some by distance home-to-main
        # Remove the rest
        # Do more precise check.

    # def locate_commuter(self, person_legs: UnSegmentedPlan) -> UnSegmentedPlan:
    #     """Gets a person's main activity and locates it.
    #     Currently just uses the Euclidean distance and potentials.
    #     Planned to also use O-D matrices.
    #     :return: Updated list of legs with located main activities.
    #     """
    #     main_activity_index, main_activity_leg = h.get_main_activity_leg(person_legs)
    #     if main_activity_leg is None:
    #         return person_legs
    #
    #     # Skip if main already located
    #     to_location = main_activity_leg.get('to_location')
    #     assert isinstance(to_location, np.ndarray), "Bad location format."
    #     if to_location.size != 0:
    #         return person_legs
    #
    #     # TODO go from here
    #     target_activity = main_activity_leg['to_act_type']
    #     home_location = person_legs[0]['from_location']
    #     estimated_distance_home_to_main = person_legs[0]['home_to_main_distance']
    #
    #     # Radii are iteratively spread by find_ring_candidates until a candidate is found
    #     radius1, radius2 = h.spread_distances(estimated_distance_home_to_main,
    #                                           estimated_distance_home_to_main)  # Initial
    #     candidates = self.target_locations.find_ring_candidates(target_activity, home_location, radius1=radius1,
    #                                                             radius2=radius2)
    #     scores = EvaluationFunction.evaluate_candidates(candidates[-2], None, len(candidates[-2]))
    #     chosen_candidate, score = (
    #         EvaluationFunction.select_candidates(candidates, scores, 1, 'monte_carlo'))
    #
    #     act_identifier, act_name, act_coord, act_pot, act_dist = chosen_candidate
    #
    #     # Update the main activity leg and the subsequent leg
    #     person_legs[main_activity_index]['to_location'] = act_coord
    #     person_legs[main_activity_index]['to_act_identifier'] = act_identifier
    #     person_legs[main_activity_index]['to_act_name'] = act_name
    #     person_legs[main_activity_index]['to_act_cap'] = act_pot
    #     person_legs[main_activity_index]['to_act_score'] = score
    #
    #     if main_activity_index + 1 < len(person_legs):  # Set from location if there is a subsequent leg
    #         person_legs[main_activity_index + 1]['from_location'] = act_coord
    #
    #     return person_legs


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


class Potentials:
    """
    Holds a dict of potentials for each location, by identifier, for each type.
    """

    # TODO: Maybe use np.arr instead
    def __init__(self, potentials: Dict[str, Dict[str, int]]):
        self.potentials = potentials

    def decrement(self, identifier, act_type):
        raise NotImplementedError

    def get(self, identifier, act_type):
        raise NotImplementedError


class TargetLocations:
    """
    Spatial index of activity locations split by type.
    This class is used to quickly find the nearest activity locations for a given location.
    """

    def __init__(self, json_folder_path: str):
        self.data: Dict[str, Dict[str, np.ndarray]] = self.load_reformatted_osmox_data(h.get_files(json_folder_path))
        self.indices: Dict[str, cKDTree] = {}

        for type, pdata in self.data.items():
            logger.info(f"Constructing spatial index for {type} ...")
            self.indices[type] = cKDTree(pdata["coordinates"])

    @staticmethod
    def load_reformatted_osmox_data(file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Convert lists back to numpy arrays
        for purpose in data:
            for key in data[purpose]:
                data[purpose][key] = np.array(data[purpose][key])
        return data

    def hoerl_query(self, purpose, location):
        _, index = self.indices[purpose].query(location.reshape(1, -1))
        index = index[0]
        identifier = self.data[purpose]["identifiers"][index]
        location = self.data[purpose]["coordinates"][index]
        return identifier, location

    def query_closest(self, type: str, location: np.ndarray, num_candidates: int = 1) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the nearest activity locations for one or more points.
        :param type: The type category to query.
        :param location: A 1D numpy array for a single point (e.g., [1.5, 2.5]) or a 2D numpy array for multiple points (e.g., [[1.5, 2.5], [3.0, 4.0]]).
        :param num_candidates: The number of nearest candidates to return.
        :return: A tuple containing numpy arrays: identifiers, coordinates, and potentials of the nearest candidates.
        """
        # Query the KDTree directly (handles both 1D and 2D inputs)
        _, indices = self.indices[type].query(location, k=num_candidates)

        # Retrieve data for the nearest candidates
        data_type = self.data[type]
        return (
            data_type["identifiers"][indices],
            data_type["coordinates"][indices],
            data_type["potentials"][indices],
        )

    def query_within_ring(self, act_type: str, location: np.ndarray, radius1: float, radius2: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the activity locations within a ring defined by two radii around a location and type.
        :param act_type: The activity category to query.
        :param location: A 1D numpy array representing the location to query (coordinates [1.5, 2.5]).
        :param radius1: Any of the two radii defining the ring.
        :param radius2: The other one.
        :return: A tuple containing identifiers, coordinates, and remaining potentials of candidates.
        """
        # Ensure location is a 2D array with a single location
        location = location.reshape(1, -1)

        outer_radius = max(radius1, radius2)
        inner_radius = min(radius1, radius2)

        # Query points within the outer radius
        tree: cKDTree = self.indices[act_type]
        outer_indices = tree.query_ball_point(location, outer_radius)[0]  # Indices of points within the outer radius

        if not outer_indices:
            return None

        # Query points within the inner radius
        inner_indices = tree.query_ball_point(location, inner_radius)[0]  # Indices of points within the inner radius

        # Filter indices to get only points in the annulus
        annulus_indices = list(set(outer_indices) - set(inner_indices))

        if not annulus_indices:
            return None

        # Retrieve corresponding activity data
        data_type = self.data[act_type]
        identifiers = data_type["identifiers"][annulus_indices]
        coordinates = data_type["coordinates"][annulus_indices]
        potentials = data_type["potentials"][annulus_indices]

        return identifiers, coordinates, potentials

    def query_within_two_overlapping_rings(self, act_type: str, location1: np.ndarray, location2: np.ndarray,
                                           radius1a: float, radius1b: float, radius2a: float, radius2b: float,
                                           max_number_of_candidates: int = None):

        location1 = location1[None, :] if location1.ndim == 1 else location1
        location2 = location2[None, :] if location2.ndim == 1 else location2

        outer_radius1, inner_radius1 = max(radius1a, radius1b), min(radius1a, radius1b)
        outer_radius2, inner_radius2 = max(radius2a, radius2b), min(radius2a, radius2b)

        outer_indices1 = self.indices[act_type].query_ball_point(location1, outer_radius1)[0]
        outer_indices2 = self.indices[act_type].query_ball_point(location2, outer_radius2)[0]

        if not outer_indices1 or not outer_indices2:
            return None

        outer_intersection = set(outer_indices1).intersection(outer_indices2)
        if not outer_intersection:
            return None

        inner_indices1 = set(self.indices[act_type].query_ball_point(location1, inner_radius1)[0])
        inner_indices2 = set(self.indices[act_type].query_ball_point(location2, inner_radius2)[0])

        overlapping_indices = list(outer_intersection - (inner_indices1.union(inner_indices2)))
        if not overlapping_indices:
            return None

        if max_number_of_candidates and len(overlapping_indices) > max_number_of_candidates:
            overlapping_indices = np.random.choice(overlapping_indices, max_number_of_candidates, replace=False)

        data = self.data[act_type]
        overlapping_indices = np.array(overlapping_indices)
        candidate_identifiers = data["identifiers"][overlapping_indices]
        candidate_coordinates = data["coordinates"][overlapping_indices]
        candidate_potentials = data["potentials"][overlapping_indices]

        return candidate_identifiers, candidate_coordinates, candidate_potentials

    def sample(self, act_type: str, random: random.Random) -> Tuple[Any, np.ndarray]:
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

    def find_ring_candidates(self, act_type: str, center: np.ndarray, radius1: float, radius2: float, max_iterations=20,
                             min_candidates=10, restrict_angle=False, direction_point=None, angle_range=math.pi / 1.5) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find candidates within a ring around a center point.
        Iteratively increase the radii until a sufficient number of candidates is found."""
        i = 0
        if logger.isEnabledFor(logging.DEBUG): logger.debug(
            f"Finding candidates for type {act_type} within a ring around {center} with radii {radius1} and {radius2}.")
        while True:
            candidates = self.query_within_ring(act_type, center, radius1, radius2)
            if candidates is not None:
                # Filter candidates by angle
                if restrict_angle:
                    angle_candidates = []
                    for j, candidate_location in enumerate(candidates[1]):
                        if is_within_angle(candidate_location, center, direction_point, angle_range):
                            angle_candidates.append(j)
                    candidates = tuple(
                        [arr[angle] if arr is not None else None for angle in angle_candidates]
                        if arr is not None
                        else None
                        for arr in candidates
                    )
                if len(candidates[0]) >= min_candidates:
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Found {len(candidates[0])} candidates.")
                    stats_tracker.log(f"Find_ring_candidates: Iterations for {act_type}", i)
                    return candidates
            radius1, radius2 = h.spread_distances(radius1, radius2, iteration=i, first_step=20)
            i += 1
            if logger.isEnabledFor(logging.DEBUG): logger.debug(
                f"Iteration {i}. Increasing radii to {radius1} and {radius2}.")
            if i > max_iterations:
                raise ValueError(f"Not enough candidates found after {max_iterations} iterations.")

    def ensure_overlap(self, location1, location2, r1a, r1b, r2a, r2b):
        """
        Ensure that two annuli (rings) defined by:
          - Ring 1: radii [r1a, r1b] around location1
          - Ring 2: radii [r2a, r2b] around location2

        If no overlap exists, we minimally adjust the radii so that they at least
        just touch or overlap.

        Parameters
        ----------
        location1 : np.ndarray
            Center of the first annulus.
        location2 : np.ndarray
            Center of the second annulus.
        r1a : float
            Inner radius of the first annulus.
        r1b : float
            Outer radius of the first annulus.
        r2a : float
            Inner radius of the second annulus.
        r2b : float
            Outer radius of the second annulus.

        Returns
        -------
        r1a, r1b, r2a, r2b : float
            Adjusted annulus radii ensuring at least a degenerate overlap.
        """
        D = np.linalg.norm(location2 - location1)
        changed_radii = False
        # Condition 1: Too far apart -> make them just touch at outer edges
        if D > (r1b + r2b):
            changed_radii = True
            if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Locations too far apart. Increasing radii to touch.")
            # Increase sum of outer radii to D
            delta = D - (r1b + r2b)
            r1b += delta / 2.0
            r2b += delta / 2.0

        # Check if one ring is fully inside the other:
        # Ring1 fully inside Ring2 if r1b < r2a
        # Ring2 fully inside Ring1 if r2b < r1a
        elif r1b < r2a:
            if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Ring 1 fully inside Ring 2. Adjusting radii.")
            # changed_radii = True
        # TODO - use midpoint formula

        elif r2b < r1a:
            if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Ring 2 fully inside Ring 1. Adjusting radii.")
            # changed_radii = True

        return r1a, r1b, r2a, r2b, changed_radii

    def find_overlapping_rings_candidates(self, act_type: str, location1: np.ndarray, location2: np.ndarray,
                                          radius1a: float, radius1b: float, radius2a: float, radius2b: float,
                                          min_candidates=1, max_candidates=None, max_iterations=15):
        """Find candidates within two overlapping rings (donuts) around two center points.
        Iteratively increase the radii until a sufficient number of candidates is found.
        """
        # original_ring_width1 = abs(radius1b - radius1a)
        # original_ring_width2 = abs(radius2b - radius2a)

        # First, ensure there's at least some initial overlap.
        # radius1a, radius1b, radius2a, radius2b, changed_radii = self.ensure_overlap(location1, location2, radius1a, radius1b, radius2a,
        #  radius2b)

        i = 0
        while True:
            candidates = self.query_within_two_overlapping_rings(
                act_type, location1, location2, radius1a, radius1b, radius2a, radius2b, max_candidates)
            if candidates is not None and len(candidates[0]) >= min_candidates:
                if logger.isEnabledFor(logging.DEBUG): logger.debug(
                    f"Found {len(candidates[0])} candidates after {i} iterations.")
                stats_tracker.log(f"Find_ring_candidates: Iterations for {act_type}", i)
                return candidates, i
            radius1a, radius1b = h.spread_distances(radius1a, radius1b, iteration=i, first_step=50)
            radius2a, radius2b = h.spread_distances(radius2a, radius2b, iteration=i, first_step=50)
            i += 1
            if logger.isEnabledFor(logging.DEBUG): logger.debug(
                f"Iteration {i}. Increasing radii to {radius1a}, {radius1b} and {radius2a}, {radius2b}.")
            if i > max_iterations:
                raise RuntimeError(f"Not enough candidates found after {max_iterations} iterations.")


class AdvancedPetreAlgorithm:
    """
    """

    def __init__(self, target_locations: TargetLocations, segmented_plans: SegmentedPlans, config):
        self.target_locations = target_locations
        self.segmented_plans = segmented_plans
        self.c_i = CircleIntersection(target_locations)

        self.number_of_branches = config['number_of_branches']
        self.min_candidates_complex_case = config['min_candidates_complex_case']
        self.candidates_two_leg_case = config['candidates_two_leg_case']
        self.max_candidates = config['max_candidates']
        self.anchor_strategy = config['anchor_strategy']
        self.selection_strategy_complex_case = config['selection_strategy_complex_case']
        self.selection_strategy_two_leg_case = config['selection_strategy_two_leg_case']
        self.max_radius_reduction_factor = config['max_radius_reduction_factor']
        self.max_iterations_complex_case = config['max_iterations_complex_case']
        self.only_return_valid_persons = config['only_return_valid_persons']

    def run(self):
        placed_dict = {}
        for person_id, segments in tqdm(self.segmented_plans.items(), desc="Processing persons"):
            placed_dict[person_id] = []
            for segment in segments:
                placed_segment, _ = self.solve_segment(segment)
                if placed_segment is not None:
                    placed_dict[person_id].append(placed_segment)
                elif not self.only_return_valid_persons:
                    raise RuntimeError("None should only be returned when only valid persons are requested.")
        return placed_dict

    def _get_anchor_index(self, num_legs: int) -> int:
        """Determine the anchor index based on strategy."""
        if self.anchor_strategy == "lower_middle":
            return num_legs // 2 - 1
        elif self.anchor_strategy == "upper_middle":
            return num_legs // 2
        elif self.anchor_strategy == "start":
            return 0
        elif self.anchor_strategy == "end":
            return num_legs - 1
        else:
            raise ValueError("Invalid anchor strategy.")

    def solve_segment(self, segment: Segment) -> Tuple[Segment, float]:
        """Recursively solve a segment for multiple candidates."""
        if len(segment) == 0:
            raise ValueError("No legs in segment.")
        elif len(segment) == 1:  # Base case for single leg
            assert segment[0].from_location.size > 0 and segment[0].to_location.size > 0, \
                "Start and end locations must be known."
            return segment, 0
        elif len(segment) == 2:  # Base case for two legs
            best_loc = self.c_i.get_best_circle_intersection_location(
                segment[0].from_location, segment[1].to_location, segment[0].to_act_type,
                segment[0].distance, segment[1].distance, self.candidates_two_leg_case,
                self.selection_strategy_two_leg_case, self.max_iterations_complex_case,
                self.only_return_valid_persons
            )
            if best_loc[0] is None:
                if self.only_return_valid_persons:
                    return None, 0
                raise RuntimeError("Reached impossible state.")
            updated_leg1 = segment[0]._replace(to_location=best_loc[1], to_act_identifier=best_loc[0])
            updated_leg2 = segment[1]._replace(from_location=best_loc[1])
            return (updated_leg1, updated_leg2), best_loc[3]  # act_score

        # Recursive case
        anchor_idx = self._get_anchor_index(len(segment))
        location1 = segment[0].from_location
        location2 = segment[-1].to_location
        act_type = segment[anchor_idx].to_act_type

        # Generate candidate locations
        distances = np.array([leg.distance for leg in segment])
        distances_start_to_act = distances[:anchor_idx + 1]  # Up to and including anchor
        distances_act_to_end = distances[anchor_idx + 1:]  # From anchor + 1 to end

        # Radii describing the search area (two overlapping donuts)
        min_possible_distance1, max_possible_distance1 = h.get_min_max_distance(distances_start_to_act)
        min_possible_distance2, max_possible_distance2 = h.get_min_max_distance(distances_act_to_end)

        # Limit the search space, as the maximum radii will almost never be needed in valid trips
        if self.max_radius_reduction_factor:
            min_possible_distance1 *= self.max_radius_reduction_factor
            max_possible_distance1 *= self.max_radius_reduction_factor

        candidates, iterations = self.target_locations.find_overlapping_rings_candidates(
            act_type, location1, location2,
            min_possible_distance1, max_possible_distance1,
            min_possible_distance2, max_possible_distance2,
            self.min_candidates_complex_case, self.max_candidates,
            self.max_iterations_complex_case)
        candidate_ids, candidate_coords, candidate_potentials = candidates

        # Evaluate candidates
        if iterations > 0:  # We need to find distance deviations of each candidate to score them
            candidate_deviations = np.zeros(len(candidate_ids))
            # We only count deviations of lowest-level legs to avoid double counting
            if len(distances_start_to_act) == 1:
                candidate_deviations += h.get_abs_distance_deviations(candidate_coords, location1,
                                                                      distances_start_to_act)
            elif len(distances_act_to_end) == 1:
                candidate_deviations += h.get_abs_distance_deviations(candidate_coords, location2,
                                                                      distances_act_to_end)
            local_scores = EvaluationFunction.evaluate_candidates(candidate_potentials, candidate_deviations)
        else:  # No distance deviations expected, just score by potentials
            candidate_deviations = np.zeros(len(candidate_ids))
            if len(distances_start_to_act) == 1:
                candidate_deviations += h.get_abs_distance_deviations(candidate_coords, location1,
                                                                      distances_start_to_act)
            if len(distances_act_to_end) == 1:
                candidate_deviations += h.get_abs_distance_deviations(candidate_coords, location2,
                                                                      distances_act_to_end)
            if np.any(candidate_deviations != 0):
                raise ValueError("Total deviations should be zero.")
            local_scores = EvaluationFunction.evaluate_candidates(candidate_potentials, None,
                                                                  len(candidate_ids))

        selected_candidates, selected_scores = EvaluationFunction.select_candidates(
            candidates, local_scores, self.number_of_branches, self.selection_strategy_complex_case
        )

        # Process each candidate and split segments
        full_segs = []
        branch_scores = []
        for i in range(len(selected_candidates[0])):
            new_coord = selected_candidates[1][i]
            new_id = selected_candidates[0][i]

            # Create updated legs (safe copies, not modifying originals)
            updated_leg1 = segment[anchor_idx]._replace(to_location=new_coord, to_act_identifier=new_id)
            updated_leg2 = segment[anchor_idx + 1]._replace(from_location=new_coord)

            # Split into subsegments with safely updated legs
            subsegment1 = (*segment[:anchor_idx], updated_leg1)
            subsegment2 = (updated_leg2, *segment[anchor_idx + 2:])

            # Recursively solve each subsegment
            located_seg1, score1 = self.solve_segment(subsegment1)
            located_seg2, score2 = self.solve_segment(subsegment2)

            if located_seg1 is None or located_seg2 is None:
                if self.only_return_valid_persons:
                    return None, 0
                raise RuntimeError("Reached impossible state.")
            # Combine results and track scores
            total_score = score1 + score2 + selected_scores[i]
            branch_scores.append(total_score)
            full_segs.append((*located_seg1, *located_seg2))

        # Return the best solution
        best_idx = np.argmax(branch_scores)
        return full_segs[best_idx], branch_scores[best_idx]

    def solve_segment_old(self, in_segment):
        """At each level, find the best n locations, solve subproblems, and choose the best solution.
        -Solve the highest current level with n candidates
        -Place their locations and resegment (split) the segment
        -Feed the new segments back into the solver"""
        segment = copy.deepcopy(in_segment)
        stats_tracker.increment("AdvancedPetreAlgorithm: Segments processed")

        if len(segment) == 0:
            raise ValueError("No legs in segment.")
        elif len(segment) == 1:
            stats_tracker.increment("AdvancedPetreAlgorithm: 1-leg segments")
            assert segment[0]['from_location'].size > 0 and segment[0][
                'to_location'].size > 0, "Both start and end locations must be known for a single leg."
            return segment, 0
        # if there are only two legs, we can find the loc immediately
        elif len(segment) == 2:
            stats_tracker.increment(f"AdvancedPetreAlgorithm: 2-leg segments")
            if logger.isEnabledFor(logging.DEBUG): logger.debug(
                f"Advanced locating. Segment has 2 legs. leg ID:{segment[0]['unique_leg_id']}")
            act = self.c_i.get_best_circle_intersection_location(segment[0]['from_location'], segment[1]['to_location'],
                                                                 segment[0]['to_act_type'], segment[0]['distance'],
                                                                 segment[1]['distance'])
            act_identifier, act_name, act_coord, act_cap, act_dist, act_score = act
            segment[0]['to_location'] = act_coord
            segment[1]['from_location'] = act_coord
            segment[0]['to_act_identifier'] = act_identifier
            # segment[0]['to_act_name'] = act_name
            # segment[0]['to_act_cap'] = act_cap
            # segment[0]['to_act_score'] = act_score
            return segment, act_score

        else:
            if logger.isEnabledFor(logging.DEBUG): logger.debug(
                f"Advanced locating. Segment has {len(segment)} legs. leg ID:{segment[0]['unique_leg_id']}")
            stats_tracker.increment(f"AdvancedPetreAlgorithm: {len(segment)}-leg segments")
            if self.anchor_strategy == "lower_middle":
                anchor_idx = len(segment) // 2 - 1
            elif self.anchor_strategy == "upper_middle":
                anchor_idx = len(segment) // 2
            elif self.anchor_strategy == "start":
                anchor_idx = 0
            elif self.anchor_strategy == "end":
                anchor_idx = len(segment) - 1
            else:
                raise ValueError("Invalid anchor strategy.")

            distances = np.array([leg['distance'] for leg in segment])

            distances_start_to_act = distances[:anchor_idx + 1]  # Up to and including anchor
            distances_act_to_end = distances[anchor_idx + 1:]  # From anchor + 1 to end

            # Radii describing the search area (two overlapping donuts)
            min_possible_distance1, max_possible_distance1 = h.get_min_max_distance(distances_start_to_act)
            min_possible_distance2, max_possible_distance2 = h.get_min_max_distance(distances_act_to_end)

            # Get candidates for the highest level
            act_type = segment[anchor_idx]['to_act_type']
            location1 = segment[0]['from_location']  # From
            location2 = segment[-1]['to_location']  # To

            candidates, iterations = self.target_locations.find_overlapping_rings_candidates(act_type,
                                                                                             location1, location2,
                                                                                             min_possible_distance1,
                                                                                             max_possible_distance1,
                                                                                             min_possible_distance2,
                                                                                             max_possible_distance2,
                                                                                             self.min_candidates_complex_case,
                                                                                             self.max_candidates,
                                                                                             self.max_iterations_complex_case)

            candidate_identifiers, candidate_coordinates, candidate_potentials = candidates

            if iterations > 0:  # We need to find distance deviations of each candidate to score them

                candidate_deviations = np.zeros(len(candidate_identifiers))
                # We only count deviations of lowest-level legs to avoid double counting
                if len(distances_start_to_act) == 1:
                    candidate_deviations += h.get_abs_distance_deviations(candidate_coordinates, location1,
                                                                          distances_start_to_act)
                elif len(distances_act_to_end) == 1:
                    candidate_deviations += h.get_abs_distance_deviations(candidate_coordinates, location2,
                                                                          distances_act_to_end)
                local_scores = EvaluationFunction.evaluate_candidates(candidate_potentials, candidate_deviations)

            else:  # No distance deviations expected, just score by potentials

                # # 'DEBUG'
                candidate_deviations = np.zeros(len(candidate_identifiers))
                if len(distances_start_to_act) == 1:
                    candidate_deviations += h.get_abs_distance_deviations(candidate_coordinates, location1,
                                                                          distances_start_to_act)
                if len(distances_act_to_end) == 1:
                    candidate_deviations += h.get_abs_distance_deviations(candidate_coordinates, location2,
                                                                          distances_act_to_end)

                if np.any(candidate_deviations != 0):
                    raise ValueError("Total deviations should be zero.")
                # # /'DEBUG'

                local_scores = EvaluationFunction.evaluate_candidates(candidate_potentials, None,
                                                                      len(candidate_identifiers))

            stats_tracker.log(f"AdvancedPetreAlgorithm: Number of candidates at segment length {len(segment)}:",
                              len(local_scores))
            if logger.isEnabledFor(logging.DEBUG): logger.debug(
                f"Advanced locating. Found {len(local_scores)} candidates at segment length {len(segment)}.")

            # # >Randomly< sample down to number_of_branches if there are too many candidates with the same score
            # # This creates a roughly equal spacial distribution of candidates
            # if len(local_scores) > number_of_branches:
            #     stats_tracker.increment(f"AdvancedPetreAlgorithm: More candidates than branches:")
            #
            #     highest_value = np.max(local_scores)
            #     highest_indices = np.where(local_scores == highest_value)[0]
            #
            #     if len(highest_indices) > number_of_branches:
            #         stats_tracker.increment(f"AdvancedPetreAlgorithm: More candidates with equal scores than branches:")
            #
            #         selected_indices = np.random.choice(highest_indices, number_of_branches, replace=False)
            #         candidates = (candidate_identifiers[selected_indices],
            #                       candidate_coordinates[selected_indices],
            #                       candidate_potentials[selected_indices])
            #         local_scores = local_scores[selected_indices]

            # Select the best-ish candidates
            selected_candidates, scores = EvaluationFunction.select_candidates(candidates, local_scores,
                                                                               self.number_of_branches, 'mixed')
            selected_identifiers, selected_coords, selected_potentials = selected_candidates

            # Split the segment at the anchor point
            subsegment1 = segment[:anchor_idx + 1]  # Up to and including anchor
            subsegment2 = segment[anchor_idx + 1:]  # From anchor + 1 to end

            full_segs = []
            branch_scores = []
            for i in range(len(selected_identifiers)):
                if len(selected_identifiers) == 1:
                    subsegment1[-1]['to_location'] = selected_coords
                    subsegment1[-1]['to_act_identifier'] = np.array(selected_identifiers[i])
                    subsegment2[0]['from_location'] = selected_coords
                else:
                    subsegment1[-1]['to_location'] = selected_coords[i]
                    subsegment1[-1]['to_act_identifier'] = np.array(selected_identifiers[i])
                    subsegment2[0]['from_location'] = selected_coords[i]

                located_seg1, score1 = self.solve_segment(subsegment1)
                located_seg2, score2 = self.solve_segment(subsegment2)

                total_branch_score = score1 + score2 + scores[i]
                branch_scores.append(total_branch_score)

                full_seg = located_seg1 + located_seg2  # List concatenation
                full_segs.append(full_seg)

            max_score = max(branch_scores)
            best_seg = full_segs[branch_scores.index(max_score)]

            return best_seg, max_score


class WeirdPetreAlgorithm:
    def __init__(self, target_locations: TargetLocations, segmented_dict: Dict[str, list[list[dict[str, Any]]]],
                 variant: Literal["advanced", "greedy"] = "greedy"):
        self.target_locations = target_locations
        self.segmented_dict = segmented_dict
        self.variant = variant
        self.c_i = CircleIntersection(target_locations)

    def run(self):
        if self.variant == "advanced":
            raise NotImplementedError
        elif self.variant == "greedy":
            pass
        else:
            raise ValueError(f"Invalid variant: {self.variant}")

    def adjust_estimation_tree(self, tree: List[List[List[float]]], real_distance: float, strong_adjust: bool = True) -> \
            List[
                List[List[float]]]:  # Tree level, Leg, Lengths
        """If there is strong overshooting or undershooting in the estimation tree, adjust the distances.
        It tries to keep given slack constraints, unless it is not possible.

        TEMP: Erstmal nur strong adjust, weak adjust ist vlt grundsätzlich nicht sinnvoll.

        Parameters:
        tree: The estimation tree to adjust.
        real_distance: The real total distance to adjust the tree to.
        strong_adjust: If True, real_bounds are used for adjustment, if False, wanted_bounds are used.
        """

        if logger.isEnabledFor(logging.DEBUG): logger.debug(tree)

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
            if logger.isEnabledFor(logging.DEBUG): logger.debug(
                "Real total distance is within bounds of highest level, no adjustment needed.")
            stats_tracker.increment("adjustment_total_runs")
            stats_tracker.increment("adjustment_no_adjustment")
            return tree
        else:
            # Traverse from one below the highest level down to including level 1 (which has index 0) - and then down to -1 just for the last check
            for level in range(len(tree) - 1, -1, -1):
                for i in range(0, len(tree[level]),
                               2):  # Traverse in pairs, skipping the last one if it's an odd number
                    if i == len(tree[level]) - 1:
                        continue

                    higher_leg_value = tree[level + 1][i // 2][2]
                    higher_leg_lower_bound = tree[level + 1][i // 2][l_bound_idx]
                    higher_leg_upper_bound = tree[level + 1][i // 2][u_bound_idx]

                    leg1_value = tree[level][i][2]
                    leg2_value = tree[level][i + 1][2]

                    if higher_leg_value < higher_leg_lower_bound:  # (Real) higher_leg_value can be zero if start and end location are the same
                        if logger.isEnabledFor(logging.DEBUG): logger.debug(
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

                        if logger.isEnabledFor(logging.DEBUG): logger.debug(
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
                    if logger.isEnabledFor(logging.DEBUG): logger.debug("Strong adjustment succeeded.")
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(tree)
                    stats_tracker.increment("adjustment_total_runs")
                    stats_tracker.increment("adjustment_strong_adjustment_success")
                    return tree
                else:
                    if logger.isEnabledFor(logging.DEBUG): logger.debug("Strong adjustment failed.")
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(tree)
                    stats_tracker.increment("adjustment_total_runs")
                    stats_tracker.increment("adjustment_strong_adjustment_failure")
                    return tree
            else:
                if plausible:
                    # These are the bounds of the first level estimation, which are set in stone by the known real distances
                    if logger.isEnabledFor(logging.DEBUG): logger.debug("Adjustment succeeded with wanted bounds.")
                    return tree
                else:
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(
                        "Adjustment failed with wanted bounds, trying strong adjustment with real bounds.")
                    return self.adjust_estimation_tree(tree, real_distance, strong_adjust=True)

    def greedy_locate_segment(self, segment):
        if len(segment) == 0:
            raise ValueError("No legs in segment.")
        elif len(segment) == 1:
            assert segment[0]['from_location'].size > 0 and segment[0][
                'to_location'].size > 0, "Both start and end locations must be known for a single leg."
            return segment
        # if there are only two legs, we can find the loc immediately
        elif len(segment) == 2:
            if logger.isEnabledFor(logging.DEBUG): logger.debug("Greedy locating. Only two legs in segment.")
            act_identifier, act_name, act_coord, act_cap, act_dist, act_score = (
                self.c_i.get_best_circle_intersection_location(
                    segment[0]['from_location'], segment[-1]['to_location'], segment[0]['to_act_type'],
                    segment[0]['distance'], segment[-1]['distance'], 5))
            segment[0]['to_location'] = act_coord
            segment[-1]['from_location'] = act_coord
            segment[0]['to_act_identifier'] = act_identifier
            # segment[0]['to_act_name'] = act_name
            # segment[0]['to_act_cap'] = act_cap
            # segment[0]['to_act_score'] = act_score
            return segment
        else:
            if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Greedy locating. Segment has {len(segment)} legs.")
            distances = [leg['distance'] for leg in segment]

            real_distance = h.euclidean_distance(segment[0]['from_location'], segment[-1]['to_location'])

            tree = h.build_estimation_tree(distances)
            tree = self.adjust_estimation_tree(tree, real_distance, strong_adjust=True)
            position_on_segment_info = self.build_position_on_segment_info(
                len(distances))  # tells us at each level which legs to look at
            assert len(tree) == len(position_on_segment_info), "Tree and position info must have the same length."

            for level in range(len(tree) - 1, -1, -1):
                for i, leg_idx in enumerate(position_on_segment_info[level]):
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Level: {level}, i: {i}, leg_idx: {leg_idx}")
                    step = 2 ** level  # How far from the current leg to look for start and end locations
                    from_location_idx = leg_idx - step + 1  # + 1 because we get the from_location
                    assert from_location_idx >= 0, "From location index must be greater or equal to 0."
                    to_location_idx = min(len(segment) - 1, leg_idx + step)

                    if level == 0:
                        dist_start_to_act = segment[leg_idx]['distance']
                        dist_act_to_end = segment[to_location_idx]['distance']
                    else:
                        # 2 * i because two lower legs are combined to one higher leg
                        # in odd cases on the current level, the last leg is skipped
                        dist_start_to_act = tree[level - 1][2 * i][2]
                        dist_act_to_end = tree[level - 1][2 * i + 1][2]

                    act_identifier, act_name, act_coord, act_cap, act_dist, act_score = \
                        self.c_i.get_best_circle_intersection_location(segment[from_location_idx]['from_location'],
                                                                       segment[to_location_idx]['to_location'],
                                                                       segment[leg_idx]['to_act_type'],
                                                                       dist_start_to_act,
                                                                       dist_act_to_end, 1)
                    segment[leg_idx]['to_location'] = act_coord
                    if leg_idx + 1 < len(segment) + 1:
                        segment[leg_idx + 1]['from_location'] = act_coord
                    segment[leg_idx]['to_act_identifier'] = act_identifier
                    segment[leg_idx]['to_act_name'] = act_name
                    segment[leg_idx]['to_act_cap'] = act_cap
                    segment[leg_idx]['to_act_score'] = act_score
            return segment

    @staticmethod
    def build_position_on_segment_info(n: int) -> list[list[int]]:
        """Based on the number of legs in a segment,
        returns a list of lists that tells us at each level which legs to process.
        Takes basically 0 time.
        n = 2
        [[0]]

        n = 3
        [[0], [1]]

        n = 4
        [[0, 2], [1]]

        n = 5
        [[0, 2], [1], [3]]

        n = 6
        [[0, 2, 4], [1], [3]]

        n = 7
        [[0, 2, 4], [1, 5], [3]]

        n = 8
        [[0, 2, 4, 6], [1, 5], [3]]

        n = 9
        [[0, 2, 4, 6], [1, 5], [3], [7]]
        """
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


class SimpleLelkeAlgorithm:
    """
    Since the algorithm is targeted for solving the whole closed trip, the "segment" is usually the whole trip.
    Skips persons where the main location is already set.
    """

    def __init__(self, target_locations: TargetLocations, segmented_dict: Dict[str, list[list[dict[str, Any]]]]):
        self.target_locations = target_locations
        self.segmented_dict = segmented_dict
        self.c_i = CircleIntersection(target_locations)

    def run(self):
        for person_id, segments in tqdm(self.segmented_dict.items(), desc="Processing persons"):
            for segment in segments:
                self.simple_locate_segment(segment)  # In-place
        return self.segmented_dict

    def simple_locate_segment(self, person_legs):
        """Assumes start and end locations of segment are identical."""
        if len(person_legs) == 0:
            raise ValueError("No legs in segment.")

        elif len(person_legs) == 1:
            assert person_legs[0]['from_location'].size > 0 and person_legs[0][
                'to_location'].size > 0, "Both start and end locations must be known for a single leg."
            return person_legs

        # if there are only two legs, get a location within the two circles around home
        elif len(person_legs) == 2:
            if logger.isEnabledFor(logging.DEBUG): logger.debug("Simple locating. Only two legs in segment.")
            distance1 = person_legs[0]['distance']
            distance2 = person_legs[1]['distance']
            if abs(distance1 - distance2) < 30:  # always meters!
                distance1, distance2 = h.spread_distances(distance1, distance2, first_step=20)
            candidates = self.target_locations.find_ring_candidates(person_legs[0]['to_act_type'],
                                                                    person_legs[0]['from_location'], distance1,
                                                                    distance2)
            act_identifier, act_coord, act_cap, act_score = (
                EvaluationFunction.monte_carlo_select_candidate(candidates))
            person_legs[0]['to_location'] = act_coord
            person_legs[1]['from_location'] = act_coord
            person_legs[0]['to_act_identifier'] = act_identifier
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
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(
                        "Home activity. Placing at, you guessed it, home (start location).")
                    act_identifier, act_name, act_coord, act_cap, act_dist, act_score = None, "home", person_legs[i][
                        'from_location'], None, None, None

                elif traveled_distance >= total_distance / 2:

                    if remaining_legs == 2:
                        if logger.isEnabledFor(logging.DEBUG): logger.debug(
                            "Selecting location using simple two-leg method.")
                        assert person_legs[-1] == person_legs[
                            i + 1], "Last leg must be the last leg."  # TODO: Remove this line in production
                        act_identifier, act_coord, act_cap, act_score = \
                            self.c_i.get_best_circle_intersection_location(person_legs[i]['from_location'],
                                                                           person_legs[-1]['to_location'],
                                                                           person_legs[i]['to_act_type'],
                                                                           person_legs[i]['distance'],
                                                                           person_legs[-1]['distance'], 5)

                    else:
                        if logger.isEnabledFor(logging.DEBUG): logger.debug(
                            "Selecting location using ring with angle restriction.")
                        distance = person_legs[i]['distance']
                        radius1, radius2 = h.spread_distances(distance, distance, iteration=0, first_step=20)
                        candidates = self.target_locations.find_ring_candidates(person_legs[i]['to_act_type'],
                                                                                person_legs[i]['from_location'],
                                                                                radius1, radius2, restrict_angle=True,
                                                                                direction_point=person_legs[-1][
                                                                                    'to_location'], max_iterations= 50)

                        act_identifier, act_coord, act_cap, act_score = (
                            EvaluationFunction.monte_carlo_select_candidate(candidates))
                else:
                    if logger.isEnabledFor(logging.DEBUG): logger.debug("Selecting location using ring.")
                    distance = person_legs[i]['distance']
                    radius1, radius2 = h.spread_distances(distance, distance, iteration=0, first_step=20)
                    candidates = self.target_locations.find_ring_candidates(person_legs[i]['to_act_type'],
                                                                            person_legs[i]['from_location'],
                                                                            radius1, radius2)
                    act_identifier, act_coord, act_cap, act_score = (
                        EvaluationFunction.monte_carlo_select_candidate(candidates))

                person_legs[i]['to_location'] = act_coord
                person_legs[i + 1]['from_location'] = act_coord
                person_legs[i]['to_act_identifier'] = act_identifier
                person_legs[i]['to_act_cap'] = act_cap
                person_legs[i]['to_act_score'] = act_score

            return person_legs


class SimpleMainLocationAlgorithm:
    """
    Since the algorithm is targeted for solving just the main location, the "segment" is usually the whole trip.
    """

    def __init__(self, target_locations: TargetLocations, legs_dict: Dict[str, list[dict[str, Any]]], config):
        self.target_locations = target_locations
        self.legs_dict = legs_dict
        self.skip_already_located = config["skip_already_located"]

    def run(self):
        for person_id, person_legs in tqdm(self.legs_dict.items(), desc="Processing persons"):
            self.locate_main(person_legs)  # In-place
        return self.legs_dict

    def locate_main(self, person_legs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Gets a person's main activity and locates it.
        Currently just uses the Euclidean distance and potentials.
        Planned to also use O-D matrices.
        :return: Updated list of legs with located main activities.
        """

        main_activity_index, main_activity_leg = h.get_main_activity_leg(person_legs)
        if main_activity_leg is None:
            return person_legs

        if self.skip_already_located:
            to_location = main_activity_leg.get('to_location')
            assert isinstance(to_location, np.ndarray), "Bad location format."
            if to_location.size != 0:
                return person_legs

        target_activity = main_activity_leg['to_act_type']
        home_location = person_legs[0]['from_location']
        estimated_distance_home_to_main = person_legs[0]['home_to_main_distance']

        # Radii are iteratively spread by find_ring_candidates until a candidate is found
        radius1, radius2 = h.spread_distances(estimated_distance_home_to_main,
                                              estimated_distance_home_to_main)  # Initial
        candidates = self.target_locations.find_ring_candidates(target_activity, home_location, radius1=radius1,
                                                                radius2=radius2)
        scores = EvaluationFunction.evaluate_candidates(candidates[2], None, len(candidates[2]))
        chosen_candidate, score = (
            EvaluationFunction.select_candidates(candidates, scores, 1, 'monte_carlo'))

        act_identifier, act_coord, act_pot = chosen_candidate

        # Update the main activity leg and the subsequent leg
        person_legs[main_activity_index]['to_location'] = act_coord[0]
        person_legs[main_activity_index]['to_act_identifier'] = act_identifier[0]
        person_legs[main_activity_index]['to_act_cap'] = act_pot[0]
        person_legs[main_activity_index]['to_act_score'] = score[0]

        if main_activity_index + 1 < len(person_legs):  # Set from location if there is a subsequent leg
            person_legs[main_activity_index + 1]['from_location'] = act_coord

        return person_legs


class OpenEndedAlgorithm:
    """
    Places the last location if it is an open trip AND if the start location is known (e.g. from home or main)
    """

    def __init__(self, target_locations: TargetLocations, legs_dict: Dict[str, list[dict[str, Any]]], config):
        self.target_locations = target_locations
        self.legs_dict = legs_dict
        self.skip_already_located = config["skip_already_located"]

    def run(self):
        for person_id, person_legs in tqdm(self.legs_dict.items(), desc="Processing persons"):
            self.locate_open(person_legs)  # In-place
        return self.legs_dict

    def locate_open(self, person_legs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Gets a person's open-ended activity and locates it.
        Currently just uses the Euclidean distance and potentials.
        Planned to also use O-D matrices.
        :return: Updated list of legs with located main activities.
        """

        last_index = len(person_legs) - 1
        last_leg = person_legs[last_index]

        # From location must be known, else skip
        from_location = last_leg['from_location']
        assert isinstance(from_location, np.ndarray), "Bad location format."
        if from_location.size == 0:
            return person_legs

        if self.skip_already_located:
            to_location = last_leg.get('to_location')
            assert isinstance(to_location, np.ndarray), "Bad location format."
            if to_location.size != 0:
                return person_legs

        target_activity = last_leg['to_act_type']
        distance = last_leg['distance']

        # Radii are iteratively spread by find_ring_candidates until a candidate is found
        radius1, radius2 = h.spread_distances(distance, distance)  # Initial
        candidates = self.target_locations.find_ring_candidates(target_activity, from_location, radius1=radius1,
                                                                radius2=radius2)
        scores = EvaluationFunction.evaluate_candidates(candidates[2], None, len(candidates[2]))
        chosen_candidate, score = (
            EvaluationFunction.select_candidates(candidates, scores, 1, 'monte_carlo'))

        act_identifier, act_coord, act_pot = chosen_candidate

        # Update the main activity leg and the subsequent leg
        person_legs[last_index]['to_location'] = act_coord[0]
        person_legs[last_index]['to_act_identifier'] = act_identifier[0]
        person_legs[last_index]['to_act_cap'] = act_pot[0]
        person_legs[last_index]['to_act_score'] = score[0]

        return person_legs


class MatrixMainLocationAlgorithm:
    def __init__(self, target_locations: TargetLocations, segmented_dict: Dict[str, list[list[dict[str, Any]]]]):
        self.target_locations = target_locations
        self.segmented_dict = segmented_dict

    def run(self):
        raise NotImplementedError


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

class EvaluationFunction:

    @staticmethod
    def evaluate_candidates(potentials: np.ndarray = None, dist_deviations: np.ndarray = None,
                            number_of_candidates: int = None) -> np.ndarray:
        """
        Scoring function collection for the candidates based on potentials and distances.

        :param potentials: Numpy array of potentials for the returned locations.
        :param dist_deviations: Distance deviations from the target (if available).
        :param number_of_candidates:
        :return: Non-normalized, absolute scores.
        """
        # if distances is not None and potentials is not None:
        #     return potentials / distances
        # if potentials is not None:
        #     return potentials
        if dist_deviations is not None:
            return np.maximum(0, 1000000 - dist_deviations)

            # return 1 / distances

            # best_index = np.argmin(distances)
            # scores = np.zeros_like(distances)
            # scores[best_index] = 1
            # return scores
        else:
            if number_of_candidates is None:
                return np.full((len(potentials),), 1000000)
            return np.full((number_of_candidates,), 1000000)

    @classmethod
    def monte_carlo_select_candidate(cls, candidates):
        """Depreciated. Use select_candidates instead."""  # TODO: Remove
        scores = cls.evaluate_candidates(candidates[2])

        # Normalize scores
        sscores = scores / np.sum(scores)
        if np.any(sscores < 0):
            logger.warning("Negative scores detected. Setting them to zero.")
            sscores = np.maximum(0.001, sscores)
            sscores = sscores / np.sum(sscores)
        if not np.isclose(np.sum(sscores), 1):
            logger.warning("Scores do not sum to 1. Renormalizing.")
            sscores = sscores / sum(sscores)

        chosen_index = np.random.choice(len(sscores), p=sscores)

        # Return the chosen candidate with its score
        chosen_candidate = tuple((arr[chosen_index] if arr is not None else None) for arr in candidates) + (
            scores[chosen_index],)

        return chosen_candidate

    @classmethod
    def select_candidate_indices(
            cls,
            scores: np.ndarray,
            num_candidates: int,
            strategy: str = 'monte_carlo',
            top_portion: float = 0.5,
            coords: np.ndarray = None,
            num_cells_x: int = 20,
            num_cells_y: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the indices of candidates based on their normalized scores using Monte Carlo sampling,
        a top-n strategy, a mixed strategy, or spatial downsampling.

        :param scores: A 1D numpy array of scores corresponding to candidates.
        :param num_candidates: The number of candidates to select.
        :param strategy: Selection strategy ('monte_carlo', 'top_n', 'mixed', or 'spatial_downsample').
        :param top_portion: Portion of candidates to select from the top scores when using the 'mixed' strategy.
        :param coords: 2D numpy array of shape (n, 2) with candidate spatial coordinates (required for spatial_downsample).
        :param num_cells_x: Number of cells along the longitude (spatial_downsample).
        :param num_cells_y: Number of cells along the latitude (spatial_downsample).
        :return: A tuple containing:
                 - The selected indices of the best candidates.
                 - A 1D array of the scores corresponding to the selected indices.
        """
        assert len(scores) > 0, "The scores array cannot be empty."
        if num_candidates >= len(scores):
            stats_tracker.increment("Select_candidates_indices: All candidates selected")
            return np.arange(len(scores)), scores

        if strategy == 'monte_carlo':
            stats_tracker.increment("Scoring runs (Monte Carlo)")
            normalized_scores = scores / np.sum(scores, dtype=np.float64)
            chosen_indices = np.random.choice(len(scores), num_candidates, p=normalized_scores, replace=False)

        elif strategy == 'top_n':
            stats_tracker.increment("Scoring runs (Top N)")
            chosen_indices = np.argsort(scores)[-num_candidates:][::-1]  # Top scores in descending order

        elif strategy == 'mixed':
            stats_tracker.increment("Scoring runs (Mixed)")
            num_top = int(np.ceil(num_candidates * top_portion))
            num_monte_carlo = num_candidates - num_top

            sorted_indices = np.argsort(scores)[-num_top:][::-1]
            remaining_indices = np.setdiff1d(np.arange(len(scores)), sorted_indices)

            if len(remaining_indices) > 0 and num_monte_carlo > 0:
                remaining_scores = scores[remaining_indices]
                normalized_remaining_scores = remaining_scores / np.sum(remaining_scores, dtype=np.float64)
                monte_carlo_indices = np.random.choice(remaining_indices, num_monte_carlo,
                                                       p=normalized_remaining_scores, replace=False)
                chosen_indices = np.concatenate((sorted_indices, monte_carlo_indices))
            else:
                chosen_indices = sorted_indices

        elif strategy == 'spatial_downsample':
            assert coords is not None, "Coordinates (coords) are required for spatial_downsample strategy."
            stats_tracker.increment("Scoring runs (Spatial Downsample)")
            chosen_indices = cls.even_spatial_downsample(
                coords, num_cells_x=num_cells_x, num_cells_y=num_cells_y
            )[:num_candidates]

        elif strategy == 'top_n_spatial_downsample':
            assert coords is not None, "Coordinates (coords) are required for top_n_spatial_downsample strategy."
            stats_tracker.increment("Scoring runs (Top N Spatial Downsample)")

            # Sort scores in descending order
            sorted_indices = np.argsort(scores)[::-1]
            sorted_scores = scores[sorted_indices]

            # Identify the cutoff score
            cutoff_score = sorted_scores[num_candidates - 1] if len(sorted_scores) >= num_candidates else sorted_scores[
                -1]

            # Find all indices with scores >= cutoff_score (this may be more than num_candidates if scores are equal)
            top_indices = np.where(scores >= cutoff_score)[0]

            # Check if spatial downsampling is needed
            if len(top_indices) > num_candidates:
                num_cells = max(1, int(np.sqrt(num_candidates)) + 1)  # Slightly above the square root of candidates
                chosen_indices = cls.even_spatial_downsample(
                    coords, num_cells_x=num_cells, num_cells_y=num_cells
                )
            else:
                # Use the sorted indices if no downsampling is needed
                chosen_indices = sorted_indices[:num_candidates]

        else:
            raise ValueError(
                "Invalid selection strategy. Use 'monte_carlo', 'top_n', 'mixed', or 'spatial_downsample'.")

        chosen_scores = scores[chosen_indices]
        return chosen_indices, chosen_scores

    @classmethod
    def select_candidates(
            cls,
            candidates: Tuple[np.ndarray, ...],
            scores: np.ndarray,
            num_candidates: int,
            strategy: str = 'monte_carlo',
            top_portion: float = 0.5,
            coords: np.ndarray = None,
            num_cells_x: int = 20,
            num_cells_y: int = 20
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """
        Selects a specified number of candidates based on their scores using various strategies.

        :param candidates: A tuple of arrays with the candidates.
        :param scores: A 1D array of scores corresponding to the candidates.
        :param num_candidates: The number of candidates to select.
        :param strategy: Selection strategy ('monte_carlo', 'top_n', 'mixed', or 'spatial_downsample').
        :param top_portion: Portion of candidates to select from the top scores when using the 'mixed' strategy.
        :param coords: 2D numpy array of candidate spatial coordinates (required for 'spatial_downsample'). If no
                        coordinates are provided, candidates[1] is used as coordinates.
        :param num_cells_x: Number of cells along the longitude (spatial_downsample).
        :param num_cells_y: Number of cells along the latitude (spatial_downsample).
        :return: A tuple containing:
            - A tuple of arrays with the selected candidates.
            - A 1D array of the scores corresponding to the selected candidates.
        """
        assert len(candidates[0]) == len(scores), "The number of candidates and scores must match."
        if strategy == 'keep_all':
            return candidates, scores
        if (strategy == 'spatial_downsample' or strategy == "top_n_spatial_downsample") and coords is None:
            coords = candidates[1]

        chosen_indices, chosen_scores = cls.select_candidate_indices(
            scores, num_candidates, strategy, top_portion, coords, num_cells_x, num_cells_y
        )

        selected_candidates = tuple(
            np.atleast_1d(arr[chosen_indices].squeeze()) if arr is not None else None for arr in candidates
        )

        if num_candidates == 1:
            return (
                tuple(
                    (
                        np.atleast_1d(selected_candidates[0]),  # IDs (n,)
                        np.atleast_2d(selected_candidates[1]),  # Coordinates (n, 2)
                        np.atleast_1d(selected_candidates[2]),  # Potentials (n,)
                    )
                ),
                np.atleast_1d(chosen_scores)  # Scores (n,)
            )

        return selected_candidates, chosen_scores

    @staticmethod
    def even_spatial_downsample(coords, num_cells_x=20, num_cells_y=20):
        """
        Downsample points and return indices of the kept points.

        Parameters:
        - coords: 2D coordinates array (n, 2)
        - num_cells_x: Number of cells along the longitude.
        - num_cells_y: Number of cells along the latitude.

        Returns:
        - A list of indices of the points that are kept after downsampling.
        """
        lats = coords[:, 0]
        lons = coords[:, 1]

        min_lat, max_lat = lats.min(), lats.max()
        min_lon, max_lon = lons.min(), lons.max()

        lat_range = max_lat - min_lat or 1e-9
        lon_range = max_lon - min_lon or 1e-9

        lat_step = lat_range / max(num_cells_y, 1)
        lon_step = lon_range / max(num_cells_x, 1)

        total_cells = num_cells_x * num_cells_y
        filled_cells = set()
        kept_indices = []

        for i in range(len(coords)):
            lat, lon = lats[i], lons[i]
            cell_x = min(int((lon - min_lon) / lon_step), num_cells_x - 1)
            cell_y = min(int((lat - min_lat) / lat_step), num_cells_y - 1)
            cell_id = cell_y * num_cells_x + cell_x

            if cell_id not in filled_cells:
                kept_indices.append(i)
                filled_cells.add(cell_id)

            # Stop early if all cells are filled
            if len(filled_cells) == total_cells:
                break

        return kept_indices


class CircleIntersection:
    def __init__(self, target_locations: TargetLocations):
        self.target_locations = target_locations

    def find_circle_intersection_candidates(self, start_coord: np.ndarray, end_coord: np.ndarray, type: str,
                                            distance_start_to_act: float, distance_act_to_end: float,
                                            num_candidates: int, only_return_valid=False):
        """
        Find n location candidates for a given activity type between two known locations.
        Returns two sets of candidates if two intersection points are found, otherwise only one set.
        """
        # Find the intersection points
        intersect1, intersect2 = self.find_circle_intersections(
            start_coord, distance_start_to_act,
            end_coord, distance_act_to_end, only_return_valid
        )
        if intersect1 is None and intersect2 is None:
            if only_return_valid:
                return None, None, None
            raise RuntimeError("Reached impossible state.")

        # Handle both intersections in a single batch query if both points exist
        if intersect2 is not None:
            # Stack intersection points into a single array
            locations_to_query = np.array([intersect1, intersect2])

            # Perform a batched query for both intersection points
            candidate_identifiers, candidate_coordinates, candidate_potentials = self.target_locations.query_closest(
                type, locations_to_query, num_candidates
            )

            if num_candidates == 1:
                return candidate_identifiers, candidate_coordinates, candidate_potentials

            # Concatenate the results from the batch query
            combined_identifiers = np.concatenate(candidate_identifiers, axis=0)
            combined_coordinates = np.concatenate(candidate_coordinates, axis=0)
            combined_potentials = np.concatenate(candidate_potentials, axis=0)
        else:
            # Query only the first intersection point
            combined_identifiers, combined_coordinates, combined_potentials = self.target_locations.query_closest(
                type, intersect1, num_candidates
            )
            if num_candidates == 1:
                return np.atleast_1d(combined_identifiers), np.atleast_2d(combined_coordinates), np.atleast_1d(
                    combined_potentials)

        return combined_identifiers, combined_coordinates, combined_potentials

    def find_circle_intersections(self, center1: np.ndarray, radius1: float, center2: np.ndarray, radius2: float,
                                  only_return_valid=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the intersection points of two circles.

        :param center1: The center of the first circle (e.g., np.array([x1, y1])).
        :param radius1: The radius of the first circle.
        :param center2: The center of the second circle (e.g., np.array([x2, y2])).
        :param radius2: The radius of the second circle.
        :param only_return_valid: If True, only return valid intersection points, else None
        :return: A tuple containing one or two intersection points (each as a np.ndarray).
        """

        x1, y1 = center1
        x2, y2 = center2
        r1 = radius1
        r2 = radius2

        # Calculate the distance between the two centers
        d = h.euclidean_distance(center1, center2)

        if logger.isEnabledFor(logging.DEBUG): logger.debug(
            f"Center 1: {center1}, Radius 1: {radius1}, Center 2: {center2}, Radius 2: {radius2}")

        # Handle non-intersection conditions:
        if d < 1e-4:
            raise RuntimeError("The case of identical start and end should be handled by the donut-function.")
            # if abs(r1 - r2) < 1e-4:
            #     if logger.isEnabledFor(logging.DEBUG): logger.debug("Infinite intersections: The start and end points and radii are identical.")
            #     if logger.isEnabledFor(logging.DEBUG): logger.debug("Choosing a point on the perimeter of the circles.")
            #     intersect = np.array([x1 + r1, y1])
            #     return intersect, None
            # else:
            #     if logger.isEnabledFor(logging.DEBUG): logger.debug("No intersection: The circles are identical but have different radii.")
            #     if logger.isEnabledFor(logging.DEBUG): logger.debug("Choosing a point on the perimeter of the circles.")
            #     intersect = np.array([x1 + r1, y1])
            #     return intersect, None

        if d > (r1 + r2):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "No direct intersection: The circles are too far apart.")
            if only_return_valid:
                return None, None

            proportional_distance = r1 / (r1 + r2)
            point_on_line = center1 + proportional_distance * (center2 - center1)

            return point_on_line, None

        # One circle inside the other with no intersection
        if d < abs(r1 - r2):
            if logger.isEnabledFor(logging.DEBUG): logger.debug(
                "No intersection: One circle is fully inside the other.")
            if only_return_valid:
                return None, None

            # Identify which circle is larger
            if r1 > r2:
                larger_center, larger_radius = center1, r1
                smaller_center, smaller_radius = center2, r2
            else:
                larger_center, larger_radius = center2, r2
                smaller_center, smaller_radius = center1, r1

            # Find a point on the line connecting the centers
            proportional_distance = (d + smaller_radius + 0.5 * (larger_radius - smaller_radius - d)) / d
            midpoint = larger_center + proportional_distance * (smaller_center - larger_center)

            if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Midpoint: {midpoint}")
            return midpoint, None

        if d == (r1 + r2) or d == abs(r1 - r2):
            logger.info("Whaaat? Tangential circles: The circles touch at exactly one point.")

            a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
            hi = 0  # Tangential circles will have h = 0 as h = sqrt(r1^2 - a^2)

            x3 = x1 + a * (x2 - x1) / d
            y3 = y1 + a * (y2 - y1) / d

            intersection = np.array([x3, y3])

            return intersection, None

        # Calculate points of intersection
        a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
        hi = np.sqrt(r1 ** 2 - a ** 2)

        x3 = x1 + a * (x2 - x1) / d
        y3 = y1 + a * (y2 - y1) / d

        intersect1 = np.array([x3 + hi * (y2 - y1) / d, y3 - hi * (x2 - x1) / d])
        intersect2 = np.array([x3 - hi * (y2 - y1) / d, y3 + hi * (x2 - x1) / d])

        return intersect1, intersect2

    def get_best_circle_intersection_location(self, start_coord: np.ndarray, end_coord: np.ndarray, act_type: str,
                                              distance_start_to_act: float, distance_act_to_end: float,
                                              num_circle_intersection_candidates=None, selection_strategy='top_n',
                                              max_iterations=15, only_return_valid=False):
        """
        Place a single activity at one of the closest locations.
        :param start_coord: Coordinates of the start location.
        :param end_coord: Coordinates of the end location.
        :param act_type: Type of activity (e.g., 'work', 'shopping').
        :param distance_start_to_act: Distance from start location to activity.
        :param distance_act_to_end: Distance from activity to end location.
        :param num_circle_intersection_candidates: Number of candidates to consider.
        :param selection_strategy: Strategy for selecting the best candidate.
        :param max_iterations: Maximum number of iterations for finding candidates.
        :param only_return_valid: If True, only return feasible locations, else None.
        :return: Tuple containing the selected identifier, coordinates, potential, and score.
        """
        # Handle home activities (special case)
        if act_type == s.ACT_HOME:
            logger.warning("Home activity detected. Returning start location.")
            return None, start_coord, None, None

        # If start and end locations are very close, fallback to ring search
        if h.euclidean_distance(start_coord, end_coord) < 1e-4:
            if only_return_valid and abs(distance_act_to_end - distance_start_to_act) > 10:  # 10m deviation is fine
                return None, None, None, None
            radius1, radius2 = h.spread_distances(distance_start_to_act, distance_act_to_end)
            candidate_ids, candidate_coords, candidate_potentials = self.target_locations.find_ring_candidates(
                act_type, start_coord, radius1, radius2, max_iterations=max_iterations, min_candidates=1
            )
        else:
            # Find intersection candidates between start and end
            candidate_ids, candidate_coords, candidate_potentials = self.find_circle_intersection_candidates(
                start_coord, end_coord, act_type, distance_start_to_act, distance_act_to_end,
                num_candidates=num_circle_intersection_candidates
            )
            if candidate_ids is None:
                if only_return_valid:
                    return None, None, None, None
                raise RuntimeError("Reached impossible state.")

        # Calculate distance deviations
        distance_deviations = (
                h.get_abs_distance_deviations(candidate_coords, start_coord, distance_start_to_act) +
                h.get_abs_distance_deviations(candidate_coords, end_coord, distance_act_to_end)
        )

        # Evaluate and select the best candidate
        scores = EvaluationFunction.evaluate_candidates(candidate_potentials, distance_deviations)
        best_index = EvaluationFunction.select_candidate_indices(scores, 1, selection_strategy)[0]

        # Extract the selected candidate's data
        best_id = candidate_ids[best_index][0]
        best_coord = candidate_coords[best_index][0]
        best_potential = candidate_potentials[best_index][0]
        best_score = scores[best_index][0]

        return best_id, best_coord, best_potential, best_score


def populate_legs_dict_from_df(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    USES OLD FORMAT.
    Uses the MiD df to populate a nested dictionary with leg information for each person.
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


def convert_to_segmented_plans(df: pd.DataFrame) -> SegmentedPlans:
    """
    Convert a DataFrame into SegmentedPlans using Leg structure.

    :param df: DataFrame containing the trip data.
    :return: SegmentedPlans (frozendict of person_id -> SegmentedPlan).
    """

    def safe_array(loc):
        """Convert location to np.array, replace None with an empty array."""
        return np.array(loc) if loc is not None else np.array([])

    def row_to_leg(leg_tuple) -> Leg:
        return Leg(
            unique_leg_id=leg_tuple[0],
            from_location=safe_array(leg_tuple[3]),
            to_location=safe_array(leg_tuple[4]),
            distance=float(leg_tuple[2]) if leg_tuple[2] is not None else 0.0,
            to_act_type=leg_tuple[1] if leg_tuple[1] is not None else "unknown",
            to_act_identifier=None
        )

    # Extract legs information into tuples
    legs_info_df = pd.DataFrame({
        s.UNIQUE_P_ID_COL: df[s.UNIQUE_P_ID_COL],
        'leg_info': list(zip(
            df[s.UNIQUE_LEG_ID_COL],
            df[s.ACT_TO_INTERNAL_COL],
            df[s.LEG_DISTANCE_METERS_COL],
            [safe_array(loc) for loc in df['from_location']],
            [safe_array(loc) for loc in df['to_location']],
        ))
    })

    # Group by unique person identifier and convert to SegmentedPlans
    grouped = legs_info_df.groupby(s.UNIQUE_P_ID_COL)['leg_info'].apply(list)
    segmented_plans = frozendict({
        person_id: tuple(
            tuple(row_to_leg(leg_tuple) for leg_tuple in segment)  # Segment as tuple of Legs
        )
        for person_id, segment in grouped.items()
    })

    return segmented_plans


def convert_to_detailed_segmented_plans(df: pd.DataFrame) -> DetailedSegmentedPlans:
    """
    Convert a DataFrame into DetailedSegmentedPlans using DetailedLeg structure.

    :param df: DataFrame containing the trip data.
    :return: DetailedSegmentedPlans (frozendict of person_id -> DetailedSegmentedPlan).
    """

    def safe_array(loc):
        """Convert location to np.array, replace None with an empty array."""
        return np.array(loc) if loc is not None else np.array([])

    def row_to_detailed_leg(leg_tuple) -> DetailedLeg:
        return DetailedLeg(
            unique_leg_id=leg_tuple[0],
            from_location=safe_array(leg_tuple[3]),
            to_location=safe_array(leg_tuple[4]),
            distance=float(leg_tuple[2]) if leg_tuple[2] is not None else 0.0,
            to_act_type=leg_tuple[1] if leg_tuple[1] is not None else "unknown",
            to_act_identifier=None,
            mode=leg_tuple[5] if leg_tuple[5] is not None else "unknown",
            is_main_activity=bool(leg_tuple[6]) if leg_tuple[6] is not None else False,
            mirrors_main_activity=bool(leg_tuple[7]) if leg_tuple[7] is not None else False,
            home_to_main_distance=float(leg_tuple[8]) if leg_tuple[8] is not None else 0.0
        )

    # Extract legs information into tuples
    legs_info_df = pd.DataFrame({
        s.UNIQUE_P_ID_COL: df[s.UNIQUE_P_ID_COL],
        'leg_info': list(zip(
            df[s.UNIQUE_LEG_ID_COL],
            df[s.ACT_TO_INTERNAL_COL],
            df[s.LEG_DISTANCE_METERS_COL],
            [safe_array(loc) for loc in df['from_location']],
            [safe_array(loc) for loc in df['to_location']],
            df[s.MODE_INTERNAL_COL],
            df[s.IS_MAIN_ACTIVITY_COL],
            df[s.MIRRORS_MAIN_ACTIVITY_COL],
            df[s.HOME_TO_MAIN_METERS_COL]
        ))
    })

    # Group by unique person identifier and convert to DetailedSegmentedPlans
    grouped = legs_info_df.groupby(s.UNIQUE_P_ID_COL)['leg_info'].apply(list)
    detailed_segmented_plans = frozendict({
        person_id: tuple(
            tuple(row_to_detailed_leg(leg_tuple) for leg_tuple in segment)  # Segment as tuple of DetailedLegs
        )
        for person_id, segment in grouped.items()
    })

    return detailed_segmented_plans




def prepare_population_df_for_location_assignment(df, filter_max_distance=None, number_of_persons=None) -> (
        pd.DataFrame, pd.DataFrame):
    """Temporarily prepare the MiD DataFrame for the leg dictionary function."""

    df["from_location"] = None
    df["to_location"] = None

    # Split persons with no leg ID into a separate DataFrame
    no_leg_df = df[df[s.LEG_ID_COL].isna()].copy()
    df = df.dropna(subset=[s.LEG_ID_COL])
    # TEMP: Remove persons that have no leg 1 (it may have been removed by enhancer)
    # TODO: Remove lines below again
    mobile_persons_with_leg_1 = df[df[s.UNIQUE_LEG_ID_COL].str.contains("_1.0")][s.UNIQUE_P_ID_COL].unique()
    df = df[df[s.UNIQUE_P_ID_COL].isin(mobile_persons_with_leg_1)]

    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"People with no legs: {no_leg_df.shape[0]}")

    # Throw out rows with missing values in the distance column
    row_count_before = df.shape[0]
    df = df.dropna(subset=[s.LEG_DISTANCE_METERS_COL])
    if logger.isEnabledFor(logging.DEBUG): logger.debug(
        f"Dropped {row_count_before - df.shape[0]} rows with missing distance values.")

    # Identify and remove records of persons with any trip exceeding the max distance if filter_max_distance is specified
    if filter_max_distance is not None:
        person_ids_to_exclude = df[df[s.LEG_DISTANCE_METERS_COL] > filter_max_distance][s.PERSON_ID_COL].unique()
        row_count_before = df.shape[0]
        df = df[~df[s.PERSON_ID_COL].isin(person_ids_to_exclude)]
        if logger.isEnabledFor(logging.DEBUG): logger.debug(
            f"Dropped {row_count_before - df.shape[0]} rows from persons with trips exceeding the max distance of {filter_max_distance} km.")

    # Ensure these columns are treated as object type to store arrays
    df["from_location"] = df["from_location"].astype(object)
    df["to_location"] = df["to_location"].astype(object)

    # Limit to the specified number of persons and keep all rows for these persons
    if number_of_persons is not None:
        person_ids = df[s.PERSON_ID_COL].unique()[:number_of_persons]
        df = df[df[s.PERSON_ID_COL].isin(person_ids)]

    # Add random home locations for each person for testing
    def generate_random_location_within_hanover():
        """Generate a random coordinate within Hanover, Germany, in EPSG:25832."""
        xmin, xmax = 546000, 556000
        ymin, ymax = 5800000, 5810000
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        return np.array([x, y])

    df[s.HOME_LOC_COL] = None
    for person_id, group in df.groupby(s.UNIQUE_P_ID_COL):
        home_location = generate_random_location_within_hanover()
        for i in group.index:
            df.at[i, s.HOME_LOC_COL] = home_location
        df.at[group.index[0], "from_location"] = home_location
        df.at[group.index[-1], "to_location"] = home_location

        home_rows_to = group[group[s.ACT_TO_INTERNAL_COL] == s.ACT_HOME].index
        if not home_rows_to.empty:
            for idx in home_rows_to:
                df.at[idx, "to_location"] = home_location

        home_rows_from = group[group[s.ACT_FROM_INTERNAL_COL] == s.ACT_HOME].index
        if not home_rows_from.empty:
            for idx in home_rows_from:
                df.at[idx, "from_location"] = home_location

    logger.info("Prepared population dataframe for location assignment.")
    if logger.isEnabledFor(logging.DEBUG): logger.debug(df.head())
    return df, no_leg_df


def segment_plans(plans):
    """
    USES OLD FORMAT
    Segment the plan of each person into separate trips where only the start and end locations are known.
    :param plans:
    :return:
    Example output:
    data = {
    '10000290_11563_10000291': [
        [
            {
                'unique_leg_id': '10000290_11563_10000291_1.0',
                'to_act_type': 'shopping',
                'distance': 950.0,
                'from_location': np.array([552452.11071084, 5807493.538159]),
                'to_location': np.array([], dtype=float64),
                'mode': 'car',
                'is_main_activity': 1,
                'home_to_main_distance': 120.0
            },
            {
                'unique_leg_id': '10000290_11563_10000291_2.0',
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
                'unique_leg_id': '10000290_11563_10000291_3.0',
                'to_act_type': 'work',
                'distance': 500.0,
                'from_location': np.array([552452.11071084, 5807493.538159]),
                'to_location': np.array([], dtype=float64),
                'mode': 'walk',
                'is_main_activity': 1,
                'home_to_main_distance': 100.0
            },
            {
                'unique_leg_id': '10000290_11563_10000291_4.0',
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
                'unique_leg_id': '10000370_11564_10000371_1.0',
                'to_act_type': 'leisure',
                'distance': 10450.0,
                'from_location': np.array([554098.49165674, 5802930.10530201]),
                'to_location': np.array([], dtype=float64),
                'mode': 'car',
                'is_main_activity': 1,
                'home_to_main_distance': 1500.0
            },
            {
                'unique_leg_id': '10000370_11564_10000371_2.0',
                'to_act_type': 'home',
                'distance': 7600.0,
                'from_location': np.array([], dtype=float64),
                'to_location': np.array([], dtype=float64),
                'mode': 'car',
                'is_main_activity': 0,
                'home_to_main_distance': 1500.0
            },
            {
                'unique_leg_id': '10000370_11564_10000371_3.0',
                'to_act_type': 'shopping',
                'distance': 13300.0,
                'from_location': np.array([], dtype=float64),
                'to_location': np.array([], dtype=float64),
                'mode': 'car',
                'is_main_activity': 0,
                'home_to_main_distance': 1500.0
            },
            {
                'unique_leg_id': '10000370_11564_10000371_4.0',
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
    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Segmenting legs for {len(plans)} persons.")
    segmented_dict = defaultdict(list)

    for person_id, legs in plans.items():
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


def new_segment_plans(plans: SegmentedPlans) -> SegmentedPlans:
    """
    Segment the plan of each person into segments where only the start and end locations are known.
    :param plans: SegmentedPlans (frozendict of person_id -> SegmentedPlan).
    :return: SegmentedPlans (frozendict of person_id -> SegmentedPlan).
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Segmenting legs for {len(plans)} persons.")

    segmented_result = {}

    for person_id, legs in plans.items():
        segments = []  # List to hold completed segments
        current_segment = []  # Temporary list for building a segment

        for leg in legs:
            current_segment.append(leg)
            if leg.to_location.size > 0:  # End of a segment when to_location is set
                segments.append(tuple(current_segment))  # Immutable tuple for the segment
                current_segment = []  # Start a new segment

        # Add any remaining legs as a final segment
        if current_segment:
            segments.append(tuple(current_segment))

        # Store the segmented plan as an immutable tuple
        segmented_result[person_id] = tuple(segments)

    # Return the result wrapped in a frozendict
    return frozendict(segmented_result)


# TODO: medium term remove
def insert_placed_distances(segment):
    """Inserts info on the actual distances between placed activities for a fully located segment.
    Optional; for debugging and evaluation."""
    for leg in segment:
        leg['placed_distance'] = h.euclidean_distance(leg['from_location'], leg['to_location'])
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


def flatten_segmented_dict(segmented_dict: Dict[str, List[List[Dict[str, Any]]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Recombine the segments of each person into a single list of legs."""
    for person_id, segments in segmented_dict.items():
        segmented_dict[person_id] = [leg for segment in segments for leg in segment]
    return segmented_dict


# def build_candidate_tree(segment, tree):
#     """Work in progress. """

#     # At the second highest level, get n candidates with score for the location
#     candidates = find_location_candidates(segment[0]['from_location'], segment[-1]['to_location'], segment[0]['to_act_type'], tree[0][0][2], tree[0][1][2], 5)
#     if logger.isEnabledFor(logging.DEBUG): logger.debug(candidates)

#     # At the next levels, get n candidates with score for each connected location
#     for level in range(1, len(tree)):
#         for i in range(0, len(tree[level]), 2):
#             candidates = find_location_candidates(segment[i]['from_location'], segment[i+1]['to_location'], segment[i]['to_act_type'], tree[level][i][2], tree[level][i+1][2], 5)
#             if logger.isEnabledFor(logging.DEBUG): logger.debug(candidates)


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
            records.extend(entry)  # works here because the entries are lists

    data_df = pd.DataFrame(records)

    # Merge the DataFrame on leg_id
    df = df.merge(data_df[[s.UNIQUE_LEG_ID_COL, 'from_location', 'to_location', 'placed_distance',
                           'placed_distance_absolute_diff',
                           'placed_distance_relative_diff']],
                  on=s.UNIQUE_LEG_ID_COL, how='left')

    return df


def write_hoerl_df_to_big_df(hoerl_df, big_df):  # TODO: write main stuff (somewhere else) to big df
    """Unites the Hoerl DataFrame with the big DataFrame."""
    hoerl_df = hoerl_df.rename(columns={'person_id': s.PERSON_ID_COL,
                                        'location_id': 'location_id_hoerl',
                                        'geometry': 'to_location_hoerl'})

    # Recreate the unique leg id column
    hoerl_df[s.LEG_NON_UNIQUE_ID_COL] = hoerl_df['activity_index'] - 1  # Starting index
    hoerl_df[s.UNIQUE_LEG_ID_COL] = (hoerl_df[s.PERSON_ID_COL] + "_" + hoerl_df[s.LEG_NON_UNIQUE_ID_COL].astype(str) +
                                     ".0")  # .0 is added to match the format of the big DataFrame :(

    hoerl_df = hoerl_df[[s.UNIQUE_LEG_ID_COL, 'location_id_hoerl', 'to_location_hoerl']]

    # Perform the merge with suffixes
    merged_df = big_df.merge(hoerl_df, on=s.UNIQUE_LEG_ID_COL, how='left', suffixes=('', '_hoerl'))

    # Combine the columns to prioritize non-NaN values from hoerl_df
    # merged_df['location_id'] = merged_df['location_id_hoerl'].combine_first(merged_df['location_id'])
    merged_df['to_location'] = merged_df['to_location_hoerl'].combine_first(merged_df['to_location'])

    # Drop the temporary hoerl columns
    merged_df = merged_df.drop(columns=['location_id_hoerl', 'to_location_hoerl'])

    return merged_df


def write_placement_results_dict_to_population_df(placement_results_dict, population_df, merge_how = 'left') -> pd.DataFrame:
    """Writes the placement results from the dictionary to the big DataFrame."""
    records = []
    for person_id, segments in placement_results_dict.items():
        for segment in segments:
            for leg in segment:
                records.append(leg)

    data_df = pd.DataFrame(records)

    # Check columns
    mandatory_columns = [s.UNIQUE_LEG_ID_COL, 'from_location', 'to_location']
    optional_columns = ['to_act_name', 'to_act_potential', 'to_act_identifier']

    for col in mandatory_columns:
        if col not in data_df.columns:
            raise ValueError(f"Mandatory column '{col}' is missing in data_df.")

    existing_optional_columns = [col for col in optional_columns if col in data_df.columns]
    existing_columns = mandatory_columns + existing_optional_columns

    # Perform the merge with the existing columns
    merged_df = population_df.merge(data_df[existing_columns], on=s.UNIQUE_LEG_ID_COL, how=merge_how)

    # Combine columns to prioritize non-NaN values from data_df (_x is the original column, _y is the new one)
    # From and to location are always expected to be present (even before placement, there will be home locations)
    merged_df['from_location'] = merged_df['from_location_y'].combine_first(merged_df['from_location_x'])
    merged_df['to_location'] = merged_df['to_location_y'].combine_first(merged_df['to_location_x'])
    merged_df = merged_df.drop(columns=['from_location_x', 'from_location_y', 'to_location_x', 'to_location_y'])

    try:
        merged_df['to_act_identifier'] = merged_df['to_act_identifier_y'].combine_first(
            merged_df['to_act_identifier_x'])
        merged_df = merged_df.drop(columns=['to_act_identifier_x', 'to_act_identifier_y'])
    except KeyError:
        pass
    try:
        merged_df['to_act_name'] = merged_df['to_act_name_y'].combine_first(merged_df['to_act_name_x'])
        merged_df = merged_df.drop(columns=['to_act_name_x', 'to_act_name_y'])
    except KeyError:
        pass
    try:
        merged_df['to_act_potential'] = merged_df['to_act_potential_y'].combine_first(merged_df['to_act_potential_x'])
        merged_df = merged_df.drop(columns=['to_act_potential_x', 'to_act_potential_y'])
    except KeyError:
        pass

    # Make sure no merge postfixes are left
    assert not any([col.endswith('_x') or col.endswith('_y') for col in merged_df.columns]), "Postfixes left."
    return merged_df
