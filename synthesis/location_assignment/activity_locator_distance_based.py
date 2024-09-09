import copy
import math
import os
import pickle
import pprint
import random
import time
import json

from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Literal

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from utils import settings as s, helpers as h, pipeline_setup
from utils.types import PlanLeg, PlanSegment, SegmentedPlan, SegmentedPlans, UnSegmentedPlan, UnSegmentedPlans
from utils.logger import logging
from utils.stats_tracker import stats_tracker

logger = logging.getLogger(__name__)


# TODO: Currently, all outputs are arrays even if not necessary. Maybe change to single values.

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

    #TODO: get station data from all of germany
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

        #TODO: DO we really?
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

    def locate_commuter(self, person_legs: UnSegmentedPlan) -> UnSegmentedPlan:
        """Gets a person's main activity and locates it.
        Currently just uses the Euclidean distance and potentials.
        Planned to also use O-D matrices.
        :return: Updated list of legs with located main activities.
        """
        main_activity_index, main_activity_leg = h.get_main_activity_leg(person_legs)
        if main_activity_leg is None:
            return person_legs

        # Skip if main already located
        to_location = main_activity_leg.get('to_location')
        assert isinstance(to_location, np.ndarray), "Bad location format."
        if to_location.size != 0:
            return person_legs

        # TODO go from here
        target_activity = main_activity_leg['to_act_type']
        home_location = person_legs[0]['from_location']
        estimated_distance_home_to_main = person_legs[0]['home_to_main_distance']

        # Radii are iteratively spread by find_ring_candidates until a candidate is found
        radius1, radius2 = h.spread_distances(estimated_distance_home_to_main,
                                              estimated_distance_home_to_main)  # Initial
        candidates = self.target_locations.find_ring_candidates(target_activity, home_location, radius1=radius1,
                                                                radius2=radius2)
        scores = EvaluationFunction.evaluate_candidates(candidates[-2], None, len(candidates[-2]))
        chosen_candidate, score = (
            EvaluationFunction.select_candidates(candidates, scores, 1, 'monte_carlo'))

        act_identifier, act_name, act_coord, act_pot, act_dist = chosen_candidate

        # Update the main activity leg and the subsequent leg
        person_legs[main_activity_index]['to_location'] = act_coord
        person_legs[main_activity_index]['to_act_identifier'] = act_identifier
        person_legs[main_activity_index]['to_act_name'] = act_name
        person_legs[main_activity_index]['to_act_cap'] = act_pot
        person_legs[main_activity_index]['to_act_score'] = score

        if main_activity_index + 1 < len(person_legs):  # Set from location if there is a subsequent leg
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

    # def __init__(self, data: Dict[str, Dict[str, np.ndarray]]):
    #     self.data: Dict[str, Dict[str, np.ndarray]] = data
    #     self.indices: Dict[str, KDTree] = {}
    #
    #     for type, pdata in self.data.items():
    #         logger.debug(f"Constructing spatial index for {type} ...")
    #         self.indices[type] = KDTree(pdata["coordinates"])

    def __init__(self, json_folder_path: str):
        self.data: Dict[str, Dict[str, np.ndarray]] = self.load_reformatted_osmox_data(h.get_files(json_folder_path))
        self.indices: Dict[str, KDTree] = {}

        for type, pdata in self.data.items():
            logger.info(f"Constructing spatial index for {type} ...")
            self.indices[type] = KDTree(pdata["coordinates"])

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
        index = self.indices[purpose].query(location.reshape(1, -1), return_distance=False)[0][0]
        identifier = self.data[purpose]["identifiers"][index]
        location = self.data[purpose]["coordinates"][index]
        return identifier, location

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
        candidate_identifiers = np.atleast_1d(np.array(self.data[type]["identifiers"])[indices].squeeze())
        candidate_names = np.atleast_1d(np.array(self.data[type]["names"])[indices].squeeze())
        candidate_coordinates = np.atleast_1d(np.array(self.data[type]["coordinates"])[indices].squeeze())
        candidate_potentials = np.atleast_1d(np.array(self.data[type]["potentials"])[indices].squeeze())

        # Reshape to 2D arrays so everything is consistent and nicely below each other
        candidate_identifiers = candidate_identifiers.reshape(-1, 1)
        candidate_names = candidate_names.reshape(-1, 1)
        candidate_potentials = candidate_potentials.reshape(-1, 1)
        # candidate_coordinates = candidate_coordinates.reshape(-1, 2)  # Coordinates stay as 2D

        return candidate_identifiers, candidate_names, candidate_coordinates, candidate_potentials, candidate_distances

    # deprecated
    # def query_within_radius(self, act_type: str, location: np.ndarray, radius: float):
    #     """
    #     Find the activity locations within a given radius of a location and type.
    #     :param act_type: The activity category to query.
    #     :param location: A 1D numpy array representing the location to query (coordinates [1.5, 2.5]).
    #     :param radius: The maximum distance from the location to search for candidates.
    #     :return: A tuple containing four numpy arrays: identifiers, coordinates, distances, and remaining potentials of the nearest candidates.
    #     """
    #     # Ensure location is a 2D array with a single location
    #     location = location.reshape(1, -1)
    #
    #     # Query the KDTree for locations within radius
    #     candidate_indices = self.indices[act_type].query_radius(location, radius)
    #     logger.debug(f"Query Indices: {candidate_indices}")
    #
    #     # Get the identifiers, coordinates, and distances for locations within the radius
    #     candidate_identifiers = np.array(self.data[act_type]["identifiers"])[candidate_indices[0]]
    #     candidate_names = np.array(self.data[act_type]["names"])[candidate_indices[0]]
    #     candidate_coordinates = np.array(self.data[act_type]["coordinates"])[candidate_indices[0]]
    #     candidate_potentials = np.array(self.data[act_type]["potentials"])[candidate_indices[0]]
    #     # candidate_distances = np.linalg.norm(candidate_coordinates - location, axis=1)
    #     candidate_distances = None
    #
    #     return candidate_identifiers, candidate_names, candidate_coordinates, candidate_potentials, candidate_distances

    def query_within_ring(self, act_type: str, location: np.ndarray, radius1: float, radius2: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the activity locations within a ring defined by two radii around a location and type.
        :param act_type: The activity category to query.
        :param location: A 1D numpy array representing the location to query (coordinates [1.5, 2.5]).
        :param radius1: Any of the two radii defining the ring.
        :param radius2: The other one.
        :return: A tuple containing four numpy arrays: identifiers, coordinates (2D array), distances, and remaining potentials of the nearest candidates.
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
        candidate_identifiers = np.atleast_1d(np.array(self.data[act_type]["identifiers"])[annulus_indices].squeeze())
        candidate_names = np.atleast_1d(np.array(self.data[act_type]["names"])[annulus_indices].squeeze())
        candidate_coordinates = np.atleast_1d(np.array(self.data[act_type]["coordinates"])[annulus_indices].squeeze())
        candidate_potentials = np.atleast_1d(np.array(self.data[act_type]["potentials"])[annulus_indices].squeeze())
        candidate_distances = None

        # Reshape to 2D arrays so everything is consistent and nicely below each other
        candidate_identifiers = candidate_identifiers.reshape(-1, 1)
        candidate_names = candidate_names.reshape(-1, 1)
        candidate_potentials = candidate_potentials.reshape(-1, 1)
        # candidate_coordinates = candidate_coordinates.reshape(-1, 2)  # Coordinates stay as 2D

        return candidate_identifiers, candidate_names, candidate_coordinates, candidate_potentials, candidate_distances

    def query_within_two_overlapping_rings(self, act_type: str, location1: np.ndarray, location2: np.ndarray,
                                           radius1a: float, radius1b: float, radius2a: float, radius2b: float,
                                           max_number_of_candidates: int = None):
        """
        Find the activity locations within a ring defined by two radii around a location and type.
        :param act_type: The activity category to query.
        :param location1: A 1D numpy array representing the first location to query (coordinates [1.5, 2.5]).
        :param location2: A 1D numpy array representing the second location to query (coordinates [1.5, 2.5]).
        :param radius1a: One of the two radii defining the first ring.
        :param radius1b: The other radius defining the first ring.
        :param radius2a: One of the two radii defining the second ring.
        :param radius2b: The other radius defining the second ring.
        :param max_number_of_candidates: Reduces the number of candidates to evaluate, for some small performance gain.
        :return: A tuple containing three numpy arrays: Identifiers, coordinates (2D array), potentials
        """
        # Ensure location is a 2D array with a single location
        location1 = location1.reshape(1, -1)
        location2 = location2.reshape(1, -1)

        outer_radius1 = max(radius1a, radius1b)
        inner_radius1 = min(radius1a, radius1b)
        outer_radius2 = max(radius2a, radius2b)
        inner_radius2 = min(radius2a, radius2b)

        outer_indices1 = self.indices[act_type].query_radius(location1, outer_radius1)
        outer_indices2 = self.indices[act_type].query_radius(location2, outer_radius2)
        if outer_indices1 is None or outer_indices2 is None:
            return None
        if len(outer_indices1[0]) == 0 or len(outer_indices2[0]) == 0:
            return None

        inner_indices1 = self.indices[act_type].query_radius(location1, inner_radius1)
        inner_indices2 = self.indices[act_type].query_radius(location2, inner_radius2)

        outer_indices_set1 = set(outer_indices1[0])
        inner_indices_set1 = set(inner_indices1[0])
        outer_indices_set2 = set(outer_indices2[0])
        inner_indices_set2 = set(inner_indices2[0])

        overlapping_rings_indices = list(outer_indices_set1.intersection(outer_indices_set2) -
                                         inner_indices_set1.union(inner_indices_set2))
        if not overlapping_rings_indices:
            return None

        if max_number_of_candidates and len(overlapping_rings_indices) > max_number_of_candidates:
            overlapping_rings_indices = random.sample(overlapping_rings_indices, max_number_of_candidates)

        # Get the identifiers, coordinates, and distances for locations within the annulus
        candidate_identifiers = np.atleast_1d(
            np.array(self.data[act_type]["identifiers"])[overlapping_rings_indices].squeeze())
        candidate_coordinates = np.atleast_1d(
            np.array(self.data[act_type]["coordinates"])[overlapping_rings_indices].squeeze())
        candidate_potentials = np.atleast_1d(
            np.array(self.data[act_type]["potentials"])[overlapping_rings_indices].squeeze())

        # Reshape to 2D arrays so everything is consistent and nicely below each other
        candidate_identifiers = candidate_identifiers.reshape(-1, 1)
        candidate_potentials = candidate_potentials.reshape(-1, 1)
        # candidate_coordinates = candidate_coordinates.reshape(-1, 2)  # Coordinates stay as 2D

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

    def find_ring_candidates(self, act_type: str, center: np.ndarray, radius1: float, radius2: float, max_iterations=15,
                             min_candidates=10, restrict_angle=False, direction_point=None, angle_range=math.pi / 2) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find candidates within a ring around a center point.
        Iteratively increase the radii until a sufficient number of candidates is found."""
        i = 0
        logger.debug(
            f"Finding candidates for type {act_type} within a ring around {center} with radii {radius1} and {radius2}.")
        while True:
            candidates = self.query_within_ring(act_type, center, radius1, radius2)
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
                    stats_tracker.log(f"Find_ring_candidates: Iterations for {act_type}", i)
                    return candidates
            radius1, radius2 = h.spread_distances(radius1, radius2, iteration=i, first_step=20)
            i += 1
            logger.debug(f"Iteration {i}. Increasing radii to {radius1} and {radius2}.")
            if i > max_iterations:
                raise ValueError(f"Not enough candidates found after {max_iterations} iterations.")

    def find_overlapping_rings_candidates(self, act_type: str, location1: np.ndarray, location2: np.ndarray,
                                          radius1a: float, radius1b: float, radius2a: float, radius2b: float,
                                          min_candidates=1, max_candidates=None, max_iterations=15):
        """Find candidates within two overlapping rings around two center points.
        Iteratively increase the radii until a sufficient number of candidates is found.
        """
        # original_ring_width1 = abs(radius1b - radius1a)
        # original_ring_width2 = abs(radius2b - radius2a)
        i = 0
        while True:
            candidates = self.query_within_two_overlapping_rings(
                act_type, location1, location2, radius1a, radius1b, radius2a, radius2b, max_candidates)
            if candidates is not None:
                if len(candidates[0]) >= min_candidates:
                    logger.debug(f"Found {len(candidates[0])} candidates.")
                    stats_tracker.log(f"Find_ring_candidates: Iterations for {act_type}", i)
                    # total_ring_width_change = (abs(radius1b - radius1a) + abs(radius2b - radius2a) -
                    #                            original_ring_width1 - original_ring_width2)
                    return candidates, i
            radius1a, radius1b = h.spread_distances(radius1a, radius1b, iteration=i, first_step=50)
            radius2a, radius2b = h.spread_distances(radius2a, radius2b, iteration=i, first_step=50)
            i += 1
            logger.debug(f"Iteration {i}. Increasing radii to {radius1a}, {radius1b} and {radius2a}, {radius2b}.")
            if i > max_iterations:
                raise RuntimeError(f"Not enough candidates found after {max_iterations} iterations.")


class AdvancedPetreAlgorithm:
    """
    """

    def __init__(self, target_locations: TargetLocations, segmented_dict: Dict[str, list[list[dict[str, Any]]]],
                 number_of_branches: int = 10, min_candidates: int = None, max_candidates: int = None,
                 anchor_strategy: Literal[
                     "lower_middle", "upper_middle", "start", "end"] = "start"):
        self.target_locations = target_locations
        self.segmented_dict = segmented_dict
        self.number_of_branches = number_of_branches
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.anchor_strategy = anchor_strategy
        self.c_i = CircleIntersection(target_locations)

    def run(self):
        placed_dict = {}
        if self.min_candidates is None:
            self.min_candidates = max(1, self.number_of_branches // 20)
        stats_tracker.log("AdvancedPetreAlgorithm: Min candidates", self.min_candidates)
        for person_id, segments in tqdm(self.segmented_dict.items(), desc="Processing persons"):
            placed_dict[person_id] = []
            for segment in segments:
                placed_segment, _ = self.solve_segment(segment, number_of_branches=self.number_of_branches,
                                                       max_candidates=self.max_candidates,
                                                       anchor_strategy=self.anchor_strategy)
                placed_dict[person_id].append(placed_segment)
        return placed_dict

    def solve_segment(self, in_segment, number_of_branches=1, max_candidates=None, anchor_strategy="lower_middle",
                      min_candidates=None):
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
            if not (segment[0]['from_location'].size > 0 and segment[0][
                'to_location'].size > 0):
                print("gdfgd")
            assert segment[0]['from_location'].size > 0 and segment[0][
                'to_location'].size > 0, "Both start and end locations must be known for a single leg."
            return segment, 0
        # if there are only two legs, we can find the loc immediately
        elif len(segment) == 2:
            stats_tracker.increment("AdvancedPetreAlgorithm: 2-leg segments")
            act_identifier, act_name, act_coord, act_cap, act_dist, act_score = self.c_i.get_best_circle_intersection_location(
                segment[0]['from_location'], segment[1]['to_location'], segment[0]['to_act_type'],
                segment[0]['distance'], segment[1]['distance'], 1)
            segment[0]['to_location'] = act_coord
            segment[1]['from_location'] = act_coord
            segment[0]['to_act_identifier'] = act_identifier
            # segment[0]['to_act_name'] = act_name
            # segment[0]['to_act_cap'] = act_cap
            # segment[0]['to_act_score'] = act_score
            return segment, act_score

        else:
            logger.debug(f"Advanced locating. Segment has {len(segment)} legs.")
            stats_tracker.increment(f"AdvancedPetreAlgorithm: {len(segment)}-leg segments")
            if anchor_strategy == "lower_middle":
                anchor_idx = len(segment) // 2 - 1
            elif anchor_strategy == "upper_middle":
                anchor_idx = len(segment) // 2
            elif anchor_strategy == "start":
                anchor_idx = 0
            elif anchor_strategy == "end":
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

            max_iterations = 15

            candidates, iterations = self.target_locations.find_overlapping_rings_candidates(act_type,
                                                                                             location1, location2,
                                                                                             min_possible_distance1,
                                                                                             max_possible_distance1,
                                                                                             min_possible_distance2,
                                                                                             max_possible_distance2,
                                                                                             min_candidates,
                                                                                             max_candidates,
                                                                                             max_iterations)

            candidate_identifiers, candidate_coordinates, candidate_potentials = candidates

            if iterations > 0:  # We need to find distance deviations of each candidate to score them
                # candidate_distances_from_start = np.linalg.norm(candidate_coordinates - location1, axis=1)
                # candidate_distances_to_end = np.linalg.norm(candidate_coordinates - location2, axis=1)
                #
                # upper_deviations1 = np.maximum(0, candidate_distances_from_start - max_possible_distance1)
                # lower_deviations1 = np.maximum(0, min_possible_distance1 - candidate_distances_from_start)
                # upper_deviations2 = np.maximum(0, candidate_distances_to_end - max_possible_distance2)
                # lower_deviations2 = np.maximum(0, min_possible_distance2 - candidate_distances_to_end)
                #
                # total_deviations = upper_deviations1 + lower_deviations1 + upper_deviations2 + lower_deviations2

                candidate_deviations = np.zeros(len(candidate_coordinates))
                # We only count deviations of lowest-level legs to avoid double counting
                if len(distances_start_to_act) == 1:
                    candidate_deviations += h.get_abs_distance_deviations(candidate_coordinates, location1,
                                                                          distances_start_to_act)
                if len(distances_act_to_end) == 1:
                    candidate_deviations += h.get_abs_distance_deviations(candidate_coordinates, location2,
                                                                          distances_act_to_end)

                local_scores = EvaluationFunction.evaluate_candidates(candidate_potentials, candidate_deviations)

            else:  # No distance deviations expected, just score by potentials
                # # 'DEBUG'
                # # TODO: REMOVE
                # candidate_distances_from_start = np.linalg.norm(candidate_coordinates - location1, axis=1)
                # candidate_distances_to_end = np.linalg.norm(candidate_coordinates - location2, axis=1)
                #
                # upper_deviations1 = np.maximum(0, candidate_distances_from_start - max_possible_distance1)
                # lower_deviations1 = np.maximum(0, min_possible_distance1 - candidate_distances_from_start)
                # upper_deviations2 = np.maximum(0, candidate_distances_to_end - max_possible_distance2)
                # lower_deviations2 = np.maximum(0, min_possible_distance2 - candidate_distances_to_end)

                candidate_deviations = np.zeros(len(candidate_coordinates))
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
                                                                      len(candidate_coordinates))

            stats_tracker.log(f"AdvancedPetreAlgorithm: Number of candidates at segment length {len(segment)}:",
                              len(local_scores))

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
                                                                               number_of_branches, 'mixed')
            selected_identifiers, selected_coords, selected_potentials = selected_candidates

            # Split the segment at the anchor point
            subsegment1 = segment[:anchor_idx + 1]  # Up to and including anchor
            subsegment2 = segment[anchor_idx + 1:]  # From anchor + 1 to end

            full_segs = []
            branch_scores = []
            for i in range(len(selected_coords)):
                subsegment1[-1]['to_location'] = selected_coords[i]
                subsegment1[-1]['to_act_identifier'] = np.array(selected_identifiers[i])
                subsegment2[0]['from_location'] = selected_coords[i]

                located_seg1, score1 = self.solve_segment(subsegment1, number_of_branches, max_candidates,
                                                          anchor_strategy)
                located_seg2, score2 = self.solve_segment(subsegment2, number_of_branches, max_candidates,
                                                          anchor_strategy)

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
            logger.debug("Greedy locating. Only two legs in segment.")
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
            logger.debug(f"Greedy locating. Segment has {len(segment)} legs.")
            distances = [leg['distance'] for leg in segment]

            real_distance = h.euclidean_distance(segment[0]['from_location'], segment[-1]['to_location'])

            tree = h.build_estimation_tree(distances)
            tree = self.adjust_estimation_tree(tree, real_distance, strong_adjust=True)
            position_on_segment_info = self.build_position_on_segment_info(
                len(distances))  # tells us at each level which legs to look at
            assert len(tree) == len(position_on_segment_info), "Tree and position info must have the same length."

            for level in range(len(tree) - 1, -1, -1):
                for i, leg_idx in enumerate(position_on_segment_info[level]):
                    logger.debug(f"Level: {level}, i: {i}, leg_idx: {leg_idx}")
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
            logger.debug("Simple locating. Only two legs in segment.")
            distance1 = person_legs[0]['distance']
            distance2 = person_legs[1]['distance']
            if abs(distance1 - distance2) < 30:  # always meters!
                distance1, distance2 = h.spread_distances(distance1, distance2, first_step=20)
            candidates = self.target_locations.find_ring_candidates(person_legs[0]['to_act_type'],
                                                                    person_legs[0]['from_location'], distance1,
                                                                    distance2)
            act_identifier, act_name, act_coord, act_cap, act_dist, act_score = (
                EvaluationFunction.monte_carlo_select_candidate(candidates))
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
                            self.c_i.get_best_circle_intersection_location(person_legs[i]['from_location'],
                                                                           person_legs[-1]['to_location'],
                                                                           person_legs[i]['to_act_type'],
                                                                           person_legs[i]['distance'],
                                                                           person_legs[-1]['distance'], 5)

                    else:
                        logger.debug("Selecting location using ring with angle restriction.")
                        distance = person_legs[i]['distance']
                        radius1, radius2 = h.spread_distances(distance, distance, iteration=0, first_step=20)
                        candidates = self.target_locations.find_ring_candidates(person_legs[i]['to_act_type'],
                                                                                person_legs[i]['from_location'],
                                                                                radius1, radius2, restrict_angle=True,
                                                                                direction_point=person_legs[-1][
                                                                                    'to_location'])

                        act_identifier, act_name, act_coord, act_cap, act_dist, act_score = (
                            EvaluationFunction.monte_carlo_select_candidate(candidates))
                else:
                    logger.debug("Selecting location using ring.")
                    distance = person_legs[i]['distance']
                    radius1, radius2 = h.spread_distances(distance, distance, iteration=0, first_step=20)
                    candidates = self.target_locations.find_ring_candidates(person_legs[i]['to_act_type'],
                                                                            person_legs[i]['from_location'],
                                                                            radius1, radius2)
                    act_identifier, act_name, act_coord, act_cap, act_dist, act_score = (
                        EvaluationFunction.monte_carlo_select_candidate(candidates))

                person_legs[i]['to_location'] = act_coord
                person_legs[i + 1]['from_location'] = act_coord
                person_legs[i]['to_act_identifier'] = act_identifier
                person_legs[i]['to_act_name'] = act_name
                person_legs[i]['to_act_cap'] = act_cap
                person_legs[i]['to_act_score'] = act_score

            return person_legs


class SimpleMainLocationAlgorithm:
    """
    Since the algorithm is targeted for solving just the main location, the "segment" is usually the whole trip.
    """

    def __init__(self, target_locations: TargetLocations, legs_dict: Dict[str, list[dict[str, Any]]]):
        self.target_locations = target_locations
        self.legs_dict = legs_dict

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

        # Skip if main already located
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
        scores = EvaluationFunction.evaluate_candidates(candidates[-2], None, len(candidates[-2]))
        chosen_candidate, score = (
            EvaluationFunction.select_candidates(candidates, scores, 1, 'monte_carlo'))

        act_identifier, act_name, act_coord, act_pot, act_dist = chosen_candidate

        # Update the main activity leg and the subsequent leg
        person_legs[main_activity_index]['to_location'] = act_coord
        person_legs[main_activity_index]['to_act_identifier'] = act_identifier
        person_legs[main_activity_index]['to_act_name'] = act_name
        person_legs[main_activity_index]['to_act_cap'] = act_pot
        person_legs[main_activity_index]['to_act_score'] = score

        if main_activity_index + 1 < len(person_legs):  # Set from location if there is a subsequent leg
            person_legs[main_activity_index + 1]['from_location'] = act_coord

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
    def select_candidates(
            candidates: Tuple[np.ndarray, ...],
            scores: np.ndarray,
            num_candidates: int,
            strategy: str = 'monte_carlo',
            top_portion: float = 0.5):
        """
        Selects a specified number of candidates based on their normalized scores using Monte Carlo sampling,
        a top-n strategy, or a mixed strategy combining top scores and Monte Carlo sampling.

        :param candidates: A tuple of arrays with the candidates.
        :param scores: A 1D array of scores corresponding to the candidates.
        :param num_candidates: The number of candidates to select.
        :param strategy: Selection strategy ('monte_carlo', 'top_n', or 'mixed').
        :param top_portion: Portion of candidates to select from the top scores when using the 'mixed' strategy (default 0.5).
        :return: A tuple containing:
            - A tuple of arrays with the selected candidates.
            - A 1D array of the scores corresponding to the selected candidates.
        """
        assert len(candidates[0]) == len(scores), "The number of candidates and scores must match."
        if num_candidates >= len(candidates[0]):
            stats_tracker.increment("Select_candidates: All candidates selected")
            return candidates, scores

        if strategy == 'monte_carlo':
            stats_tracker.increment("Scoring runs (Monte Carlo)")
            a_scores = np.array(scores, dtype=np.float64)  # Make floating-point array
            m_scores = a_scores / np.sum(a_scores)
            chosen_indices = np.random.choice(len(m_scores), num_candidates, p=m_scores, replace=False)

        elif strategy == 'top_n':
            stats_tracker.increment("Scoring runs (Top N)")
            chosen_indices = np.argsort(scores)[-num_candidates:][::-1]

        elif strategy == 'mixed':
            stats_tracker.increment("Scoring runs (Mixed)")
            # Determine the number of top candidates to select
            num_top = int(np.ceil(num_candidates * top_portion))
            num_monte_carlo = num_candidates - num_top

            # Identify the top score and find all indices with that score
            highest_value = np.max(scores)
            highest_indices = np.where(scores == highest_value)[0]

            if len(highest_indices) == len(scores):
                stats_tracker.increment("Select_candidates: All scores equal")
                chosen_indices = np.random.choice(len(scores), num_candidates, replace=False)
            else:
                if len(highest_indices) > num_top:
                    stats_tracker.increment("Select_candidates: More top scores than num_top")
                    top_indices = np.random.choice(highest_indices, num_top, replace=False)
                else:
                    stats_tracker.increment("Select_candidates: Fewer or eq top scores than num_top")
                    top_indices = np.argsort(scores)[-num_top:][::-1]

                # Monte Carlo selection from the remaining candidates
                remaining_indices = np.setdiff1d(np.arange(len(scores)), top_indices)
                if len(remaining_indices) > 0 and num_monte_carlo > 0:
                    m_scores = scores[remaining_indices] / np.sum(scores[remaining_indices])
                    monte_carlo_indices = np.random.choice(remaining_indices, num_monte_carlo, p=m_scores,
                                                           replace=False)
                    chosen_indices = np.concatenate((top_indices, monte_carlo_indices))
                else:
                    # If no remaining candidates, fallback to top indices only
                    chosen_indices = top_indices

        else:
            raise ValueError("Invalid selection strategy. Use 'monte_carlo', 'top_n', or 'mixed'.")

        selected_candidates = tuple(
            np.atleast_1d(arr[chosen_indices].squeeze()) if arr is not None else None for arr in candidates
        )

        return selected_candidates, np.atleast_1d(scores[chosen_indices].squeeze())

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
        #     return potentials / distances  # TODO: Improve scoring function
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
    def monte_carlo_select_candidate(cls, candidates, use_distance=True):
        """Depreciated. Use select_candidates instead."""  # TODO: Remove
        if use_distance:
            scores = cls.evaluate_candidates(candidates[-2], candidates[-1])
        else:
            scores = cls.evaluate_candidates(candidates[-2])

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


class CircleIntersection:
    def __init__(self, target_locations: TargetLocations):
        self.target_locations = target_locations

    def find_circle_intersection_candidates(self, start_coord: np.ndarray, end_coord: np.ndarray, type: str,
                                            distance_start_to_act: float, distance_act_to_end: float,
                                            num_candidates: int):
        """
        Find n location candidates for a given activity type between two known locations.
        Returns two sets of candidates if two intersection points are found, otherwise only one set.
        """
        intersect1, intersect2 = self.find_circle_intersections(start_coord, distance_start_to_act, end_coord,
                                                                distance_act_to_end)

        candidates1 = self.target_locations.query_closest(type, intersect1, num_candidates)

        if intersect2 is not None:
            candidates2 = self.target_locations.query_closest(type, intersect2, num_candidates)
            combined_candidates = tuple(
                np.vstack((np.atleast_1d(arr1), np.atleast_1d(arr2))) for arr1, arr2 in zip(candidates1, candidates2))

        else:
            combined_candidates = candidates1

        candidate_identifiers, candidate_names, candidate_coordinates, candidate_potentials, candidate_distances = combined_candidates
        return candidate_identifiers, candidate_names, candidate_coordinates, candidate_potentials, candidate_distances

    def find_circle_intersections(self, center1: np.ndarray, radius1: float, center2: np.ndarray, radius2: float) -> \
            Tuple[
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
        d = h.euclidean_distance(center1, center2)

        logger.debug(f"Center 1: {center1}, Radius 1: {radius1}, Center 2: {center2}, Radius 2: {radius2}")

        # Handle non-intersection conditions:
        if d == 0:
            if abs(r1 - r2) < 1e-4:
                logger.debug("Infinite intersections: The start and end points and radii are identical.")
                logger.debug("Choosing a point on the perimeter of the circles.")
                intersect = np.array([x1 + r1, y1])
                return intersect, None
            else:
                logger.debug("No intersection: The circles are identical but have different radii.")
                logger.debug("Choosing a point on the perimeter of the circles.")
                intersect = np.array([x1 + r1, y1])
                return intersect, None

        if d > (r1 + r2):
            logger.debug("No direct intersection: The circles are too far apart.")
            logger.debug("Finding point on the line with distances proportional to radii as fallback.")

            proportional_distance = r1 / (r1 + r2)
            point_on_line = center1 + proportional_distance * (center2 - center1)

            return point_on_line, None

        if d < abs(r1 - r2):
            logger.debug("No direct intersection: One circle is contained within the other.")
            logger.debug("Returning closest point on the circumference of the inner circle.")

            if r1 > r2:
                closest_point = center2 + r2 * (center1 - center2) / d
                return closest_point, None
            else:
                closest_point = center1 + r1 * (center2 - center1) / d
                return closest_point, None

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
                                              num_candidates: int):
        """Place a single activity at one of the closest locations."""
        # TODO: Maybe depreciate returning potential, name and score
        # Home locations aren't among the targets and are for now replaced by the start location
        if act_type == s.ACT_HOME:
            logger.warning(
                "Home activity detected. Secondary locator shouldn't be used for that. Returning start location.")
            return None, "home", start_coord, None, None, None

        candidates = self.find_circle_intersection_candidates(start_coord, end_coord, act_type, distance_start_to_act,
                                                              distance_act_to_end, num_candidates)
        candidate_potentials = candidates[-2]
        candidate_coords = candidates[-3]
        candidate_distance_deviations = h.get_abs_distance_deviations(candidate_coords, start_coord,
                                                                      distance_start_to_act)
        candidate_distance_deviations += h.get_abs_distance_deviations(candidate_coords, end_coord, distance_act_to_end)

        scores = EvaluationFunction.evaluate_candidates(candidate_potentials, candidate_distance_deviations,
                                                        num_candidates)
        candidate, score = EvaluationFunction.select_candidates(candidates, scores, 1, 'top_n')

        identifier, name, coord, potential, distance = candidate

        return identifier, name, coord, potential, distance, score


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
    x = random.uniform(xmin, xmax)
    y = random.uniform(ymin, ymax)
    return np.array([x, y])


def prepare_population_df_for_location_assignment(df, filter_max_distance=None, number_of_persons=None) -> (
        pd.DataFrame, pd.DataFrame):
    """Temporarily prepare the MiD DataFrame for the leg dictionary function."""

    # Initialize columns with empty objects to ensure compatibility
    df["from_location"] = None
    df["to_location"] = None

    # Split persons with no leg ID into a separate DataFrame
    no_leg_df = df[df[s.LEG_ID_COL].isna()].copy()
    df = df.dropna(subset=[s.LEG_ID_COL])
    # TEMP: Remove persons that have no leg 1 (it has by accident been removed by enhancer)
    # TODO: Remove lines below again
    mobile_persons_with_leg_1 = df[df[s.UNIQUE_LEG_ID_COL].str.contains("_1.0")][s.UNIQUE_P_ID_COL].unique()
    df = df[df[s.UNIQUE_P_ID_COL].isin(mobile_persons_with_leg_1)]

    logger.debug(f"People with no legs: {no_leg_df.shape[0]}")

    # Throw out rows with missing values in the distance column
    row_count_before = df.shape[0]
    df = df.dropna(subset=[s.LEG_DISTANCE_METERS_COL])
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
    # TODO: write function that properly assigns homes (very similar to now, from popsim)
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
    logger.debug(df.head())
    return df, no_leg_df


def segment_plans(nested_dict: UnSegmentedPlans) -> SegmentedPlans:
    """
    Segment the plan of each person into separate trips where only the start and end locations are known.
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
#     logger.debug(candidates)

#     # At the next levels, get n candidates with score for each connected location
#     for level in range(1, len(tree)):
#         for i in range(0, len(tree[level]), 2):
#             candidates = find_location_candidates(segment[i]['from_location'], segment[i+1]['to_location'], segment[i]['to_act_type'], tree[level][i][2], tree[level][i+1][2], 5)
#             logger.debug(candidates)


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


def write_placement_results_dict_to_population_df(placement_results_dict, population_df):  # TODO: finish
    """Writes the placement results from the dictionary to the big DataFrame."""
    records = []
    for person_id, segments in placement_results_dict.items():
        for segment in segments:
            for leg in segment:
                records.append(leg)

    data_df = pd.DataFrame(records)

    # Check columns
    mandatory_columns = [s.UNIQUE_LEG_ID_COL, 'from_location', 'to_location', 'to_act_identifier']
    optional_columns = ['to_act_name', 'to_act_potential']

    for col in mandatory_columns:
        if col not in data_df.columns:
            raise ValueError(f"Mandatory column '{col}' is missing in data_df.")

    existing_optional_columns = [col for col in optional_columns if col in data_df.columns]
    existing_columns = mandatory_columns + existing_optional_columns

    # Perform the merge with the existing columns
    merged_df = population_df.merge(data_df[existing_columns], on=s.UNIQUE_LEG_ID_COL, how='left')

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
