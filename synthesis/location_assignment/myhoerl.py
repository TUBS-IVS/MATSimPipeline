import sklearn.neighbors
import numpy as np
import shapely.geometry as geo
import geopandas as gpd
import pandas as pd
import numpy.linalg as la
from typing import Dict, List, Any
from typing import List, Dict, Any

from collections import defaultdict
# COMPONENTS

# TODO: Give him my locations
# TODO: Give him my segments in the right format

# For retrieving locations quickly
class CandidateIndex:
    def __init__(self, data):
        self.data = data
        self.indices = {}

        for purpose, pdata in self.data.items():
            print("Constructing spatial index for %s ..." % purpose)
            self.indices[purpose] = sklearn.neighbors.KDTree(pdata["coordinates"])

    def query(self, purpose, location):
        index = self.indices[purpose].query(location.reshape(1, -1), return_distance=False)[0][0]
        identifier = self.data[purpose]["identifiers"][index]
        location = self.data[purpose]["coordinates"][index]
        return identifier, location

    def sample(self, purpose, random):
        index = random.randint(0, len(self.data[purpose]["locations"]))
        identifier = self.data[purpose]["identifiers"][index]
        location = self.data[purpose]["coordinates"][index]
        return identifier, location


# Simply gets the closest location
class CustomDiscretizationSolver:

    def __init__(self, index):
        self.index = index

    def solve(self, problem, locations):
        discretized_locations = []
        discretized_identifiers = []

        for location, purpose in zip(locations, problem["purposes"]):
            identifier, location = self.index.query(purpose, location.reshape(1,
                                                                              -1))

            discretized_identifiers.append(identifier)
            discretized_locations.append(location)

        assert len(discretized_locations) == problem["size"]

        return dict(
            valid=True, locations=np.vstack(discretized_locations), identifiers=discretized_identifiers
        )


def format_segmented_legs(segmented_dict: Dict[str, List[List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    mode_mapping = {
        1: "walk",
        2: "bike",
        3: "car_passenger",
        4: "car",
        5: "pt",
        9: "no_information"
    }
    formatted_problems = []

    for person_id, segments in segmented_dict.items():
        for trip_index, segment in enumerate(segments):
            purposes = [leg['to_act_purpose'] for leg in segment]
            purposes = purposes[
                       :-1]  # Removing the last purpose (in segments, the last to purpose heads to a fixed activity)

            modes = [mode_mapping.get(leg['mode'], "unknown") for leg in segment]
            distances = [leg['distance'] for leg in segment]  # Assuming 'distance' acts as a proxy for travel time

            # Finding the origins and destinations
            origin_location = segment[0]['from_location'] if np.size(segment[0]['from_location']) > 0 else None
            destination_location = segment[-1]['to_location'] if np.size(segment[-1]['to_location']) > 0 else None

            # Convert locations to numpy arrays if they exist
            if origin_location is not None:
                origin_location = np.array(origin_location).reshape(1, -1)
            if destination_location is not None:
                destination_location = np.array(destination_location).reshape(1, -1)

            problem = {
                'person_id': person_id,
                'trip_index': trip_index,
                'purposes': purposes,
                'modes': modes,
                'travel_times': distances,
                'size': len(purposes),
                'origin': origin_location,
                'destination': destination_location,
                'activity_index': trip_index + 1  # Starting index from 1
            }

            formatted_problems.append(problem)

    return formatted_problems



# ---------------------------------


# -----------------------------------------------------------------------------------------


# ---------------------------------


# LOCATIONS


# the ooooooverall run function
# def process(context, arguments):
def process(destinations, segmented_dict):
    """Destinations is the reformatted version of my locations. Segmented_dict is the reformatted version of my segments."""
    # Set up RNG
    # random = np.random.RandomState(context.config("random_seed"))
    random = np.random.RandomState()
    # maximum_iterations = context.config("secloc_maximum_iterations")
    maximum_iterations = 1000

    # Set up discretization solver
    candidate_index = CandidateIndex(destinations)  # TODO: Give him my locations
    discretization_solver = CustomDiscretizationSolver(candidate_index)

    # Set up distance sampler -- not needed for now
    # distance_distributions = context.data("distance_distributions")
    # distance_sampler = CustomDistanceSampler(
    #     maximum_iterations=min(1000, maximum_iterations),
    #     random=random,
    #     distributions=distance_distributions)

    # Set up relaxation solver; currently, we do not consider tail problems.
    chain_solver = GravityChainSolver(
        random=random, eps=10.0, lateral_deviation=10.0, alpha=0.1,
        maximum_iterations=min(1000, maximum_iterations)
    )

    # Only looking at chains for now
    # tail_solver = AngularTailSolver(random=random)
    # free_solver = CustomFreeChainSolver(random, candidate_index)

    relaxation_solver = GeneralRelaxationSolver(chain_solver)

    # Set up assignment solver
    thresholds = dict(
        car=200.0, car_passenger=200.0, pt=200.0,
        bike=100.0, walk=100.0
    )

    assignment_objective = DiscretizationErrorObjective(thresholds=thresholds)
    assignment_solver = AssignmentSolver(
        distance_sampler=None,
        relaxation_solver=relaxation_solver,
        discretization_solver=discretization_solver,
        objective=assignment_objective,
        maximum_iterations=min(20, maximum_iterations)
    )

    df_locations = []
    df_convergence = []

    last_person_id = None

    # for problem in find_assignment_problems(df_trips, df_primary):
    #     result = assignment_solver.solve(problem)

    for problem in format_segmented_legs(segmented_dict):
        result = assignment_solver.solve(problem)

        starting_activity_index = problem["activity_index"]

        for index, (identifier, location) in enumerate(
                zip(result["discretization"]["identifiers"], result["discretization"]["locations"])):
            df_locations.append((
                problem["person_id"], starting_activity_index + index, identifier, geo.Point(location)
            ))

        df_convergence.append((
            result["valid"], problem["size"]
        ))

        if problem["person_id"] != last_person_id:
            last_person_id = problem["person_id"]

    df_locations = pd.DataFrame.from_records(df_locations,
                                             columns=["person_id", "activity_index", "location_id", "geometry"])
    df_locations = gpd.GeoDataFrame(df_locations, crs="EPSG:2154")
    assert not df_locations["geometry"].isna().any()

    df_convergence = pd.DataFrame.from_records(df_convergence, columns=["valid", "size"])
    return df_locations, df_convergence


# ---------------------------------


# -----------------------------------------------------------------------------------------


# ---------------------------------


# PROBLEMS


FIELDS = ["person_id", "trip_index", "preceding_purpose", "following_purpose", "mode", "travel_time"]
FIXED_PURPOSES = ["home", "work", "education"]


# instead of this maybe just feed it with my "segments" but reformatted.

def find_bare_assignment_problems(df):
    problem = None

    for row in df[FIELDS].itertuples(index=False):
        person_id, trip_index, preceding_purpose, following_purpose, mode, travel_time = row

        if not problem is None and person_id != problem["person_id"]:
            # We switch person, but we're still tracking a problem. This is a tail!
            yield problem
            problem = None

        if problem is None:
            # Start a new problem
            problem = dict(
                person_id=person_id, trip_index=trip_index, purposes=[preceding_purpose],
                modes=[], travel_times=[]
            )

        problem["purposes"].append(following_purpose)
        problem["modes"].append(mode)
        problem["travel_times"].append(travel_time)

        if problem["purposes"][-1] in FIXED_PURPOSES:
            # The current chain (or initial tail) ends with a fixed activity.
            yield problem
            problem = None

    if not problem is None:
        yield problem


LOCATION_FIELDS = ["person_id", "home", "work", "education"]


def find_assignment_problems(df, df_locations):
    """
        Enriches assignment problems with:
          - Locations of the fixed activities
          - Size of the problem
          - Reduces purposes to the variable ones

          df_locations contains the locations of the fixed, primary activities
    """
    location_iterator = df_locations[LOCATION_FIELDS].itertuples(index=False)
    current_location = None

    for problem in find_bare_assignment_problems(df):
        origin_purpose = problem["purposes"][0]
        destination_purpose = problem["purposes"][-1]

        # Reduce purposes
        if origin_purpose in FIXED_PURPOSES and destination_purpose in FIXED_PURPOSES:
            problem["purposes"] = problem["purposes"][1:-1]

        elif origin_purpose in FIXED_PURPOSES:
            problem["purposes"] = problem["purposes"][1:]

        elif destination_purpose in FIXED_PURPOSES:
            problem["purposes"] = problem["purposes"][:-1]

        else:
            pass  # Neither chain nor tail

        # Define size
        problem["size"] = len(problem["purposes"])

        if problem["size"] == 0:
            continue  # We can skip if there are no variable activities

        # Advance location iterator until we arrive at the current problem's person
        while current_location is None or current_location[0] != problem["person_id"]:
            current_location = next(location_iterator)

        # Define origin and destination locations if they have fixed purposes
        problem["origin"] = None
        problem["destination"] = None

        if origin_purpose in FIXED_PURPOSES:
            problem["origin"] = current_location[LOCATION_FIELDS.index(origin_purpose)]  # Shapely POINT
            problem["origin"] = np.array([[problem["origin"].x, problem["origin"].y]])

        if destination_purpose in FIXED_PURPOSES:
            problem["destination"] = current_location[LOCATION_FIELDS.index(destination_purpose)]  # Shapely POINT
            problem["destination"] = np.array([[problem["destination"].x, problem["destination"].y]])

        if problem["origin"] is None:
            problem["activity_index"] = problem["trip_index"]
        else:
            problem["activity_index"] = problem["trip_index"] + 1

        yield problem


# ---------------------------------


# -----------------------------------------------------------------------------------------


# ---------------------------------


# RDA


def check_feasibility(distances, direct_distance, consider_total_distance=True):
    return calculate_feasibility(distances, direct_distance, consider_total_distance) == 0.0


def calculate_feasibility(distances, direct_distance, consider_total_distance=True):
    # Really elegant way to calculate the feasibility of any chain

    total_distance = np.sum(distances)
    delta_distance = 0.0

    # Remaining is the diff between each individual dist and the sum of all dists (so remaining is the sum of all distances except itself)
    remaining_distance = total_distance - distances
    # So this checks if we can get "back" to the end if one dist is very large and gets us far away
    # If delta is larger than one, we can't get back to the end
    delta = max(distances - direct_distance - remaining_distance)

    # Delta gets positive if the real dist is larger than the sum of all distances
    if consider_total_distance:
        delta = max(delta, direct_distance - total_distance)

    return float(max(delta, 0))


class DiscretizationSolver:
    def solve(self, problem, locations):
        raise NotImplementedError()


class RelaxationSolver:
    def solve(self, problem, distances):
        raise NotImplementedError()


class DistanceSampler:
    def sample(self, problem):
        raise NotImplementedError()


class FeasibleDistanceSampler(DistanceSampler):
    def __init__(self, random, maximum_iterations=1000):
        self.maximum_iterations = maximum_iterations
        self.random = random

    def sample_distances(self, problem):
        # Return distance chains per row
        raise NotImplementedError()

    def sample(self, problem):
        # Extract necessary information from the problem
        origin = problem["origin"]
        destination = problem["destination"]
        distances = problem["travel_times"]

        # Ensure distances are in the correct format, directly returning them
        if problem["size"] == 1 and np.linalg.norm(destination - origin, axis=1) < 1e-3:
            distances = np.array([distances[0], distances[0]])

        return {
            "valid": True,
            "distances": distances,
            "iterations": None  # No iterations needed as distances are known and valid
        }

#
# class CustomDistanceSampler(FeasibleDistanceSampler):
#     def __init__(self, random, distributions, maximum_iterations=1000):
#         FeasibleDistanceSampler.__init__(self, random=random, maximum_iterations=maximum_iterations)
#
#         self.random = random
#         self.distributions = distributions
#
#     def sample_distances(self, problem):
#         distances = np.zeros((len(problem["modes"])))
#
#         for index, (mode, travel_time) in enumerate(zip(problem["modes"], problem["travel_times"])):
#             mode_distribution = self.distributions[mode]
#
#             bound_index = np.count_nonzero(travel_time > mode_distribution["bounds"])
#             mode_distribution = mode_distribution["distributions"][bound_index]
#
#             distances[index] = mode_distribution["values"][
#                 np.count_nonzero(self.random.random_sample() > mode_distribution["cdf"])
#             ]
#
#         return distances


class AssignmentObjective:
    def evaluate(self, problem, distance_result, relaxation_result, discretization_result):
        raise NotImplementedError()


# The overall solver
class AssignmentSolver:
    def __init__(self, distance_sampler, relaxation_solver, discretization_solver, objective, maximum_iterations=1000):
        self.maximum_iterations = maximum_iterations

        self.relaxation_solver = relaxation_solver
        self.distance_sampler = distance_sampler
        self.discretization_solver = discretization_solver
        self.objective = objective

        if distance_sampler is None:
            self.distance_sampler = FeasibleDistanceSampler(random=None)
    def solve(self, problem):
        best_result = None

        for assignment_iteration in range(self.maximum_iterations):
            distance_result = self.distance_sampler.sample(problem)  # dict mit "distances", "valid", "iterations"

            relaxation_result = self.relaxation_solver.solve(problem, distance_result["distances"])
            discretization_result = self.discretization_solver.solve(problem, relaxation_result["locations"])

            assignment_result = self.objective.evaluate(problem, distance_result, relaxation_result,
                                                        discretization_result)

            if best_result is None or assignment_result["objective"] < best_result["objective"]:
                best_result = assignment_result

                assignment_result["distance"] = distance_result
                assignment_result["relaxation"] = relaxation_result
                assignment_result["discretization"] = discretization_result
                assignment_result["iterations"] = assignment_iteration

            if best_result["valid"]:
                break

        return best_result


class GeneralRelaxationSolver(RelaxationSolver):
    def __init__(self, chain_solver, tail_solver=None, free_solver=None):
        self.chain_solver = chain_solver
        self.tail_solver = tail_solver
        self.free_solver = free_solver

    def solve(self, problem, distances):
        # Only looking at chains for now

        # if problem["origin"] is None and problem["destination"] is None:
        #     return self.free_solver.solve(problem, distances)
        #
        # elif problem["origin"] is None or problem["destination"] is None:
        #     return self.tail_solver.solve(problem, distances)
        #
        # else:
        return self.chain_solver.solve(problem, distances)


# The actual relaxation solver.
class GravityChainSolver:
    def __init__(self, random, alpha=0.3, eps=1.0, maximum_iterations=1000, lateral_deviation=None):
        self.alpha = 0.3
        self.eps = 1e-2
        self.maximum_iterations = maximum_iterations
        self.random = random
        self.lateral_deviation = lateral_deviation

    # Very similar to my two-leg solver (just better written).
    # When the two circles overlap, it chooses one intersection at random.
    def solve_two_points(self, problem, origin, destination, distances, direction, direct_distance):
        if direct_distance == 0.0:
            location = origin + direction * distances[0]

            return dict(
                valid=distances[0] == distances[1],
                locations=location.reshape(-1, 2), iterations=None
            )

        elif direct_distance > np.sum(distances):
            ratio = 1.0

            if distances[0] > 0.0 or distances[1] > 0.0:
                ratio = distances[0] / np.sum(distances)

            location = origin + direction * ratio * direct_distance

            return dict(
                valid=False, locations=location.reshape(-1, 2), iterations=None
            )

        elif direct_distance < np.abs(distances[0] - distances[1]):
            ratio = 1.0

            if distances[0] > 0.0 or distances[1] > 0.0:
                ratio = distances[0] / np.sum(distances)

            maximum_distance = max(distances)
            location = origin + direction * ratio * maximum_distance

            return dict(
                valid=False, locations=location.reshape(-1, 2), iterations=None
            )

        else:
            A = 0.5 * (distances[0] ** 2 - distances[1] ** 2 + direct_distance ** 2) / direct_distance
            H = np.sqrt(max(0, distances[0] ** 2 - A ** 2))
            r = self.random.random_sample()

            center = origin + direction * A
            offset = direction * H
            offset = np.array([offset[0, 1], -offset[0, 0]])

            location = center + (1.0 if r < 0.5 else -1.0) * offset

            return dict(
                valid=True, locations=location.reshape(-1, 2), iterations=None
            )

    # This is the actual relaxation algorithm.
    def solve(self, problem, distances):
        origin, destination = problem["origin"], problem["destination"]

        if origin is None or destination is None:
            raise RuntimeError("Invalid chain for GravityChainSolver")

        # Prepare direction and normal direction
        direct_distance = la.norm(destination - origin)

        if direct_distance < 1e-12:  # We have a zero direct distance, choose a direction randomly
            angle = self.random.random() * np.pi * 2.0

            direction = np.array([
                np.cos(angle), np.sin(angle)
            ]).reshape((1, 2))

        else:
            direction = (destination - origin) / direct_distance

        normal = np.array([direction[0, 1], -direction[0, 0]])

        # If we have only one variable point, take a short cut
        if problem["size"] == 1:
            return self.solve_two_points(problem, origin, destination, distances, direction, direct_distance)

        # Prepare initial locations
        if np.sum(distances) < 1e-12:
            shares = np.linspace(0, 1, len(distances) - 1)
        else:
            shares = np.cumsum(distances[:-1]) / np.sum(distances)

        locations = origin + direction * shares[:, np.newaxis] * direct_distance
        locations = np.vstack([origin, locations, destination])

        if not check_feasibility(distances, direct_distance):
            return dict(  # We still return some locations, although they may not be perfect
                valid=False, locations=locations[1:-1], iterations=None
            )

        # Add lateral devations
        lateral_deviation = self.lateral_deviation if not self.lateral_deviation is None else max(direct_distance, 1.0)
        locations[1:-1] += normal * 2.0 * (
                self.random.normal(size=len(distances) - 1)[:, np.newaxis] - 0.5) * lateral_deviation

        # Prepare gravity simulation
        valid = False

        origin_weights = np.ones((len(distances) - 1, 2))
        origin_weights[0, :] = 2.0

        destination_weights = np.ones((len(distances) - 1, 2))
        destination_weights[-1, :] = 2.0

        # Run gravity simulation
        for k in range(self.maximum_iterations):
            directions = locations[:-1] - locations[1:]
            lengths = la.norm(directions, axis=1)

            offset = distances - lengths
            lengths[lengths < 1.0] = 1.0
            directions /= lengths[:, np.newaxis]

            if np.all(np.abs(offset) < self.eps):  # Check if we have converged
                valid = True
                break

            # Apply adjustment to locations
            adjustment = np.zeros((len(distances) - 1, 2))
            adjustment -= 0.5 * self.alpha * offset[:-1, np.newaxis] * directions[:-1] * origin_weights
            adjustment += 0.5 * self.alpha * offset[1:, np.newaxis] * directions[1:] * destination_weights

            locations[1:-1] += adjustment

            if np.isnan(locations).any() or np.isinf(locations).any():
                raise RuntimeError("NaN/Inf value encountered during gravity simulation")

        return dict(
            valid=valid, locations=locations[1:-1], iterations=k
        )


# Returns valid if the discretization error is below the threshold
class DiscretizationErrorObjective(AssignmentObjective):
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def evaluate(self, problem, distance_result, relaxation_result, discretization_result):
        sampled_distances = distance_result["distances"]

        discretized_locations = []
        if not problem["origin"] is None: discretized_locations.append(problem["origin"])
        discretized_locations.append(discretization_result["locations"])
        if not problem["destination"] is None: discretized_locations.append(problem["destination"])
        discretized_locations = np.vstack(discretized_locations)

        discretized_distances = la.norm(discretized_locations[:-1] - discretized_locations[1:], axis=1)
        discretization_error = np.abs(sampled_distances - discretized_distances)

        objective = 0.0 # TODO: In die segments die modes reinbringen für hörl problem, aber bei mir wieder entfernen (?)
        for error, mode in zip(discretization_error, problem["modes"]):
            target_error = self.thresholds[mode]
            excess_error = max(0.0, error - target_error)
            objective = max(objective, excess_error)

        valid = objective == 0.0
        valid &= distance_result["valid"]
        valid &= relaxation_result["valid"]
        valid &= discretization_result["valid"]

        return dict(valid=valid, objective=objective)
