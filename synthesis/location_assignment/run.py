import sys
import logging
import time

from utils.config import Config
from utils.logger import setup_logging
from utils.stats_tracker import StatsTracker

import pandas as pd
import os

from utils import column_names as s
from utils.helpers import Helpers
from synthesis.location_assignment import activity_locator_distance_based as al
from synthesis.location_assignment import hoerl

# From the minimal united locations datafile (which contains centre points, polygons, MiD hh ids, ALKIS oi, and allowed activities)
# get buildings/locations where households exist
# associate any needed data with the households from enhanced mid, including trip info
# get all buildings with just their centre point and the allowed activities

def run_location_assignment():

    locations_json_folder = config.get("location_assignment.input.locations_json")
    algorithms_to_run = config.get("location_assignment.algorithms_to_run")

    save_intermediate_results = config.get("location_assignment.save_intermediate_results")
    assert_no_missing_locations = config.get("location_assignment.assert_no_missing_locations")
    filter_max_distance = config.get("location_assignment.filter_max_distance")
    filter_number_of_persons = config.get("location_assignment.filter_number_of_persons")
    filter_by_person = config.get("location_assignment.filter.filter_by_person")
    skip_loading_full_population = config.get("location_assignment.skip_loading_full_population")
    write_to_csv = config.get("location_assignment.write_to_csv")

    # Early check if all algorithms are valid
    valid_algorithms = ['load_intermediate', 'filter', 'remove_unfeasible', 'hoerl', 'simple_lelke', 'greedy_petre',
                        'simple_main', 'CARLA', 'open_ended', 'nothing']

    if not all(algorithm in valid_algorithms for algorithm in algorithms_to_run):
        raise ValueError(f"Invalid algorithm. Valid algorithms are: {valid_algorithms}")

    # Build the common KDTree for the locations
    target_locations = al.TargetLocations(h.get_files(locations_json_folder))

    if not skip_loading_full_population:
        # Load the population dataframe
        population_df = pd.read_csv(h.get_files(config.get("location_assignment.population_df_folder")))

        # Prepare the population dataframe, split off non-mobile persons
        mobile_population_df, non_mobile_population_df = (al.prepare_population_df_for_location_assignment
                                                          (population_df,
                                                           number_of_persons=filter_number_of_persons,
                                                           filter_max_distance=filter_max_distance))
        mobile_population_df[s.LEG_DISTANCE_METERS_COL] = mobile_population_df[s.LEG_DISTANCE_METERS_COL] / \
                                                          config.get("location_assignment.detour_factor")

    for algorithm in algorithms_to_run:
        if algorithm == "load_intermediate":
            mobile_population_df = load_intermediate()
            non_mobile_population_df = pd.DataFrame()
        elif algorithm == 'nothing':
            logger.info("Doing nothing.")
        elif algorithm == 'filter':
            mobile_population_df = mobile_population_df[mobile_population_df[s.UNIQUE_P_ID_COL] == filter_by_person]
        elif algorithm == 'remove_unfeasible':
            mobile_population_df = remove_unfeasible_persons(mobile_population_df)
        elif algorithm == 'hoerl':
            mobile_population_df = run_hoerl(
                mobile_population_df, target_locations,
                config)
        elif algorithm == 'simple_lelke':
            mobile_population_df = run_simple_lelke(
                mobile_population_df, target_locations)
        elif algorithm == 'greedy_petre':
            mobile_population_df = run_greedy_petre(
                mobile_population_df, target_locations)
        elif algorithm == 'main':
            mobile_population_df = run_simple_main(
                mobile_population_df, target_locations,
                config)  # TODO: config object will not work -> adjust inner code
        elif algorithm == 'open_ended':
            mobile_population_df = run_open_ended(
                mobile_population_df, target_locations,
                config)  # TODO: config object will not work -> adjust inner code
        elif algorithm == 'CARLA':
            mobile_population_df = run_carla(
                mobile_population_df, target_locations,
                config)
        else:
            raise ValueError("Invalid algorithm.")

        # Make sure algorithm results are in the correct format
        mobile_population_df['to_location'] = mobile_population_df['to_location'].apply(
            lambda x: h.convert_to_point(x, target='array'))
        mobile_population_df['from_location'] = mobile_population_df['from_location'].apply(
            lambda x: h.convert_to_point(x, target='array'))
        if save_intermediate_results:
            mobile_population_df.to_csv(os.path.join(output_folder, f"mobile_population_{algorithm}.csv"),
                                        index=False)

    if assert_no_missing_locations:
        assert mobile_population_df['to_location'].notna().all(), "Some persons have no location assigned."

    # Recombine the population dataframes
    result_df = pd.concat([mobile_population_df, non_mobile_population_df], ignore_index=True)
    result_df.sort_values(by=[s.UNIQUE_HH_ID_COL, s.UNIQUE_P_ID_COL, s.UNIQUE_LEG_ID_COL], ascending=[True, True, True],
                          inplace=True)

    # Write the result to a CSV file
    if write_to_csv:
        algos_string = "_".join(algorithms_to_run)
        if "CARLA" in algorithms_to_run:
            num_branches_string = f"_{config.get('location_assignment.CARLA.number_of_branches')}-branches"
            min_candidates_complex_string = f"_{config.get('location_assignment.CARLA.min_candidates_complex_case')}-min-cand-complex"
            candidates_two_leg_string = f"_{config.get('location_assignment.CARLA.candidates_two_leg_case')}-cand-two-leg"
        else:
            num_branches_string = ""
            candidates_two_leg_string = ""
            min_candidates_complex_string = ""
        result_df.to_csv(os.path.join(output_folder, f"location_assignment_result_{algos_string}"
                                                     f"{num_branches_string}"
                                                     f"{candidates_two_leg_string}"
                                                     f"{min_candidates_complex_string}.csv"),
                         index=False)
        logger.info(f"Wrote location assignment result to {output_folder}.")

    return result_df


def load_intermediate():
    mobile_population_df = pd.read_csv(h.get_files(r"data/intermediates"))
    if "to_location" in mobile_population_df.columns:
        mobile_population_df["to_location"] = mobile_population_df["to_location"].apply(
            lambda x: h.convert_to_point(x, target='array'))
    if "from_location" in mobile_population_df.columns:
        mobile_population_df["from_location"] = mobile_population_df["from_location"].apply(
            lambda x: h.convert_to_point(x, target='array'))
    return mobile_population_df


def remove_unfeasible_persons(population_df):
    logger.info("Removing unfeasible persons.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    segmented_dict = al.segment_plans(legs_dict)
    logger.info("Dict segmented.")
    feasible_dict = h.filter_feasible_data(segmented_dict)
    population_df = al.write_placement_results_dict_to_population_df(feasible_dict, population_df, merge_how='right')
    return population_df


def run_hoerl(population_df, target_locations, config):
    """Runs the Hoerl algorithm on the given population and locations CSV files."""
    logger.info("Starting Hoerl algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    segmented_dict = al.segment_plans(legs_dict)
    logger.info("Dict segmented, starting hoerl")
    time_start = time.time()
    df_location, df_convergence = hoerl.process(target_locations, segmented_dict, config)
    algo_time = time.time() - time_start
    logger.info(f"Hoerl done in {algo_time} seconds.")
    stats_tracker.log("runtimes.hoerl_time", algo_time)
    population_df['to_location'] = population_df['to_location'].apply(
        lambda x: h.convert_to_point(x, target='array'))  # Needed currently so [] becomes None
    population_df['from_location'] = population_df['from_location'].apply(
        lambda x: h.convert_to_point(x, target='array'))  # Needed currently so [] becomes None
    population_df = al.write_hoerl_df_to_big_df(df_location, population_df)
    population_df = h.add_from_location(population_df, 'to_location', 'from_location')
    return population_df


def run_greedy_petre(population_df, target_locations):
    """Runs the Greedy Petre algorithm on the given population and locations CSV files."""
    logger.info("Starting Greedy Petre algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    segmented_dict = al.segment_plans(legs_dict)
    logger.info("Dict segmented.")
    greedy_petre_algorithm = al.WeirdPetreAlgorithm(target_locations, segmented_dict, variant="greedy")
    result_dict = greedy_petre_algorithm.run()
    population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
    return h.add_from_location(population_df, 'to_location', 'from_location')


def run_simple_lelke(population_df, target_locations):
    """Runs the Simple Lelke algorithm on the given population and locations CSV files."""
    logger.info("Starting Simple Lelke algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    segmented_dict = al.segment_plans(legs_dict)
    logger.info("Dict segmented.")
    lelke_algorithm = al.SimpleLelkeAlgorithm(target_locations, segmented_dict)
    result_dict = lelke_algorithm.run()
    population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
    return h.add_from_location(population_df, 'to_location', 'from_location')


def run_simple_main(population_df, target_locations, config):
    """Runs the Main algorithm on the given population and locations CSV files."""
    logger.info("Starting Main algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    simple_main_algorithm = al.SimpleMainLocationAlgorithm(target_locations, legs_dict,
                                                           config)  # It wants unsegmented legs
    result_dict = simple_main_algorithm.run()
    result_dict = al.segment_plans(result_dict)  # Needed as writer expects segmented legs
    population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
    return h.add_from_location(population_df, 'to_location', 'from_location')


def run_open_ended(population_df, target_locations, config):
    logger.info("Starting open-ended algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    open_ended_algorithm = al.OpenEndedAlgorithm(target_locations, legs_dict, config)
    result_dict = open_ended_algorithm.run()
    result_dict = al.segment_plans(result_dict)  # Needed as writer expects segmented legs
    population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
    return h.add_from_location(population_df, 'to_location', 'from_location')


def run_carla(population_df, target_locations, config):
    """Runs the CARLA algorithm on the given population and locations CSV files."""
    logger.info("Starting CARLA algorithm.")
    legs_dict = al.convert_to_segmented_plans(population_df)
    logger.info("Dict populated.")
    segmented_dict = al.new_segment_plans(legs_dict)
    logger.info("Dict segmented.")
    time_start = time.time()
    advance_petre_algorithm = al.CARLA(target_locations, segmented_dict, config)
    result_dict = advance_petre_algorithm.run()
    algo_time = time.time() - time_start
    logger.info(f"CARLA done in {algo_time} seconds.")
    stats_tracker.log("runtimes.carla_time", algo_time)
    population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
    return h.add_from_location(population_df, 'to_location', 'from_location')


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: run.py <output_folder> <project_root> <config_yaml>")
        print("Absolute paths, folders must exist.")
        sys.exit(1)

    output_folder = sys.argv[1]
    project_root = sys.argv[2]
    config_yaml = sys.argv[3]
    step_name = "location_assignment"

    # Each step sets up its own logging, Config object and StatsTracker
    config = Config(output_folder, project_root, config_yaml)
    config.resolve_paths()

    setup_logging(output_folder, console_level=config.get("settings.logging.console_level"),
                  file_level=config.get("settings.logging.file_level"))
    logger = logging.getLogger(step_name)

    stats_tracker = StatsTracker(output_folder)

    h = Helpers(project_root, output_folder, config, stats_tracker, logger)

    logger.info(f"Starting step {step_name}")
    time_start = time.time()
    run_location_assignment()
    time_end = time.time()
    time_step = time_end - time_start
    stats_tracker.log("runtimes.location_assignment_time", time_step)
    stats_tracker.write_stats()
    config.write_used_config()

    logger.info(f"Step {step_name} finished in {time_step:.2f} seconds.")
