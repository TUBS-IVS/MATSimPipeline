import pandas as pd
import os
import time
from typing import Literal
from utils import settings as s
from utils import helpers as h
from utils import pipeline_setup
from utils.logger import logging
from synthesis.location_assignment import activity_locator_distance_based as al
from synthesis.location_assignment import myhoerl
from utils.stats_tracker import stats_tracker

logger = logging.getLogger(__name__)


def run_location_assignment():
    """Runs the location assignment algorithm(s) on the given population and locations CSV files."""
    population_df = h.read_csv(h.get_files(r"data/mid/enhanced"))
    locations_json_path = r"playground/reformatted_data2.json"
    algorithms_to_run = ['load_main','advanced_petre']  # prepend "load_" to load intermediate results
    save_intermediate_results = True
    assert_no_missing_locations = False

    # Early check if all algorithms are valid
    valid_algorithms = ['hoerl', 'simple_lelke', 'greedy_petre', 'main', 'advanced_petre']
    algos_to_check = [
        algorithm[len("load_"):] if algorithm.startswith("load_") else algorithm
        for algorithm in algorithms_to_run
    ]
    if not all(algorithm in valid_algorithms for algorithm in algos_to_check):
        raise ValueError(f"Invalid algorithm. Valid algorithms are: {valid_algorithms}")

    # Build the common KDTree for the locations
    target_locations = al.TargetLocations(locations_json_path)

    # Prepare the population dataframe, split off non-mobile persons
    mobile_population_df, non_mobile_population_df = (al.prepare_population_df_for_location_assignment
                                                      (population_df,
                                                       number_of_persons=1000,
                                                       filter_max_distance=20000))

    for algorithm in algorithms_to_run:
        if algorithm.startswith("load"):
            mobile_population_df = load_intermediate(algorithm)
            non_mobile_population_df = pd.DataFrame()
        elif algorithm == 'hoerl':
            mobile_population_df = run_hoerl(
                mobile_population_df, target_locations)
        elif algorithm == 'simple_lelke':
            mobile_population_df = run_simple_lelke(
                mobile_population_df, target_locations)
        elif algorithm == 'greedy_petre':
            mobile_population_df = run_greedy_petre(
                mobile_population_df, target_locations)
        elif algorithm == 'main':
            mobile_population_df = run_main(
                mobile_population_df, target_locations)
        elif algorithm == 'advanced_petre':
            number_of_branches = 100
            max_candidates = None
            anchor_strategy = "lower_middle"
            min_candidates = 100

            stats_tracker.log("number_of_branches", number_of_branches)
            stats_tracker.log("max_candidates", max_candidates)
            stats_tracker.log("anchor_strategy", anchor_strategy)

            mobile_population_df = run_advanced_petre(
                mobile_population_df, target_locations, number_of_branches=number_of_branches,
                max_candidates=max_candidates, anchor_strategy=anchor_strategy, min_candidates=min_candidates)
        else:
            raise ValueError("Invalid algorithm.")

        # Make sure algorithm results are in the correct format
        mobile_population_df['to_location'] = mobile_population_df['to_location'].apply(
            lambda x: h.convert_to_point(x, target='array'))
        mobile_population_df['from_location'] = mobile_population_df['from_location'].apply(
            lambda x: h.convert_to_point(x, target='array'))
        if save_intermediate_results:
            mobile_population_df.to_csv(os.path.join(pipeline_setup.OUTPUT_DIR, f"mobile_population_{algorithm}.csv"),
                                        index=False)

    if assert_no_missing_locations:
        assert mobile_population_df['to_location'].notna().all(), "Some persons have no location assigned."

    # Recombine the population dataframes
    result_df = pd.concat([mobile_population_df, non_mobile_population_df], ignore_index=True)
    result_df.sort_values(by=[s.UNIQUE_HH_ID_COL, s.UNIQUE_P_ID_COL, s.UNIQUE_LEG_ID_COL], ascending=[True, True, True],
                          inplace=True)
    algos_string = "_".join(algorithms_to_run)
    if "advanced_petre" in algorithms_to_run:
        num_branches_string = f"_{number_of_branches}-branches"
    else:
        num_branches_string = ""
    result_df.to_csv(os.path.join(pipeline_setup.OUTPUT_DIR, f"location_assignment_result_{algos_string}"
                                                             f"{num_branches_string}.csv"),
                     index=False)
    logger.info(f"Wrote location assignment result to {pipeline_setup.OUTPUT_DIR}.")
    stats_tracker.write_stats_to_file(os.path.join(pipeline_setup.OUTPUT_DIR, "location_assignment_stats.txt"))

def load_intermediate(algorithm: str):
    intermediate_to_load = algorithm[len("load_"):]
    mobile_population_df = h.read_csv(f"data\\intermediates\\mobile_population_{intermediate_to_load}.csv")
    if "to_location" in mobile_population_df.columns:
        mobile_population_df["to_location"] = mobile_population_df["to_location"].apply(
            lambda x: h.convert_to_point(x, target='array'))
    if "from_location" in mobile_population_df.columns:
        mobile_population_df["from_location"] = mobile_population_df["from_location"].apply(
            lambda x: h.convert_to_point(x, target='array'))
    return mobile_population_df


def run_hoerl(population_df, target_locations):
    """Runs the Hoerl algorithm on the given population and locations CSV files."""
    logger.info("Starting Hoerl algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    segmented_dict = al.segment_plans(legs_dict)
    logger.info("Dict segmented.")
    df_location, df_convergence = myhoerl.process(target_locations, segmented_dict)
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


def run_main(population_df, target_locations):
    """Runs the Main algorithm on the given population and locations CSV files."""
    logger.info("Starting Main algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    simple_main_algorithm = al.SimpleMainLocationAlgorithm(target_locations, legs_dict)  # It wants unsegmented legs
    result_dict = simple_main_algorithm.run()
    result_dict = al.segment_plans(result_dict)  # Needed as writer expects segmented legs
    population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
    return h.add_from_location(population_df, 'to_location', 'from_location')


def run_advanced_petre(population_df, target_locations, number_of_branches: int = 10, max_candidates=None,
                       anchor_strategy: Literal["lower_middle", "upper_middle", "start", "end"] = "lower_middle", min_candidates=None):
    """Runs the Advanced Petre algorithm on the given population and locations CSV files."""
    logger.info("Starting Advanced Petre algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    segmented_dict = al.segment_plans(legs_dict)
    logger.info("Dict segmented.")
    advance_petre_algorithm = al.AdvancedPetreAlgorithm(target_locations, segmented_dict,
                                                        number_of_branches=number_of_branches,
                                                        max_candidates=max_candidates,
                                                        anchor_strategy=anchor_strategy,
                                                        min_candidates=min_candidates)
    result_dict = advance_petre_algorithm.run()
    population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
    return h.add_from_location(population_df, 'to_location', 'from_location')


if __name__ == "__main__":
    run_location_assignment()
