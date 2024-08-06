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

logger = logging.getLogger(__name__)


def run_location_assignment():
    """Runs the location assignment algorithm(s) on the given population and locations CSV files."""
    population_df = h.read_csv(h.get_files(r"C:\Users\petre\Documents\GitHub\MATSimPipeline\data\mid\enhanced"))
    locations_json_path = r"C:\Users\petre\Documents\GitHub\MATSimPipeline\playground\reformatted_data2.json"
    algorithms_to_run = ['advanced_petre']
    save_intermediate_results = True

    valid_algorithms = ['hoerl', 'simple_lelke', 'greedy_petre', 'main', 'advanced_petre']

    # Check if all algorithms are valid
    if not all(algorithm in valid_algorithms for algorithm in algorithms_to_run):
        raise ValueError(f"Invalid algorithm. Valid algorithms are: {valid_algorithms}")

    # Build the common KDTree for the locations
    target_locations = al.TargetLocations(locations_json_path)

    # Prepare the population dataframe, split off non-mobile persons
    mobile_population_df, non_mobile_population_df = al.prepare_population_df_for_location_assignment(population_df,
                                                                                                      number_of_persons=100,
                                                                                                      filter_max_distance=30000)

    for algorithm in algorithms_to_run:
        if algorithm == 'hoerl':
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
            mobile_population_df = run_advanced_petre(
                mobile_population_df, target_locations, number_of_branches=100, max_candidates=None,
                anchor_strategy="start")
            # 'lower_middle', 'upper_middle', 'start', 'end'
        else:
            raise ValueError("Invalid algorithm.")

        if save_intermediate_results:
            mobile_population_df.to_csv(os.path.join(pipeline_setup.OUTPUT_DIR, f"mobile_population_{algorithm}.csv"),
                                        index=False)

    # Recombine the population dataframes
    result_df = pd.concat([mobile_population_df, non_mobile_population_df], ignore_index=True)
    result_df.sort_values(by=[s.UNIQUE_HH_ID_COL, s.UNIQUE_P_ID_COL, s.UNIQUE_LEG_ID_COL], ascending=[True, True, True],
                          inplace=True)
    result_df.to_csv(os.path.join(pipeline_setup.OUTPUT_DIR, "location_assignment_result.csv"), index=False)


def run_hoerl(population_df, target_locations):
    """Runs the Hoerl algorithm on the given population and locations CSV files."""
    logger.info("Starting Hoerl algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    segmented_dict = al.segment_legs(legs_dict)
    logger.info("Dict segmented.")
    df_location, df_convergence = myhoerl.process(target_locations, segmented_dict)
    population_df['to_location'] = population_df['to_location'].apply(
        h.convert_to_shapely_point)  # Needed currently so [] becomes None
    population_df['from_location'] = population_df['from_location'].apply(
        h.convert_to_shapely_point)  # Needed currently so [] becomes None
    population_df = al.write_hoerl_df_to_big_df(df_location, population_df)
    population_df = h.add_from_location(population_df, 'to_location', 'from_location')
    return population_df


def run_greedy_petre(population_df, target_locations):
    """Runs the Greedy Petre algorithm on the given population and locations CSV files."""
    logger.info("Starting Greedy Petre algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    segmented_dict = al.segment_legs(legs_dict)
    logger.info("Dict segmented.")
    greedy_petre_algorithm = al.WeirdPetreAlgorithm(target_locations, segmented_dict, variant="greedy")
    result_dict = greedy_petre_algorithm.run()
    population_df = al.write_placement_results_dict_to_big_df(result_dict, population_df)
    return h.add_from_location(population_df, 'to_location', 'from_location')


def run_simple_lelke(population_df, target_locations):
    """Runs the Simple Lelke algorithm on the given population and locations CSV files."""
    logger.info("Starting Simple Lelke algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    segmented_dict = al.segment_legs(legs_dict)
    logger.info("Dict segmented.")
    lelke_algorithm = al.SimpleLelkeAlgorithm(target_locations, segmented_dict)
    result_dict = lelke_algorithm.run()
    population_df = al.write_placement_results_dict_to_big_df(result_dict, population_df)
    return h.add_from_location(population_df, 'to_location', 'from_location')


def run_main(population_df, target_locations):
    """Runs the Main algorithm on the given population and locations CSV files."""
    logger.info("Starting Main algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    segmented_dict = al.segment_legs(legs_dict)
    logger.info("Dict segmented.")
    simple_main_algorithm = al.SimpleMainLocationAlgorithm(target_locations, segmented_dict)
    result_dict = simple_main_algorithm.run()
    population_df = al.write_placement_results_dict_to_big_df(result_dict, population_df)
    return h.add_from_location(population_df, 'to_location', 'from_location')


def run_advanced_petre(population_df, target_locations, number_of_branches: int = 10, max_candidates=None,
                       anchor_strategy: Literal["lower_middle", "upper_middle", "start", "end"] = "start"):
    """Runs the Advanced Petre algorithm on the given population and locations CSV files."""
    logger.info("Starting Advanced Petre algorithm.")
    legs_dict = al.populate_legs_dict_from_df(population_df)
    logger.info("Dict populated.")
    segmented_dict = al.segment_legs(legs_dict)
    logger.info("Dict segmented.")
    advance_petre_algorithm = al.AdvancedPetreAlgorithm(target_locations, segmented_dict,
                                                        number_of_branches=number_of_branches,
                                                        max_candidates=max_candidates,
                                                        anchor_strategy=anchor_strategy)
    result_dict = advance_petre_algorithm.run()
    population_df = al.write_placement_results_dict_to_big_df(result_dict, population_df)
    return h.add_from_location(population_df, 'to_location', 'from_location')


if __name__ == "__main__":
    run_location_assignment()
