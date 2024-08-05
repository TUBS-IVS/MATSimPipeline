import pandas as pd
import os
from typing import Literal
from utils import settings as s
from utils import helpers as h
from utils import pipeline_setup
from synthesis.location_assignment import activity_locator_distance_based as al

def run_location_assignment():
    """Runs the location assignment algorithm(s) on the given population and locations CSV files."""
    population_df = h.read_csv(s.SYNTHETIC_POPULATION_CSV_PATH)
    locations_json_path = r"C:\Users\petre\Documents\GitHub\MATSimPipeline\playground\reformatted_data.json"
    algorithms_to_run = ['main', 'advanced_petre']
    save_intermediate_results = True

    valid_algorithms = ['hoerl', 'simple_lelke', 'greedy_petre', 'main', 'advanced_petre']

    # Check if all algorithms are valid
    if not all(algorithm in valid_algorithms for algorithm in algorithms_to_run):
        raise ValueError(f"Invalid algorithm. Valid algorithms are: {valid_algorithms}")

    # Build the common KDTree for the locations
    my_target_locations = al.TargetLocations(locations_json_path)
    
    # Prepare the population dataframe, split off non-mobile persons
    mobile_population_df, non_mobile_population_df = al.prepare_population_df_for_location_assignment(population_df, number_of_persons=100, filter_max_distance=30000)

    for algorithm in algorithms_to_run:
        if algorithm == 'hoerl':
            mobile_population_df = run_hoerl(
                mobile_population_df, my_target_locations)
        elif algorithm == 'simple_lelke':
            mobile_population_df = run_simple_lelke(
                mobile_population_df, my_target_locations)
        elif algorithm == 'greedy_petre':
            mobile_population_df = run_greedy_petre(
                mobile_population_df, my_target_locations)
        elif algorithm == 'main':
            mobile_population_df = run_main(
                mobile_population_df, my_target_locations)
        elif algorithm == 'advanced_petre':
            mobile_population_df = run_advanced_petre(
                mobile_population_df, my_target_locations, number_of_branches=100, max_candidates=None, anchor_strategy="start")
                # 'lower_middle', 'upper_middle', 'start', 'end'
        else:
            raise ValueError("Invalid algorithm.")

        if save_intermediate_results:
            mobile_population_df.to_csv(os.path.join(pipeline_setup.OUTPUT_DIR, f"mobile_population_{algorithm}.csv"), index=False)

    # Recombine the population dataframes
    result_df = pd.concat([mobile_population_df, non_mobile_population_df], ignore_index=True)
    result_df.sort_values(by=[s.UNIQUE_HH_ID_COL, s.UNIQUE_P_ID_COL, s.UNIQUE_LEG_ID_COL], ascending=[True, True, True], inplace=True)
    result_df.to_csv(os.path.join(pipeline_setup.OUTPUT_DIR, "location_assignment_result.csv"), index=False)

def run_hoerl(population_df, target_locations):
    """Runs the Hoerl algorithm on the given population and locations CSV files."""
    population_df = h.read_csv(s.POPULATION_CSV_PATH)
    locations_df = h.read_csv(s.LOCATIONS_CSV_PATH)
    population_df = prepare_mid_df_for_legs_dict(population_df, number_of_persons=100, filter_max_distance=30000)
    dictu = populate_legs_dict_from_df(population_df)
    with_main_dict = locate_main_activities(dictu)
    segmented_dict = segment_legs(with_main_dict)
    result = myhoerl.process(locations_df, segmented_dict)
    result.to_csv(s.LOCATIONS_OUTPUT_PATH, index=False)

def run_greedy_petre(population_df, target_locations):
    """Runs the Greedy Petre algorithm on the given population and locations CSV files."""
    population_df = h.read_csv(s.POPULATION_CSV_PATH)
    locations_df = h.read_csv(s.LOCATIONS_CSV_PATH)
    population_df = prepare_mid_df_for_legs_dict(population_df, number_of_persons=100, filter_max_distance=30000)
    dictu = populate_legs_dict_from_df(population_df)
    with_main_dict = locate_main_activities(dictu)
    segmented_dict = segment_legs(with_main_dict)
    all_dict = {}
    for person_id, segments in segmented_dict.items():
        all_dict[person_id] = []  # Initialize an empty list for each person_id
        for segment in segments:
            placed_segment, _ = solve_segment(segment, number_of_branches=100, max_candidates=None,
                                              anchor_strategy="start")
            with_from_segment = add_from_locations(placed_segment)
            with_all_segment = insert_placed_distances(with_from_segment)
            all_dict[person_id].append(with_all_segment)
    result = flatten_segmented_dict(all_dict)
    result.to_csv(s.LOCATIONS_OUTPUT_PATH, index=False)

def run_simple_lelke(population_df, target_locations):
    """Runs the Simple Lelke algorithm on the given population and locations CSV files."""
    population_df = h.read_csv(s.POPULATION_CSV_PATH)
    locations_df = h.read_csv(s.LOCATIONS_CSV_PATH)
    population_df = prepare_mid_df_for_legs_dict(population_df, number_of_persons=100, filter_max_distance=30000)
    dictu = populate_legs_dict_from_df(population_df)
    with_main_dict = locate_main_activities(dictu)
    segmented_dict = segment_legs(with_main_dict)
    all_dict = {}
    for person_id, segments in segmented_dict.items():
        all_dict[person_id] = []  # Initialize an empty list for each person_id
        for segment in segments:
            placed_segment, _ = solve_segment(segment, number_of_branches=100, max_candidates=None,
                                              anchor_strategy="start")
            with_from_segment = add_from_locations(placed_segment)
            with_all_segment = insert_placed_distances(with_from_segment)
            all_dict[person_id].append(with_all_segment)
    result = flatten_segmented_dict(all_dict)
    result.to_csv(s.LOCATIONS_OUTPUT_PATH, index=False)

def run_main(population_df, target_locations):
    """Runs the Main algorithm on the given population and locations CSV files."""
    population_df = h.read_csv(s.POPULATION_CSV_PATH)
    locations_df = h.read_csv(s.LOCATIONS_CSV_PATH)
    population_df = prepare_mid_df_for_legs_dict(population_df, number_of_persons=100, filter_max_distance=30000)
    dictu = populate_legs_dict_from_df(population_df)
    with_main_dict = locate_main_activities(dictu)
    segmented_dict = segment_legs(with_main_dict)
    all_dict = {}
    for person_id, segments in segmented_dict.items():
        all_dict[person_id] = []  # Initialize an empty list for each person_id
        for segment in segments:
            placed_segment, _ = solve_segment(segment, number_of_branches=100, max_candidates=None,
                                              anchor_strategy="start")
            with_from_segment = add_from_locations(placed_segment)
            with_all_segment = insert_placed_distances(with_from_segment)
            all_dict[person_id].append(with_all_segment)
    result = flatten_segmented_dict(all_dict)
    result.to_csv(s.LOCATIONS_OUTPUT_PATH, index=False)
    
def run_advanced_petre(population_df, target_locations, number_of_branches: int, max_candidates: int, 
                       anchor_strategy: Literal["lower_middle", "upper_middle", "start", "end"]):
    """Runs the Advanced Petre algorithm on the given population and locations CSV files."""
    population_df = h.read_csv(s.POPULATION_CSV_PATH)
    locations_df = h.read_csv(s.LOCATIONS_CSV_PATH)
    population_df = prepare_mid_df_for_legs_dict(population_df, number_of_persons=100, filter_max_distance=30000)
    dictu = populate_legs_dict_from_df(population_df)
    with_main_dict = locate_main_activities(dictu)
    segmented_dict = segment_legs(with_main_dict)
    all_dict = {}
    for person_id, segments in segmented_dict.items():
        all_dict[person_id] = []  # Initialize an empty list for each person_id
        for segment in segments:
            placed_segment, _ = solve_segment(segment, number_of_branches=100, max_candidates=None,
                                              anchor_strategy="start")
            with_from_segment = add_from_locations(placed_segment)
            with_all_segment = insert_placed_distances(with_from_segment)
            all_dict[person_id].append(with_all_segment)
    result = flatten_segmented_dict(all_dict)
    result.to_csv(s.LOCATIONS_OUTPUT_PATH, index=False)
    
if __name__ == "__main__":
    run_location_assignment()