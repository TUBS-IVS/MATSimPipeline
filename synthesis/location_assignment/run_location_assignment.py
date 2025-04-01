"""Soon to be deleted."""

# import pandas as pd
# import os
# import time
# import sys
# from typing import Literal
#
# from utils import settings as s
# from utils import helpers as h
# import logging
# from utils.logger import setup_logging
# from synthesis.location_assignment import activity_locator_distance_based as al
# from synthesis.location_assignment import hoerl
# from utils.stats_tracker import StatsTracker
#
# output_folder = sys.argv[1] # Absolute path to the output folder
# project_root = sys.argv[2] # Absolute path to the project root
#
# setup_logging(output_folder)
# logger = logging.getLogger(__name__)
# stats_tracker = StatsTracker(output_folder)
#
# def run_location_assignment(configs):
#     logger.info("Starting location assignment.")
#     for config in configs.items():
#         logger.info(f"{config}")
#
#     locations_json_folder = configs["general"]["locations_json_folder"]
#     algorithms_to_run = configs["general"]["algorithms_to_run"]
#
#     save_intermediate_results = configs["general"]["save_intermediate_results"]
#     assert_no_missing_locations = configs["general"]["assert_no_missing_locations"]
#     filter_max_distance = configs["general"]["filter_max_distance"]
#     filter_number_of_persons = configs["general"]["filter_number_of_persons"]
#     skip_loading_full_population = configs["general"]["skip_loading_full_population"]
#     write_to_csv = configs["general"]["write_to_csv"]
#
#     # New Stlye
#     # locations_json_folder = config.get("general.locations_json_folder")
#     # algorithms_to_run = config.get("general.algorithms_to_run")
#     #
#     # save_intermediate_results = config.get("general.save_intermediate_results")
#     # assert_no_missing_locations = config.get("general.assert_no_missing_locations")
#     # filter_max_distance = config.get("general.filter_max_distance")
#     # filter_number_of_persons = config.get("general.filter_number_of_persons")
#     # filter_by_person = config.get("general.filter_by_person")
#     # skip_loading_full_population = config.get("general.skip_loading_full_population")
#     # write_to_csv = config.get("general.write_to_csv")
#
#     algo_time = 0
#
#     # Early check if all algorithms are valid
#     valid_algorithms = ['load_intermediate', 'filter', 'remove_unfeasible', 'hoerl', 'simple_lelke', 'greedy_petre',
#                         'main', 'advanced_petre', 'open_ended', 'nothing']
#
#     if not all(algorithm in valid_algorithms for algorithm in algorithms_to_run):
#         raise ValueError(f"Invalid algorithm. Valid algorithms are: {valid_algorithms}")
#
#     # Build the common KDTree for the locations
#     target_locations = al.TargetLocations(h.get_files(locations_json_folder))
#
#     if not skip_loading_full_population:
#         # Load the population dataframe
#         population_df = h.read_csv(h.get_files(configs["general"]["population_df_folder"]))
#
#         # Prepare the population dataframe, split off non-mobile persons
#         mobile_population_df, non_mobile_population_df = (al.prepare_population_df_for_location_assignment
#                                                           (population_df,
#                                                            number_of_persons=filter_number_of_persons,
#                                                            filter_max_distance=filter_max_distance))
#         mobile_population_df[s.LEG_DISTANCE_METERS_COL] = mobile_population_df[s.LEG_DISTANCE_METERS_COL]/configs["general"]["detour_factor"]
#
#     for algorithm in algorithms_to_run:
#         if algorithm == "load_intermediate":
#             mobile_population_df = load_intermediate()
#             non_mobile_population_df = pd.DataFrame()
#         elif algorithm == 'nothing':
#             logger.info("Doing nothing.")
#         elif algorithm == 'filter':
#             mobile_population_df = mobile_population_df[mobile_population_df[s.UNIQUE_P_ID_COL] == configs["general"]["filter_by_person"]]
#         elif algorithm == 'remove_unfeasible':
#             mobile_population_df = remove_unfeasible_persons(mobile_population_df)
#         elif algorithm == 'hoerl':
#             mobile_population_df, algo_time = run_hoerl(
#                 mobile_population_df, target_locations, configs.get("hoerl"))
#         elif algorithm == 'simple_lelke':
#             mobile_population_df = run_simple_lelke(
#                 mobile_population_df, target_locations)
#         elif algorithm == 'greedy_petre':
#             mobile_population_df = run_greedy_petre(
#                 mobile_population_df, target_locations)
#         elif algorithm == 'main':
#             mobile_population_df = run_main(
#                 mobile_population_df, target_locations, configs.get("main"))
#         elif algorithm == 'open_ended':
#             mobile_population_df = run_open_ended(
#                 mobile_population_df, target_locations, configs.get("open_ended"))
#         elif algorithm == 'advanced_petre':
#             mobile_population_df, algo_time = run_advanced_petre(
#                 mobile_population_df, target_locations, configs.get("advanced_petre"))
#         else:
#             raise ValueError("Invalid algorithm.")
#
#         # Make sure algorithm results are in the correct format
#         mobile_population_df['to_location'] = mobile_population_df['to_location'].apply(
#             lambda x: h.convert_to_point(x, target='array'))
#         mobile_population_df['from_location'] = mobile_population_df['from_location'].apply(
#             lambda x: h.convert_to_point(x, target='array'))
#         if save_intermediate_results:
#             mobile_population_df.to_csv(os.path.join(output_folder, f"mobile_population_{algorithm}.csv"),
#                                         index=False)
#
#     if assert_no_missing_locations:
#         assert mobile_population_df['to_location'].notna().all(), "Some persons have no location assigned."
#
#     # Recombine the population dataframes
#     result_df = pd.concat([mobile_population_df, non_mobile_population_df], ignore_index=True)
#     result_df.sort_values(by=[s.UNIQUE_HH_ID_COL, s.UNIQUE_P_ID_COL, s.UNIQUE_LEG_ID_COL], ascending=[True, True, True],
#                           inplace=True)
#
#     # Write the result to a CSV file
#     if write_to_csv:
#         algos_string = "_".join(algorithms_to_run)
#         if "advanced_petre" in algorithms_to_run:
#             num_branches_string = f"_{configs['advanced_petre']['number_of_branches']}-branches"
#             min_candidates_complex_string = f"_{configs['advanced_petre']['min_candidates_complex_case']}-min-cand-complex"
#             candidates_two_leg_string = f"_{configs['advanced_petre']['candidates_two_leg_case']}-min-cand-two-leg"
#         else:
#             num_branches_string = ""
#             candidates_two_leg_string = ""
#             min_candidates_complex_string = ""
#         result_df.to_csv(os.path.join(output_folder, f"location_assignment_result_{algos_string}"
#                                                                  f"{num_branches_string}"
#                                                                  f"{candidates_two_leg_string}"
#                                                                  f"{min_candidates_complex_string}.csv"),
#                          index=False)
#         logger.info(f"Wrote location assignment result to {output_folder}.")
#         stats_tracker.write_stats_to_file(os.path.join(output_folder, "location_assignment_stats.txt"))
#
#     return result_df, algo_time
#
#
# def load_intermediate():
#     mobile_population_df = h.read_csv(h.get_files(r"data/intermediates"))
#     if "to_location" in mobile_population_df.columns:
#         mobile_population_df["to_location"] = mobile_population_df["to_location"].apply(
#             lambda x: h.convert_to_point(x, target='array'))
#     if "from_location" in mobile_population_df.columns:
#         mobile_population_df["from_location"] = mobile_population_df["from_location"].apply(
#             lambda x: h.convert_to_point(x, target='array'))
#     return mobile_population_df
#
#
# def remove_unfeasible_persons(population_df):
#     logger.info("Removing unfeasible persons.")
#     legs_dict = al.populate_legs_dict_from_df(population_df)
#     logger.info("Dict populated.")
#     segmented_dict = al.segment_plans(legs_dict)
#     logger.info("Dict segmented.")
#     feasible_dict = h.filter_feasible_data(segmented_dict)
#     population_df = al.write_placement_results_dict_to_population_df(feasible_dict, population_df, merge_how='right')
#     return population_df
#
#
# def run_hoerl(population_df, target_locations, config):
#     """Runs the Hoerl algorithm on the given population and locations CSV files."""
#     logger.info("Starting Hoerl algorithm.")
#     legs_dict = al.populate_legs_dict_from_df(population_df)
#     logger.info("Dict populated.")
#     segmented_dict = al.segment_plans(legs_dict)
#     logger.info("Dict segmented, starting hoerl")
#     time_start = time.time()
#     df_location, df_convergence = myhoerl.process(target_locations, segmented_dict, config)
#     algo_time = time.time() - time_start
#     logger.info(f"Hoerl done in {algo_time} seconds.")
#     population_df['to_location'] = population_df['to_location'].apply(
#         lambda x: h.convert_to_point(x, target='array'))  # Needed currently so [] becomes None
#     population_df['from_location'] = population_df['from_location'].apply(
#         lambda x: h.convert_to_point(x, target='array'))  # Needed currently so [] becomes None
#     population_df = al.write_hoerl_df_to_big_df(df_location, population_df)
#     population_df = h.add_from_location(population_df, 'to_location', 'from_location')
#     return population_df, algo_time
#
#
# def run_greedy_petre(population_df, target_locations):
#     """Runs the Greedy Petre algorithm on the given population and locations CSV files."""
#     logger.info("Starting Greedy Petre algorithm.")
#     legs_dict = al.populate_legs_dict_from_df(population_df)
#     logger.info("Dict populated.")
#     segmented_dict = al.segment_plans(legs_dict)
#     logger.info("Dict segmented.")
#     greedy_petre_algorithm = al.WeirdPetreAlgorithm(target_locations, segmented_dict, variant="greedy")
#     result_dict = greedy_petre_algorithm.run()
#     population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
#     return h.add_from_location(population_df, 'to_location', 'from_location')
#
#
# def run_simple_lelke(population_df, target_locations):
#     """Runs the Simple Lelke algorithm on the given population and locations CSV files."""
#     logger.info("Starting Simple Lelke algorithm.")
#     legs_dict = al.populate_legs_dict_from_df(population_df)
#     logger.info("Dict populated.")
#     segmented_dict = al.segment_plans(legs_dict)
#     logger.info("Dict segmented.")
#     lelke_algorithm = al.SimpleLelkeAlgorithm(target_locations, segmented_dict)
#     result_dict = lelke_algorithm.run()
#     population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
#     return h.add_from_location(population_df, 'to_location', 'from_location')
#
#
# def run_main(population_df, target_locations, config):
#     """Runs the Main algorithm on the given population and locations CSV files."""
#     logger.info("Starting Main algorithm.")
#     legs_dict = al.populate_legs_dict_from_df(population_df)
#     logger.info("Dict populated.")
#     simple_main_algorithm = al.SimpleMainLocationAlgorithm(target_locations, legs_dict,
#                                                            config)  # It wants unsegmented legs
#     result_dict = simple_main_algorithm.run()
#     result_dict = al.segment_plans(result_dict)  # Needed as writer expects segmented legs
#     population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
#     return h.add_from_location(population_df, 'to_location', 'from_location')
#
#
# def run_open_ended(population_df, target_locations, config):
#     logger.info("Starting open-ended algorithm.")
#     legs_dict = al.populate_legs_dict_from_df(population_df)
#     logger.info("Dict populated.")
#     open_ended_algorithm = al.OpenEndedAlgorithm(target_locations, legs_dict, config)
#     result_dict = open_ended_algorithm.run()
#     result_dict = al.segment_plans(result_dict)  # Needed as writer expects segmented legs
#     population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
#     return h.add_from_location(population_df, 'to_location', 'from_location')
#
#
# def run_advanced_petre(population_df, target_locations, config):
#     """Runs the Advanced Petre algorithm on the given population and locations CSV files."""
#     logger.info("Starting Advanced Petre algorithm.")
#     legs_dict = al.convert_to_segmented_plans(population_df)
#     logger.info("Dict populated.")
#     segmented_dict = al.new_segment_plans(legs_dict)
#     logger.info("Dict segmented.")
#     time_start = time.time()
#     advance_petre_algorithm = al.CARLA(target_locations, segmented_dict, config)
#     result_dict = advance_petre_algorithm.run()
#     algo_time = time.time() - time_start
#     logger.info(f"CARLA done in {algo_time} seconds.")
#     population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
#     return h.add_from_location(population_df, 'to_location', 'from_location'), algo_time
#
#
# def main():
#     configs = {
#         "general": {
#             "population_df_folder": r"data/mid/enhanced",
#             "locations_json_folder": r"data/locations",
#             "algorithms_to_run": ['load_intermediate', 'advanced_petre'],
#             "save_intermediate_results": True,
#             "assert_no_missing_locations": True,
#             "filter_by_person": "10474610_12005_10474614",
#             "filter_number_of_persons": 1000,
#             "filter_max_distance": 30000,
#             "detour_factor": 1.4,
#             "skip_loading_full_population": True,
#             "write_to_csv": True
#         },
#         "advanced_petre": {
#             "number_of_branches": 70,
#             "min_candidates_complex_case": 20,  # 10-20, more not useful
#             "candidates_two_leg_case": 30,
#             "max_candidates": None,
#             # If more candidates are found, gets a random sample of this size, only these are evaluated. (negligible speedup, should be None)
#             "anchor_strategy": "lower_middle",
#             "selection_strategy_complex_case": "top_n_spatial_downsample",
#             "selection_strategy_two_leg_case": "top_n",
#             "max_radius_reduction_factor": None,
#             "max_iterations_complex_case": 15,
#             # How long rings are expanded before raising error (as this points to data issues)
#             "only_return_valid_persons": False
#             # Very rough validity check - are all lowest-level circle intersections valid?
#         },
#         "main": {
#             "skip_already_located": True
#         },
#         "open_ended": {
#             "skip_already_located": False
#         },
#         "hoerl": {
#             "max_iterations": 1000
#         }
#     }
#     run_location_assignment(configs)
#
#
# if __name__ == "__main__":
#     import cProfile
#     import pstats
#
#     # Wrap the function call with arguments in a lambda
#     cProfile.run('main()', r'C:\Users\petre\Documents\GitHub\MATSimPipeline\output\profile_stats')
#
#     # Read the stats
#     # stats = pstats.Stats('profile_stats')
#     # stats.strip_dirs().sort_stats('cumulative').print_stats(10)
#
# #--------- Evaluation ---------
# # import copy
# # from analysis import analysis as ays
# #
# # def generate_variations(base_config, parameter_variations):
# #     """
# #     Generate configurations by varying parameters relative to the base config.
# #     """
# #     variations = []
# #     for param, values in parameter_variations.items():
# #         for value in values:
# #             config = copy.deepcopy(base_config)
# #             keys = param.split(".")
# #             current = config
# #             for key in keys[:-1]:
# #                 current = current[key]
# #             current[keys[-1]] = value
# #             variations.append((param, value, config))
# #     return variations
# #
# # def evaluate_config(config, param, value, repetitions):
# #     """
# #     Evaluate a single configuration multiple times and return average results.
# #     """
# #     qualities = []
# #     runtimes = []
# #     for _ in range(repetitions):
# #         try:
# #             df, runtime = run_location_assignment(config)  # Replace with your function
# #             analysis = ays.DataframeAnalysis()
# #             analysis.df = df
# #             analysis.df['to_location'] = analysis.df['to_location'].apply(lambda x: h.convert_to_point(x))
# #             analysis.df['from_location'] = analysis.df['from_location'].apply(lambda x: h.convert_to_point(x))
# #             analysis.evaluate_distance_deviations_from_df()
# #             quality = analysis.df['discretization_error'].describe()
# #             qualities.append(quality)
# #             runtimes.append(runtime)
# #         except Exception as e:
# #             return {
# #                 "param": param,
# #                 "value": value,
# #                 "error": str(e),
# #                 "runtimes": None,
# #                 "qualities": None,
# #             }
# #     return {
# #         "param": param,
# #         "value": value,
# #         "error": None,
# #         "runtimes": runtimes,
# #         "qualities": qualities
# #     }
# #
# # def main():
# #     # Base configuration
# #     base_config = {
# #         "general": {
# #             "population_df_folder": r"data/mid/enhanced",
# #             "locations_json_folder": r"data/locations",
# #             "algorithms_to_run": ['load_intermediate', 'hoerl'],
# #             "save_intermediate_results": False,
# #             "assert_no_missing_locations": True,
# #             "filter_by_person": "10474610_12005_10474614",
# #             "filter_number_of_persons": 1000,
# #             "filter_max_distance": 30000,
# #             "detour_factor": 1.4,
# #             "skip_loading_full_population": True,
# #             "write_to_csv": False
# #         },
# #         "advanced_petre": {
# #             "number_of_branches": 50,
# #             "min_candidates_complex_case": 10,
# #             "candidates_two_leg_case": 20,
# #             "max_candidates": None,
# #             "anchor_strategy": "lower_middle",
# #             "selection_strategy_complex_case": "mixed",
# #             "selection_strategy_two_leg_case": "top_n",
# #             "max_radius_reduction_factor": None,
# #             "max_iterations_complex_case": 15,
# #             "only_return_valid_persons": False # Only return persons where ALL tours are feasible. OVERRIDES max_radius_reduction_factor
# #         },
# #         "hoerl": {
# #             "max_iterations": 100
# #         }
# #     }
# #
# #     # Define parameter variations
# #     parameter_variations_hoerl_maxsetting = {
# #         "hoerl.max_iterations": [1, 5, 10, 15, 20, 40, 60, 80, 100, 200, 300, 500],
# #         }
# #     parameter_variations = { # hoerl std
# #         "hoerl.max_iterations": [1, 5, 10, 15, 20, 40, 60, 80, 100, 200, 300, 500, 1000, 2000],
# #         }
# #
# #     parameter_variations_ka = {
# #         "advanced_petre.selection_strategy_complex_case": ['keep_all'],
# #         }
# #
# #     parameter_variations_p = {
# #          "advanced_petre.number_of_branches": [1,2,5,10,20,30,50,70,100,300,1000],
# #          "advanced_petre.min_candidates_complex_case": [2,5,10,20,30,50,70,100,200],
# #          "advanced_petre.candidates_two_leg_case": [1,2,5,10,20,30,50,70,100,200],
# #          #"advanced_petre.max_candidates": [None, 1, 10,100],
# #          "advanced_petre.anchor_strategy": ["lower_middle", "upper_middle"],
# #          "advanced_petre.selection_strategy_complex_case": ["top_n", "mixed", "monte_carlo", "spatial_downsample", "top_n_spatial_downsample"], # keep_all
# #          "advanced_petre.selection_strategy_two_leg_case": ["top_n", "monte_carlo"],
# #          "advanced_petre.max_radius_reduction_factor": [None, 0.5, 0.7, 0.8, 0.9],
# #     }
# #     # Only reasonably fast ones
# #     parameter_variations_fast = {
# #         "advanced_petre.number_of_branches": [1,2,5,10,20,30,50,70,100,200],
# #         "advanced_petre.min_candidates_complex_case": [2,5,10,20,30,50,70,100,200],
# #         "advanced_petre.candidates_two_leg_case": [1,2,5,10,20,30,50,70,100,200],
# #         "advanced_petre.max_candidates": [None, 1, 10,100],
# #         "advanced_petre.anchor_strategy": ["lower_middle", "upper_middle"],
# #         "advanced_petre.selection_strategy_complex_case": ["top_n", "mixed", "monte_carlo", "spatial_downsample", "top_n_spatial_downsample"],
# #         "advanced_petre.selection_strategy_two_leg_case": ["top_n", "monte_carlo"],
# #         "advanced_petre.max_radius_reduction_factor": [None, 0.5, 0.7, 0.8, 0.9],
# #     }
# #
# #     # Generate configurations to test
# #     variations = generate_variations(base_config, parameter_variations)
# #
# #     # Evaluate configurations
# #     results = []
# #     for param, value, config in variations:
# #         repetitions = 3
# #         result = evaluate_config(config, param, value, repetitions)
# #         results.append(result)
# #
# #     # Save results to a DataFrame
# #     df = pd.DataFrame(results)
# #     df.to_csv("structured_evaluation_result_unvalidunplaced_hoerl_std.csv", index=False)
# #
# #     print("Evaluation complete.")
# #
# # if __name__ == "__main__":
# #     main()
#
