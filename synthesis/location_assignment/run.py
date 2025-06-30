import sys
import logging
import time
import cProfile
import io
import pstats
import networkx as nx
import matplotlib.pyplot as plt
import uuid
from utils.config import Config
from utils.logger import setup_logging
from utils.stats_tracker import StatsTracker
import folium
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

    locations_json_path = os.path.join(project_root, config.get("location_assignment.input.locations_json", ""))
    locations_pkl_path = os.path.join(project_root, config.get("location_assignment.input.locations_pkl"))

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
    target_locations = al.TargetLocations(locations_json_path, locations_pkl_path, stats_tracker)

    if not skip_loading_full_population:
        # Load the population dataframe
        try:
            logger.info("Loading population dataframe from pickle...")
            population_df = pd.read_pickle(os.path.join(project_root, config.get("location_assignment.input.population_pkl")))
        except (FileNotFoundError, TypeError):
            logger.info("Pickle not found, loading population dataframe from CSV...")
            population_df = pd.read_csv(os.path.join(project_root, config.get("location_assignment.input.population_csv")))

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

        # # Make sure algorithm results are in the correct format
        # mobile_population_df['to_location'] = mobile_population_df['to_location'].apply(
        #     lambda x: h.convert_to_point(x, target='array'))
        # mobile_population_df['from_location'] = mobile_population_df['from_location'].apply(
        #     lambda x: h.convert_to_point(x, target='array'))
        # if save_intermediate_results:
        #     mobile_population_df.to_csv(os.path.join(output_folder, f"mobile_population_{algorithm}.csv"),
        #                                 index=False)

    if assert_no_missing_locations:
        assert mobile_population_df[s.TO_X_COL].notna().all(), "Some persons have no location assigned."
        assert mobile_population_df[s.TO_Y_COL].notna().all(), "Some persons have no location assigned."

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
    # population_df['to_location'] = population_df['to_location'].apply(
    #     lambda x: h.convert_to_point(x, target='array'))  # Needed currently so [] becomes None
    # population_df['from_location'] = population_df['from_location'].apply(
    #     lambda x: h.convert_to_point(x, target='array'))  # Needed currently so [] becomes None
    population_df = al.write_hoerl_df_to_big_df(df_location, population_df)
    population_df = h.add_from_location(population_df)
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
    return h.add_from_location(population_df)


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
    visualizer = CarlaVisualizer() if config.get("location_assignment.CARLA.visualize") else None
    time_start = time.time()
    CARLA_algo = al.CARLA(target_locations, segmented_dict, config, visualizer)
    result_dict = CARLA_algo.run()
    algo_time = time.time() - time_start
    logger.info(f"CARLA done in {algo_time} seconds.")
    if visualizer:
        visualizer.visualize()
        visualizer.visualize_levels()
    stats_tracker.log("runtimes.carla_time", algo_time)
    population_df = al.write_placement_results_dict_to_population_df(result_dict, population_df)
    return h.add_from_location(population_df)


import folium
from pyproj import Transformer
import matplotlib.colors as mcolors
import uuid

class CarlaVisualizer:
    def __init__(self):
        self.tree = nx.DiGraph()
        self.locations = {}  # node_id -> {coords, metadata, label, level}
        self.transformer = Transformer.from_crs(25832, 4326, always_xy=True)

    def add_node(self, parent_id, label, location=None, metadata=None):
        node_id = str(uuid.uuid4())
        self.tree.add_node(node_id, label=label)
        # Determine level
        if parent_id:
            self.tree.add_edge(parent_id, node_id)
            parent_level = self.locations[parent_id]["level"] if parent_id in self.locations else 0
            level = parent_level + 1
        else:
            level = 0
        # Always store node metadata, even if coords is None
        self.locations[node_id] = {
            "coords": location,
            "metadata": metadata or {},
            "label": label,
            "level": level
        }
        return node_id

    def visualize(self):
        if not self.locations:
            print("No locations to visualize.")
            return

        # Find first node with valid coordinates
        root_node = next((info for info in self.locations.values() if info["coords"] is not None), None)
        if not root_node:
            print("No valid coordinates found for visualization.")
            return

        lon, lat = self.transformer.transform(root_node["coords"][0], root_node["coords"][1])
        m = folium.Map(location=[lat, lon], zoom_start=13)

        # Set up colormap
        levels = [info["level"] for info in self.locations.values()]
        max_level = max(levels) if levels else 1
        from matplotlib import cm
        cmap = cm.get_cmap('viridis', max_level + 1)
        norm = mcolors.Normalize(vmin=0, vmax=max_level)

        # Add nodes (skip ones without coords)
        for node_id, info in self.locations.items():
            if info["coords"] is None:
                continue
            easting, northing = info["coords"]
            lon, lat = self.transformer.transform(easting, northing)
            color = mcolors.to_hex(cmap(norm(info["level"])))
            popup = (f"{info['label']}<br>"
                     f"Metadata: {info['metadata']}<br>"
                     f"Level: {info['level']}")
            folium.CircleMarker([lat, lon], radius=5, color=color, fill=True, popup=popup).add_to(m)

        # Add edges between parent and child nodes
        for parent_id, child_id in self.tree.edges():
            parent_info = self.locations.get(parent_id)
            child_info = self.locations.get(child_id)
            if parent_info and child_info:
                if parent_info["coords"] is None or child_info["coords"] is None:
                    continue
                if parent_info["level"] == 0:
                    continue  # Skip edge from root
                e1, n1 = parent_info["coords"]
                e2, n2 = child_info["coords"]
                lon1, lat1 = self.transformer.transform(e1, n1)
                lon2, lat2 = self.transformer.transform(e2, n2)
                folium.PolyLine([(lat1, lon1), (lat2, lon2)], color='black', weight=3).add_to(m)

        m.save("carla_branching_map.html")
        print("Map saved as carla_branching_map.html")

    def visualize_levels(self):
        if not self.locations:
            print("No locations to visualize.")
            return

        # Find root location for centering
        root_node = next((info for info in self.locations.values() if info["coords"] is not None), None)
        if not root_node:
            print("No valid coordinates found for visualization.")
            return

        lon, lat = self.transformer.transform(root_node["coords"][0], root_node["coords"][1] - 1000) # centre more south

        # Determine max level
        levels = [info["level"] for info in self.locations.values() if info["coords"] is not None]
        max_level = max(levels)

        from matplotlib import cm
        cmap = cm.get_cmap('viridis', max_level + 1)
        norm = mcolors.Normalize(vmin=0, vmax=max_level)

        # Group nodes by level
        level_groups = {lvl: [] for lvl in range(max_level + 1)}
        for node_id, info in self.locations.items():
            if info["coords"] is not None:
                level_groups[info["level"]].append((node_id, info))

        # Step 1: Individual level maps
        for lvl in range(max_level + 1):
            m = folium.Map(location=[lat, lon], zoom_start=13)
            for node_id, info in level_groups[lvl]:
                easting, northing = info["coords"]
                lon_, lat_ = self.transformer.transform(easting, northing)
                color = mcolors.to_hex(cmap(norm(info["level"])))
                popup = f"{info['label']}<br>Metadata: {info['metadata']}<br>Level: {info['level']}"
                folium.CircleMarker([lat_, lon_], radius=5, color=color, fill=True, popup=popup).add_to(m)

            for parent_id, child_id in self.tree.edges():
                p_info = self.locations.get(parent_id)
                c_info = self.locations.get(child_id)
                if p_info and c_info and p_info["coords"] is not None and c_info["coords"] is not None:
                    if p_info["level"] == lvl or c_info["level"] == lvl:
                        e1, n1 = p_info["coords"]
                        e2, n2 = c_info["coords"]
                        lon1, lat1 = self.transformer.transform(e1, n1)
                        lon2, lat2 = self.transformer.transform(e2, n2)
                        folium.PolyLine([(lat1, lon1), (lat2, lon2)], color='gray', weight=1).add_to(m)

            m.save(f"carla_map_level_{lvl}.html")
            print(f"Map for level {lvl} saved as carla_map_level_{lvl}.html")

        # Step 2: Cumulative level maps
        for lvl in range(max_level + 1):
            m = folium.Map(location=[lat, lon], zoom_start=13)


            for parent_id, child_id in self.tree.edges():
                p_info = self.locations.get(parent_id)
                c_info = self.locations.get(child_id)
                if p_info and c_info and p_info["coords"] is not None and c_info["coords"] is not None:
                    if p_info["level"] == 0:
                        continue  # Skip edge from root
                    if p_info["level"] <= lvl and c_info["level"] <= lvl:
                        e1, n1 = p_info["coords"]
                        e2, n2 = c_info["coords"]
                        lon1, lat1 = self.transformer.transform(e1, n1)
                        lon2, lat2 = self.transformer.transform(e2, n2)
                        folium.PolyLine([(lat1, lon1), (lat2, lon2)], color='black', weight=2).add_to(m)
            for l in range(lvl + 1):
                for node_id, info in level_groups[l]:
                    easting, northing = info["coords"]
                    lon_, lat_ = self.transformer.transform(easting, northing)
                    color = mcolors.to_hex(cmap(norm(info["level"])))
                    popup = f"{info['label']}<br>Metadata: {info['metadata']}<br>Level: {info['level']}"
                    folium.CircleMarker(
                        [lat_, lon_],
                        radius=8,  # larger marker
                        color='black',  # border color
                        fill=True,
                        fill_color=color,  # internal color from colormap
                        fill_opacity=1.0,  # fully opaque
                        weight=1,  # border thickness
                        popup=popup
                    ).add_to(m)
            m.save(f"carla_map_levels_0_to_{lvl}.html")
            print(f"Cumulative map for levels 0 to {lvl} saved as carla_map_levels_0_to_{lvl}.html")


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

    profile_enabled = config.get("settings.profiling.enabled", default=False)
    save_txt = config.get("settings.profiling.save_txt", default=True)
    save_raw = config.get("settings.profiling.save_raw", default=False)
    profiler = None
    if profile_enabled:
        logger.info("Profiling enabled — running with cProfile.")
        profiler = cProfile.Profile()
        profiler.enable()
    logger.info(f"Starting step {step_name}")
    time_start = time.time()
    run_location_assignment()
    time_end = time.time()
    time_step = time_end - time_start
    if profile_enabled and profiler:
        profiler.disable()
        if save_txt:
            stats_path = os.path.join(output_folder, "profile_stats.txt")
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()
            with open(stats_path, "w") as f:
                f.write(s.getvalue())
            logging.info(f"Profiling summary saved to {stats_path}")
        if save_raw:
            raw_path = os.path.join(output_folder, "profile_stats.prof")
            profiler.dump_stats(raw_path)
            logging.info(f"Raw profiler data saved to {raw_path}")
    stats_tracker.log("runtimes.location_assignment_time", time_step)
    stats_tracker.write_stats()
    config.write_used_config()

    logger.info(f"Step {step_name} finished in {time_step:.2f} seconds.")
