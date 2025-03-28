"""
Creates only the raw synthetic population, combining inputs and setting up and running populationsim.
"""
import sys
import logging
import time

from utils.config import Config
from utils.logger import setup_logging
from utils.stats_tracker import StatsTracker
from utils import column_names as s
from utils.helpers import Helpers

# def generate_random_location_within_hanover():
#     """Generate a random coordinate within Hanover, Germany, in EPSG:25832."""
#     xmin, xmax = 546000, 556000
#     ymin, ymax = 5800000, 5810000
#     x = random.uniform(xmin, xmax)
#     y = random.uniform(ymin, ymax)
#     return np.array([x, y])
#
#
# df[s.HOME_LOC_COL] = None
def assign_random_locations(df, polygon):
    for person_id, group in df.groupby(s.UNIQUE_P_ID_COL):
        home_location = h.random_point_in_polygon(polygon)
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

def run_synthetic_population():
    """
    Creates only the raw synthetic population, combining inputs and setting up and running populationsim.
    """
    if config.get("synthetic_population.assign_random_locations"):
        logger.info("Assigning random locations to synthetic population.")
        # Given all buildings and all MiD households we want to assign, randomly assign them to buildings
        # Add random home locations for each person for testing
    elif config.get("synthetic_population.assign_random_buildings"):
        logger.info("Assigning random buildings to synthetic population.")
        # Given all buildings and all MiD households we want to assign, randomly assign them to buildings
        # Add random home locations for each person for testing
        raise NotImplementedError("Assigning random buildings is not yet implemented.")
    else:
        raise NotImplementedError("PopulationSim and surrounding code is not yet integrated and runs separately.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: run.py <output_folder> <project_root> <config_yaml>")
        print("Absolute paths, folders must exist.")
        sys.exit(1)

    output_folder = sys.argv[1]
    project_root = sys.argv[2]
    config_yaml = sys.argv[3]
    step_name = "synthetic_population"

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
    run_synthetic_population(config)
    time_end = time.time()
    time_step = time_end - time_start
    stats_tracker.log("runtimes.synthetic_population_time", time_step)
    stats_tracker.write_stats()
    config.write_used_config()

    logger.info(f"Step {step_name} finished in {time_step:.2f} seconds.")
