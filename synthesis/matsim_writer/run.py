"""
Does any necessary data post-processing.
Writes the population, households, vehicles and facilities data to MATSim XML format.
"""

import sys
import logging
import time
import os
import pandas as pd
import geopandas as gpd

from synthesis.matsim_writer.matsim_writer import MATSimWriter
from synthesis.matsim_writer.population_post_processor import PopulationPostProcessor
from utils.config import Config
from utils.logger import setup_logging
from utils.stats_tracker import StatsTracker
from utils.helpers import Helpers
from utils import column_names as s

def run_matsim_writer():
    population_post_processor = PopulationPostProcessor(stats_tracker, logger)
    population_post_processor.load_df_from_csv(config.get("matsim_writer.input.population_df"))
    # population_post_processor.change_last_leg_activity_to_home()
    population_post_processor.vary_times_by_household(s.UNIQUE_HH_ID_COL, [s.LEG_START_TIME_COL, s.LEG_END_TIME_COL])
    population_df = population_post_processor.df

    if config.get("matsim_writer.write_locations"):
        logger.info("Loading locations...")
        try:
            locations_gdf = pd.read_pickle(os.path.join(output_folder, config.get("matsim_writer.input.locations_pkl")))
        except FileNotFoundError:
            locations_gdf = gpd.read_file(os.path.join(output_folder, config.get("matsim_writer.input.locations_gpkg")))
        matsim_writer = MATSimWriter(population_df, config, logger, h, locations_gdf)
        logger.info("Writing locations...")
        matsim_writer.write_facilities_to_matsim_xml()
    else:
        matsim_writer = MATSimWriter(population_df, config, logger, h)
    if config.get("matsim_writer.write_plans"):
        logger.info("Writing plans")
        matsim_writer.write_plans_to_matsim_xml()
    if config.get("matsim_writer.write_households"):
        logger.info("Writing households")
        matsim_writer.write_households_to_matsim_xml()
    if config.get("matsim_writer.write_vehicles"):
        logger.info("Writing vehicles")
        matsim_writer.write_vehicles_to_matsim_xml()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: run.py <output_folder> <project_root> <config_yaml>")
        print("Absolute paths, folders must exist.")
        sys.exit(1)

    output_folder = sys.argv[1]  # Absolute path
    project_root = sys.argv[2]
    config_yaml = sys.argv[3]  # Just the filename
    step_name = "matsim_writer"

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

    run_matsim_writer()

    time_end = time.time()
    time_step = time_end - time_start
    stats_tracker.log("runtimes.matsim_writer_time", time_step)
    stats_tracker.write_stats()

    logger.info(f"Step {step_name} finished in {time_step:.2f} seconds.")
