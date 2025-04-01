"""
Does any necessary data post-processing.
Writes the population, households, vehicles and facilities data to MATSim XML format.
"""

import sys
import logging
import time
from synthesis.matsim_writer.matsim_writer import MATSimWriter
from synthesis.matsim_writer.population_post_processor import PopulationPostProcessor
from utils.config import Config
from utils.logger import setup_logging
from utils.stats_tracker import StatsTracker
from utils.helpers import Helpers

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

    population_post_processor = PopulationPostProcessor(population, config, logger, h)
    population_post_processor.load_df_from_csv()
    # population_post_processor.change_last_leg_activity_to_home()
    population_post_processor.vary_times_by_household()

    matsim_writer = MATSimWriter(population, config, logger, h)
    matsim_writer.write_plans_to_matsim_xml()
    matsim_writer.write_households_to_matsim_xml()
    matsim_writer.write_vehicles_to_matsim_xml()

    time_end = time.time()
    time_step = time_end - time_start
    stats_tracker.log("runtimes.matsim_writer_time", time_step)
    stats_tracker.write_stats()

    logger.info(f"Step {step_name} finished in {time_step:.2f} seconds.")
