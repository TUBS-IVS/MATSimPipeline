"""
Does any necessary data post-processing.
Writes the population, households, vehicles and facilities data to MATSim XML format.
"""

import sys
import logging
import time
import os
import matsim.writers
import pandas as pd

from utils import helpers as h, column_names as s
from utils.config import Config
from utils.logger import setup_logging
from utils.stats_tracker import StatsTracker

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

    logger.info(f"Starting step {step_name}")
    time_start = time.time()

    # Run the MATSim writer
    matsim_writer = MATSimWriter(config, output_folder)
    matsim_writer.write_plans_to_matsim_xml()
    matsim_writer.write_households_to_matsim_xml()
    matsim_writer.write_vehicles_to_matsim_xml()

    time_end = time.time()
    time_step = time_end - time_start
    stats_tracker.log("runtimes.matsim_writer_time", time_step)
    stats_tracker.write_stats()

    logger.info(f"Step {step_name} finished in {time_step:.2f} seconds.")
