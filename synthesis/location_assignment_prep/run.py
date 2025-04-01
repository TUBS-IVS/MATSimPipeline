"""
Prepares population and building data for efficient location assignment.
Separate step for modularity and if location assignment is run many times (e.g. testing)
"""
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import sys
import time
import pickle
from tqdm import tqdm
from utils.config import Config
from utils.stats_tracker import StatsTracker
from utils.logger import setup_logging
import logging
from utils import column_names as s


def reformat_locations_from_df(df: pd.DataFrame) -> dict:
    """
    Reformats a DataFrame into a structured dictionary format for fast access by activity purpose.
    Falls back to '' for missing names and 0.0 for missing capacities.
    """
    reformatted_data = {}

    for row in tqdm(df.itertuples(index=False), desc="Reformatting locations", total=len(df)):
        raw_acts = getattr(row, s.FACILITY_ACTIVITIES_COL, "") or ""
        activities = [a.strip() for a in raw_acts.split(";") if a.strip()]  # Split by semicolon into list

        identifier = getattr(row, "OI")

        x = getattr(row, s.FACILITY_X_COL)
        y = getattr(row, s.FACILITY_Y_COL)

        name = getattr(row, s.FACILITY_NAME_COL, "") or ""

        raw_pots = getattr(row, s.FACILITY_POTENTIAL_COL, "") or ""
        potentials = [float(p) for p in raw_pots.split(";") if p.strip()]
        if not potentials:
            potentials = [0.0] * len(activities)
        elif len(potentials) < len(activities):
            # Pad with zeros if fewer potentials than activities
            potentials.extend([0.0] * (len(activities) - len(potentials)))
            logger.error(f"Potentials for {identifier} ({name}) have fewer entries than activities.")
        elif len(potentials) > len(activities):
            # Truncate if too many
            potentials = potentials[:len(activities)]
            logger.error(f"Potentials for {identifier} ({name}) have more entries than activities.")

        for i, purpose in enumerate(activities):
            if purpose not in reformatted_data:
                reformatted_data[purpose] = {
                    'identifiers': [],
                    'names': [],
                    'coordinates': [],
                    'potentials': []
                }

            reformatted_data[purpose]['identifiers'].append(identifier)
            reformatted_data[purpose]['names'].append(name)
            reformatted_data[purpose]['coordinates'].append(np.array([x, y]))
            reformatted_data[purpose]['potentials'].append(potentials[i])

    # Convert lists to numpy arrays
    for purpose, data in reformatted_data.items():
        data['identifiers'] = np.array(data['identifiers'], dtype=object)
        data['names'] = np.array(data['names'], dtype=str)
        data['coordinates'] = np.array(data['coordinates'], dtype=float)
        data['potentials'] = np.array(data['potentials'], dtype=float)

    return reformatted_data


def save_reformatted_data_to_json(data_dict: dict, file_path: str):
    serializable_data = {
        purpose: {
            key: value.tolist() for key, value in data.items()
        } for purpose, data in data_dict.items()
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    logger.debug(f"Reformatted data saved to {file_path}")


def run_prepare_location_assignment():
    # Load building GeoDataFrame
    try:
        gdf = pd.read_pickle(
            os.path.join(project_root, config.get("location_assignment_prep.input.locations_pkl"))
        )
    except FileNotFoundError:
        gdf = gpd.read_file(
            os.path.join(project_root, config.get("location_assignment_prep.input.locations_gpkg"))
        )

    # Ensure consistent CRS
    common_crs = config.get("settings.common_crs")
    if gdf.crs is None:
        logger.info(f"No CRS detected. Assuming {common_crs}")
        gdf.set_crs(common_crs, inplace=True)
    elif gdf.crs != common_crs:
        gdf = gdf.to_crs(common_crs)

    reformatted = reformat_locations_from_df(gdf)

    if config.get("location_assignment_prep.save_pkl"):
        output_pkl_path = os.path.join(output_folder, config.get("location_assignment_prep.output.location_pkl"))
        logger.info(f"Saving location dict data to {output_pkl_path}")
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(reformatted, f)

    if config.get("location_assignment_prep.save_json"):
        output_dict_path = os.path.join(output_folder, config.get("location_assignment_prep.output.location_json"))
        logger.info(f"Saving location dict data to {output_dict_path}")
        save_reformatted_data_to_json(reformatted, output_dict_path)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: run.py <output_folder> <project_root> <config_yaml>")
        print("Absolute paths, folders must exist.")
        sys.exit(1)

    output_folder = sys.argv[1]  # Absolute path
    project_root = sys.argv[2]
    config_yaml = sys.argv[3]  # Just the filename
    step_name = "location_assignment_prep"

    # Each step sets up its own logging, Config object and StatsTracker
    config = Config(output_folder, project_root, config_yaml)
    config.resolve_paths()

    setup_logging(output_folder, console_level=config.get("settings.logging.console_level"),
                  file_level=config.get("settings.logging.file_level"))
    logger = logging.getLogger(step_name)

    stats_tracker = StatsTracker(output_folder)

    logger.info(f"Starting step {step_name}")
    time_start = time.time()

    run_prepare_location_assignment()

    time_end = time.time()
    time_step = time_end - time_start
    stats_tracker.log("runtimes.location_assignment_prep_time", time_step)
    stats_tracker.write_stats()

    logger.info(f"Step {step_name} finished in {time_step:.2f} seconds.")
