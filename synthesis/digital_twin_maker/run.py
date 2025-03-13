"""
Makes any changes and additions to the raw synthetic population, integrates building data and potentials.
Creates a full population/building dataset with all necessary attributes.
"""
# TODO: make sure this runs properly.
# TODO: make sure location assignment works with this data and simplify loc ass setup if possible
import geopandas as gpd
import pandas as pd
import numpy as np
import random
import os
import pickle
from utils.config import Config


class RandomHouseholdAssigner:
    def __init__(self, config: Config):
        hh_ids_csv = config.get("data.mid_hh_ids_csv")
        hh_id_col = config.get("settings.hh_id_column", "mid_hh_id")

        df = pd.read_csv(hh_ids_csv)

        if hh_id_col not in df.columns:
            raise ValueError(f"Column '{hh_id_col}' not found in the CSV file.")

        self.hh_ids = df[hh_id_col].dropna().unique().tolist()

    def assign(self):
        num_ids = random.randint(0, 5)
        return random.sample(self.hh_ids, num_ids) if num_ids > 0 else []


class RandomActivityAssigner:
    def __init__(self):
        self.activities = [
            "work", "business", "education", "shopping", "errands",
            "pick_up_drop_off", "leisure", "home", "return_journey",
            "other", "early_education", "daycare", "accompany_adult",
            "sports", "meetup", "lessons", "unspecified"
        ]
    def assign(self):
        num_activities = random.randint(1, 6)
        return random.sample(self.activities, num_activities)


class LocationTranslater:
    """
    Reformats location data and saves it as a Pickle file for efficient reloading.
    """

    def __init__(self, gdf: gpd.GeoDataFrame):
        self.gdf = gdf
        self.data = self.reformat_locations_from_df(self.gdf)

    @staticmethod
    def reformat_locations_from_df(df: pd.DataFrame) -> dict:
        """
        Reformats a DataFrame into a structured dictionary format.

        :param df: DataFrame with location information.
        :return: Nested dictionary with formatted location data.
        """
        reformatted_data = {}

        for row in df.itertuples(index=False):
            activities = row.assigned_activities  # Assigned activities are already lists
            for purpose in activities:
                if purpose not in reformatted_data:
                    reformatted_data[purpose] = {
                        'identifiers': [],
                        'names': [],
                        'coordinates': [],
                        'potentials': []
                    }

                reformatted_data[purpose]['identifiers'].append(row.identifier)
                reformatted_data[purpose]['names'].append(row.name)
                reformatted_data[purpose]['coordinates'].append(np.array([row.x, row.y]))
                reformatted_data[purpose]['potentials'].append(row.capacity)

        # Convert lists to numpy arrays for efficiency
        for purpose in reformatted_data:
            reformatted_data[purpose]['identifiers'] = np.array(reformatted_data[purpose]['identifiers'], dtype=object)
            reformatted_data[purpose]['names'] = np.array(reformatted_data[purpose]['names'], dtype=str)
            reformatted_data[purpose]['coordinates'] = np.array(reformatted_data[purpose]['coordinates'], dtype=float)
            reformatted_data[purpose]['potentials'] = np.array(reformatted_data[purpose]['potentials'], dtype=float)

        return reformatted_data

    def save_as_pickle(self, output_path: str):
        """
        Saves the translated location data as a Pickle file for fast reloading.

        :param output_path: Path to the output .pkl file.
        """
        with open(output_path, 'wb') as f:
            pickle.dump(self.data, f)

        print(f"Location data saved as Pickle: {output_path}")


def make_twin():
    # Load paths from config
    building_shp_path = config.get("data.building_shp")

    # Load the building shapefile
    gdf = gpd.read_file(building_shp_path)

    # Ensure correct coordinate reference system
    if gdf.crs != "EPSG:25832":
        gdf = gdf.to_crs("EPSG:25832")

    # Extract x and y coordinates
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y

    if config.get("digital_twin_maker.assign_random_households"):
        household_assigner = RandomHouseholdAssigner(config)
        gdf["assigned_hh_ids"] = gdf.apply(lambda _: household_assigner.assign(), axis=1)
    else:
        raise NotImplementedError("Proper Household assignment is not implemented yet.")

    if config.get("digital_twin_maker.assign_random_activities"):
        activity_assigner = RandomActivityAssigner()
        gdf["assigned_activities"] = gdf.apply(lambda _: activity_assigner.assign(), axis=1)
    else:
        raise NotImplementedError("Proper Activity assignment is not implemented yet.")

    # Perform location translation and save as Pickle
    if config.get("digital_twin_maker.save_pkl"):
        translator = LocationTranslater(gdf)
        output_pkl_path = os.path.join(output_folder, config.get("digital_twin_maker.output.location_pkl"))
        translator.save_as_pickle(output_pkl_path)
        logger.info(f"Location data saved as Pickle: {output_pkl_path}")

    if config.get("digital_twin_maker.save_shp"):
        # Save the modified GeoDataFrame
        output_shp_path= os.path.join(output_folder, config.get("digital_twin_maker.output.location_shp"))
        gdf.to_file(output_shp_path)
        logger.info(f"Location data saved as Shapefile: {output_shp_path}")

    if config.get("digital_twin_maker.save_csv"):
        # Save the modified GeoDataFrame as CSV
        output_csv_path = os.path.join(output_folder, config.get("digital_twin_maker.output.location_csv"))
        gdf.to_csv(output_csv_path, index=False)
        logger.info(f"Location data saved as CSV: {output_csv_path}")

import sys
import logging
import time
from utils.config import Config
from utils.logger import setup_logging
from utils.stats_tracker import StatsTracker

# def distribute_by_weights(self, weights_df: pd.DataFrame, cell_id_col: str, cut_missing_ids: bool = False):
#     result = h.distribute_by_weights(self.df, weights_df, cell_id_col, cut_missing_ids)
#     self.df = self.df.merge(result[[s.UNIQUE_HH_ID_COL, 'home_loc']], on=s.UNIQUE_HH_ID_COL, how='left')


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: run.py <output_folder> <project_root> <config_yaml>")
        print("Absolute paths, folders must exist.")
        sys.exit(1)

    output_folder = sys.argv[1]  # Absolute path
    project_root = sys.argv[2]
    config_yaml = sys.argv[3]  # Just the filename
    step_name = "twin_maker"

    # Each step sets up its own logging, Config object and StatsTracker
    config = Config(output_folder, project_root, config_yaml)
    config.resolve_paths()

    setup_logging(output_folder, console_level=config.get("settings.logging.console_level"),
                  file_level=config.get("settings.logging.file_level"))
    logger = logging.getLogger(step_name)

    stats_tracker = StatsTracker(output_folder)

    logger.info(f"Starting step {step_name}")
    time_start = time.time()

    # Run
    make_twin()

    time_end = time.time()
    time_step = time_end - time_start
    stats_tracker.log("runtimes.twin_maker_time", time_step)
    stats_tracker.write_stats()

    logger.info(f"Step {step_name} finished in {time_step:.2f} seconds.")