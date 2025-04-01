"""
Adds households to the given locations.
"""
import os.path
import sys
import logging
import time
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from utils.config import Config
from utils.logger import setup_logging
from utils.stats_tracker import StatsTracker
from utils import column_names as s
from utils.helpers import Helpers
import fiona.errors


def assign_population_to_random_buildings(population_df, buildings_gdf):
    """
    Assigns each household in the population to a valid random building (activity includes 'home'),
    and updates the population_df with home/from/to locations.
    """
    valid_buildings = buildings_gdf[buildings_gdf[s.FACILITY_ACTIVITIES_COL].apply(lambda acts: s.ACT_HOME in acts)].copy()

    if valid_buildings.empty:
        raise ValueError("No buildings with 'home' in activity field.")

    # One row per household (just using mid id as this is just a test sample)
    households_df = population_df[[s.HOUSEHOLD_MID_ID_COL]].drop_duplicates()

    # Sample buildings with replacement (several households can be assigned to the same building)
    assigned_buildings = valid_buildings.sample(n=len(households_df), replace=True).reset_index(drop=True)
    households_df = households_df.reset_index(drop=True)

    households_df["geometry"] = assigned_buildings.geometry.centroid
    households_df["building_idx"] = assigned_buildings.index  # index from original buildings_gdf

    # Merge back into population_df by hh mid id
    population_df = population_df.merge(households_df[[s.HOUSEHOLD_MID_ID_COL, "geometry"]], on=s.HOUSEHOLD_MID_ID_COL, how="left")

    # Assign locations for each person
    for person_id, group in tqdm(population_df.groupby(s.PERSON_MID_ID_COL)):
        location = group.iloc[0]["geometry"]

        for i in group.index:
            population_df.at[i, s.HOME_LOC_COL] = location

        population_df.at[group.index[0], "from_location"] = location
        population_df.at[group.index[-1], "to_location"] = location

        home_rows_to = group[group[s.ACT_TO_INTERNAL_COL] == s.ACT_HOME].index
        if not home_rows_to.empty:
            for idx in home_rows_to:
                population_df.at[idx, "to_location"] = location

        home_rows_from = group[group[s.ACT_FROM_INTERNAL_COL] == s.ACT_HOME].index
        if not home_rows_from.empty:
            for idx in home_rows_from:
                population_df.at[idx, "from_location"] = location

    # Group hh_ids by building index
    groups = households_df.groupby("building_idx")[s.HOUSEHOLD_MID_ID_COL].apply(list)

    # Add hh_ids to the buildings
    valid_buildings["hh_ids"] = valid_buildings.index.map(groups).fillna("").apply(
        lambda x: x if isinstance(x, list) else [])

    return valid_buildings, population_df


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
    return df

def population_sim():
    """
    Placeholder for PopulationSim code.
    """
    raise NotImplementedError("PopulationSim and surrounding code is not yet integrated and runs separately.")
    # def distribute_by_weights(self, weights_df: pd.DataFrame, cell_id_col: str, cut_missing_ids: bool = False):
    #     result = h.distribute_by_weights(self.df, weights_df, cell_id_col, cut_missing_ids)
    #     self.df = self.df.merge(result[[s.UNIQUE_HH_ID_COL, 'home_loc']], on=s.UNIQUE_HH_ID_COL, how='left')



def run_synthetic_population():
    """
    Produces both:
      - the building-based GeoDataFrame with added hh_ids per building (the universal main file)
      - a population DataFrame with locations for each person (optional, for speeding up location assignment)
    """

    if config.get("synthetic_population.assign_random_locations"):
        logger.info("Assigning random locations to sample population.")
        population_sample_df = pd.read_csv(
            os.path.join(project_root, config.get("synthetic_population.input.sample_population_csv")))
        region_border_shp = gpd.read_file(
            os.path.join(project_root, config.get("synthetic_population.input.region_border_shp"))
        )
        region_border_polygon = region_border_shp.geometry.iloc[0]

        population_df = assign_random_locations(population_sample_df, region_border_polygon)
        building_gdf_with_hhs = None

    elif config.get("synthetic_population.assign_random_buildings"):
        logger.info("Assigning random buildings to sample population.")
        population_sample_df = pd.read_csv(
            os.path.join(project_root, config.get("synthetic_population.input.sample_population_csv")))
        try:
            buildings_gdf = pd.read_pickle(
                os.path.join(project_root, config.get("synthetic_population.input.buildings_pkl")))
        except (FileNotFoundError, fiona.errors.DriverError):
            buildings_gdf = gpd.read_file(
                os.path.join(project_root, config.get("synthetic_population.input.buildings_gpkg")))
        if config.get("synthetic_population.limit_buildings_to_region"):
            logger.info("Limiting buildings to region border.")
            region_border_shp = gpd.read_file(
                os.path.join(project_root, config.get("synthetic_population.input.region_border_shp"))
            )
            region_border_polygon = region_border_shp.geometry.iloc[0]
            buildings_gdf = buildings_gdf[buildings_gdf.intersects(region_border_polygon)]
        # Ensure consistent CRS
        common_crs = config.get("settings.common_crs")
        if buildings_gdf.crs is None:
            logger.info(f"No CRS detected. Assuming {common_crs}")
            buildings_gdf.set_crs(common_crs, inplace=True)
        elif buildings_gdf.crs != common_crs:
            logger.info(f"Converting CRS to {common_crs}")
            buildings_gdf = buildings_gdf.to_crs(common_crs)
        building_gdf_with_hhs, population_df = assign_population_to_random_buildings(population_sample_df,
                                                                                     buildings_gdf)

    else:
        raise NotImplementedError("PopulationSim and surrounding code is not yet integrated and runs separately.")

    # Save outputs depending on config
    if config.get("synthetic_population.save_population_with_locations_csv"):
        output_csv = os.path.join(output_folder,
                                  config.get("synthetic_population.output.population_with_locations_csv"))
        logger.info(f"Saving population with assigned locations to {output_csv}")
        population_df.to_csv(output_csv, index=False)
    if config.get("synthetic_population.save_population_with_locations_pkl"):
        output_pkl = os.path.join(output_folder,
                                  config.get("synthetic_population.output.population_with_locations_pkl"))
        logger.info(f"Saving population with assigned locations to {output_pkl}")
        population_df.to_pickle(output_pkl)

    if config.get("synthetic_population.save_building_assignments_gpkg"):
        if building_gdf_with_hhs is None:
            logger.warning("No building assignments to save.")
        else:
            output_gpkg = os.path.join(output_folder,
                                       config.get("synthetic_population.output.building_assignment_gpkg"))
            logger.info(f"Saving building assignments to {output_gpkg}")
            building_gdf_with_hhs.to_file(output_gpkg, layer="data", driver="GPKG")
    if config.get("synthetic_population.save_building_assignments_pkl"):
        if building_gdf_with_hhs is None:
            logger.warning("No building assignments to save.")
        else:
            output_pkl = os.path.join(output_folder,
                                      config.get("synthetic_population.output.building_assignment_pkl"))
            logger.info(f"Saving building assignments to {output_pkl}")
            building_gdf_with_hhs.to_pickle(output_pkl)


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
    run_synthetic_population()
    time_end = time.time()
    time_step = time_end - time_start
    stats_tracker.log("runtimes.synthetic_population_time", time_step)
    stats_tracker.write_stats()
    config.write_used_config()

    logger.info(f"Step {step_name} finished in {time_step:.2f} seconds.")
