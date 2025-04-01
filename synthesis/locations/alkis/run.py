import sys
import os
import logging
import time
import pandas as pd
import geopandas as gpd

from utils import column_names as s
from utils.helpers import Helpers
from utils.config import Config
from utils.logger import setup_logging
from utils.stats_tracker import StatsTracker

"""Get a usable building dataset, incl. purposes, from ALKIS data.
https://ni-lgln-opengeodata.hub.arcgis.com/documents/lgln-opengeodata::alkis-hausumringe/about"""


def ensure_required_activities(row):
    required_activities = [
        "pick_up_drop_off", "return_journey", "accompany_adult", "unspecified", "other"
    ]
    current = row if pd.notna(row) else ""
    current_acts = set([a.strip() for a in current.split(";") if a.strip()])
    updated_acts = current_acts.union(required_activities)
    return ";".join(sorted(updated_acts))


def run_alkis_locations():
    """
    Produces the base building-based GeoDataFrame (the universal main file) with possible activities per location.
    """
    gdf = gpd.read_file(os.path.join(project_root, config.get("locations_alkis.input.hausumringe_shp")))
    common_crs = config.get("settings.common_crs")

    if gdf.crs is None:
        logger.info(f"No CRS detected. Assuming {common_crs}")
        gdf.set_crs(common_crs, inplace=True)
    elif gdf.crs != common_crs:
        gdf = gdf.to_crs(common_crs)

    mapping_df = pd.read_csv(os.path.join(project_root, config.get("locations_alkis.input.gfk_mapping_csv")))
    mapping_df["activities"] = mapping_df["activities"].apply(ensure_required_activities)

    # Ensure semicolon-separated string (already is coming from CSV, but be sure)
    gfk_to_activities = mapping_df.set_index("gfk_code")["activities"].apply(str).to_dict()

    logger.info("Loaded ALKIS shapefile and mapping file. Mapping...")

    # Map activities as string field (semicolon-separated)
    gdf[s.FACILITY_ACTIVITIES_COL] = gdf["GFK"].map(gfk_to_activities).astype(str)

    logger.info("Mapping done. Adding centroids...")

    gdf[s.FACILITY_CENTROID_COL] = gdf.geometry.centroid
    gdf[s.FACILITY_X_COL] = gdf[s.FACILITY_CENTROID_COL].x
    gdf[s.FACILITY_Y_COL] = gdf[s.FACILITY_CENTROID_COL].y
    gdf.drop(columns=[s.FACILITY_CENTROID_COL], inplace=True)  # Keep only one geometry column

    if config.get("locations_alkis.save_gpkg"):
        output_path = os.path.join(output_folder, config.get("locations_alkis.output.locations_gpkg"))
        logger.info(f"Saving GeoPackage to {output_path}")
        gdf.to_file(output_path, layer="data", driver="GPKG")

    if config.get("locations_alkis.save_pkl"):
        output_path = os.path.join(output_folder, config.get("locations_alkis.output.locations_pkl"))
        logger.info(f"Saving pickle to {output_path}")
        gdf.to_pickle(output_path)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: run.py <output_folder> <project_root> <config_yaml>")
        print("Absolute paths, folders must exist.")
        sys.exit(1)

    output_folder = sys.argv[1]
    project_root = sys.argv[2]
    config_yaml = sys.argv[3]
    step_name = "locations_alkis"

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

    run_alkis_locations()

    time_end = time.time()
    time_step = time_end - time_start
    stats_tracker.log("runtimes.locations_alkis_time", time_step)
    stats_tracker.write_stats()
    config.write_used_config()

    logger.info(f"Step {step_name} finished in {time_step:.2f} seconds.")
