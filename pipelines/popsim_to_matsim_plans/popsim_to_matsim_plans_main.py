import os
import time
import multiprocessing as mp

import pandas as pd
import geopandas as gpd
import winsound
from pipelines.common import rules

from pipelines.common import helpers as h
from pipelines.popsim_to_matsim_plans.main_activity_locator import ActivityLocator
from pipelines.popsim_to_matsim_plans.population_frame_processor import PopulationFrameProcessor
from utils import matsim_pipeline_setup
from utils import settings_values as s
from utils.logger import logging

logger = logging.getLogger(__name__)

# Set working dir
os.chdir(matsim_pipeline_setup.PROJECT_ROOT)
write_to_file = True if s.POPULATION_ANALYSIS_OUTPUT_FILE else False


def popsim_to_matsim_plans_main():
    """
    Main function for the popsim_to_matsim_plans pipeline.
    :return:
    Notes:
        - The ids of households, persons and trips must be unique within the population sample (e.g. MiD)
        (MiD: H_ID, HP_ID, and a previously added HPW_ID for legs)
    """

    logger.info(f"Starting popsim_to_matsim_plans pipeline")

    # Load data from PopSim, concat different PopSim results if necessary
    # Lowest level of geography must be named the same in all input files, if there are multiple
    population = PopulationFrameProcessor()
    for file in s.EXPANDED_HOUSEHOLDS_FILES:
        population.load_df_from_csv(file, test_col=s.HOUSEHOLD_POPSIM_ID_COL, if_df_exists="concat")

    # SINGLE
    # Assign household points to households (random if there are no points in the household cell)
    # Expand enhanced mid file

    # Give households cells based on RH shapefile

    # Split file into chunks
    # MULTI
    # Do plan location assignment in parallel

    population.downsample_population(s.SAMPLE_SIZE)

    # Add household attributes from MiD
    # if s.HH_COLUMNS:
    #     population.add_csv_data_on_id(s.FILE, s.HH_COLUMNS, id_column=s.HOUSEHOLD_MID_ID_COL,
    #                                   drop_duplicates_from_source=True, delete_later=True)
    # logger.info(f"Population df after adding HH attributes: \n{population.df.head()}")

    # Distribute buildings to households (if PopSim assigned hhs to a larger geography)
    weights_df = h.read_csv(s.BUILDINGS_IN_LOWEST_GEOGRAPHY_WITH_WEIGHTS_FILE, s.LOWEST_LEVEL_GEOGRAPHY)
    population.distribute_by_weights(weights_df, s.LOWEST_LEVEL_GEOGRAPHY, True)  # "home_loc" is a col in this file

    # If "home_loc" is NaN after this, assign a random location within its cell (lowest level of geography)
    population.assign_random_location()  # TODO: Assign within their cell!

    # Where the "hom_loc" is NaN in the df, take the value from "random_loc"
    population.df["home_loc"] = population.df["home_loc"].fillna(population.df["random_point"])

    # Add/edit household-specific rule-based attributes
    population.apply_row_wise_rules([rules.home_loc, rules.unique_household_id])

    # Add all data of each household (increases the number of rows)
    population.add_csv_data_on_id(s.ENHANCED_MID_FILE, id_column=s.HOUSEHOLD_MID_ID_COL,
                                  drop_duplicates_from_source=False)
    logger.info(f"Population df after adding detailed data: \n{population.df.head()}")

    # Add/edit rule-based attributes
    apply_me = [rules.unique_person_id, rules.unique_leg_id]
    population.apply_row_wise_rules(apply_me)

    population.df[s.CONNECTED_LEGS_COL] = population.df[s.CONNECTED_LEGS_COL].apply(h.convert_to_list)
    population.df[s.LIST_OF_CARS_COL] = population.df[s.LIST_OF_CARS_COL].apply(h.convert_to_list)

    # Temp for testing
    logger.debug(f"Number of rows after adding detailed data: {len(population.df)}")
    logger.debug(f"Number of nan leg ids after adding detailed data: {population.df[s.LEG_ID_COL].isna().sum()}")
    logger.debug(
        f"Number of legs called nan after adding detailed data: {len(population.df[population.df[s.LEG_ID_COL] == 'nan'])}")
    logger.debug(f"Number of empty leg ids after adding detailed data: {len(population.df[population.df[s.LEG_ID_COL] == ''])}")
    logger.debug(f"Number of unique leg ids after adding detailed data: {len(population.df[s.LEG_ID_COL].unique())}")

    # There might be people with 0 legs, meaning they didn't travel on the survey day - remove them.
    # All people where there are 0 legs for other reasons, e.g. because of missing data, must be removed in the inputs.
    # For MiD, all people with 0 legs can be assumed to not have travelled.

    population.df = population.df[population.df[s.LEG_ID_COL].notna()].reset_index()

    # Remove legs that are "regelmäßiger beruflicher Weg", commercial traffic (duration is marked as 70701.
    # Shouldn't be there anyway, but to make sure)
    population.filter_out_rows(s.LEG_DURATION_MINUTES_COL, [70701])
    # Remove imputed home legs, we don't use them here
    population.filter_out_rows(s.IMPUTED_LEG_COL, [1])

    # Convert time columns to datetime
    # population.convert_time_to_datetime([s.LEG_START_TIME_COL, s.LEG_END_TIME_COL])  # is already in that format from enhanced
    # population.convert_datetime_to_seconds([s.LEG_START_TIME_COL])

    population.df[s.LEG_START_TIME_COL] = pd.to_datetime(population.df[s.LEG_START_TIME_COL], format='ISO8601')
    population.df[s.LEG_END_TIME_COL] = pd.to_datetime(population.df[s.LEG_END_TIME_COL], format='ISO8601')

    population.change_last_leg_activity_to_home()

    population.assign_random_location()

    # Set coord_to to home_loc for all legs where activity is home
    population.df.loc[population.df[s.LEG_TO_ACTIVITY_COL] == s.ACTIVITY_HOME, s.COORD_TO_COL] = (
        population.df.loc)[population.df[s.LEG_TO_ACTIVITY_COL] == s.ACTIVITY_HOME, "home_loc"]

    # Set coord_from
    population.df = h.add_from_coord(population.df)

    # Assign each now known home coord a cell id using spatial join (make home cell col)
    population_gdf = gpd.GeoDataFrame(population.df, geometry="home_loc", crs="EPSG:25832")

    # Perform spatial join to find the cell each person is in
    cells_gdf = gpd.read_file(s.CAPA_CELLS_SHP_PATH)
    persons_with_cells = gpd.sjoin(population_gdf, cells_gdf, how="left", op="within").dropna(
        subset=["NAME"])
    # Rename the cell id column s.HOME_CELL_COL (from "NAME")
    persons_with_cells.rename(columns={'NAME': s.HOME_CELL_COL}, inplace=True)

    # Check if there are persons without a cell
    missing_cells_count = len(population_gdf) - len(persons_with_cells)
    if missing_cells_count > 0:
        logger.warning(f"{missing_cells_count} persons without a cell. They will be ignored.")

    # To_cell
    persons_with_cells.loc[persons_with_cells[s.LEG_TO_ACTIVITY_COL] == s.ACTIVITY_HOME, s.CELL_TO_COL] = (
        persons_with_cells.loc)[persons_with_cells[s.LEG_TO_ACTIVITY_COL] == s.ACTIVITY_HOME, s.HOME_CELL_COL]
    # From_cell
    persons_with_cells = h.add_from_cell(persons_with_cells)
    persons_with_cells = pd.DataFrame(persons_with_cells)
    # keep only needed cols
    # persons_with_cells = persons_with_cells[]

    # <test>
    persons_with_cells.to_csv("testdata/persons_with_cells.csv", index=False)

    persons_with_cells = pd.read_csv("testdata/persons_with_cells.csv")
    persons_with_cells[s.CONNECTED_LEGS_COL] = persons_with_cells[s.CONNECTED_LEGS_COL].apply(h.convert_to_list)
    persons_with_cells[s.LIST_OF_CARS_COL] = persons_with_cells[s.LIST_OF_CARS_COL].apply(h.convert_to_list)
    persons_with_cells[s.LEG_START_TIME_COL] = pd.to_datetime(persons_with_cells[s.LEG_START_TIME_COL], format='ISO8601')
    persons_with_cells[s.LEG_END_TIME_COL] = pd.to_datetime(persons_with_cells[s.LEG_END_TIME_COL], format='ISO8601')
    # </test>

    # With this information, plans can now be located
    # locator = ActivityLocator(persons_with_cells)
    # located_pop = locator.locate_activities()

    located_pop = parallel_locate_activities(persons_with_cells, num_processes=2)

    # All legs are located to cells. Find appropriate points (to_coord) within the cells for each leg.
    located_pop_with_points = h.assign_points(located_pop, s.CAPA_CELLS_SHP_PATH, "NAME")  # TODO: make better?

    population = PopulationFrameProcessor(located_pop_with_points)

    # population.impute_cars_in_household()
    population.list_cars_in_household()



    # apply_me = [rules.add_return_home_leg]  # Adds rows, so safe_apply=False
    # population.apply_group_wise_rules(apply_me, groupby_column="unique_person_id", safe_apply=False)
    logger.info(f"Population df after locating: \n{population.df.head()}")

    population.vary_times_by_household(s.UNIQUE_HH_ID_COL, [s.LEG_START_TIME_COL, s.LEG_END_TIME_COL])

    # Write plans to MATSim XML format
    pop_output_file = population.write_plans_to_matsim_xml()
    # Replace start_time with max_dur (temp workaround until matsim writer does max_dur)
    h.modify_text_file(pop_output_file, pop_output_file, 'start_time', 'max_dur')
    h.compress_to_gz(pop_output_file, delete_original=False)

    # Write stats
    population.write_stats()

    # Write dataframe to csv file if desired
    if write_to_file:
        population.df.to_csv(os.path.join(matsim_pipeline_setup.OUTPUT_DIR, s.POPULATION_ANALYSIS_OUTPUT_FILE), index=False)
        logger.info(f"Wrote population analysis output file to {s.POPULATION_ANALYSIS_OUTPUT_FILE}")

    logger.info(f"Finished popsim_to_matsim_plans pipeline")
    return


def process_chunk(chunk_df):
    locator = ActivityLocator(chunk_df)
    return locator.locate_activities()


def create_balanced_chunks(df, num_chunks):
    # Group the DataFrame by hh_id
    grouped = df.groupby(s.UNIQUE_HH_ID_COL)

    # Sort groups by size to help balance the chunks
    sorted_groups = sorted(grouped, key=lambda x: len(x[1]), reverse=True)

    # Initialize chunks
    chunks = [[] for _ in range(num_chunks)]

    # Distribute groups to chunks
    for i, group in enumerate(sorted_groups):
        chunks[i % num_chunks].append(group[1])

    # Combine group DataFrames in each chunk into a single DataFrame
    return [pd.concat(chunk, ignore_index=True) for chunk in chunks]


def parallel_locate_activities(df, num_processes=None):
    chunks = create_balanced_chunks(df, num_processes)

    # Create a multiprocessing pool and process chunks in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)

    # Combine results into a single DataFrame
    return pd.concat(results, ignore_index=True)


if __name__ == '__main__':
    output_dir = matsim_pipeline_setup.create_output_directory()
    try:

        popsim_to_matsim_plans_main()

    except Exception as e:
        if s.DUN_DUN_DUUUN:
            winsound.Beep(600, 500)
            time.sleep(0.1)
            winsound.Beep(500, 500)
            time.sleep(0.2)
            winsound.Beep(400, 1500)

        raise
else:
    output_dir = matsim_pipeline_setup.OUTPUT_DIR
