import os
import multiprocessing as mp

import pandas as pd
import geopandas as gpd

from synthesis.location_assignment.main_activity_locator import ActivityLocator
from synthesis.location_assignment.population_frame_processor import PopulationProcessor
from utils import pipeline_setup, helpers as h
from utils import settings as s
from utils.logger import logging

logger = logging.getLogger(__name__)

# Set working dir
os.chdir(pipeline_setup.PROJECT_ROOT)
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
    population = PopulationProcessor()
    for file in s.EXPANDED_HOUSEHOLDS_FILES:
        population.load_df_from_csv(file, test_col=s.HOUSEHOLD_POPSIM_ID_COL, if_df_exists="concat")

    population.downsample_population(s.SAMPLE_SIZE)
    # population.apply_row_wise_rules([rules.unique_household_id])
    # population.generate_unique_household_id()
    population.df = h.generate_unique_household_id(population.df)

    # Distribute buildings to households (if PopSim assigned hhs to a larger geography)
    weights_df = h.read_csv(s.BUILDINGS_IN_LOWEST_GEOGRAPHY_WITH_WEIGHTS_FILE, s.LOWEST_LEVEL_GEOGRAPHY)
    population.distribute_by_weights(weights_df, s.LOWEST_LEVEL_GEOGRAPHY, True)  # s.HOME_LOC_COL is a col in this file

    # If s.HOME_LOC_COL is NaN after this, assign a random location within its cell (lowest level of geography)
    # This is the case for the surrounding region
    population.df = h.assign_points(population.df, s.REGION_WITHOUT_CITY_GPKG_FILE, s.LOWEST_LEVEL_GEOGRAPHY, "id", s.HOME_LOC_COL)

    population.assign_random_location()  # Creates random location in a new column "random_point" for fallback

    # Where the "hom_loc" is still NaN in the df, take the value from "random_point". This should never happen.
    nan_count_before = population.df[s.HOME_LOC_COL].isna().sum()
    population.df[s.HOME_LOC_COL] = population.df[s.HOME_LOC_COL].fillna(population.df["random_point"])
    nan_count_after = population.df[s.HOME_LOC_COL].isna().sum()

    logger.info(f"Number of NaN home_loc before assigning random points: {nan_count_before}")
    logger.info(f"Number of NaN home_loc after assigning random points: {nan_count_after}")

    # Turn the point string at home_loc into a shapely point (this is sometimes necessary)
    # population.apply_row_wise_rules([rules.home_loc])
    population.df[s.HOME_LOC_COL] = population.df[s.HOME_LOC_COL].apply(h.convert_to_point)

    # Add all data of each household (increases the number of rows)
    logger.info("Loading enhanced MiD data")
    population.add_csv_data_on_id(s.ENHANCED_MID_FILE, id_column=s.HOUSEHOLD_MID_ID_COL,
                                  drop_duplicates_from_source=False)
    logger.info(f"Population df after adding detailed data: \n{population.df.head()}")

    # Add unique ids for persons and legs
    # apply_me = [rules.unique_person_id, rules.unique_leg_id]
    # population.apply_row_wise_rules(apply_me)
    # population.generate_unique_person_id()
    population.df = h.generate_unique_person_id(population.df)
    # population.generate_unique_leg_id()
    population.df = h.generate_unique_leg_id(population.df)

    population.df[s.CONNECTED_LEGS_COL] = population.df[s.CONNECTED_LEGS_COL].apply(h.convert_to_list)
    population.df[s.LIST_OF_CARS_COL] = population.df[s.LIST_OF_CARS_COL].apply(h.convert_to_list)

    # There might be people with 0 legs, meaning they didn't travel on the survey day - remove them.
    # All people where there are 0 legs for other reasons, e.g. because of missing data, must be removed in the inputs.
    # For MiD, all people with 0 legs can be assumed to not have travelled.
    population.df = population.df[population.df[s.LEG_ID_COL].notna()].reset_index(drop=True)

    # Remove legs that are "regelmäßiger beruflicher Weg", commercial traffic (duration is marked as 70701.
    # Shouldn't be there anyway, but to make sure)
    population.filter_out_rows(s.LEG_DURATION_MINUTES_COL, [70701])
    # Remove imputed home legs, we don't use them here
    population.filter_out_rows(s.IS_IMPUTED_LEG_COL, [1])

    # Convert time columns to datetime
    # population.convert_time_to_datetime([s.LEG_START_TIME_COL, s.LEG_END_TIME_COL])  # is already in that format from enhanced
    # population.convert_datetime_to_seconds([s.LEG_START_TIME_COL])

    population.df[s.LEG_START_TIME_COL] = pd.to_datetime(population.df[s.LEG_START_TIME_COL], format='ISO8601')
    population.df[s.LEG_END_TIME_COL] = pd.to_datetime(population.df[s.LEG_END_TIME_COL], format='ISO8601')

    population.change_last_leg_activity_to_home()

    # Set coord_to to home_loc for all legs where activity is home
    population.df.loc[population.df[s.ACT_TO_INTERNAL_COL] == s.ACT_HOME, s.COORD_TO_COL] = (
        population.df.loc)[population.df[s.ACT_TO_INTERNAL_COL] == s.ACT_HOME, s.HOME_LOC_COL]

    # Set coord_from
    population.df = h.add_from_coord(population.df)

    # Assign each now known home coord a cell id using spatial join (make home cell col)
    population_gdf = gpd.GeoDataFrame(population.df, geometry=s.HOME_LOC_COL, crs="EPSG:25832")
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
    persons_with_cells.loc[persons_with_cells[s.ACT_TO_INTERNAL_COL] == s.ACT_HOME, s.CELL_TO_COL] = (
        persons_with_cells.loc)[persons_with_cells[s.ACT_TO_INTERNAL_COL] == s.ACT_HOME, s.HOME_CELL_COL]
    # From_cell
    persons_with_cells = h.add_from_cell(persons_with_cells)
    persons_with_cells = pd.DataFrame(persons_with_cells)

    # # # <test>  Kept for simpler "integration" testing
    # # persons_with_cells.to_csv("testdata/persons_with_cells.csv", index=False)
    #
    # persons_with_cells = pd.read_csv("testdata/persons_with_cells.csv")
    # persons_with_cells[s.CONNECTED_LEGS_COL] = persons_with_cells[s.CONNECTED_LEGS_COL].apply(h.convert_to_list)
    # persons_with_cells[s.LIST_OF_CARS_COL] = persons_with_cells[s.LIST_OF_CARS_COL].apply(h.convert_to_list)
    # persons_with_cells[s.LEG_START_TIME_COL] = pd.to_datetime(persons_with_cells[s.LEG_START_TIME_COL], format='ISO8601')
    # persons_with_cells[s.LEG_END_TIME_COL] = pd.to_datetime(persons_with_cells[s.LEG_END_TIME_COL], format='ISO8601')
    # population = PopulationFrameProcessor(persons_with_cells)
    # population.apply_row_wise_rules([rules.home_loc])
    # persons_with_cells = population.df
    # # </test>

    # With this information, plans can now be located
    persons_with_cells.sort_values(by=[s.UNIQUE_LEG_ID_COL], inplace=True)
    persons_with_cells.reset_index(inplace=True, drop=True)

    # Single-processing
    # locator = ActivityLocator(persons_with_cells)
    # located_pop = locator.locate_activities()

    # Multi-processing
    located_pop = parallel_locate_activities(persons_with_cells, num_processes=20)

    # All legs are located to cells. Find appropriate points (to_coord) within the cells for each leg.
    located_pop_with_points = h.assign_points(located_pop, s.CAPA_CELLS_SHP_PATH, s.CELL_TO_COL, "NAME", s.COORD_TO_COL)

    # Check if there are legs without a coord_to
    missing_coords_count = len(located_pop_with_points[located_pop_with_points[s.COORD_TO_COL].isna()])
    if missing_coords_count > 0:
        logger.warning(f"{missing_coords_count} legs without a coord_to. Assigning their home_loc.")
        located_pop_with_points.loc[located_pop_with_points[s.COORD_TO_COL].isna(), s.COORD_TO_COL] = (
            located_pop_with_points.loc)[located_pop_with_points[s.COORD_TO_COL].isna(), s.HOME_LOC_COL]
    else:
        logger.info("All legs have a coord_to.")

    population = PopulationProcessor(located_pop_with_points)

    # population.impute_cars_in_household()
    # population.list_cars_in_household()

    logger.info(f"Population df after locating: \n{population.df.head()}")

    population.vary_times_by_household(s.UNIQUE_HH_ID_COL, [s.LEG_START_TIME_COL, s.LEG_END_TIME_COL])
    population.df[s.COORD_TO_COL] = population.df[s.COORD_TO_COL].apply(h.convert_to_point)

    # Write plans to MATSim XML format
    population.df.sort_values(by=[s.UNIQUE_LEG_ID_COL], inplace=True, ignore_index=True)
    pop_output_file = population.write_plans_to_matsim_xml()
    # Replace start_time with max_dur (temp workaround until matsim writer does max_dur)
    h.modify_text_file(pop_output_file, pop_output_file, 'start_time', 'max_dur')
    h.compress_to_gz(pop_output_file, delete_original=False)
    # Write households to MATSim XML format
    population.write_households_to_matsim_xml()

    # Write stats
    # population.write_stats()

    # Write dataframe to csv file if desired
    if write_to_file:
        population.df.to_csv(os.path.join(pipeline_setup.OUTPUT_DIR, s.POPULATION_ANALYSIS_OUTPUT_FILE), index=False)
        logger.info(f"Wrote population analysis output file to {s.POPULATION_ANALYSIS_OUTPUT_FILE}")

    logger.info(f"Finished popsim_to_matsim_plans pipeline")
    


def process_chunk(chunk_df):
    locator = ActivityLocator(chunk_df)
    return locator.locate_activities()


def create_balanced_chunks(df, num_chunks):
    grouped = df.groupby(s.UNIQUE_HH_ID_COL)
    sorted_groups = sorted(grouped, key=lambda x: len(x[1]), reverse=True)

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
    output_dir = pipeline_setup.create_output_directory()
    popsim_to_matsim_plans_main()

else:
    output_dir = pipeline_setup.OUTPUT_DIR
