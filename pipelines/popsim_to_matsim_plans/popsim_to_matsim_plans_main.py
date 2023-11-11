import os

from pipelines.common import rules
from pipelines.popsim_to_matsim_plans.population_frame_processor import PopulationFrameProcessor
from utils import matsim_pipeline_setup
from utils.logger import logging

logger = logging.getLogger(__name__)

# Set working dir
os.chdir(matsim_pipeline_setup.PROJECT_ROOT)

# Load settings
settings = matsim_pipeline_setup.load_yaml_config('settings.yaml')

EXPANDED_HOUSEHOLDS_FILES: list = settings['expanded_households_files']
MiD_HH_FILE = settings['mid_hh_file']
MiD_PERSONS_FILE = settings['mid_persons_file']
MiD_TRIPS_FILE = settings['mid_trips_file']

ID_COLUMNS = settings['id_columns']
HH_COLUMNS = settings['hh_columns']
P_COLUMNS = settings['person_columns']
L_COLUMNS = settings['leg_columns']

HOUSEHOLD_ID_COLUMN = ID_COLUMNS['household_mid_id_column']
PERSON_ID_COLUMN = ID_COLUMNS['person_id_column']
LEG_ID_COLUMN = ID_COLUMNS['leg_id_column']

LEG_START_TIME_COL = L_COLUMNS['leg_start_time']
LEG_END_TIME_COL = L_COLUMNS['leg_end_time']
LEG_DURATION_MINUTES_COL = L_COLUMNS['leg_duration_minutes']

LOWEST_LEVEL_GEOGRAPHY = settings['lowest_level_geography']

POPULATION_ANALYSIS_OUTPUT_FILE = settings['population_analysis_output_file']
write_to_file = True if POPULATION_ANALYSIS_OUTPUT_FILE else False


def popsim_to_matsim_plans_main():
    """
    Main function for the popsim_to_matsim_plans pipeline.
    :return:
    Notes:
        - The ids of households, persons and trips must be unique within the population sample (e.g. MiD)
        (MiD: H_ID, HP_ID, and a previously added HPW_ID for legs)
    """
    # Create unique leg ids in the leg input file if necessary
    #matsim_pipeline_setup.create_unique_leg_ids()

    # Load data from PopSim, concat different PopSim results if necessary
    # Lowest level of geography must be named the same in all input files, if there are multiple
    population = PopulationFrameProcessor()
    population.id_column = HOUSEHOLD_ID_COLUMN
    for csv_path in EXPANDED_HOUSEHOLDS_FILES:
        population.load_df_from_csv(csv_path, "concat")


    # Add household attributes from MiD
    if HH_COLUMNS:
        population.add_csv_data_on_id(MiD_HH_FILE, HH_COLUMNS, id_column=HOUSEHOLD_ID_COLUMN,
                                      drop_duplicates_from_source=True, delete_later=True)
    logger.info(f"Population df after adding HH attributes: \n{population.df.head()}")

    # Add/edit household-specific rule-based attributes
    apply_me = [rules.unique_household_id]
    population.apply_row_wise_rules(apply_me)
    logger.info(f"Population df after applying HH rules: \n{population.df.head()}")

    # Distribute buildings to households (if PopSim assigned them to a larger geography)
    # buildings_in_lowest_geography_with_weights_df =
    # population.distribute_by_weights(buildings_in_lowest_geography_with_weights_df, LOWEST_LEVEL_GEOGRAPHY)

    # Add people to households (increases the number of rows)
    population.add_csv_data_on_id(MiD_PERSONS_FILE, [PERSON_ID_COLUMN], id_column=HOUSEHOLD_ID_COLUMN,
                                  drop_duplicates_from_source=False)
    logger.info(f"Population df after adding persons: \n{population.df.head()}")

    # Add person attributes from MiD
    if P_COLUMNS:
        population.add_csv_data_on_id(MiD_PERSONS_FILE, P_COLUMNS, id_column=PERSON_ID_COLUMN,
                                      drop_duplicates_from_source=True, delete_later=True)
    logger.info(f"Population df after adding P attributes: \n{population.df.head()}")

    # Add/edit person-specific rule-based attributes
    apply_me = [rules.unique_person_id]  # rules.has_license_imputed
    population.apply_row_wise_rules(apply_me)
    logger.info(f"Population df after applying P row rules: \n{population.df.head()}")

    apply_me = []
    population.apply_group_wise_rules(apply_me, groupby_column="unique_household_id")
    logger.info(f"Population df after applying P group rules: \n{population.df.head()}")

    # Add MiD-trips to people (increases the number of rows)
    population.add_csv_data_on_id(MiD_TRIPS_FILE, [LEG_ID_COLUMN], id_column=PERSON_ID_COLUMN,
                                  drop_duplicates_from_source=False)
    logger.info(f"Population df after adding trips: \n{population.df.head()}")

    # Temp for testing
    logger.debug(f"Number of rows after adding L attributes: {len(population.df)}")
    logger.debug(f"Number of nan leg ids after adding L attributes: {population.df[LEG_ID_COLUMN].isna().sum()}")
    logger.debug(f"Number of legs called nan after adding L attributes: {len(population.df[population.df[LEG_ID_COLUMN] == 'nan'])}")
    logger.debug(f"Number of empty leg ids after adding L attributes: {len(population.df[population.df[LEG_ID_COLUMN] == ''])}")
    logger.debug(f"Number of unique leg ids after adding L attributes: {len(population.df[LEG_ID_COLUMN].unique())}")

    # There might be people with 0 legs, meaning they didn't travel on the survey day - remove them.
    # All people where there are 0 legs for other reasons, e.g. because of missing data, must be removed in the inputs.
    # For MiD, all people with 0 legs can be assumed to not have travelled.

    population.df = population.df[population.df[LEG_ID_COLUMN].notna()].reset_index()

    # Add trip attributes from MiD
    if L_COLUMNS:
        population.add_csv_data_on_id(MiD_TRIPS_FILE, L_COLUMNS, id_column=LEG_ID_COLUMN,
                                      drop_duplicates_from_source=True, delete_later=True)
    logger.info(f"Population df after adding L attributes: \n{population.df.head()}")

    # Convert time columns to datetime
    population.convert_time_columns_to_datetime([LEG_START_TIME_COL, LEG_END_TIME_COL])

    # Add/edit trip-specific rule-based attributes
    apply_me = [rules.unique_leg_id]
    population.apply_row_wise_rules(apply_me)
    logger.info(f"Population df after applying L row rules: \n{population.df.head()}")

    # apply_me = [rules.activity_duration_in_minutes, rules.is_main_activity]
    # population.apply_group_wise_rules(apply_me, groupby_column="unique_person_id")

    apply_me = [rules.connected_activities]
    population.apply_group_wise_rules(apply_me, groupby_column="unique_household_id")

    apply_me = [rules.add_return_home_leg]  # Adds rows, so safe_apply=False
    population.apply_group_wise_rules(apply_me, groupby_column="unique_person_id", safe_apply=False)
    logger.info(f"Population df after applying L group rules: \n{population.df.head()}")

    # Remove cols that were used by rules, to keep the df clean
    # population.remove_columns_marked_for_later_deletion()

    #

    # Write plans to MATSim XML format
    population.write_plans_to_matsim_xml()

    # Write dataframe to csv file if desired
    if write_to_file:
        population.df.to_csv(POPULATION_ANALYSIS_OUTPUT_FILE, index=False)


#
if __name__ == '__main__':
    output_dir = matsim_pipeline_setup.create_output_directory()
    popsim_to_matsim_plans_main()
else:
    output_dir = matsim_pipeline_setup.OUTPUT_DIR
