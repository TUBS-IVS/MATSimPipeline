import os
import time

import winsound

from pipelines.common import helpers as h
from pipelines.common import rules
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

    # Create unique leg ids in the leg input file if necessary
    h.create_unique_leg_ids()

    # Load data from PopSim, concat different PopSim results if necessary
    # Lowest level of geography must be named the same in all input files, if there are multiple
    population = PopulationFrameProcessor()
    population.id_column = s.HOUSEHOLD_MID_ID_COL
    for csv_path in s.EXPANDED_HOUSEHOLDS_FILES:
        population.load_df_from_csv(csv_path, "concat")

    population.downsample_population(s.SAMPLE_SIZE)

    # Add household attributes from MiD
    if s.HH_COLUMNS:
        population.add_csv_data_on_id(s.MiD_HH_FILE, s.HH_COLUMNS, id_column=s.HOUSEHOLD_MID_ID_COL,
                                      drop_duplicates_from_source=True, delete_later=True)
    logger.info(f"Population df after adding HH attributes: \n{population.df.head()}")

    # Distribute buildings to households (if PopSim assigned hhs to a larger geography)
    weights_df = h.read_csv(s.BUILDINGS_IN_LOWEST_GEOGRAPHY_WITH_WEIGHTS_FILE, s.LOWEST_LEVEL_GEOGRAPHY)
    population.distribute_by_weights(weights_df, s.LOWEST_LEVEL_GEOGRAPHY, True)

    # Add/edit household-specific rule-based attributes
    apply_me = [rules.unique_household_id, rules.home_loc]
    population.apply_row_wise_rules(apply_me)
    logger.info(f"Population df after applying HH rules: \n{population.df.head()}")

    # Add people to households (increases the number of rows)
    population.add_csv_data_on_id(s.MiD_PERSONS_FILE, [s.PERSON_ID_COL], id_column=s.HOUSEHOLD_MID_ID_COL,
                                  drop_duplicates_from_source=False)
    logger.info(f"Population df after adding persons: \n{population.df.head()}")

    # Add person attributes from MiD
    if s.P_COLUMNS:
        population.add_csv_data_on_id(s.MiD_PERSONS_FILE, s.P_COLUMNS, id_column=s.PERSON_ID_COL,
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
    population.add_csv_data_on_id(s.MiD_TRIPS_FILE, [s.LEG_ID_COL], id_column=s.PERSON_ID_COL,
                                  drop_duplicates_from_source=False)
    logger.info(f"Population df after adding trips: \n{population.df.head()}")

    # Temp for testing
    logger.debug(f"Number of rows after adding L attributes: {len(population.df)}")
    logger.debug(f"Number of nan leg ids after adding L attributes: {population.df[s.LEG_ID_COL].isna().sum()}")
    logger.debug(
        f"Number of legs called nan after adding L attributes: {len(population.df[population.df[s.LEG_ID_COL] == 'nan'])}")
    logger.debug(f"Number of empty leg ids after adding L attributes: {len(population.df[population.df[s.LEG_ID_COL] == ''])}")
    logger.debug(f"Number of unique leg ids after adding L attributes: {len(population.df[s.LEG_ID_COL].unique())}")

    # There might be people with 0 legs, meaning they didn't travel on the survey day - remove them.
    # All people where there are 0 legs for other reasons, e.g. because of missing data, must be removed in the inputs.
    # For MiD, all people with 0 legs can be assumed to not have travelled.

    population.df = population.df[population.df[s.LEG_ID_COL].notna()].reset_index()

    # Add trip attributes from MiD
    if s.L_COLUMNS:
        list_L_COLUMNS = list(s.L_COLUMNS.values())
        list_L_COLUMNS.append(s.LEG_NON_UNIQUE_ID_COL)
        population.add_csv_data_on_id(s.MiD_TRIPS_FILE, list_L_COLUMNS, id_column=s.LEG_ID_COL,
                                      drop_duplicates_from_source=True, delete_later=True)
    logger.info(f"Population df after adding L attributes: \n{population.df.head()}")

    # Remove legs that are "regelmäßiger beruflicher Weg" (duration is marked as 70701)
    population.filter_out_rows(s.LEG_DURATION_MINUTES_COL, [70701])

    # Convert time columns to datetime
    population.convert_time_to_datetime([s.LEG_START_TIME_COL, s.LEG_END_TIME_COL])
    # population.convert_datetime_to_seconds([s.LEG_START_TIME_COL])

    # Add/edit trip-specific rule-based attributes
    apply_me = [rules.unique_leg_id]
    population.apply_row_wise_rules(apply_me)
    logger.info(f"Population df after applying L row rules: \n{population.df.head()}")

    population.calculate_activity_duration()
    population.estimate_leg_times()
    # Recalculate after estimating leg times
    population.calculate_activity_duration()

    apply_me = [rules.is_main_activity]
    # population.apply_group_wise_rules(apply_me, groupby_column="unique_person_id")

    apply_me = [rules.connected_activities]
    # population.apply_group_wise_rules(apply_me, groupby_column="unique_household_id")

    population.adjust_mode_based_on_age()
    population.change_last_leg_activity_to_home()
    population.assign_random_location()
    population.translate_modes()
    population.translate_activities()

    # apply_me = [rules.add_return_home_leg]  # Adds rows, so safe_apply=False
    # population.apply_group_wise_rules(apply_me, groupby_column="unique_person_id", safe_apply=False)
    logger.info(f"Population df after applying L group rules: \n{population.df.head()}")

    # Remove cols that were used by rules, to keep the df clean
    # population.remove_columns_marked_for_later_deletion()

    #population.vary_times_by_person("unique_person_id", [s.LEG_START_TIME_COL, s.LEG_END_TIME_COL])

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
