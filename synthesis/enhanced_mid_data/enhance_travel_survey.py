import os
import time

import winsound

from utils.population_frame_processor import PopulationFrameProcessor
from utils import matsim_pipeline_setup, rules
from utils import settings_values as s
from utils.logger import logging

logger = logging.getLogger(__name__)

# Set working dir
os.chdir(matsim_pipeline_setup.PROJECT_ROOT)
write_to_file = True if s.POPULATION_ANALYSIS_OUTPUT_FILE else False

add_all_columns = True


def enhance_travel_survey():
    """
    Creates a combined leg file from the MiD-survey household, person and trip files.
    Imputes values, adds attributes.
    :return:
    Notes:
        - The ids of households, persons and trips must be unique within the population sample (e.g. MiD)
        (MiD: H_ID, HP_ID, and a previously added HPW_ID for legs)
    """

    logger.info(f"Starting popsim_to_matsim_plans pipeline")

    # Create unique leg ids in the leg input file if necessary
    # h.create_unique_leg_ids()

    population = PopulationFrameProcessor()

    # Only load necessary household columns to not blow up the final file size
    # Otherwise, use_cols=None to load all columns
    if add_all_columns:
        household_cols_to_load = None
    else:
        household_cols_to_load = list(s.HH_COLUMNS.values())
        if s.HOUSEHOLD_MID_ID_COL not in household_cols_to_load:
            household_cols_to_load.append(s.HOUSEHOLD_MID_ID_COL)

    population.load_df_from_csv(s.MiD_HH_FILE, test_col=s.HOUSEHOLD_MID_ID_COL, use_cols=household_cols_to_load)

    logger.info(f"Population df after adding HH attributes: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    # Add/edit household-specific rule-based attributes
    apply_me = [rules.unique_household_id]
    population.apply_row_wise_rules(apply_me)
    logger.info(f"Population df after applying HH rules: \n{population.df.head()}")

    # Add persons to households (increases the number of rows)
    population.add_csv_data_on_id(s.MiD_PERSONS_FILE, [s.PERSON_ID_COL], id_column=s.HOUSEHOLD_MID_ID_COL,
                                  drop_duplicates_from_source=False)
    logger.info(f"Population df after adding persons: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    # Add person attributes
    if add_all_columns:
        person_cols_to_load = None
    else:
        person_cols_to_load = list(s.P_COLUMNS.values())
    if s.P_COLUMNS:
        population.add_csv_data_on_id(s.MiD_PERSONS_FILE, columns_to_add=person_cols_to_load, id_column=s.PERSON_ID_COL,
                                      drop_duplicates_from_source=True, delete_later=True)
    logger.info(f"Population df after adding P attributes: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    # Add/edit person-specific rule-based attributes
    apply_me = [rules.unique_person_id]  # rules.has_license_imputed
    population.apply_row_wise_rules(apply_me)
    logger.info(f"Population df after applying P row rules: \n{population.df.head()}")

    apply_me = []
    population.apply_group_wise_rules(apply_me, groupby_column="unique_household_id")
    logger.info(f"Population df after applying P group rules: \n{population.df.head()}")

    population.impute_license_status()

    # Add MiD-trips to people (increases the number of rows)
    population.add_csv_data_on_id(s.MiD_TRIPS_FILE, [s.LEG_ID_COL], id_column=s.PERSON_ID_COL,
                                  drop_duplicates_from_source=False)
    logger.info(f"Population df after adding trips: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    # Add trip attributes from MiD
    if add_all_columns:
        list_L_COLUMNS = None
    else:
        list_L_COLUMNS = list(s.L_COLUMNS.values())
        list_L_COLUMNS.append(s.LEG_NON_UNIQUE_ID_COL)
    population.add_csv_data_on_id(s.MiD_TRIPS_FILE, list_L_COLUMNS, id_column=s.LEG_ID_COL,
                                      drop_duplicates_from_source=True)
    logger.info(f"Population df after adding L attributes: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    # Remove legs that are "regelmäßiger beruflicher Weg" (duration is marked as 70701)
    population.filter_out_rows(s.LEG_DURATION_MINUTES_COL, [70701])

    # Convert time columns to datetime
    population.convert_time_to_datetime([s.LEG_START_TIME_COL, s.LEG_END_TIME_COL])
    # population.convert_datetime_to_seconds([s.LEG_START_TIME_COL])

    # Add/edit trip-specific rule-based attributes
    population.apply_row_wise_rules([rules.unique_leg_id])
    logger.info(f"Population df after applying L row rules: \n{population.df.head()}")

    population.add_from_activity()
    population.filter_home_to_home_legs()  # should be early (but after adding from activity)
    population.update_number_of_legs()

    population.calculate_activity_duration()
    population.activity_times_distribution_seconds()
    population.leg_duration_distribution_seconds()
    population.estimate_leg_times()
    # Recalculate after estimating leg times
    population.calculate_activity_duration()

    population.apply_group_wise_rules([rules.is_main_activity], groupby_column=s.UNIQUE_P_ID_COL)

    population.adjust_mode_based_on_age()
    population.adjust_mode_based_on_license()

    population.find_connected_legs()
    population.close_connected_leg_groups()
    population.mark_connected_persons_and_hhs()
    population.count_connected_legs_per_person()
    population.adjust_mode_based_on_connected_legs()

    population.add_return_home_leg()
    population.update_number_of_legs(s.NUMBER_OF_LEGS_INCL_IMPUTED_COL)  # Writes new column

    population.apply_group_wise_rules([rules.is_protagonist], groupby_column=s.UNIQUE_HH_ID_COL)

    population.mark_mirroring_main_activities()

    population.find_main_mode_to_main_act()
    population.find_home_to_main_time()

    population.update_activity_for_prot_legs()

    population.translate_modes()
    population.translate_activities()

    logger.info(f"Final population df: \n{population.df.head()}")

    # Write stats
    # population.write_stats()

    population.df.to_csv(os.path.join(matsim_pipeline_setup.OUTPUT_DIR, "enhanced_frame_final.csv"), index=False)
    logger.info(f"Wrote population output file.")

    logger.info(f"Finished enhance_travel_survey pipeline")
    return


if __name__ == '__main__':
    output_dir = matsim_pipeline_setup.create_output_directory()
    try:

        enhance_travel_survey()

    except Exception as e:
        if s.DUN_DUN_DUUUN:
            winsound.Beep(600, 500)
            time.sleep(0.1)
            winsound.Beep(500, 500)
            time.sleep(0.2)
            winsound.Beep(400, 1500)

        raise
