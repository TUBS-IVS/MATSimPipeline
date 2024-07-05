import os
import time

import winsound

from utils.population_frame_processor import MiDDataEnhancer
from utils import pipeline_setup, settings as s, helpers as h
from utils.logger import logging
from utils.stats_tracker import stats_tracker

logger = logging.getLogger(__name__)

# Set working dir
os.chdir(pipeline_setup.PROJECT_ROOT)


def enhance_travel_survey():
    """
    Creates a combined leg file from the MiD-survey household, person and trip files.
    Imputes values, adds attributes.
    :return:
    Notes:
        - The ids of households, persons and trips must be unique within the population sample (e.g. MiD)
        (MiD: H_ID, HP_ID, and a previously added HPW_ID for legs)
    """

    logger.info(f"Starting enhance_travel_survey module")

    # Create unique leg ids in the leg input file if necessary
    # h.create_unique_leg_ids()

    population = MiDDataEnhancer()
    population.load_df_from_csv(h.get_files(s.MiD_HH_FOLDER), test_col=s.HOUSEHOLD_MID_ID_COL)

    logger.info(f"Population df after adding HH attributes: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    population.df = h.generate_unique_household_id(population.df)

    # Add persons to households (increases the number of rows)
    population.add_csv_data_on_id(h.get_files(s.MiD_PERSONS_FOLDER), [s.PERSON_ID_COL], id_column=s.HOUSEHOLD_MID_ID_COL,
                                  drop_duplicates_from_source=False)
    logger.info(f"Population df after adding persons: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    # Add person attributes from MiD
    population.add_csv_data_on_id(h.get_files(s.MiD_PERSONS_FOLDER), id_column=s.PERSON_ID_COL,
                                  drop_duplicates_from_source=True, delete_later=True)
    logger.info(f"Population df after adding P attributes: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    population.df = h.generate_unique_person_id(population.df)
    population.impute_license_status()

    # Add MiD-trips to people (increases the number of rows)
    population.add_csv_data_on_id(h.get_files(s.MiD_TRIPS_FOLDER), [s.LEG_ID_COL], id_column=s.PERSON_ID_COL,
                                  drop_duplicates_from_source=False)
    logger.info(f"Population df after adding trips: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    # Add trip attributes from MiD
    population.add_csv_data_on_id(h.get_files(s.MiD_TRIPS_FOLDER), id_column=s.LEG_ID_COL,
                                  drop_duplicates_from_source=True)
    logger.info(f"Population df after adding L attributes: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    # Remove legs that are "regelmäßiger beruflicher Weg" (duration is marked as 70701)
    population.filter_out_rows(s.LEG_IS_RBW_COL, 1)

    # Translate MiD activities and modes to internal ones (only internal ones are to be used in processing)
    population.df = h.translate_column(population.df, s.ACT_MID_COL, s.ACT_TO_INTERNAL_COL, "activities", "mid",
                                       "internal")
    population.df = h.translate_column(population.df, s.MODE_MID_COL, s.MODE_INTERNAL_COL, "modes", "mid", "internal")

    # Convert time columns to datetime
    population.convert_time_to_datetime([s.LEG_START_TIME_COL, s.LEG_END_TIME_COL])
    # population.convert_datetime_to_seconds([s.LEG_START_TIME_COL])

    # Add/edit trip-specific rule-based attributes
    # population.apply_row_wise_rules([rules.unique_leg_id])
    # logger.info(f"Population df after applying L row rules: \n{population.df.head()}")
    population.df = h.generate_unique_leg_id(population.df)

    population.add_from_activity()
    population.filter_home_to_home_legs()  # should be early (but after adding from activity)
    population.update_number_of_legs()

    population.convert_minutes_to_seconds(s.LEG_DURATION_MINUTES_COL, s.LEG_DURATION_SECONDS_COL)
    population.convert_kilometers_to_meters(s.LEG_DISTANCE_KM_COL, s.LEG_DISTANCE_METERS_COL)

    population.calculate_activity_duration()
    # population.activity_times_distribution_seconds()
    # population.leg_duration_distribution_seconds()
    population.write_short_overview()
    population.estimate_leg_times()
    # Recalculate after estimating leg times
    population.calculate_activity_duration()

    # population.apply_group_wise_rules([rules.is_main_activity], groupby_column=s.UNIQUE_P_ID_COL)

    population.adjust_mode_based_on_age()
    population.adjust_mode_based_on_license()

    population.find_connected_legs()
    population.close_connected_leg_groups()
    population.mark_connected_persons_and_hhs()
    population.count_connected_legs_per_person()
    population.adjust_mode_based_on_connected_legs()

    population.add_return_home_leg()
    population.update_number_of_legs(s.NUMBER_OF_LEGS_INCL_IMPUTED_COL)  # Writes new column

    # population.apply_group_wise_rules([rules.is_protagonist], groupby_column=s.UNIQUE_HH_ID_COL)

    population.mark_mirroring_main_activities()

    population.find_main_mode_to_main_act()
    population.find_home_to_main_time_and_distance()

    population.update_activity_for_prot_legs()

    # Translate modes from internal to MiD (because modes may have been updated)
    population.df = h.translate_column(population.df, s.MODE_INTERNAL_COL, s.MODE_MID_COL, "modes", "internal", "mid")
    population.df = h.translate_column(population.df, s.ACT_TO_INTERNAL_COL, s.ACT_MID_COL, "activities", "internal",
                                       "mid")

    logger.info(f"Final population df: \n{population.df.head()}")

    # population.write_overview()

    population.df.to_csv(os.path.join(pipeline_setup.OUTPUT_DIR, s.ENHANCED_MID_FILE), index=False)
    logger.info(f"Wrote population output file.")

    stats_tracker.write_stats_to_file(os.path.join(pipeline_setup.OUTPUT_DIR, s.STATS_FILE))

    logger.info(f"Finished enhance_travel_survey pipeline")
    return


if __name__ == '__main__':
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
