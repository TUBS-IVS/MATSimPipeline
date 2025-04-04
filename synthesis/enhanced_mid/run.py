import sys
import logging
import time
import pandas as pd

from utils import column_names as s
from utils.helpers import Helpers
from utils.config import Config
from utils.logger import setup_logging
from utils.stats_tracker import StatsTracker
from synthesis.enhanced_mid.mid_data_enhancer import MiDDataEnhancer

def enhance_travel_survey(config: Config):
    """
    Creates a combined leg file from the MiD-survey household, person and trip files.
    Imputes values, adds attributes.
    :return:
    Notes:
        - The ids of households, persons and trips must be unique within the population sample (e.g. MiD)
        (MiD: H_ID, HP_ID, and a previously added HPW_ID for legs)
    """
    # TODO: Unique ids for legs, hh and persons should be added LATER and consistently in the syn pop step.
    # Create unique leg ids in the leg input file if necessary
    # df = h.read_csv(h.get_files(s.MiD_TRIPS_FOLDER))
    # df = h.create_leg_ids(df)
    # df.to_csv(h.get_files(s.MiD_TRIPS_FOLDER), index=False)

    population = MiDDataEnhancer(stats_tracker=stats_tracker, logger=logger, helpers=h, output_folder=output_folder)
    hh_folder = config.get("enhanced_mid.input.mid_hh_folder")
    population.load_df_from_csv(h.get_files(hh_folder), test_col=s.HOUSEHOLD_MID_ID_COL)

    logger.info(f"Population df after adding HH attributes: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    population.df = h.generate_unique_household_id(population.df)

    # Add persons to households (increases the number of rows)
    persons_folder = config.get("enhanced_mid.input.mid_persons_folder")
    population.add_csv_data_on_id(h.get_files(persons_folder), [s.PERSON_MID_ID_COL],
                                  id_column=s.HOUSEHOLD_MID_ID_COL,
                                  drop_duplicates_from_source=False)
    logger.info(f"Population df after adding persons: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    # Add person attributes from MiD
    population.add_csv_data_on_id(h.get_files(persons_folder), id_column=s.PERSON_MID_ID_COL,
                                  drop_duplicates_from_source=True, delete_later=True)
    logger.info(f"Population df after adding P attributes: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    population.df = h.generate_unique_person_id(population.df)
    population.impute_license_status()

    # Add MiD-trips to people (increases the number of rows)
    trips_folder = config.get("enhanced_mid.input.mid_trips_folder")
    population.add_csv_data_on_id(h.get_files(trips_folder), [s.LEG_ID_COL], id_column=s.PERSON_MID_ID_COL,
                                  drop_duplicates_from_source=False)
    logger.info(f"Population df after adding trips: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    # Add trip attributes from MiD
    population.add_csv_data_on_id(h.get_files(trips_folder), id_column=s.LEG_ID_COL,
                                  drop_duplicates_from_source=True)
    logger.info(f"Population df after adding L attributes: \n{population.df.head()}")
    population.check_for_merge_suffixes()

    # Remove legs that are "regelmäßiger beruflicher Weg" (duration is marked as 70701)
    population.filter_out_rows(s.LEG_IS_RBW_COL, [1])

    # Translate MiD activities and modes to internal ones (only internal ones are to be used in processing)
    population.df = h.translate_column(population.df, s.ACT_MID_COL, s.ACT_TO_INTERNAL_COL, "activities", "mid",
                                       "internal")
    population.df = h.translate_column(population.df, s.MODE_MID_COL, s.MODE_INTERNAL_COL, "modes", "mid", "internal")

    # Convert time columns to datetime
    population.convert_time_to_datetime([s.LEG_START_TIME_COL, s.LEG_END_TIME_COL])
    # population.convert_datetime_to_seconds([s.LEG_START_TIME_COL])

    # Add/edit trip-specific rule-based attributes
    # logger.info(f"Population df after applying L row rules: \n{population.df.head()}")
    population.df[s.LEG_NUMBER_COL] = pd.to_numeric(population.df[s.LEG_NUMBER_COL], errors='coerce')
    population.df = h.generate_unique_leg_id(population.df)

    population.add_from_activity()
    # population.filter_home_to_home_legs()  Home-to-home should be dealt with in placement itself. Also, removing it
    # here may cause issues because it may remove the first leg.
    # population.expand_home_to_home_legs() TODO: We may do this instead. After that, we must update number of legs, leg numbers and leg IDs.
    population.update_number_of_legs()

    population.convert_minutes_to_seconds(s.LEG_DURATION_MINUTES_COL, s.LEG_DURATION_SECONDS_COL)
    population.convert_kilometers_to_meters(s.LEG_DISTANCE_KM_COL, s.LEG_DISTANCE_METERS_COL)

    population.calculate_activity_duration()
    # population.activity_times_distribution_seconds()
    # population.leg_duration_distribution_seconds()
    population.write_short_overview()
    # population.estimate_leg_times()
    population.mark_bad_times_as_nan()
    population.correct_times()
    # Recalculate after estimating leg times
    population.calculate_activity_duration()

    population.mark_main_activity()

    population.adjust_mode_based_on_age()
    population.adjust_mode_based_on_license()

    population.find_connected_legs()
    population.close_connected_leg_groups()
    population.mark_connected_persons_and_hhs()
    population.count_connected_legs_per_person()
    population.adjust_mode_based_on_connected_legs()

    # population.add_return_home_leg() # needs to be reworked
    # population.update_number_of_legs(s.NUMBER_OF_LEGS_INCL_IMPUTED_COL)  # Writes new column

    population.mark_protagonist_leg()

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
    output_file = config.get("enhanced_mid.output.enhanced_mid_file")
    population.df.to_csv(output_file, index=False)
    logger.info(f"Wrote population output file: {output_file}")
    return

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: run.py <output_folder> <project_root> <config_yaml>")
        print("Absolute paths, folders must exist.")
        sys.exit(1)

    output_folder = sys.argv[1]  # Absolute path
    project_root = sys.argv[2]
    config_yaml = sys.argv[3]  # Just the filename
    step_name = "enhanced_mid"

    # Each step sets up its own logging, Config object, StatsTracker and Helpers
    config = Config(output_folder, project_root, config_yaml)
    config.resolve_paths()

    setup_logging(output_folder, console_level=config.get("settings.logging.console_level"),
                  file_level=config.get("settings.logging.file_level"))
    logger = logging.getLogger(step_name)

    stats_tracker = StatsTracker(output_folder)

    h = Helpers(project_root, output_folder, config, stats_tracker, logger)

    logger.info(f"Starting step {step_name}")
    time_start = time.time()

    # Run the MiD data enhancer
    enhance_travel_survey(config)

    time_end = time.time()
    time_step = time_end - time_start
    stats_tracker.log("runtimes.enhanced_mid_time", time_step)
    stats_tracker.write_stats()
    config.write_used_config()

    logger.info(f"Step {step_name} finished in {time_step:.2f} seconds.")
