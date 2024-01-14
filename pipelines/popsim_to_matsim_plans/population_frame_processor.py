import itertools
import os.path
from collections import deque
from datetime import timedelta

import geopandas as gpd
import matsim.writers
import numpy as np
import pandas as pd

from pipelines.common import helpers as h
from pipelines.common import rules
from pipelines.common.data_frame_processor import DataFrameProcessor
from utils import matsim_pipeline_setup
from utils import settings_values as s
from utils.logger import logging

logger = logging.getLogger(__name__)


class PopulationFrameProcessor(DataFrameProcessor):
    """
    A class to process population dataframes.
    Contains methods that are specific to population dataframes and need full access to the dataframe and/or
    are too complex to be implemented as rules.
    """

    def __init__(self, df: pd.DataFrame = None, id_column: str = None):
        super().__init__(df, id_column)
        self.sf = h.SlackFactors(s.SLACK_FACTORS_FILE)

    def distribute_by_weights(self, weights_df: pd.DataFrame, cell_id_col: str, cut_missing_ids: bool = False):
        result = h.distribute_by_weights(self.df, weights_df, cell_id_col, cut_missing_ids)
        self.df = self.df.merge(result[[s.UNIQUE_HH_ID_COL, 'home_loc']], on=s.UNIQUE_HH_ID_COL, how='left')


    def write_plans_to_matsim_xml(self):
        """
        Write to MATSim xml.gz directly from the dataframe.
        The design of this method decides which data from the population frame is written and which is not.
        All trips are assumed to start at home.
        All trips should end at home. If not, we warn the user but use the given activity.
        """
        logger.info("Writing plans to MATSim xml...")
        logger.info("All trips are assumed to start at home.")

        self.df.reset_index(drop=True, inplace=True)

        output_file = os.path.join(matsim_pipeline_setup.OUTPUT_DIR, "population.xml")
        with open(output_file, 'wb+') as f_write:
            writer = matsim.writers.PopulationWriter(f_write)

            writer.start_population(attributes={"coordinateReferenceSystem": "UTM-32N"})

            for _, group in self.df.groupby([s.UNIQUE_P_ID_COL]):
                writer.start_person(group[s.UNIQUE_P_ID_COL].iloc[0])
                writer.start_plan(selected=True)

                # Add home activity
                writer.add_activity(
                    type="home",
                    x=group['home_loc'].iloc[0].x, y=group['home_loc'].iloc[0].y,
                    end_time=h.seconds_from_datetime(group[s.LEG_START_TIME_COL].iloc[0]))
                # One row in the df contains the leg and the following activity
                for idx, row in group.iterrows():
                    writer.add_leg(mode=row['mode_translated_string'])
                    if not pd.isna(row[s.ACT_DUR_SECONDS_COL]):
                        # Create an own activity type for each duration (for correct matsim scoring)
                        # Rounding must fit the matsim config
                        max_dur: int = round(row[s.ACT_DUR_SECONDS_COL] / 600) * 600
                        writer.add_activity(
                            type=f"{row['activity_translated_string']}_{max_dur}",
                            x=row[s.COORD_TO_COL].x, y=row[s.COORD_TO_COL].y,
                            # The writer expects seconds. Also, we mean max_dur here, but the writer doesn't have that yet.
                            start_time=max_dur)
                    else:
                        # No time for the last activity
                        writer.add_activity(
                            type=row['activity_translated_string'],
                            x=row[s.COORD_TO_COL].x, y=row[s.COORD_TO_COL].y)

                writer.end_plan()
                writer.end_person()

            writer.end_population()
        logger.info(f"Wrote plans to MATSim xml: {output_file}")
        return output_file

    def write_households_to_matsim_xml(self):
        logger.info("Writing households to MATSim xml...")
        output_file = os.path.join(matsim_pipeline_setup.OUTPUT_DIR, "households.xml")
        with open(output_file, 'wb+') as f_write:
            households_writer = matsim.writers.HouseholdsWriter(f_write)
            households_writer.start_households()

            for _, hh in self.df.groupby([s.UNIQUE_HH_ID_COL]):
                household_id = hh[s.UNIQUE_HH_ID_COL].iloc[0]
                person_ids = hh[s.UNIQUE_P_ID_COL].unique().tolist()
                vehicle_ids = hh[s.LIST_OF_CARS_COL].iloc[0]

                households_writer.start_household(household_id)
                households_writer.add_members(person_ids)
                # if vehicle_ids:
                #     households_writer.add_vehicles(vehicle_ids)
                households_writer.end_household()

            households_writer.end_households()

    def write_facilities_to_matsim_xml(self, facilities_df: pd.DataFrame):
        with open(os.path.join(matsim_pipeline_setup.OUTPUT_DIR, "facilities.xml"), 'wb+') as f_write:
            facilities_writer = matsim.writers.FacilitiesWriter(f_write)
            facilities_writer.start_facilities()

            for row in facilities_df.itertuples():
                # Using itertuples() with getattr() is much faster than iterrows(), even if it's a bit uglier
                facility_id = getattr(row, s.FACILITY_ID_COL)
                x = getattr(row, s.FACILITY_X_COL)
                y = getattr(row, s.FACILITY_Y_COL)
                activities = list(getattr(row, s.FACILITY_ACTIVITIES_COL))

                facilities_writer.start_facility(facility_id, x, y)

                for activity in activities:
                    facilities_writer.add_activity(activity)

                facilities_writer.end_facility()

            facilities_writer.end_facilities()

    def write_vehicles_to_matsim_xml(self):
        logger.info("Writing vehicles to MATSim xml...")
        output_file = os.path.join(matsim_pipeline_setup.OUTPUT_DIR, "vehicles.xml")
        with open(output_file, 'wb+') as f_write:
            vehicle_writer = matsim.writers.VehiclesWriter(f_write)
            vehicle_writer.start_vehicle_definitions()

            vehicle_id = "car"
            length = 7.5
            width = 1.0
            pce = 1.0
            network_mode = "car"

            vehicle_writer.add_vehicle_type(vehicle_id=vehicle_id, length=length, width=width, pce=pce,
                                            network_mode=network_mode)

            for _, hh in self.df.groupby([s.UNIQUE_HH_ID_COL]):
                vehicle_ids: list = hh[s.LIST_OF_CARS_COL].iloc[0]
                if vehicle_ids:
                    for vehicle_id in vehicle_ids:
                        vehicle_writer.add_vehicle(vehicle_id=vehicle_id, vehicle_type="car")

            vehicle_writer.end_vehicle_definitions()

    def change_last_leg_activity_to_home(self) -> None:
        """
        Change the target activity of the last leg to home. Alternative to add_return_home_leg().
        Assumes LEG_ID is ascending in order of legs (which it is in MiD and should be in other datasets).
        """
        logger.info("Changing last leg activity to home...")
        self.df = self.df.sort_values(by=[s.UNIQUE_LEG_ID_COL])

        is_last_leg = self.df[s.PERSON_ID_COL].ne(self.df[s.PERSON_ID_COL].shift(-1))

        number_of_rows_to_change = len(self.df[is_last_leg & (self.df[s.LEG_TO_ACTIVITY_COL] != s.ACTIVITY_HOME)])

        self.df.loc[is_last_leg, s.LEG_TO_ACTIVITY_COL] = s.ACTIVITY_HOME
        self.df.loc[is_last_leg, 'activity_translated_string'] = "home"
        # We also need to remove markers for main or mirroring main activities, because home is never main
        self.df.loc[is_last_leg, s.IS_MAIN_ACTIVITY_COL] = 0
        self.df.loc[is_last_leg, s.MIRRORS_MAIN_ACTIVITY_COL] = 0
        # This means there might be some persons with no main activity. This is not a problem for the current model.
        logger.info(f"Changed last leg activity to home for {number_of_rows_to_change} of {len(self.df)} rows.")

    def adjust_mode_based_on_age(self):
        """
        Change the mode of transportation from car to ride if age < 17.
        """
        logger.info("Adjusting mode based on age...")
        conditions = (self.df[s.LEG_MAIN_MODE_COL] == s.MODE_CAR) & (self.df[s.PERSON_AGE_COL] < 17)
        self.df.loc[conditions, s.LEG_MAIN_MODE_COL] = s.MODE_RIDE
        logger.info(f"Adjusted mode based on age for {conditions.sum()} of {len(self.df)} rows.")

    def adjust_mode_based_on_license(self):
        """
        Change the mode of transportation from car to ride if person has no license.
        """
        logger.info("Adjusting mode based on license...")
        conditions = (self.df[s.LEG_MAIN_MODE_COL] == s.MODE_CAR) & (self.df["imputed_license"] == s.LICENSE_NO)
        self.df.loc[conditions, s.LEG_MAIN_MODE_COL] = s.MODE_RIDE
        logger.info(f"Adjusted mode based on license for {conditions.sum()} of {len(self.df)} rows.")

    def adjust_mode_based_on_connected_legs(self):
        """
        Change the mode of transportation from undefined to ride if the leg is connected to other legs.
        This works because connection analysis only matches undefined legs to car legs.
        """
        logger.info("Adjusting mode based on connected legs...")
        conditions = (self.df[s.LEG_MAIN_MODE_COL] == s.MODE_UNDEFINED) & (isinstance(self.df[s.CONNECTED_LEGS_COL], list))
        self.df.loc[conditions, s.LEG_MAIN_MODE_COL] = s.MODE_RIDE
        logger.info(f"Adjusted mode based on connected legs for {conditions.sum()} of {len(self.df)} rows.")

    def calculate_activity_duration(self):
        """
        Calculate the time between the end of one leg and the start of the next leg seconds.
        :return:
        """
        logger.info("Calculating activity duration...")
        self.df.sort_values(by=['unique_household_id', s.PERSON_ID_COL, s.LEG_NON_UNIQUE_ID_COL], inplace=True,
                            ignore_index=True)

        # Group by person and calculate the time difference within each group
        self.df[s.ACT_DUR_SECONDS_COL] = self.df.groupby(s.PERSON_ID_COL)[s.LEG_START_TIME_COL].shift(-1) - self.df[
            s.LEG_END_TIME_COL]

        self.df[s.ACT_DUR_SECONDS_COL] = self.df[s.ACT_DUR_SECONDS_COL].dt.total_seconds()
        self.df[s.ACT_DUR_SECONDS_COL] = pd.to_numeric(self.df[s.ACT_DUR_SECONDS_COL], downcast='integer',
                                                       errors='coerce')

        # Set the activity time of the last leg to None
        is_last_leg = self.df["unique_person_id"] != self.df["unique_person_id"].shift(-1)
        self.df.loc[is_last_leg, s.ACT_DUR_SECONDS_COL] = None
        logger.info(f"Calculated activity duration in secs.")

    def assign_random_location(self):
        """
        Assign a random location to each activity.
        :return:
        """
        gdf = gpd.read_file(s.SHAPE_BOUNDARY_FILE)
        polygon = h.find_outer_boundary(gdf)
        self.df['random_point'] = self.df.apply(lambda row: h.random_point_in_polygon(polygon), axis=1)

    def translate_modes(self):
        """
        Translate the modes from the MiD codes to the MATSim strings.
        Recommended to do this just before writing to MATSim xml.
        :return:
        """
        logger.info(f"Translating modes...")
        defined_modes = [s.MODE_CAR, s.MODE_PT, s.MODE_RIDE, s.MODE_BIKE, s.MODE_WALK, s.MODE_UNDEFINED]
        count_non_matching = (~self.df[s.LEG_MAIN_MODE_COL].isin(defined_modes)).sum()
        count_nan = self.df[s.LEG_MAIN_MODE_COL].isna().sum()
        count_non_leg = self.df[s.LEG_ID_COL].isna().sum()
        if count_non_matching > 0:
            logger.warning(f"{count_non_matching} rows have a mode that is not in the defined modes."
                           f"{count_nan} rows have no mode."
                           f"{count_non_leg} rows have no leg.")
        mode_translation = {
            s.MODE_CAR: "car",
            s.MODE_PT: "pt",
            s.MODE_RIDE: "ride",
            s.MODE_BIKE: "bike",
            s.MODE_WALK: "walk",
        }
        self.df['mode_translated_string'] = self.df[s.LEG_MAIN_MODE_COL].map(mode_translation)
        logger.info(f"Translated modes.")

    def translate_activities(self):
        """
        Translate the activities from the MiD codes to the MATSim strings.
        :return:
        """
        logger.info(f"Translating activities...")
        activity_translation = {
            s.ACTIVITY_WORK: "work",
            s.ACTIVITY_BUSINESS: "work",
            s.ACTIVITY_EDUCATION: "education",
            s.ACTIVITY_SHOPPING: "shopping",
            s.ACTIVITY_ERRANDS: "leisure",
            s.ACTIVITY_PICK_UP_DROP_OFF: "other",
            s.ACTIVITY_LEISURE: "leisure",
            s.ACTIVITY_HOME: "home",
            s.ACTIVITY_RETURN_JOURNEY: "other",
            s.ACTIVITY_OTHER: "other",
            s.ACTIVITY_EARLY_EDUCATION: "education",
            s.ACTIVITY_DAYCARE: "education",
            s.ACTIVITY_ACCOMPANY_ADULT: "other",
            s.ACTIVITY_SPORTS: "leisure",
            s.ACTIVITY_MEETUP: "leisure",
            s.ACTIVITY_LESSONS: "leisure",
            s.ACTIVITY_UNSPECIFIED: "other",
        }
        self.df['activity_translated_string'] = self.df[s.TO_ACTIVITY_WITH_CONNECTED_COL].map(activity_translation)
        logger.info(f"Translated activities.")

    def write_stats(self, stat_by_columns: list = None):
        logger.info(f"Exporting stats...")

        stat_by_columns = [col for col in s.GEO_COLUMNS if col in self.df.columns]
        stat_by_columns.append(s.LEG_TO_ACTIVITY_COL)
        # stat_by_columns.extend(["unique_household_id", "unique_person_id"])  # Very large files
        # non_stat_by_columns = [col for col in self.df.columns if col not in stat_by_columns]

        for col in stat_by_columns:
            stats_df = self.df.groupby(col).describe()

            # Flattening MultiIndex columns
            stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns]

            file_path = f"{matsim_pipeline_setup.OUTPUT_DIR}/{col}_stats.csv"
            stats_df.to_csv(file_path)
            logger.info(f"Exported stats to {file_path}.")

    def activity_times_distribution_seconds(self):
        """
        Returns a pd DataFrame with the average duration and standard deviation for each activity type.
        """
        # Ignore negative and zero values
        result = self.df[self.df[s.ACT_DUR_SECONDS_COL] > 0].groupby(s.LEG_TO_ACTIVITY_COL)[
            s.ACT_DUR_SECONDS_COL].agg(
            ['mean', 'std'])
        logger.info(f"Activity times distribution in seconds (mean and std): \n{result}")
        return result

    def leg_duration_distribution_seconds(self):
        """
        Returns a pd DataFrame with the average leg time (travel time) and standard deviation
        towards each activity type in seconds.
        """
        # Ignore negative values and values > 5 hours (these might be errors or error codes)
        filtered_df = self.df[(self.df[s.LEG_DURATION_MINUTES_COL] > 0) & (self.df[s.LEG_DURATION_MINUTES_COL] <= 300)]
        result = filtered_df.groupby(s.LEG_TO_ACTIVITY_COL)[s.LEG_DURATION_MINUTES_COL].agg(['mean', 'std']) * 60
        logger.info(f"Leg times distribution in seconds (mean and std): \n{result}")
        return result

    def first_leg_start_time_distribution(self):
        """
        Returns a pd DataFrame with the average start time and standard deviation for each activity type.
        """
        # Only look at the first leg of each person
        first_legs = self.df[self.df[s.LEG_NON_UNIQUE_ID_COL] == 1]

        # Convert datetime to numeric (timestamp) for calculation
        first_legs['timestamp'] = first_legs['leg_start_time'].view(np.int64)

        # Group by activity type and calculate mean and std of timestamps
        result = first_legs.groupby(s.LEG_TO_ACTIVITY_COL)['timestamp'].agg(['mean', 'std'])

        result['mean'] = pd.to_datetime(result['mean'])
        result['std'] = pd.to_timedelta(result['std'])

        logger.info(f"First leg start time distribution (mean and std): \n{result}")
        return result

    def add_return_home_leg(self):
        """
        Add a home leg at the end of the day, if it doesn't exist. Alternative to change_last_leg_target_to_home().
        The length of the activity and the leg duration are estimated.
        Requires is_main_activity() to be run first.
        :return: DataFrame with added home legs
        """
        logger.info("Adding return home legs...")
        self.df[s.IMPUTED_LEG_COL] = 0
        new_rows = []

        for person_id, group in self.df.groupby(s.UNIQUE_P_ID_COL):
            if pd.isna(group.at[group.index[0], s.LEG_NON_UNIQUE_ID_COL]):
                logger.debug(f"Person {person_id} has no legs. Skipping...")
                continue
            if group[s.LEG_TO_ACTIVITY_COL].iloc[-1] == s.ACTIVITY_HOME:
                logger.debug(f"Person {person_id} already has a home leg. Skipping...")
                continue
            logger.debug(f"Adding return home leg for person {person_id}...")
            main_activity_index = group[group['is_main_activity'] == 1].index[0]  # There should only be one main activity
            sum_durations_before_main = group.loc[:main_activity_index, s.LEG_DURATION_MINUTES_COL].sum()
            sum_durations_after_main = group.loc[main_activity_index:, s.LEG_DURATION_MINUTES_COL].sum()

            # Estimate leg duration:
            average_leg_duration = group[s.LEG_DURATION_MINUTES_COL].mean()
            average_leg_duration_after_main = group.loc[main_activity_index:, s.LEG_DURATION_MINUTES_COL].mean()
            if average_leg_duration_after_main:
                home_leg_duration = average_leg_duration_after_main
            else:
                home_leg_duration = average_leg_duration
            # We assume the trip home is equal or longer than the trip to the main activity
            if sum_durations_before_main > sum_durations_after_main + home_leg_duration:
                # max() so the trip doesn't get crazy short
                home_leg_duration = max(sum_durations_before_main - sum_durations_after_main, average_leg_duration / 4)

            # Estimate activity duration:
            last_leg = group.iloc[-1]
            similar_persons = self.find_similar_persons(last_leg, 100)
            activity_time = similar_persons[similar_persons[s.LEG_TO_ACTIVITY_COL] == last_leg[s.LEG_TO_ACTIVITY_COL]][
                s.ACT_DUR_SECONDS_COL].mean()
            if pd.isna(activity_time):
                logger.debug(f"Person {person_id} has no similar persons with the same last activity. ")
                activity_time = 3600  # 1 hour default

            # Create home_leg with the calculated duration
            home_leg = last_leg.copy()
            home_leg[s.LEG_NON_UNIQUE_ID_COL] = last_leg[s.LEG_NON_UNIQUE_ID_COL] + 1
            home_leg[s.UNIQUE_LEG_ID_COL] = rules.unique_leg_id(home_leg)
            home_leg[s.LEG_START_TIME_COL] = last_leg[s.LEG_END_TIME_COL] + pd.Timedelta(seconds=activity_time)
            home_leg[s.LEG_END_TIME_COL] = home_leg[s.LEG_START_TIME_COL] + pd.Timedelta(minutes=home_leg_duration)
            home_leg[s.LEG_TO_ACTIVITY_COL] = s.ACTIVITY_HOME
            home_leg[s.LEG_DURATION_MINUTES_COL] = home_leg_duration
            home_leg[s.LEG_DISTANCE_COL] = None  # Could also be estimated, but isn't necessary for the current use case
            home_leg[s.IS_MAIN_ACTIVITY_COL] = 0
            home_leg[s.IMPUTED_LEG_COL] = 1

            new_rows.append(home_leg)

        new_rows_df = pd.DataFrame(new_rows)
        logger.info(f"Adding {len(new_rows_df)} return home legs for {len(self.df[s.UNIQUE_P_ID_COL].unique())} persons.")

        # Sorting by person_id and leg_id_col will insert the new rows in the correct place
        self.df = pd.concat([self.df, new_rows_df]).sort_values([s.UNIQUE_P_ID_COL, s.LEG_ID_COL]).reset_index(drop=True)
        logger.info(f"Added return home legs.")

    # def estimate_leg_times_averages(self):
    #     """
    #     Estimates leg_start_time and leg_end_time if they are missing.
    #     """
    #     persons = self.df.groupby("unique_person_id")
    #     logger.info(f"Estimating times, where missing, for {len(persons)} persons...")
    #
    #     first_leg_start_time_distribution = (self.first_leg_start_time_distribution()).astype(int)
    #     activity_times_distribution_seconds = (self.activity_times_distribution_seconds()).astype(int)
    #     leg_duration_distribution_seconds = (self.leg_duration_distribution_seconds()).astype(int)
    #
    #     # Initialize an empty list for updates (significantly faster than updating the original df each time)
    #     updated_persons = []
    #
    #     for person_id, person in persons:
    #         person = person.copy()  # Work on a copy to avoid SettingWithCopyWarning
    #
    #         if len(person) == 1:
    #             if pd.isna(person.at[person.index[0], s.LEG_NON_UNIQUE_ID_COL]):
    #                 logger.debug(f"Person {person_id} has no legs. Skipping...")
    #                 continue
    #             # Persons with one leg might be problematic, but impute times for them anyway
    #             logger.warning(f"Person {person_id} has only one leg.")
    #
    #         # Check for negative activity times
    #         if (person[s.ACT_DUR_SECONDS_COL] < 0).any():
    #             first_negative_time_index = person[person[s.ACT_DUR_SECONDS_COL] < 0].index[0]
    #             logger.debug(f"Person {person_id} has negative activity times. Removing all times after the first bad time.")
    #             for col in [s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]:
    #                 person.loc[first_negative_time_index:, col] = None
    #
    #         if person[[s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]].isna().any().any():
    #             first_index = person.index[0]
    #             first_missing_time_index = person[person[s.LEG_START_TIME_COL].isna() | person[s.LEG_END_TIME_COL].isna()].index[
    #                 0]
    #             if first_missing_time_index == first_index:
    #                 logger.debug(
    #                     f"Person {person_id} has no time information, imputation from index {first_missing_time_index}...")
    #             else:
    #                 logger.debug(
    #                     f"Person {person_id} has some time information, imputation from index {first_missing_time_index}...")
    #
    #             # Start updating times from the first missing time
    #             for idx in range(first_missing_time_index, first_index + len(person)):
    #                 if idx == first_missing_time_index:
    #                     if idx == first_index:  # Start of the day
    #                         random_day_start = pd.Timestamp(s.BASE_DATE) + pd.Timedelta(hours=random.randint(5, 9),
    #                                                                                     minutes=random.randint(0, 59))
    #                         next_start_time = random_day_start if pd.isna(person.at[idx, s.LEG_START_TIME_COL]) else \
    #                             person.at[idx, s.LEG_START_TIME_COL]
    #                     else:
    #                         prev_end_time = person.at[idx - 1, s.LEG_END_TIME_COL]
    #                         next_start_time = prev_end_time + pd.Timedelta(
    #                             seconds=average_activity_times[person.at[idx - 1, s.LEG_ACTIVITY_COL]])
    #                 else:
    #                     prev_end_time = person.at[idx - 1, s.LEG_END_TIME_COL]
    #                     next_start_time = prev_end_time + pd.Timedelta(
    #                         seconds=average_activity_times[person.at[idx - 1, s.LEG_ACTIVITY_COL]])
    #
    #                 person.at[idx, s.LEG_START_TIME_COL] = next_start_time
    #                 person.at[idx, s.LEG_END_TIME_COL] = next_start_time + pd.Timedelta(
    #                     seconds=average_leg_times[person.at[idx, s.LEG_ACTIVITY_COL]])
    #
    #             logger.debug(f"Person {person_id} updated times: \n{person[[s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]]}")
    #             if person[[s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]].isna().any().any():
    #                 logger.warning(f"Person {person_id} still has missing times. "
    #                                f"Check the data and try again. Skipping...")
    #                 continue
    #             updated_persons.append(person)
    #     if updated_persons:
    #         logger.debug(f"Concatenating {len(updated_persons)} updated persons...")
    #         updated_df = pd.concat(updated_persons)
    #         logger.debug(f"Updating original df...")
    #         self.df.update(updated_df)
    #     logger.info("Time estimation completed.")

    def estimate_leg_times(self):
        """
        Estimates leg_start_time and leg_end_time if they are missing, using data from similar persons.
        The function lowers the matching criteria if insufficient similar persons are found.
        """
        self.df[s.IMPUTED_TIME_COL] = 0
        persons = self.df.groupby("unique_person_id")
        logger.info(f"Estimating times, where missing, for {len(persons)} persons...")

        updated_persons = []  # For storing updates

        for person_id, person in persons:
            person = person.copy()  # Avoid SettingWithCopyWarning

            if len(person) == 1:
                if pd.isna(person.at[person.index[0], s.LEG_NON_UNIQUE_ID_COL]):
                    logger.debug(f"Person {person_id} has no legs. Skipping...")
                    continue
                # Persons with one leg might be problematic, but impute times for them anyway
                logger.debug(f"Person {person_id} has only one leg.")

            # Check for negative activity times
            if (person[s.ACT_DUR_SECONDS_COL] < 0).any():
                first_bad_time_index = person[person[s.ACT_DUR_SECONDS_COL] < 0].index[0]
                logger.debug(f"Person {person_id} has negative activity times. Removing all times after the first bad time.")
                for col in [s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]:
                    person.loc[first_bad_time_index:, col] = None

            # Check for bad leg times (MiD-codes)  #TODO. rework this
            if (person[s.LEG_DURATION_MINUTES_COL] > 1000).any():
                first_bad_time_index = person[person[s.LEG_DURATION_MINUTES_COL] > 1000].index[0]
                logger.debug(f"Person {person_id} has bad leg times. Removing all times after the first bad time.")
                for col in [s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]:
                    person.loc[first_bad_time_index:, col] = None

            # Process each person
            if person[[s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]].isna().any().any():
                first_index = person.index[0]
                first_missing_time_index = person[person[s.LEG_START_TIME_COL].isna() | person[s.LEG_END_TIME_COL].isna()].index[
                    0]
                if first_missing_time_index == first_index:
                    logger.info(
                        f"Person {person_id} has no time information, imputation from index {first_missing_time_index}...")
                else:
                    logger.info(
                        f"Person {person_id} has some time information, imputation from index {first_missing_time_index}...")

                # Find similar persons
                orig_min_similar = 400
                for min_similar in range(orig_min_similar, 10000,
                                         1000):  # Find more similar persons if there are too few with good data
                    similar_persons: pd.DataFrame = self.find_similar_persons(person, min_similar)

                    # Filter similar persons for valid data
                    similar_persons_with_last_legs = similar_persons[
                        similar_persons[s.LEG_NON_UNIQUE_ID_COL].notna() &
                        similar_persons[s.LEG_START_TIME_COL].notna() &
                        (similar_persons[s.LEG_DURATION_MINUTES_COL] > 0) &
                        (similar_persons[s.LEG_DURATION_MINUTES_COL] <= 300)]
                    # Removing rows with na activity durs removes the last leg, so we need a separate df to keep quality
                    similar_persons_no_last_legs = similar_persons_with_last_legs[
                        (similar_persons_with_last_legs[s.ACT_DUR_SECONDS_COL].notna()) &
                        (similar_persons_with_last_legs[s.ACT_DUR_SECONDS_COL] > 0)]

                    if len(similar_persons_no_last_legs) > orig_min_similar / 2:
                        break
                    else:
                        logger.info(f"Person {person_id} has too few similar persons. Lowering standards.")
                else:  # No break
                    logger.warning(f"Person {person_id} misses times and has no even slightly similar persons. "
                                   f"Removing the person to avoid errors.")
                    self.df.drop(person.index, inplace=True)
                    continue

                # Impute times
                loops_max = 5
                loop = 0
                while loop < loops_max:  # Loop until imputed times pass checks
                    for idx, row in person.iterrows():
                        if idx >= first_missing_time_index:

                            if idx == first_index:  # Start of the day
                                similar_persons_same_activity = similar_persons_with_last_legs.loc[
                                    (similar_persons_with_last_legs[s.LEG_TO_ACTIVITY_COL] == row[s.LEG_TO_ACTIVITY_COL]) &
                                    (similar_persons_with_last_legs[s.LEG_NON_UNIQUE_ID_COL] == 1), s.LEG_START_TIME_COL]
                                if similar_persons_same_activity.empty:
                                    logger.info(f"Person {person_id} has no similar persons with the same first activity."
                                                f"Lowering standards.")
                                    similar_persons_same_activity = similar_persons_with_last_legs.loc[
                                        (similar_persons_with_last_legs[s.LEG_NON_UNIQUE_ID_COL] == 1), s.LEG_START_TIME_COL]

                                typical_day_start = similar_persons_same_activity.sample(1).iloc[0]
                                my_start_time = typical_day_start if pd.isna(person.at[idx, s.LEG_START_TIME_COL]) else \
                                    person.at[idx, s.LEG_START_TIME_COL]

                            else:  # Other leg start times
                                prev_end_time = person.at[idx - 1, s.LEG_END_TIME_COL]
                                similar_persons_same_activity = similar_persons_no_last_legs.loc[
                                    similar_persons_no_last_legs[s.LEG_TO_ACTIVITY_COL] == person.loc[
                                        idx - 1, s.LEG_TO_ACTIVITY_COL],
                                    s.ACT_DUR_SECONDS_COL]
                                if similar_persons_same_activity.empty:
                                    logger.info(f"Person {person_id} has no similar persons with the same activity. "
                                                f"Lowering standards.")
                                    similar_persons_same_activity = similar_persons_no_last_legs[s.ACT_DUR_SECONDS_COL]
                                my_start_time = prev_end_time + pd.Timedelta(
                                    # Sample an activity duration from a similar person with the same activity
                                    seconds=similar_persons_same_activity.sample(1).iloc[0])

                            # Leg duration
                            # Utilize imputed leg duration from MiD if available
                            similar_persons_same_activity = similar_persons_with_last_legs.loc[
                                similar_persons_with_last_legs[s.LEG_TO_ACTIVITY_COL] == row[
                                    s.LEG_TO_ACTIVITY_COL], s.LEG_DURATION_MINUTES_COL]
                            if similar_persons_same_activity.empty:
                                logger.info(f"Person {person_id} has no similar persons with the same activity. "
                                            f"Lowering standards.")
                                similar_persons_same_activity = similar_persons_with_last_legs[s.LEG_DURATION_MINUTES_COL]
                            my_leg_duration = similar_persons_same_activity.sample(1).iloc[0]

                            # Assign times
                            person.at[idx, s.LEG_START_TIME_COL] = my_start_time
                            person.at[idx, s.LEG_END_TIME_COL] = my_start_time + pd.Timedelta(
                                minutes=my_leg_duration)
                            person.at[idx, s.LEG_DURATION_MINUTES_COL] = my_leg_duration
                            person.at[idx, s.IMPUTED_TIME_COL] = 1

                    # Check if times are valid (for now, all tours that end before 2am are valid)
                    if person[s.LEG_END_TIME_COL].iloc[-1] < pd.Timestamp(s.BASE_DATE) + pd.Timedelta(days=1, hours=2):
                        break
                    logger.info(f"Person {person_id} imputed invalid times. Trying again...")
                    loop += 1

                logger.info(
                    f"Person {person_id} updated times: \n{person[[s.LEG_START_TIME_COL, s.LEG_END_TIME_COL, s.LEG_TO_ACTIVITY_COL]]}")
                if person[[s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]].isna().any().any():
                    logger.warning(f"Person {person_id} still has missing times. "
                                   f"Check the data and try again. Skipping...")
                    continue
                updated_persons.append(person)

        if updated_persons:
            logger.info(f"Concatenating {len(updated_persons)} updated persons...")
            updated_df = pd.concat(updated_persons)
            logger.info(f"Updating original df...")
            self.df.update(updated_df)
        logger.info("Time estimation completed.")

    def vary_times_by_household(self, hh_id_col, time_cols, max_shift_minutes=15):
        """
        Varies times in the DataFrame by the same random amount (±max_shift_minutes) for each household.

        :param hh_id_col: String, the column name for the unique hh identifier.
        :param time_cols: List of strings, the names of the columns containing time data.
        :param max_shift_minutes: Integer, the maximum number of minutes for the time shift.
        :return: pandas DataFrame with varied times.
        """

        logger.info("Varying times by household...")

        def apply_time_shift(group):
            # Generate a random time shift between -max_shift_minutes and +max_shift_minutes
            time_shift = timedelta(minutes=np.random.randint(-max_shift_minutes, max_shift_minutes + 1))

            # Apply this time shift to all time columns
            for col in time_cols:
                group[col] = group[col].apply(lambda x: x + time_shift if pd.notnull(x) else x)
            return group

        # Group by person and apply the function
        self.df = self.df.groupby(hh_id_col).apply(apply_time_shift)
        logger.info("Times varied by person.")

    def downsample_population(self, sample_percentage):
        """
        Downsample the population to a given sample percentage size of the original population.
        Recommended to sample households (e.g. at the point when only households are loaded),
        not persons, to keep the household structure intact.
        """
        logger.info("Downsampling population...")
        self.df = self.df.sample(frac=sample_percentage)
        logger.info(f"Downsampled population to {sample_percentage * 100}% of the original population.")

    def find_similar_persons(self, person, min_similar, attributes: list = None):
        """
        Find similar persons (or other entries) based on a dynamic number of matching attributes.
        :param person: Dataframe or Series, the person to find similar persons for.
        :param min_similar: Integer, the minimum number of similar persons to find.
        :param attributes: List of strings, the attributes to match on. If None, uses default attributes.
        """
        if isinstance(person, pd.DataFrame):
            person = person.iloc[0]
        if attributes is None:
            attributes = [s.H_REGION_TYPE_COL, s.NUMBER_OF_LEGS_COL, s.HAS_LICENSE_COL, s.H_CAR_IN_HH_COL]

        logger.debug(f"Finding similar persons for {person[s.UNIQUE_P_ID_COL]}...")
        for min_matches in range(len(attributes), 0, -1):  # Decrease criteria to a minimum of 1 attributes
            attribute_combinations = itertools.combinations(attributes, min_matches)
            for combination in attribute_combinations:

                condition = (self.df[list(combination)] == person[list(combination)]).all(axis=1)
                similar_persons = self.df[condition]
                # similar_persons = self.df  # old slower way
                # for attr in combination:
                #     similar_persons = similar_persons[similar_persons[attr] == person[attr]]

                if len(similar_persons) >= min_similar:
                    logger.debug(f"Found {len(similar_persons)} similar persons for {person[s.UNIQUE_P_ID_COL]} "
                                 f"based on {combination}.")
                    # Drop the person itself from the similar persons
                    return similar_persons[similar_persons[s.UNIQUE_P_ID_COL] != person[s.UNIQUE_P_ID_COL]]

        logger.debug(f"Found no similar persons for {person[s.PERSON_ID_COL]}.")
        return pd.DataFrame()  # Return an empty DataFrame if no similar persons found

    def impute_license_status(self):
        """
        Vectorized function to impute license status based on age and statistical probabilities.
        Uses the same representation of license status as the survey data.
        Adds a new column 'imputed_license' to the dataframe.
        :return: None
        """
        logger.info("Imputing license status...")

        # Calculate license likelihoods (valid entries are mostly based on adults in MiD, which is good, because we
        # assign non-adults no license; this means unknowns are mostly adults)
        valid_entries = self.df[self.df[s.HAS_LICENSE_COL].isin([s.LICENSE_YES, s.LICENSE_NO])]
        likelihoods = h.calculate_value_frequencies_df(valid_entries, s.H_CAR_IN_HH_COL, s.HAS_LICENSE_COL)
        licence_likelihood_with_car = likelihoods.at[s.CAR_IN_HH_YES, s.LICENSE_YES]
        logger.debug(f"Likelihood of having a license with a car: {licence_likelihood_with_car}")
        licence_likelihood_without_car = likelihoods.at[s.CAR_IN_HH_NO, s.LICENSE_YES]
        logger.debug(f"Likelihood of having a license without a car: {licence_likelihood_without_car}")

        # Log cases where license status was wrongly reported based on age
        self.df.loc[
            (self.df[s.HAS_LICENSE_COL] == s.LICENSE_YES) & (self.df[s.PERSON_AGE_COL] < 17), 'imputed_license'] = s.LICENSE_NO
        logger.info(f"Changed {self.df['imputed_license'].eq(s.LICENSE_NO).sum()} license status to no license based on age.")

        # Cases where license status is known
        self.df['imputed_license'] = valid_entries[s.HAS_LICENSE_COL]
        self.df.loc[
            (self.df[s.HAS_LICENSE_COL] == s.PERSON_UNDER_16) | (
                    self.df[s.PERSON_AGE_COL] < 17), 'imputed_license'] = s.LICENSE_NO

        logger.info(f"{self.df['imputed_license'].eq(s.LICENSE_YES).sum()} rows with license and "
                    f"{self.df['imputed_license'].eq(s.LICENSE_NO).sum()} rows without license before imputation.")

        # Impute for still unknown license cases (s.LICENSE_UNKNOWN, s.ADULT_OVER_16_PROXY, but also any other unknown values)
        unknown_license = self.df['imputed_license'].isna()
        no_car = self.df[s.H_CAR_IN_HH_COL] == s.CAR_IN_HH_NO

        # Weighed random choice for unknown license with/without car
        condition = unknown_license & no_car
        num_rows = self.df[condition].shape[0]
        self.df.loc[condition, 'imputed_license'] = np.random.choice(
            [s.LICENSE_NO, s.LICENSE_YES], size=num_rows,
            p=[1 - licence_likelihood_without_car, licence_likelihood_without_car]
        )
        logger.info(f"Imputed license status for {num_rows} rows without car of {self.df.shape[0]} total rows.")

        condition = unknown_license & ~no_car
        num_rows = self.df[condition].shape[0]
        self.df.loc[condition, 'imputed_license'] = np.random.choice(
            [s.LICENSE_NO, s.LICENSE_YES], size=num_rows, p=[1 - licence_likelihood_with_car, licence_likelihood_with_car]
        )
        logger.info(f"Imputed license status for {num_rows} rows with car of {self.df.shape[0]} total rows.")

        logger.info(f"{self.df['imputed_license'].eq(s.LICENSE_YES).sum()} rows with license and "
                    f"{self.df['imputed_license'].eq(s.LICENSE_NO).sum()} rows without license.")

        assert self.df['imputed_license'].isna().sum() == 0, "There are still unknown license statuses."

    def close_connected_leg_groups(self):
        """
        Close connected leg groups by ensuring each leg has a list of all unique leg IDs in its connected group.
        """
        logger.info("Closing connected leg groups...")

        for household_id, household_group in self.df.groupby(s.UNIQUE_HH_ID_COL):
            if household_group[s.CONNECTED_LEGS_COL].isna().all():
                logger.debug(f"Household {household_id} has no connected legs. Skipping...")
                continue
            else:
                logger.debug(f"Household {household_id} has connected legs. Closing...")

            checked_legs = set()
            for row in household_group.itertuples():
                unique_leg_id = getattr(row, s.UNIQUE_LEG_ID_COL)
                if not isinstance(getattr(row, s.CONNECTED_LEGS_COL), list):
                    continue
                if unique_leg_id in checked_legs:
                    continue

                queue = deque([unique_leg_id])
                connected_legs = {unique_leg_id}

                # Collect all connected legs
                while queue:
                    current_leg = queue.popleft()

                    connected = set(self.df.loc[self.df[s.UNIQUE_LEG_ID_COL] == current_leg, s.CONNECTED_LEGS_COL].iloc[0])
                    new_connections = connected - connected_legs
                    queue.extend(new_connections)
                    connected_legs.update(new_connections)

                # Update connected_legs for all legs in the group
                updated_connected_legs = list(connected_legs)
                for leg in connected_legs:
                    existing_connected = self.df.loc[self.df[s.UNIQUE_LEG_ID_COL] == leg, s.CONNECTED_LEGS_COL].iloc[0]
                    leg_indexes = self.df.loc[self.df[s.UNIQUE_LEG_ID_COL] == leg].index
                    if len(leg_indexes) > 1:
                        print("Multiple indexes found:", leg_indexes)
                        logger.warning(f"Multiple indexes found for leg {leg}. This shouldn't happen.")
                    leg_index = leg_indexes[0]

                    # self.df.loc[self.df[s.UNIQUE_LEG_ID_COL] == leg, s.CONNECTED_LEGS_COL][0] = updated_connected_legs
                    self.df.at[leg_index, s.CONNECTED_LEGS_COL] = updated_connected_legs

                    if existing_connected != updated_connected_legs:
                        logger.debug(
                            f"Updated connected legs for leg {leg} from {existing_connected} to {updated_connected_legs}.")

                    checked_legs.add(leg)

    def update_activity_for_prot_legs(self):
        """
        Update the activity of connected legs to match the activity of the protagonist leg in the group.
        """
        logger.info("Updating activity for protagonist legs...")

        # Make a copy of the activity column that will be updated
        self.df[s.TO_ACTIVITY_WITH_CONNECTED_COL] = self.df[s.LEG_TO_ACTIVITY_COL]

        # if s.IS_PROTAGONIST_COL in self.df.columns:  # Just for debugging
        #     logger.debug("Protagonist column already exists.")
        prot_legs = self.df[self.df[s.IS_PROTAGONIST_COL] == 1]

        for row in prot_legs.itertuples():
            protagonist_activity = getattr(row, s.LEG_TO_ACTIVITY_COL)
            protagonist_leg_id = getattr(row, s.UNIQUE_LEG_ID_COL)
            connected_legs_list = getattr(row, s.CONNECTED_LEGS_COL)

            if not isinstance(connected_legs_list, list):
                logger.error(f"Protagonist leg {protagonist_leg_id} has no connected legs. This shouldn't happen. Skipping...")
                continue

            connected_legs = set(connected_legs_list)
            connected_legs.discard(protagonist_leg_id)

            # Assign the protagonist's activity to all connected legs
            self.df.loc[
                self.df[s.UNIQUE_LEG_ID_COL].isin(connected_legs), s.TO_ACTIVITY_WITH_CONNECTED_COL] = protagonist_activity

        logger.info("Updated activity for protagonist legs.")

    def add_from_activity(self):  # MA DONE
        """
        Add a 'from_activity' column to the DataFrame, which is the to_activity of the previous leg.
        For the first leg of each person, set 'from_activity' based on 'starts_at_home' (-> home or unspecified).
        :return:
        """
        logger.info("Adding from_activity column...")
        # Sort the DataFrame by person ID and leg number (the df should usually already be sorted this way)
        self.df.sort_values(by=[s.PERSON_ID_COL, s.LEG_NON_UNIQUE_ID_COL], inplace=True)

        # Shift the 'to_activity' down to create 'from_activity' for each group
        self.df[s.LEG_FROM_ACTIVITY_COL] = self.df.groupby(s.PERSON_ID_COL)[s.LEG_TO_ACTIVITY_COL].shift(1)

        # For the first leg of each person, set 'from_activity' based on 'starts_at_home'
        self.df.loc[(self.df[s.LEG_NON_UNIQUE_ID_COL] == 1) & (
                self.df[s.FIRST_LEG_STARTS_AT_HOME_COL] == s.FIRST_LEG_STARTS_AT_HOME), s.LEG_FROM_ACTIVITY_COL] = s.ACTIVITY_HOME
        self.df.loc[(self.df[s.LEG_NON_UNIQUE_ID_COL] == 1) & (
                self.df[
                    s.FIRST_LEG_STARTS_AT_HOME_COL] != s.FIRST_LEG_STARTS_AT_HOME), s.LEG_FROM_ACTIVITY_COL] = s.ACTIVITY_UNSPECIFIED

        # Handle cases with no legs (NA in leg_id)
        self.df.loc[self.df[s.LEG_NON_UNIQUE_ID_COL].isna(), s.LEG_FROM_ACTIVITY_COL] = None

        logger.info("Added from_activity column.")

    def calculate_slack_factors(self):
        slack_factors = []

        df = self.df[self.df[s.LEG_DISTANCE_COL] < 500]

        for person_id, person_trips in df.groupby(s.PERSON_ID_COL):
            logger.debug(f"Searching sf at person {person_id}...")
            # Sort by ordered_id to ensure sequence
            person_trips = person_trips.sort_values(by=s.LEG_NON_UNIQUE_ID_COL)

            # Find indirect routes by checking consecutive legs
            for i in range(len(person_trips) - 1):
                first_leg = person_trips.iloc[i]
                second_leg = person_trips.iloc[i + 1]

                # This should always be true, except for missing data
                if first_leg[s.LEG_TO_ACTIVITY_COL] == second_leg[s.LEG_FROM_ACTIVITY_COL]:

                    direct_trip = self.df[
                        (self.df[s.PERSON_ID_COL] == person_id) &
                        # Exclude the two legs we're checking
                        (self.df[s.LEG_NON_UNIQUE_ID_COL] != first_leg[s.LEG_NON_UNIQUE_ID_COL]) &
                        (self.df[s.LEG_NON_UNIQUE_ID_COL] != second_leg[s.LEG_NON_UNIQUE_ID_COL]) &
                        # Find direct trip in both directions
                        ((self.df[s.LEG_FROM_ACTIVITY_COL] == first_leg[s.LEG_FROM_ACTIVITY_COL]) &
                         (self.df[s.LEG_TO_ACTIVITY_COL] == second_leg[s.LEG_TO_ACTIVITY_COL]) |
                         (self.df[s.LEG_FROM_ACTIVITY_COL] == second_leg[s.LEG_TO_ACTIVITY_COL]) &
                         (self.df[s.LEG_TO_ACTIVITY_COL] == first_leg[s.LEG_FROM_ACTIVITY_COL]))
                        ]

                    if not direct_trip.empty:
                        direct_distance = direct_trip.iloc[0][s.LEG_DISTANCE_COL]
                        indirect_distance = first_leg[s.LEG_DISTANCE_COL] + second_leg[s.LEG_DISTANCE_COL]
                        slack_factor = indirect_distance / direct_distance
                        slack_factors.append((person_id,
                                              first_leg[s.H_REGION_TYPE_COL],
                                              first_leg[s.PERSON_AGE_COL],
                                              first_leg[s.LEG_FROM_ACTIVITY_COL],
                                              first_leg[s.LEG_TO_ACTIVITY_COL],
                                              second_leg[s.LEG_TO_ACTIVITY_COL],
                                              first_leg[s.LEG_MAIN_MODE_COL],
                                              second_leg[s.LEG_MAIN_MODE_COL],
                                              direct_trip.iloc[0][s.LEG_MAIN_MODE_COL],
                                              slack_factor))
                        logger.debug(f"Found a slack factor of {slack_factor} for person {person_id} ")

        return pd.DataFrame(slack_factors, columns=[s.PERSON_ID_COL,
                                                    s.H_REGION_TYPE_COL,
                                                    s.PERSON_AGE_COL,
                                                    'start_activity',
                                                    'via_activity',
                                                    'end_activity',
                                                    'start_mode',
                                                    'end_mode',
                                                    'direct_mode',
                                                    'slack_factor'])

    def list_cars_in_household(self):  # MA DONE
        """
        Create a list of cars with unique ids in each household and add it to the DataFrame.
        """
        logger.info("Listing cars in household...")
        self.df[s.LIST_OF_CARS_COL] = None
        # Group by household
        hhs = self.df.groupby(s.UNIQUE_HH_ID_COL)
        total_cars = 0
        for household_id, hh in hhs:
            number_of_cars: int = int(hh[s.H_NUMBER_OF_CARS_COL].iloc[0])
            if number_of_cars == 0:
                continue
            if number_of_cars > 30:
                logger.debug(f"Household {household_id} has {number_of_cars} cars. "
                             f"This is either a code for unknown number of cars or likely an error. Skipping...")
                continue

            total_cars += number_of_cars

            # Generate unique car IDs for each household
            car_ids = [f"{household_id}_veh_{i + 1}" for i in range(number_of_cars)]
            self.df.at[hh.index[0], s.LIST_OF_CARS_COL] = car_ids

        logger.info(f"Listed {total_cars} cars in {len(hhs)} households for {self.df[s.UNIQUE_P_ID_COL].nunique()} persons, "
                    f"meaning {total_cars / self.df[s.UNIQUE_P_ID_COL].nunique()} cars per person on average.")

    def impute_cars_in_household(self):  # TODO: add some actual imputation
        """
        Impute the number of cars in a household if unknown, based on the number of cars in similar households.
        """
        self.df.loc[self.df[s.H_NUMBER_OF_CARS_COL] == 99, s.H_NUMBER_OF_CARS_COL] = None
        logger.info(f"Imputing cars in household for {self.df[s.H_NUMBER_OF_CARS_COL].isna().sum()} of "
                    f"{len(self.df)} rows...")

        # Set all other unknown values to 0
        self.df.loc[self.df[s.H_NUMBER_OF_CARS_COL].isna, s.H_NUMBER_OF_CARS_COL] = 0

    def mark_mirroring_main_activities(self, duration_threshold_seconds=7200):  # C DONE, UNTESTED, MA NOT
        """
        Mark activities that mirror the peron's main activity; activities that still likely represent the same main activity, but
        are separated by a different, short, activity (e.g. a lunch break between two work activities).
        :param duration_threshold_seconds: Integer, the maximum duration of the short activity in seconds.
        :return:
        """

        logger.info("Marking mirroring main activities...")
        # Vectorized because it's insanely faster than looping
        # Create shifted columns for comparison
        self.df['next_person_id'] = self.df[s.UNIQUE_P_ID_COL].shift(-1)
        self.df['next_act_dur'] = self.df[s.ACT_DUR_SECONDS_COL].shift(-1)
        self.df['next_next_activity'] = self.df[s.LEG_TO_ACTIVITY_COL].shift(-2)
        self.df['next_next_person_id'] = self.df[s.UNIQUE_P_ID_COL].shift(-2)
        self.df['next_leg_distance'] = self.df[s.LEG_DISTANCE_COL].shift(-1)
        self.df['next_next_leg_distance'] = self.df[s.LEG_DISTANCE_COL].shift(-2)

        # Make sure we are checking the same person, and based on main activity
        person_id_condition = (self.df['next_person_id'] == self.df[s.UNIQUE_P_ID_COL]) & \
                              (self.df['next_next_person_id'] == self.df[s.UNIQUE_P_ID_COL]) & \
                              (self.df[s.IS_MAIN_ACTIVITY_COL] == 1)

        # Time threshold for the in-between activity
        short_duration_condition = (self.df['next_act_dur'] < duration_threshold_seconds)

        # Candidate activity must be the same as the main activity
        same_activity_condition = (self.df['next_next_activity'] == self.df[s.LEG_TO_ACTIVITY_COL])

        # Leg distance to the in-between activity and from it to the candidate activity must be the same
        same_leg_distance_condition = (self.df['next_leg_distance'] == self.df['next_next_leg_distance'])

        self.df[s.MIRRORS_MAIN_ACTIVITY_COL] = (
            (person_id_condition & short_duration_condition & same_activity_condition & same_leg_distance_condition).shift(
                2).fillna(False)).astype(int)

        # Drop temporary columns
        self.df.drop(['next_person_id', 'next_act_dur', 'next_next_activity', 'next_next_person_id', 'next_leg_distance',
                      'next_next_leg_distance'], axis=1, inplace=True)

        logger.info("Marked mirroring main mischief, my merry miscreant mate.")

    def find_home_to_main_time(self):
        """
        Determines the time (and distance) between home and main activity for each person in the DataFrame.
        It may not look it, but this was a pain to write.
        Main directly after home: Leg distance of main leg
        Main directly before home: Leg distance of home leg
        :return: Adds column with the distance to the DataFrame.
        """

        logger.info("Determining home to main activity times/distances...")

        persons = self.df.groupby(s.UNIQUE_P_ID_COL)
        distances = {}
        times = {}
        is_estimated = {}

        for pid, person in persons:
            # Extract the indices for main activity, mirroring main and home activities
            main_activity_idx = np.where(person[s.IS_MAIN_ACTIVITY_COL])[0]  # where returns a tuple. Legs to main
            mirroring_main_idx = np.where(person[s.MIRRORS_MAIN_ACTIVITY_COL])[0]  # Legs to mirrored main
            home_indices = np.where(person[s.LEG_TO_ACTIVITY_COL] == s.ACTIVITY_HOME)[0]  # legs to home

            if main_activity_idx.size == 0:
                logger.warning(f"Person {pid} has no main activity. Skipping this person.")
                continue
            if main_activity_idx.size > 1:  # should not happen but still works
                logger.warning(f"Person {pid} has more than one main activity. Using the first one.")

            if home_indices.size == 0:
                logger.debug(f"Person {pid} has no home activities. Using the first activity instead.")
                closest_home_idx = -1
                closest_home_row = person.index[0] - 1
            else:

                idx_distances_to_main = np.abs(home_indices - main_activity_idx[0])
                closest_to_main = np.min(idx_distances_to_main)

                # If there is a mirroring main activity, calculate the distance to that as well
                if not mirroring_main_idx.size == 0:
                    idx_distances_to_mirroring = np.abs(home_indices - mirroring_main_idx[0])
                    closest_to_mirroring = np.min(idx_distances_to_mirroring)
                    if closest_to_mirroring < closest_to_main:
                        main_activity_idx = mirroring_main_idx
                        idx_distances_to_main = idx_distances_to_mirroring

                # Determine the closest home activity by number of legs
                if (person.at[person.index[0], s.FIRST_LEG_STARTS_AT_HOME_COL] == s.FIRST_LEG_STARTS_AT_HOME and
                        main_activity_idx[0] < np.min(
                            idx_distances_to_main)):  # This means starting home is same or closer than any other home to main
                    closest_home_idx = -1
                    closest_home_row = person.index[0] - 1
                else:
                    closest_home_idx = home_indices[np.argmin(idx_distances_to_main)]
                    closest_home_row = person.index[closest_home_idx]

            main_activity_row = person.index[main_activity_idx[0]]

            # Calculate the distance between home and main activity
            if closest_home_idx == main_activity_idx[0] or closest_home_row == main_activity_row:
                logger.error(f"Person {pid} has a home activity marked as main activity. This should never be the case. "
                             f"time/distance cannot be determined and are arbitrarily set to 1.")
                home_to_main_distance = 1
                home_to_main_time = 1
                time_is_estimated = 1

            elif closest_home_idx - main_activity_idx[0] == 1:  # Main to home
                logger.debug(f"Person {pid} has a home activity directly after main. ")
                home_to_main_distance = person.at[closest_home_row, s.LEG_DISTANCE_COL]
                home_to_main_time = person.at[closest_home_row, s.LEG_DURATION_MINUTES_COL]
                time_is_estimated = 0
            elif closest_home_idx - main_activity_idx[0] == -1:  # Home to main
                logger.debug(f"Person {pid} has a home activity directly before main. ")
                home_to_main_distance = person.at[main_activity_row, s.LEG_DISTANCE_COL]
                home_to_main_time = person.at[main_activity_row, s.LEG_DURATION_MINUTES_COL]
                time_is_estimated = 0

            else:
                logger.debug(f"Person {pid} has a main activity and home activity more than one leg apart. "
                             f"Distance will be estimated.")
                # Get all legs between home and main activity (thus exclude leg towards first activity)
                if closest_home_row < main_activity_row:  # Home to main
                    legs = person.loc[closest_home_row + 1:main_activity_row]

                    updated_legs, level = self.sf.get_all_estimated_times_with_slack(legs)
                    home_to_main_time = updated_legs[f"level_{level}"].dropna().iloc[0]
                    home_to_main_distance = None  # We cannot correctly determine this without an own function (yes, really)
                    # and it's not needed nor worth the effort here. Also, this serves as a marker for estimated time.
                    time_is_estimated = 1

                else:  # Main to home
                    legs = person.loc[main_activity_row + 1:closest_home_row]

                    updated_legs, level = self.sf.get_all_estimated_times_with_slack(legs)
                    home_to_main_time = updated_legs[f"level_{level}"].dropna().iloc[0]
                    home_to_main_distance = None
                    time_is_estimated = 1

            distances[pid] = home_to_main_distance
            times[pid] = home_to_main_time
            is_estimated[pid] = time_is_estimated

        self.df[s.HOME_TO_MAIN_DIST_COL] = self.df[s.UNIQUE_P_ID_COL].map(distances)
        self.df[s.HOME_TO_MAIN_TIME_COL] = self.df[s.UNIQUE_P_ID_COL].map(times)
        self.df[s.HOME_TO_MAIN_TIME_ESTIMATED_COL] = self.df[s.UNIQUE_P_ID_COL].map(is_estimated)

        # Set the distance to NaN for all rows except the first one for each person
        # mask = self.df.groupby(UNIQUE_P_ID_COL).cumcount() == 0
        # self.df['HOME_TO_MAIN_DISTANCE'] = self.df['HOME_TO_MAIN_DISTANCE'].where(mask, np.nan)

        logger.info("Determining home to main activity distances/times completed.")

    def find_main_mode_to_main_act(self):
        """
        Determines the main mode used to reach the main activity from either the closest previous home activity
        or the start of the day if there is no home activity before.
        Main mode is the mode with the longest total use time.
        """
        logger.info("Determining main mode to main activity...")
        # Group by unique person ID
        persons = self.df.groupby(s.UNIQUE_P_ID_COL)

        # Initialize a dictionary to store the main mode for each person
        main_modes_time = {}
        main_modes_dist = {}

        for pid, person in persons:
            if len(person) == 1:
                if person[s.LEG_NON_UNIQUE_ID_COL].isna().all():
                    # If the person has no legs, there is no main activity
                    logger.debug(f"Person {pid} has no legs. Skipping...")
                    continue
                if person[s.LEG_TO_ACTIVITY_COL].iloc[0] == s.ACTIVITY_HOME:
                    # If the person has only one leg, and it's home, there is no main activity
                    logger.debug(f"Person {pid} has only one leg and it's home. Skipping...")
                    continue

            # Find the main activity
            try:
                main_activity_idx = person[person[s.IS_MAIN_ACTIVITY_COL] == 1].index[0]
            except IndexError:
                logger.warning(f"Person {pid} has no main activity for unknown reasons. Skipping this person.")
                continue

            # Find the closest previous home activity or the start of the day. FROM_activity so the trip to home is excluded.
            home_indices = person[person[s.LEG_FROM_ACTIVITY_COL] == s.ACTIVITY_HOME].index
            start_idx = home_indices[home_indices < main_activity_idx].max() if not home_indices.empty else person.index[0]
            if pd.isna(start_idx):
                start_idx = person.index[0]

            # Calculate the total use time for each mode. Slicing is inclusive of both start and end index.
            mode_times = person.loc[start_idx:main_activity_idx].groupby(s.LEG_MAIN_MODE_COL)[s.LEG_DURATION_MINUTES_COL].sum()
            mode_distances = person.loc[start_idx:main_activity_idx].groupby(s.LEG_MAIN_MODE_COL)[s.LEG_DISTANCE_COL].sum()

            main_mode_time_base = mode_times.idxmax()
            main_mode_dist_base = mode_distances.idxmax()

            # Store the main mode for the person
            main_modes_time[pid] = main_mode_time_base
            main_modes_dist[pid] = main_mode_dist_base

        # Add a new column to the dataframe with the main mode for each person
        self.df[s.MAIN_MODE_TO_MAIN_ACT_TIMEBASED_COL] = self.df[s.UNIQUE_P_ID_COL].map(main_modes_time)
        self.df[s.MAIN_MODE_TO_MAIN_ACT_DISTBASED_COL] = self.df[s.UNIQUE_P_ID_COL].map(main_modes_dist)
        logger.info("Determined main mode to main activity.")

    def filter_home_to_home_legs(self):  # TODO: test PASSED
        """
        Filters out 'home to home' legs from the DataFrame.
        """
        logger.info(f"Filtering out 'home to home' legs from {len(self.df)} rows...")
        home_to_home_condition = (self.df[s.LEG_FROM_ACTIVITY_COL] == s.ACTIVITY_HOME) & \
                                 (self.df[s.LEG_TO_ACTIVITY_COL] == s.ACTIVITY_HOME)
        self.df = self.df[~home_to_home_condition].reset_index(drop=True)
        logger.info(f"Filtered out 'home to home' legs. {len(self.df)} rows remaining.")

    def update_number_of_legs(self, col_to_write_to=s.NUMBER_OF_LEGS_COL):  # TODO: test PASSED. Repeat run.
        """
        Updates the NUMBER_OF_LEGS_COL with the correct number of legs for each person;
        or writes a new col with the given name.
        """
        persons = self.df.groupby(s.UNIQUE_P_ID_COL)

        # Number of legs for each person, this actually counts the number of rows for each person
        number_of_legs = persons.size()

        self.df[col_to_write_to] = self.df[s.UNIQUE_P_ID_COL].map(number_of_legs)

        # Set the number to 0 for rows with no legs
        self.df.loc[self.df[s.LEG_ID_COL].isna(), col_to_write_to] = 0

    def find_connected_legs(self):
        """
        Find connections between trip legs in a household.
        Uses unique_leg_id; lists all legs that are connected to each leg.
        """
        # Group by household
        households = self.df.groupby(s.UNIQUE_HH_ID_COL)
        num_households = len(households)

        # Initialize connections series and checks_df
        connections = pd.Series(index=self.df.index, dtype='object')
        checks_data = []
        for household_id, household_group in households:
            num_households -= 1
            if num_households % 1000 == 0:
                logger.info(f"------ {num_households} households remaining. ------")
            if household_group[s.PERSON_ID_COL].nunique() == 1:
                logger.debug(f"Household {household_id} has only one person. No connections.")
                continue

            for idx_a, person_a_leg in household_group.iterrows():
                for idx_b, person_b_leg in household_group.iterrows():
                    if person_a_leg[s.PERSON_ID_COL] == person_b_leg[s.PERSON_ID_COL] or idx_b <= idx_a:
                        continue  # So we don't compare a leg to itself or to a leg it's already been compared to

                    dist_match = h.check_distance(person_a_leg, person_b_leg)
                    time_match = h.check_time(person_a_leg, person_b_leg)
                    mode_match = h.check_mode(person_a_leg, person_b_leg)
                    activity_match = h.check_activity(person_a_leg, person_b_leg)
                    logger.debug(f"Legs {person_a_leg[s.UNIQUE_LEG_ID_COL]} and {person_b_leg[s.UNIQUE_LEG_ID_COL]}: "
                                 f"distance {dist_match}, time {time_match}, mode {mode_match}, activity {activity_match}")
                    checks_data.append({  # List of dics
                        'leg_id_a': person_a_leg[s.UNIQUE_LEG_ID_COL],
                        'leg_id_b': person_b_leg[s.UNIQUE_LEG_ID_COL],
                        'dist_match': dist_match,
                        'time_match': time_match,
                        'mode_match': mode_match,
                        'activity_match': activity_match
                    })

                    if dist_match and time_match and mode_match and activity_match:
                        if not isinstance(connections.at[idx_a], list):  # Checking for NaN doesn't work here
                            connections.at[idx_a] = []
                        if not isinstance(connections.at[idx_b], list):
                            connections.at[idx_b] = []
                        connections.at[idx_a].append(person_b_leg[s.UNIQUE_LEG_ID_COL])
                        connections.at[idx_b].append(person_a_leg[s.UNIQUE_LEG_ID_COL])

        # Save checks_df to a CSV file
        checks_df = pd.DataFrame(checks_data)
        file_loc = os.path.join(matsim_pipeline_setup.OUTPUT_DIR, 'leg_connections_logs.csv')
        checks_df.to_csv(file_loc, index=False)

        # Add connections as a new column to self.df
        self.df[s.CONNECTED_LEGS_COL] = connections

    def mark_connected_persons_and_hhs(self):
        logger.info("Marking connected persons and households...")
        self.df[s.HH_HAS_CONNECTIONS_COL] = 0
        self.df[s.P_HAS_CONNECTIONS_COL] = 0

        for person_id in self.df[s.PERSON_ID_COL].unique():
            if any(self.df[self.df[s.PERSON_ID_COL] == person_id][s.CONNECTED_LEGS_COL].apply(lambda x: isinstance(x, list))):
                self.df.loc[self.df[s.PERSON_ID_COL] == person_id, s.P_HAS_CONNECTIONS_COL] = 1
                logger.debug(f"Person {person_id} has connections.")

        for hh_id in self.df[s.UNIQUE_HH_ID_COL].unique():
            if any(self.df[self.df[s.UNIQUE_HH_ID_COL] == hh_id][s.CONNECTED_LEGS_COL].apply(lambda x: isinstance(x, list))):
                self.df.loc[self.df[s.UNIQUE_HH_ID_COL] == hh_id, s.HH_HAS_CONNECTIONS_COL] = 1
                logger.debug(f"Household {hh_id} has connections.")

    def count_connected_legs_per_person(self):
        logger.info("Counting connected legs per person...")
        self.df[s.NUM_CONNECTED_LEGS_COL] = 0

        for person_id in self.df[s.PERSON_ID_COL].unique():
            person_rows = self.df[self.df[s.PERSON_ID_COL] == person_id]
            num_connections = person_rows[s.CONNECTED_LEGS_COL].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
            self.df.loc[self.df[s.PERSON_ID_COL] == person_id, s.NUM_CONNECTED_LEGS_COL] = num_connections
            logger.debug(f"Person {person_id} has {num_connections} connected legs.")


