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

    def distribute_by_weights(self, weights_df: pd.DataFrame, cell_id_col: str, cut_missing_ids: bool = False):
        self.df = h.distribute_by_weights(self.df, weights_df, cell_id_col, cut_missing_ids)

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

            writer.start_population(attributes={"coordinateReferenceSystem": "UTM-32N"})  # TODO: verify CRS everywhere

            for _, group in self.df.groupby(['unique_person_id']):
                writer.start_person(group['unique_person_id'].iloc[0])
                writer.start_plan(selected=True)

                # Add home activity
                writer.add_activity(
                    type="home",
                    x=group['home_loc'].iloc[0].x, y=group['home_loc'].iloc[0].y,
                    end_time=h.seconds_from_datetime(group[s.LEG_START_TIME_COL].iloc[0]))
                # One row in the df contains the leg and the following activity
                for idx, row in group.iterrows():
                    writer.add_leg(mode=row['mode_translated_string'])
                    if not pd.isna(row['activity_duration_seconds']):
                        # Create an own activity type for each duration (for correct matsim scoring)
                        # Rounding must fit the matsim config
                        max_dur: int = round(row["activity_duration_seconds"] / 600) * 600
                        writer.add_activity(
                            type=f"{row['activity_translated_string']}_{max_dur}",
                            x=row["random_point"].x, y=row["random_point"].y,
                            # The writer expects seconds. Also, we mean max_dur here, but the writer doesn't have that yet.
                            start_time=max_dur)
                    else:
                        # No time for the last activity
                        writer.add_activity(
                            type=row['activity_translated_string'],
                            x=row["random_point"].x, y=row["random_point"].y)

                writer.end_plan()
                writer.end_person()

            writer.end_population()
        logger.info(f"Wrote plans to MATSim xml: {output_file}")
        return output_file

    def write_households_to_matsim_xml(self):  # TODO: finish
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
                households_writer.add_vehicles(vehicle_ids)
                households_writer.end_household()

            households_writer.end_households()

    def write_facilities_to_matsim_xml(self, facilities_df):  # TODO: finish
        logger.info("Writing facilities to MATSim xml...")
        output_file = os.path.join(matsim_pipeline_setup.OUTPUT_DIR, "facilities.xml")
        with open(output_file, 'wb+') as f_write:
            facilities_writer = matsim.writers.FacilitiesWriter(f_write)
            facilities_writer.start_facilities()

            for _, row in self.df.iterrows():
                facility_id = row['facility_id']
                x = row['x']
                y = row['y']
                activities = row['activities']  # Assuming this is a list of activities

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
                for vehicle_id in vehicle_ids:
                    vehicle_writer.add_vehicle(vehicle_id=vehicle_id, vehicle_type="car")

            vehicle_writer.end_vehicle_definitions()

    def change_last_leg_activity_to_home(self) -> None:
        """
        Change the target activity of the last leg to home. Alternative to add_return_home_leg().
        Assumes LEG_ID is ascending in order of legs (which it is in MiD and should be in other datasets).
        """
        logger.info("Changing last leg activity to home...")
        self.df = self.df.sort_values(by=['unique_household_id', s.PERSON_ID_COL, s.LEG_NON_UNIQUE_ID_COL])

        is_last_leg = self.df[s.PERSON_ID_COL].ne(self.df[s.PERSON_ID_COL].shift(-1))

        number_of_rows_to_change = len(self.df[is_last_leg & (self.df[s.LEG_TO_ACTIVITY_COL] != s.ACTIVITY_HOME)])

        self.df.loc[is_last_leg, s.LEG_TO_ACTIVITY_COL] = s.ACTIVITY_HOME
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
        conditions = (self.df[s.LEG_MAIN_MODE_COL] == s.MODE_UNDEFINED) & (isinstance(self.df["connected_legs"], list))
        self.df.loc[conditions, s.LEG_MAIN_MODE_COL] = s.MODE_RIDE
        logger.info(f"Adjusted mode based on connected legs for {conditions.sum()} of {len(self.df)} rows.")

    def calculate_activity_duration(self):
        """
        Calculate the time between the end of one leg and the start of the next leg seconds.
        :return:
        """
        self.df.sort_values(by=['unique_household_id', s.PERSON_ID_COL, s.LEG_NON_UNIQUE_ID_COL], inplace=True,
                            ignore_index=True)

        # Group by person and calculate the time difference within each group
        self.df['activity_duration_seconds'] = self.df.groupby(s.PERSON_ID_COL)[s.LEG_START_TIME_COL].shift(-1) - self.df[
            s.LEG_END_TIME_COL]

        self.df['activity_duration_seconds'] = self.df['activity_duration_seconds'].dt.total_seconds()
        self.df['activity_duration_seconds'] = pd.to_numeric(self.df['activity_duration_seconds'], downcast='integer',
                                                             errors='coerce')

        # Set the activity time of the last leg to None
        is_last_leg = self.df["unique_person_id"] != self.df["unique_person_id"].shift(-1)
        self.df.loc[is_last_leg, 'activity_duration_seconds'] = None

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
        if count_non_matching > 0:
            logger.warning(f"{count_non_matching} rows have a mode that is not in the defined modes."
                           f"They will not be translated. This might cause errors in MATSim.")
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
        self.df['activity_translated_string'] = self.df[s.LEG_TO_ACTIVITY_COL].map(activity_translation)
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
        result = self.df[self.df["activity_duration_seconds"] > 0].groupby(s.LEG_TO_ACTIVITY_COL)[
            "activity_duration_seconds"].agg(
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
                'activity_duration_seconds'].mean()
            if pd.isna(activity_time):
                logger.info(f"Person {person_id} has no similar persons with the same last activity. ")
                activity_time = 3600  # 1 hour default

            # Create home_leg with the calculated duration
            home_leg = last_leg.copy()
            home_leg[s.LEG_NON_UNIQUE_ID_COL] = last_leg[s.LEG_NON_UNIQUE_ID_COL] + 1
            home_leg["unique_leg_id"] = rules.unique_leg_id(home_leg)
            home_leg[s.LEG_START_TIME_COL] = last_leg[s.LEG_END_TIME_COL] + pd.Timedelta(seconds=activity_time)
            home_leg[s.LEG_END_TIME_COL] = home_leg[s.LEG_START_TIME_COL] + pd.Timedelta(minutes=home_leg_duration)
            home_leg[s.LEG_TO_ACTIVITY_COL] = s.ACTIVITY_HOME
            home_leg[s.LEG_DURATION_MINUTES_COL] = home_leg_duration
            home_leg[s.LEG_DISTANCE_COL] = None  # Could also be estimated, but isn't necessary for the current use case
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
    #         if (person["activity_duration_seconds"] < 0).any():
    #             first_negative_time_index = person[person["activity_duration_seconds"] < 0].index[0]
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
            if (person["activity_duration_seconds"] < 0).any():
                first_bad_time_index = person[person["activity_duration_seconds"] < 0].index[0]
                logger.debug(f"Person {person_id} has negative activity times. Removing all times after the first bad time.")
                for col in [s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]:
                    person.loc[first_bad_time_index:, col] = None

            # Check for bad leg times (MiD-codes)
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
                        (similar_persons_with_last_legs["activity_duration_seconds"].notna()) &
                        (similar_persons_with_last_legs["activity_duration_seconds"] > 0)]

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
                                    "activity_duration_seconds"]
                                if similar_persons_same_activity.empty:
                                    logger.info(f"Person {person_id} has no similar persons with the same activity. "
                                                f"Lowering standards.")
                                    similar_persons_same_activity = similar_persons_no_last_legs["activity_duration_seconds"]
                                my_start_time = prev_end_time + pd.Timedelta(
                                    # Sample an activity duration from a similar person with the same activity
                                    seconds=similar_persons_same_activity.sample(1).iloc[0])

                            # Leg duration
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

    def vary_times_by_household(self, person_id_col, time_cols, max_shift_minutes=3):
        """
        Varies times in the DataFrame by the same random amount (±max_shift_minutes) for each household.

        :param person_id_col: String, the column name for the unique person identifier.
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
        self.df = self.df.groupby(person_id_col).apply(apply_time_shift)
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
                similar_persons = self.df
                for attr in combination:
                    similar_persons = similar_persons[similar_persons[attr] == person[attr]]

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
        Close connected leg groups by assigning the same total connected leg ids to all legs in the group.
        """
        logger.info("Closing connected leg groups...")

        for household_id, household_group in self.df.groupby(s.UNIQUE_HH_ID_COL):
            if household_group['connected_legs'].isna().all():
                logger.debug(f"Household {household_id} has no connected legs. Skipping...")
                continue
            else:
                logger.debug(f"Household {household_id} has connected legs. Closing...")

            checked_legs = set()
            for idx, row in household_group.iterrows():
                if not isinstance(row['connected_legs'], list):  # Checking for NaN doesn't work here
                    continue
                if row["unique_leg_id"] in checked_legs:
                    continue

                queue = deque([row["unique_leg_id"]])
                connected_legs = set()

                # Collect all connected legs
                while queue:
                    current_leg = queue.popleft()
                    connected_legs.add(current_leg)

                    connected = set(
                        household_group.loc[household_group["unique_leg_id"] == current_leg, 'connected_legs'].iloc[0])
                    for leg in connected:
                        if leg not in connected_legs:
                            queue.append(leg)

                # Assign the same total connected leg ids to all legs in the group
                self.df.loc[self.df["unique_leg_id"].isin(connected_legs), 'connected_legs'] = list(connected_legs)
                checked_legs = checked_legs.union(connected_legs)

        logger.info("Closed connected leg groups.")

    def add_from_activity(self):
        """
        Add a 'from_activity' column to the DataFrame, which is the to_activity of the previous leg.
        For the first leg of each person, set 'from_activity' based on 'starts_at_home' (-> home or unspecified).
        :return:
        """
        logger.info("Adding from_activity column...")
        # Sort the DataFrame by person ID and leg number (the df should usually already be sorted this way)
        self.df.sort_values(by=[s.UNIQUE_P_ID_COL, s.LEG_NON_UNIQUE_ID_COL], inplace=True)

        # Shift the 'to_activity' down to create 'from_activity' for each group
        self.df[s.LEG_FROM_ACTIVITY_COL] = self.df.groupby(s.UNIQUE_P_ID_COL)[s.LEG_TO_ACTIVITY_COL].shift(1)

        # For the first leg of each person, set 'from_activity' based on 'starts_at_home'
        self.df.loc[(self.df[s.LEG_NON_UNIQUE_ID_COL] == 1) & (
                self.df[s.FIRST_LEG_STARTS_AT_HOME_COL] == s.FIRST_LEG_STARTS_AT_HOME), s.LEG_FROM_ACTIVITY_COL] = s.ACTIVITY_HOME
        self.df.loc[(self.df[s.LEG_NON_UNIQUE_ID_COL] == 1) & (
                self.df[
                    s.FIRST_LEG_STARTS_AT_HOME_COL] != s.FIRST_LEG_STARTS_AT_HOME), s.LEG_FROM_ACTIVITY_COL] = s.ACTIVITY_UNSPECIFIED

        # Handle cases with no legs (NA in leg_number)
        self.df.loc[self.df[s.LEG_NON_UNIQUE_ID_COL].isna(), s.LEG_FROM_ACTIVITY_COL] = None
        logger.info("Added from_activity column.")

    def calculate_slack_factors(self):
        slack_factors = []

        for person_id, person_trips in self.df.groupby(s.UNIQUE_P_ID_COL):
            logger.debug(f"Searching slack factors at person {person_id}...")
            # Sort by ordered_id to ensure sequence
            person_trips = person_trips.sort_values(by=s.LEG_NON_UNIQUE_ID_COL)

            # Find indirect routes by checking consecutive legs
            for i in range(len(person_trips) - 1):
                first_leg = person_trips.iloc[i]
                second_leg = person_trips.iloc[i + 1]

                # This should always be true, except for missing data
                if first_leg[s.LEG_TO_ACTIVITY_COL] == second_leg[s.LEG_FROM_ACTIVITY_COL]:

                    direct_trip = self.df[
                        (self.df[s.UNIQUE_P_ID_COL] == person_id) &
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
                        direct_distance = direct_trip.iloc[0][s.LEG_DURATION_MINUTES_COL]
                        indirect_distance = first_leg[s.LEG_DURATION_MINUTES_COL] + second_leg[s.LEG_DURATION_MINUTES_COL]
                        slack_factor = indirect_distance / direct_distance
                        if slack_factor < 1 or slack_factor > 50:
                            logger.debug(f"Found an unrealistic slack factor of {slack_factor} for person {person_id} "
                                         f"Skipping...")
                            continue
                        slack_factors.append((person_id,
                                              first_leg[s.H_REGION_TYPE_COL],
                                              first_leg[s.PERSON_AGE_COL],
                                              first_leg[s.LEG_FROM_ACTIVITY_COL],
                                              first_leg[s.LEG_TO_ACTIVITY_COL],
                                              second_leg[s.LEG_TO_ACTIVITY_COL],
                                              slack_factor))
                        logger.debug(f"Found a slack factor of {slack_factor} for person {person_id} ")

        return pd.DataFrame(slack_factors, columns=[s.UNIQUE_P_ID_COL,
                                                    s.H_REGION_TYPE_COL,
                                                    s.PERSON_AGE_COL,
                                                    'start_activity',
                                                    'via_activity',
                                                    'end_activity',
                                                    'slack_factor'])

    def list_cars_in_household(self):
        """
        Create a list of cars with unique ids in each household and add it to the DataFrame.
        """
        logger.info("Listing cars in household...")
        # Group by household
        hhs = self.df.groupby(s.UNIQUE_HH_ID_COL)
        total_cars = 0
        for household_id, hh in hhs:
            number_of_cars: int = hh[s.H_NUMBER_OF_CARS_COL].iloc[0]
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

        logger.info(f"Listed {total_cars} cars in {len(hhs)} households.")

    def impute_cars_in_household(self):  # TODO: add some actual imputation
        """
        Impute the number of cars in a household if unknown, based on the number of cars in similar households.
        """
        self.df.loc[self.df[s.H_NUMBER_OF_CARS_COL] == 99, s.H_NUMBER_OF_CARS_COL] = None
        logger.info(f"Imputing cars in household for {self.df[s.H_NUMBER_OF_CARS_COL].isna().sum()} of "
                    f"{len(self.df)} rows...")

        # Set all other unknown values to 0
        self.df.loc[self.df[s.H_NUMBER_OF_CARS_COL].isna, s.H_NUMBER_OF_CARS_COL] = 0

    def mark_mirroring_main_activities(self, duration_threshold=60):  # TODO sat. morning: finish this
        """
        Mark activities that mirror the peron's main activity; activities that still likely represent the same main activity, but
        are separated by a different, short, activity (e.g. a lunch break between two work activities).
        :param duration_threshold:
        :return:
        """
        # Vectorized because it's insanely faster than looping
        # Create shifted columns for comparison
        self.df['next_person_id'] = self.df['person_id'].shift(-1)
        self.df['next_act_dur'] = self.df['act_dur'].shift(-1)
        self.df['next_next_activity'] = self.df['activity'].shift(-2)
        self.df['next_next_person_id'] = self.df['person_id'].shift(-2)

        # Condition for short duration activity following a main activity
        short_duration_condition = (self.df['is_main'] == 1) & \
                                   (self.df['next_person_id'] == self.df['person_id']) & \
                                   (self.df['next_act_dur'] < duration_threshold)

        # Condition for the same activity after the short duration activity
        same_activity_condition = (self.df['next_next_activity'] == self.df['activity']) & \
                                  (self.df['next_next_person_id'] == self.df['person_id'])

        # Combine conditions and assign to the new column
        self.df['mirrors_main_activity'] = ((short_duration_condition & same_activity_condition).shift(2).fillna(False)).astype(
            int)

        # Drop the temporary shifted columns
        self.df.drop(['next_person_id', 'next_act_dur', 'next_next_activity', 'next_next_person_id'], axis=1, inplace=True)
