import os.path
import random
from datetime import timedelta

import geopandas as gpd
import matsim.writers
import numpy as np
import pandas as pd
from shapely.geometry import Point

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

    def distribute_by_weights(self, weights_df, external_id_column, cut_missing_ids=False):
        """
        Distribute data points from `weights_df` across the population dataframe based on weights (e.g. assign buildings to households).

        The function modifies the internal population dataframe by appending the point IDs from the weights dataframe
        based on their weights and the count of each ID in the population dataframe.

        Args:
            weights_df (pd.DataFrame): DataFrame containing the ID of the geography, point IDs, and their weights.
            Must contain all geography IDs in the population dataframe. Is allowed to contain more (they will be skipped).
            external_id_column (str): The column name of the ID in the weights dataframe (e.g. 'BLOCK_NR').
            cut_missing_ids (bool): If True, IDs in the population dataframe that are not in the weights dataframe are cut from the population dataframe.
        """
        logger.info("Starting distribution by weights...")

        if not self.df[external_id_column].isin(weights_df[external_id_column]).all():
            if cut_missing_ids:
                logger.warning(f"Not all geography IDs in the population dataframe are in the weights dataframe. "
                               f"Cutting missing IDs: {set(self.df[external_id_column]) - set(weights_df[external_id_column])}")
                self.df = self.df[self.df[external_id_column].isin(weights_df[external_id_column])]
            else:
                raise ValueError(f"Not all geography IDs in the population dataframe are in the weights dataframe. "
                                 f"Missing IDs: {set(self.df[external_id_column]) - set(weights_df[external_id_column])}")

        # Count of each ID in population_df
        id_counts = self.df[external_id_column].value_counts().reset_index()
        id_counts.columns = [external_id_column, '_processing_count']
        logger.info(f"Computed ID counts for {len(id_counts)} unique IDs.")

        # Merge with weights_df
        weights_df = pd.merge(weights_df, id_counts, on=external_id_column, how='left')

        def distribute_rows(group):
            total_count = group['_processing_count'].iloc[0]
            if total_count == 0 or pd.isna(total_count):
                logger.debug(f"Geography ID {group[external_id_column].iloc[0]} is not in the given dataframe, "
                             f"likely because no person/activity etc. exists there. Skipping distribution for this ID.")
                return []
            # Compute distribution
            group['_processing_repeat_count'] = (group['ewzahl'] / group['ewzahl'].sum()) * total_count
            group['_processing_int_part'] = group['_processing_repeat_count'].astype(int)
            group['_processing_frac_part'] = group['_processing_repeat_count'] - group['_processing_int_part']

            # Distribute remainder
            remainder = total_count - group['_processing_int_part'].sum()
            assert remainder >= 0 and remainder % 1 == 0, f"Remainder is {remainder}, should be a positive integer."
            remainder = int(remainder)
            top_indices = group['_processing_frac_part'].nlargest(remainder).index
            group.loc[top_indices, '_processing_int_part'] += 1

            # Expand rows based on int_part
            expanded = []
            for _, row in group.iterrows():
                expanded.extend([row.to_dict()] * int(row['_processing_int_part']))
            return expanded

        expanded_rows = []
        for _, group in weights_df.groupby(external_id_column):
            expanded_rows.extend(distribute_rows(group))

        expanded_weights_df = pd.DataFrame(expanded_rows).drop(
            columns=['_processing_count', '_processing_repeat_count', '_processing_int_part', '_processing_frac_part'])
        logger.info(f"Generated expanded weights DataFrame with {len(expanded_weights_df)} rows.")
        if len(expanded_weights_df) != self.df.shape[0]:
            raise ValueError(f"Expanded weights DataFrame has {len(expanded_weights_df)} rows, "
                             f"but the population DataFrame has {self.df.shape[0]} rows.")

        # Add a sequence column to both dataframes to prevent cartesian product on merge
        self.df['_processing_seq'] = self.df.groupby(external_id_column).cumcount()
        expanded_weights_df['_processing_seq'] = expanded_weights_df.groupby(external_id_column).cumcount()

        # Merge using the ID column and the sequence
        self.df = pd.merge(self.df, expanded_weights_df, on=[external_id_column, '_processing_seq'],
                           how='left').drop(columns='_processing_seq')

        logger.info("Completed distribution by weights.")

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

            writer.start_population()  # (attributes={"coordinateReferenceSystem": "UTM-32N"})  # TODO: verify CRS everywhere

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
                        writer.add_activity(
                            type=row['activity_translated_string'],
                            x=row["random_point"].x, y=row["random_point"].y,
                            # The writer expects seconds. Also, we mean max_dur here, but the writer doesn't have that yet.
                            start_time=row["activity_duration_seconds"])
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

    def change_last_leg_activity_to_home(self) -> None:
        """
        Change the target activity of the last leg to home. Alternative to add_return_home_leg().
        Assumes LEG_ID is ascending in order of legs (which it is in MiD and should be in other datasets).
        """
        logger.info("Changing last leg activity to home...")
        self.df = self.df.sort_values(by=['unique_household_id', s.PERSON_ID_COL, s.LEG_NON_UNIQUE_ID_COL])

        is_last_leg = self.df[s.PERSON_ID_COL].ne(self.df[s.PERSON_ID_COL].shift(-1))

        number_of_rows_to_change = len(self.df[is_last_leg & (self.df[s.LEG_ACTIVITY_COL] != s.ACTIVITY_HOME)])

        self.df.loc[is_last_leg, s.LEG_ACTIVITY_COL] = s.ACTIVITY_HOME
        logger.info(f"Changed last leg activity to home for {number_of_rows_to_change} of {len(self.df)} rows.")

    def adjust_mode_based_on_age(self):
        """
        Change the mode of transportation from car to ride if age < 17.
        """
        logger.info("Adjusting mode based on age...")
        conditions = (self.df[s.LEG_MAIN_MODE_COL] == s.MODE_CAR) & (self.df[s.PERSON_AGE_COL] < 17)
        self.df.loc[conditions, s.LEG_MAIN_MODE_COL] = s.MODE_RIDE
        logger.info(f"Adjusted mode based on age for {conditions.sum()} of {len(self.df)} rows.")

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

        def random_point_in_polygon(polygon):
            if not polygon.is_valid or polygon.is_empty:
                raise ValueError("Invalid polygon")

            min_x, min_y, max_x, max_y = polygon.bounds

            while True:
                random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
                if polygon.contains(random_point):
                    return random_point

        gdf = gpd.read_file(s.SHAPE_BOUNDARY_FILE)
        polygon = h.find_outer_boundary(gdf)
        self.df['random_point'] = self.df.apply(lambda row: random_point_in_polygon(polygon), axis=1)

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
        Not strictly necessary, but makes the output more readable.
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
        self.df['activity_translated_string'] = self.df[s.LEG_ACTIVITY_COL].map(activity_translation)
        logger.info(f"Translated activities.")

    def write_stats(self):
        logger.info(f"Exporting stats...")

        stat_by_columns = [col for col in s.GEO_COLUMNS if col in self.df.columns]
        stat_by_columns.append(s.LEG_ACTIVITY_COL)
        # stat_by_columns.extend(["unique_household_id", "unique_person_id"])  # Very large files
        # non_stat_by_columns = [col for col in self.df.columns if col not in stat_by_columns]

        for geo_col in stat_by_columns:
            stats_df = self.df.groupby(geo_col).describe()

            # Flattening MultiIndex columns
            stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns]

            file_path = f"{matsim_pipeline_setup.OUTPUT_DIR}/{geo_col}_stats.csv"
            stats_df.to_csv(file_path)
            logger.info(f"Exported stats to {file_path}.")

    def average_activity_times_seconds(self):
        """
        Returns a pd Series with the average duration for each activity type.
        """
        # Ignore negative values
        times = self.df[self.df["activity_duration_seconds"] > 0].groupby(s.LEG_ACTIVITY_COL)["activity_duration_seconds"].mean()
        logger.debug(f"Average activity times in seconds: \n{times}")
        return times

    def average_leg_duration_seconds(self):
        """
        Returns a pd Series with the average leg time (travel time) towards each activity type.
        """
        # Ignore negative values and values > 5 hours (these might be errors or error codes)
        times = self.df[(self.df[s.LEG_DURATION_MINUTES_COL] > 0) & (self.df[s.LEG_DURATION_MINUTES_COL] <= 300)].groupby(
            s.LEG_ACTIVITY_COL)[
                    s.LEG_DURATION_MINUTES_COL].mean() * 60
        logger.debug(f"Average leg times in seconds: \n{times}")
        return times

    def add_return_home_leg(self):
        """
        Add a home leg at the end of the day, if it doesn't exist. Alternative to change_last_leg_target_to_home().
        The length of the activity and the leg duration are estimated.
        Requires is_main_activity() to be run first.
        :return: DataFrame with added home legs
        """
        new_rows = []

        for person_id, group in self.df.groupby(s.PERSON_ID_COL):
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
            try:
                activity_time = int(self.average_activity_times_seconds()[last_leg[s.LEG_ACTIVITY_COL]] / 60)
            except KeyError:
                activity_time = 60  # 1 hour default

            # Create home_leg with the calculated duration
            home_leg = last_leg.copy()
            home_leg[s.LEG_ID_COL] = last_leg['LEG_ID'] + 1
            home_leg["unique_leg_id"] = rules.unique_leg_id(home_leg)
            home_leg[s.LEG_START_TIME_COL] = last_leg[s.LEG_END_TIME_COL] + pd.Timedelta(minutes=activity_time)
            home_leg[s.LEG_END_TIME_COL] = home_leg[s.LEG_START_TIME_COL] + pd.Timedelta(minutes=home_leg_duration)
            home_leg[s.LEG_ACTIVITY_COL] = s.ACTIVITY_HOME
            home_leg[s.LEG_DURATION_MINUTES_COL] = home_leg_duration
            home_leg[s.LEG_DISTANCE_COL] = None  # Could also be estimated, but isn't necessary for the current use case

            new_rows.append(home_leg)

        new_rows_df = pd.DataFrame(new_rows)

        # Sorting by person_id and leg_id_col will insert the new rows in the correct place
        self.df = pd.concat([self.df, new_rows_df]).sort_values([s.PERSON_ID_COL, s.LEG_ID_COL]).reset_index(drop=True)

    def estimate_leg_times(self):
        """
        Estimates leg_start_time and leg_end_time if they are missing.
        Times strings must have been converted to datetime before.
        """
        persons = self.df.groupby("unique_person_id")
        logger.info(f"Estimating times, where missing, for {len(persons)} persons...")

        average_activity_times = (self.average_activity_times_seconds()).astype(int)
        average_leg_times = (self.average_leg_duration_seconds()).astype(int)

        # Initialize an empty list for updates (significantly faster than updating the original df each time)
        updated_persons = []

        for person_id, person in persons:
            person = person.copy()  # Work on a copy to avoid SettingWithCopyWarning

            # Persons with one leg are problematic, but make up times for them anyway
            if len(person) == 1:
                logger.warning(f"Person {person_id} has only one leg. Remove them or add more data.")

            # Check for negative activity times
            if (person["activity_duration_seconds"] < 0).any():
                first_negative_time_index = person[person["activity_duration_seconds"] < 0].index[0]
                logger.debug(f"Person {person_id} has negative activity times. Removing all times after the first bad time.")
                for col in [s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]:
                    person.loc[first_negative_time_index:, col] = None

            if person[[s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]].isna().any().any():
                first_index = person.index[0]
                first_missing_time_index = person[person[s.LEG_START_TIME_COL].isna() | person[s.LEG_END_TIME_COL].isna()].index[
                    0]
                logger.debug(
                    f"Person {person_id} has some time information, filling in the blanks starting from index {first_missing_time_index}...")

                # Start updating times from the first missing time
                for idx in range(first_missing_time_index, first_index + len(person)):
                    if idx == first_missing_time_index:
                        if idx == first_index:  # Start of the day
                            random_day_start = pd.Timestamp(s.BASE_DATE) + pd.Timedelta(hours=random.randint(5, 9),
                                                                                        minutes=random.randint(0, 59))
                            next_start_time = random_day_start if pd.isna(person.at[idx, s.LEG_START_TIME_COL]) else \
                                person.at[idx, s.LEG_START_TIME_COL]
                        else:
                            prev_end_time = person.at[idx - 1, s.LEG_END_TIME_COL]
                            next_start_time = prev_end_time + pd.Timedelta(
                                seconds=average_activity_times[person.at[idx - 1, s.LEG_ACTIVITY_COL]])
                    else:
                        prev_end_time = person.at[idx - 1, s.LEG_END_TIME_COL]
                        next_start_time = prev_end_time + pd.Timedelta(
                            seconds=average_activity_times[person.at[idx - 1, s.LEG_ACTIVITY_COL]])

                    person.at[idx, s.LEG_START_TIME_COL] = next_start_time
                    person.at[idx, s.LEG_END_TIME_COL] = next_start_time + pd.Timedelta(
                        seconds=average_leg_times[person.at[idx, s.LEG_ACTIVITY_COL]])

                logger.debug(f"Person {person_id} updated times: \n{person[[s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]]}")
                if person[[s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]].isna().any().any():
                    logger.warning(f"Person {person_id} still has missing times. "
                                   f"Check the data and try again. Skipping...")
                    continue
                updated_persons.append(person)
        if updated_persons:
            logger.debug(f"Concatenating {len(updated_persons)} updated persons...")
            updated_df = pd.concat(updated_persons)
            logger.debug(f"Updating original df...")
            self.df.update(updated_df)
        logger.info("Time estimation completed.")

    def vary_times_by_person(self, person_id_col, time_cols):
        """
        Varies times in the DataFrame by the same random amount (±3 minutes) for each person.

        :param person_id_col: String, the column name for the unique person identifier.
        :param time_cols: List of strings, the names of the columns containing time data.
        :return: pandas DataFrame with varied times.
        """

        # Apply the random time shift for each person
        logger.info("Varying times by person...")

        def apply_time_shift(group):
            # Generate a random time shift between -3 and +3 minutes
            time_shift = timedelta(minutes=np.random.randint(-3, 4))

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
        Recommended to sample households, not persons, to keep the household structure intact.
        """
        logger.info("Downsampling population...")
        self.df = self.df.sample(frac=sample_percentage)
        logger.info(f"Downsampled population to {sample_percentage * 100}% of the original population.")
