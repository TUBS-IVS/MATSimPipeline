import itertools
import os.path
import random
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
        self.df['activity_translated_string'] = self.df[s.LEG_ACTIVITY_COL].map(activity_translation)
        logger.info(f"Translated activities.")

    def write_stats(self, stat_by_columns: list = None):
        logger.info(f"Exporting stats...")

        stat_by_columns = [col for col in s.GEO_COLUMNS if col in self.df.columns]
        stat_by_columns.append(s.LEG_ACTIVITY_COL)
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
        result = self.df[self.df["activity_duration_seconds"] > 0].groupby(s.LEG_ACTIVITY_COL)["activity_duration_seconds"].agg(
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
        result = filtered_df.groupby(s.LEG_ACTIVITY_COL)[s.LEG_DURATION_MINUTES_COL].agg(['mean', 'std']) * 60
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
        result = first_legs.groupby(s.LEG_ACTIVITY_COL)['timestamp'].agg(['mean', 'std'])

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
                activity_time = int(self.activity_times_distribution_seconds()[last_leg[s.LEG_ACTIVITY_COL]] / 60)
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
        """
        persons = self.df.groupby("unique_person_id")
        logger.info(f"Estimating times, where missing, for {len(persons)} persons...")

        first_leg_start_time_distribution = (self.first_leg_start_time_distribution()).astype(int)
        activity_times_distribution_seconds = (self.activity_times_distribution_seconds()).astype(int)
        leg_duration_distribution_seconds = (self.leg_duration_distribution_seconds()).astype(int)

        # Initialize an empty list for updates (significantly faster than updating the original df each time)
        updated_persons = []

        for person_id, person in persons:
            person = person.copy()  # Work on a copy to avoid SettingWithCopyWarning

            if len(person) == 1:
                if pd.isna(person.at[person.index[0], s.LEG_NON_UNIQUE_ID_COL]):
                    logger.debug(f"Person {person_id} has no legs. Skipping...")
                    continue
                # Persons with one leg might be problematic, but impute times for them anyway
                logger.warning(f"Person {person_id} has only one leg.")

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
                if first_missing_time_index == first_index:
                    logger.debug(
                        f"Person {person_id} has no time information, imputation from index {first_missing_time_index}...")
                else:
                    logger.debug(
                        f"Person {person_id} has some time information, imputation from index {first_missing_time_index}...")

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

    def estimate_leg_times(self):
        """
        Estimates leg_start_time and leg_end_time if they are missing, using data from similar persons.
        The function lowers the matching criteria if insufficient similar persons are found.
        """
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
                logger.warning(f"Person {person_id} has only one leg.")

            # Check for negative activity times
            if (person["activity_duration_seconds"] < 0).any():
                first_negative_time_index = person[person["activity_duration_seconds"] < 0].index[0]
                logger.debug(f"Person {person_id} has negative activity times. Removing all times after the first bad time.")
                for col in [s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]:
                    person.loc[first_negative_time_index:, col] = None

            # Process each person
            if person[[s.LEG_START_TIME_COL, s.LEG_END_TIME_COL]].isna().any().any():
                first_index = person.index[0]
                first_missing_time_index = person[person[s.LEG_START_TIME_COL].isna() | person[s.LEG_END_TIME_COL].isna()].index[
                    0]
                if first_missing_time_index == first_index:
                    logger.debug(
                        f"Person {person_id} has no time information, imputation from index {first_missing_time_index}...")
                else:
                    logger.debug(
                        f"Person {person_id} has some time information, imputation from index {first_missing_time_index}...")

                orig_min_similar = 200
                for min_similar in range(orig_min_similar, 2000,
                                         200):  # Find more similar persons if there are too few with good data
                    similar_persons: pd.DataFrame = self.find_similar_persons(person, min_similar)

                    # Filter similar persons for valid data
                    similar_persons_with_last_legs = similar_persons[
                        similar_persons[s.LEG_START_TIME_COL].notna() &
                        similar_persons[s.LEG_DURATION_MINUTES_COL] > 0 &
                        similar_persons[s.LEG_DURATION_MINUTES_COL] <= 300]
                    # Removing rows with na activity durs removes the last leg, so we need a separate df to keep quality
                    similar_persons_no_last_legs = similar_persons_with_last_legs[
                        similar_persons_with_last_legs["activity_duration_seconds"].notna() &
                        similar_persons_with_last_legs["activity_duration_seconds"] > 0]

                    if len(similar_persons_no_last_legs) > orig_min_similar / 2:
                        break
                    else:
                        logger.debug(f"Person {person_id} has too few similar persons. Lowering standards.")
                else:  # No break
                    logger.warning(f"Person {person_id} misses times and has no even slightly similar persons. "
                                   f"Removing the person to avoid errors.")
                    self.df.drop(person.index, inplace=True)
                    continue

                for idx, row in person.iterrows():
                    if idx == first_missing_time_index:
                        if idx == first_index:  # Start of the day TODO: Select by just the fitting activity
                            typical_day_start = similar_persons_with_last_legs[similar_persons[s.LEG_NON_UNIQUE_ID_COL] == 1].sample(1).iloc[0]
                            my_start_time = typical_day_start if pd.isna(person.at[idx, s.LEG_START_TIME_COL]) else \
                                person.at[idx, s.LEG_START_TIME_COL]
                        else:
                            prev_end_time = person.at[idx - 1, s.LEG_END_TIME_COL]
                            my_start_time = prev_end_time + pd.Timedelta(
                                seconds=similar_persons_no_last_legs["activity_duration_seconds"].sample(1).iloc[0])
                    else:
                        prev_end_time = person.at[idx - 1, s.LEG_END_TIME_COL]
                        my_start_time = prev_end_time + pd.Timedelta(
                            seconds=similar_persons_no_last_legs["activity_duration_seconds"].sample(1).iloc[0])

                    person.at[idx, s.LEG_START_TIME_COL] = my_start_time
                    person.at[idx, s.LEG_END_TIME_COL] = my_start_time + pd.Timedelta(
                        minutes=similar_persons_with_last_legs[s.LEG_DURATION_MINUTES_COL].sample(1).iloc[0])

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


    def update_missing_times(self, similar_persons, person, idx, row):
        """
        Update missing leg start or end times using similar person's data.
        """
        similar_legs = similar_persons[similar_persons[s.LEG_ACTIVITY_COL] == row[s.LEG_ACTIVITY_COL]]
        if not similar_legs.empty:
            similar_leg = similar_legs.sample(1).iloc[0]
            person.at[idx, s.LEG_START_TIME_COL] = similar_leg[s.LEG_START_TIME_COL]
            person.at[idx, s.LEG_END_TIME_COL] = similar_leg[s.LEG_END_TIME_COL]


    def find_similar_persons(self, person: pd.DataFrame, min_similar, attributes: list = None):
        """
        Find similar persons (or other entries) based on a dynamic number of matching attributes.
        """
        if attributes is None:
            attributes = [s.H_REGION_TYPE_COL]

        logger.debug(f"Finding similar persons for {person[s.PERSON_ID_COL].iloc[0]}...")
        for min_matches in range(len(attributes), 0, -1):  # Decrease criteria to a minimum of 1 attributes
            attribute_combinations = itertools.combinations(attributes, min_matches)
            for combination in attribute_combinations:
                similar_persons = self.df
                for attr in combination:
                    similar_persons = similar_persons[similar_persons[attr] == person[attr].iloc[0]]

                if len(similar_persons) >= min_similar:
                    logger.debug(f"Found {len(similar_persons)} similar persons for {person[s.PERSON_ID_COL].iloc[0]} "
                                 f"based on {combination}.")
                    return similar_persons.drop(person.index)  # Drop the person itself from the similar persons

        logger.debug(f"Found no similar persons for {person[s.PERSON_ID_COL].iloc[0]}.")
        return pd.DataFrame()  # Return an empty DataFrame if no similar persons found


    def impute_license_status(self):
        """
        Vectorized function to impute license status based on age and statistical probabilities.
        Uses the same representation of license status as the survey data.
        Adds a new column 'imputed_license' to the dataframe.
        :return: None
        """
        logger.info("Imputing license status...")

        self.df['imputed_license'] = s.LICENSE_YES  # Default to no license

        # Calculate license likelihoods
        valid_entries = self.df[self.df[s.HAS_LICENSE_COL].isin([s.LICENSE_YES, s.LICENSE_NO])]
        likelihoods = h.calculate_value_frequencies_df(valid_entries, s.H_CAR_IN_HH_COL, s.HAS_LICENSE_COL)
        licence_likelihood_with_car = likelihoods.at[s.CAR_IN_HH_YES, s.LICENSE_YES]
        licence_likelihood_without_car = likelihoods.at[s.CAR_IN_HH_NO, s.LICENSE_YES]

        # Case where license status is explicitly known
        self.df.loc[self.df[s.HAS_LICENSE_COL] == s.LICENSE_YES, 'imputed_license'] = s.LICENSE_YES
        self.df.loc[
            (self.df[s.HAS_LICENSE_COL] == s.LICENSE_YES) & (self.df[s.PERSON_AGE_COL] < 17), 'imputed_license'] = s.LICENSE_NO
        logger.info(f"Changed {self.df['imputed_license'].eq(s.LICENSE_NO).sum()} license status to no license based on age.")

        # Impute for unknown license cases
        unknown_license = self.df[s.HAS_LICENSE_COL].isin([s.LICENSE_UNKNOWN, s.ADULT_OVER_16_PROXY])
        adults = self.df[s.PERSON_AGE_COL] >= 17
        no_car = self.df[s.H_CAR_IN_HH_COL] == s.CAR_IN_HH_NO

        # Random choice for unknown license with/without car
        condition = unknown_license & adults & no_car
        num_rows = self.df[condition].shape[0]
        self.df.loc[condition, 'imputed_license'] = np.random.choice(
            [s.LICENSE_NO, s.LICENSE_YES], size=num_rows,
            p=[1 - licence_likelihood_without_car, licence_likelihood_without_car]
        )
        logger.info(f"Imputed license status for {num_rows} rows without car.")

        condition = unknown_license & adults & ~no_car
        num_rows = self.df[condition].shape[0]
        self.df.loc[condition, 'imputed_license'] = np.random.choice(
            [s.LICENSE_NO, s.LICENSE_YES], size=num_rows, p=[1 - licence_likelihood_with_car, licence_likelihood_with_car]
        )
        logger.info(f"Imputed license status for {num_rows} rows with car.")
