import itertools
from collections import deque
import numpy as np
import pandas as pd
from os import path
import ivs_helpers as ivs
from tqdm import tqdm

from utils import settings as s, pipeline_setup, helpers as h
from utils.data_frame_processor import DataFrameProcessor
from utils.stats_tracker import stats_tracker
from utils.logger import logging

logger = logging.getLogger(__name__)

class MiDDataEnhancer(DataFrameProcessor):
    """
    Collects all methods that are applied only, directly, to the unexpanded, raw MiD data.
    """

    def __init__(self, df: pd.DataFrame = None, id_column: str = None):
        super().__init__(df, id_column)
        # self.sf = h.SlackFactors(s.SLACK_FACTORS_FILE)

    def filter_home_to_home_legs(self):  # TODO: remove?
        """
        Filters out 'home to home' legs from the DataFrame.
        """
        logger.info(f"Filtering out 'home to home' legs from {len(self.df)} rows...")
        home_to_home_condition = (self.df[s.ACT_FROM_INTERNAL_COL] == s.ACT_HOME) & \
                                 (self.df[s.ACT_TO_INTERNAL_COL] == s.ACT_HOME)
        self.df = self.df[~home_to_home_condition].reset_index(drop=True)
        logger.info(f"Filtered out 'home to home' legs. {len(self.df)} rows remaining.")

    def reset_leg_ids(self):
        """
        Resets the leg ids to start from 1 for each person.
        """
        logger.info("Resetting leg ids...")
        # Sort by person id and leg id
        self.df.sort_values(by=[s.PERSON_ID_COL, s.LEG_NON_UNIQUE_ID_COL], inplace=True, ignore_index=True)
        self.df[s.LEG_NON_UNIQUE_ID_COL] = self.df.groupby(s.PERSON_ID_COL).cumcount() + 1
        logger.info("Reset leg ids.")

    def convert_minutes_to_seconds(self, minute_col, seconds_col):
        logger.info(f"Converting {minute_col} to seconds...")

        self.df[minute_col] = pd.to_numeric(self.df[minute_col], errors='coerce')
        nan_count = self.df[minute_col].isna().sum()
        if nan_count > 0:
            logger.warning(f"There were {nan_count} non-numeric values in {minute_col} which have been set to NaN.")
        if seconds_col not in self.df.columns:
            self.df[seconds_col] = self.df[minute_col] * 60
            logger.info(f"Created new column {seconds_col}.")
        else:
            self.df[seconds_col] = self.df[minute_col] * 60
            logger.info(f"Overwrote existing column {seconds_col}.")

    def convert_hours_to_seconds(self, hour_col, seconds_col):
        logger.info(f"Converting {hour_col} to seconds...")
        self.df[hour_col] = self.df[hour_col].str.replace(',', '.')

        self.df[hour_col] = pd.to_numeric(self.df[hour_col], errors='coerce')
        nan_count = self.df[hour_col].isna().sum()
        if nan_count > 0:
            logger.warning(f"There were {nan_count} non-numeric values in {hour_col} which have been set to NaN.")
        if seconds_col not in self.df.columns:
            self.df[seconds_col] = self.df[hour_col] * 3600
            logger.info(f"Created new column {seconds_col}.")
        else:
            self.df[seconds_col] = self.df[hour_col] * 3600
            logger.info(f"Overwrote existing column {seconds_col}.")

    def convert_kilometers_to_meters(self, km_col, m_col):
        logger.info(f"Converting {km_col} to meters...")
        # Needed for string replacement
        self.df[km_col] = self.df[km_col].astype(str)
        # Replace commas with dots for decimal conversion
        self.df[km_col] = self.df[km_col].str.replace(',', '.')
        self.df[km_col] = pd.to_numeric(self.df[km_col], errors='coerce')
        nan_count = self.df[km_col].isna().sum()
        if nan_count > 0:
            logger.warning(f"There were {nan_count} non-numeric values in {km_col} which have been set to NaN.")

        if m_col not in self.df.columns:
            self.df[m_col] = self.df[km_col] * 1000
            logger.info(f"Created new column {m_col}.")
        else:
            self.df[m_col] = self.df[km_col] * 1000
            logger.info(f"Overwrote existing column {m_col}.")

    def adjust_mode_based_on_age(self):
        """
        Change the mode of transportation from car to ride if age < 17.
        """
        logger.info("Adjusting mode based on age...")
        conditions = (self.df[s.MODE_INTERNAL_COL] == s.MODE_CAR) & (self.df[s.PERSON_AGE_COL] < 17)
        self.df.loc[conditions, s.MODE_INTERNAL_COL] = s.MODE_RIDE
        logger.info(f"Adjusted mode based on age for {conditions.sum()} of {len(self.df)} rows.")

    def adjust_mode_based_on_license(self):
        """
        Change the mode of transportation from car to ride if person has no license.
        """
        logger.info("Adjusting mode based on license...")
        conditions = (self.df[s.MODE_INTERNAL_COL] == s.MODE_CAR) & (self.df["imputed_license"] == s.LICENSE_NO)
        self.df.loc[conditions, s.MODE_INTERNAL_COL] = s.MODE_RIDE
        logger.info(f"Adjusted mode based on license for {conditions.sum()} of {len(self.df)} rows.")

    def adjust_mode_based_on_connected_legs(self):
        """
        Change the mode of transportation from undefined to ride if the leg is connected to other legs.
        This works because connection analysis only matches undefined legs to car legs.
        """
        logger.info("Adjusting mode based on connected legs...")
        conditions = (self.df[s.MODE_INTERNAL_COL] == s.MODE_UNDEFINED) & (
            isinstance(self.df[s.CONNECTED_LEGS_COL], list))
        self.df.loc[conditions, s.MODE_INTERNAL_COL] = s.MODE_RIDE
        logger.info(f"Adjusted mode based on connected legs for {conditions.sum()} of {len(self.df)} rows.")

    def calculate_activity_duration(self):
        """
        Calculate the time between the end of one leg and the start of the next leg seconds.
        Writes to a separate column different to the given MiD duration.
        :return:
        """
        logger.info("Calculating activity duration...")
        self.df.sort_values(by=['unique_household_id', s.PERSON_ID_COL, s.LEG_NON_UNIQUE_ID_COL], inplace=True,
                            ignore_index=True)

        # Group by person and calculate the time difference within each group
        self.df[s.ACT_DUR_SECONDS_COL] = self.df.groupby(s.PERSON_ID_COL)[s.LEG_START_TIME_COL].shift(-1) - \
                                         self.df[
                                             s.LEG_END_TIME_COL]

        self.df[s.ACT_DUR_SECONDS_COL] = self.df[s.ACT_DUR_SECONDS_COL].dt.total_seconds()
        self.df[s.ACT_DUR_SECONDS_COL] = pd.to_numeric(self.df[s.ACT_DUR_SECONDS_COL],
                                                       downcast='integer',
                                                       errors='coerce')

        # Set the activity time of the last leg to None
        is_last_leg = self.df["unique_person_id"] != self.df["unique_person_id"].shift(-1)
        self.df.loc[is_last_leg, s.ACT_DUR_SECONDS_COL] = None
        logger.info(f"Calculated activity duration in secs.")

    def write_short_overview(self):
        """
        Generates summary statistics before estimating leg times.
        """
        # Group by unique person ID
        persons = self.df.groupby("unique_person_id")

        num_persons = len(persons)
        num_legs = self.df[s.LEG_NON_UNIQUE_ID_COL].notna().sum()
        num_persons_one_leg = sum(
            len(person) == 1 and pd.notna(person[s.LEG_NON_UNIQUE_ID_COL]).any() for _, person in persons)
        num_persons_no_leg = sum(pd.isna(person[s.LEG_NON_UNIQUE_ID_COL]).all() for _, person in persons)

        logger.info(f"Number of persons: {num_persons}")
        logger.info(f"Number of legs: {num_legs}")
        logger.info(f"Number of persons with one leg: {num_persons_one_leg}")
        logger.info(f"Number of persons with no leg: {num_persons_no_leg}")
        stats_tracker.log("num_persons", num_persons)
        stats_tracker.log("num_legs", num_legs)
        stats_tracker.log("num_persons_one_leg", num_persons_one_leg)
        stats_tracker.log("num_persons_no_leg", num_persons_no_leg)

    def write_overview(self):
        logger.info(f"Exporting stats...")

        stat_by_columns = [col for col in s.GEO_COLUMNS if col in self.df.columns]
        stat_by_columns.append(s.ACT_TO_INTERNAL_COL)
        # stat_by_columns.extend(["unique_household_id", "unique_person_id"])  # Very large files
        # non_stat_by_columns = [col for col in self.df.columns if col not in stat_by_columns]

        for col in stat_by_columns:
            stats_df = self.df.groupby(col).describe()

            # Flattening MultiIndex columns
            stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns]

            file_path = f"{pipeline_setup.OUTPUT_DIR}/{col}_stats.csv"
            stats_df.to_csv(file_path)
            logger.info(f"Exported stats to {file_path}.")

    def activity_times_distribution_seconds(self):
        """
        Returns a pd DataFrame with the average duration and standard deviation for each activity type.
        """
        # Ignore negative and zero values
        result = self.df[self.df[s.ACT_DUR_SECONDS_COL] > 0].groupby(s.ACT_TO_INTERNAL_COL)[
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
        result = filtered_df.groupby(s.ACT_TO_INTERNAL_COL)[s.LEG_DURATION_MINUTES_COL].agg(['mean', 'std']) * 60
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

        result = first_legs.groupby(s.ACT_TO_INTERNAL_COL)['timestamp'].agg(['mean', 'std'])

        result['mean'] = pd.to_datetime(result['mean'])
        result['std'] = pd.to_timedelta(result['std'])

        logger.info(f"First leg start time distribution (mean and std): \n{result}")
        return result

    def add_return_home_leg(self):  # TODO: rework (if this is needed at all)
        """
        Add a home leg at the end of the day, if it doesn't exist. Alternative to change_last_leg_target_to_home().
        The length of the activity and the leg duration are estimated.
        Requires is_main_activity() to be run first.
        :return: DataFrame with added home legs
        """
        logger.info("Adding return home legs...")
        self.df[s.IS_IMPUTED_LEG_COL] = 0
        new_rows = []

        for person_id, group in self.df.groupby(s.UNIQUE_P_ID_COL):
            if pd.isna(group.at[group.index[0], s.LEG_NON_UNIQUE_ID_COL]):
                logger.debug(f"Person {person_id} has no legs. Skipping...")
                continue
            if group[s.ACT_TO_INTERNAL_COL].iloc[-1] == s.ACT_HOME:
                logger.debug(f"Person {person_id} already has a home leg. Skipping...")
                continue
            logger.debug(f"Adding return home leg for person {person_id}...")
            main_activity_index = group[group['is_main_activity'] == 1].index[
                0]  # There should only be one main activity
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
            activity_time = similar_persons[similar_persons[s.ACT_TO_INTERNAL_COL] == last_leg[s.ACT_TO_INTERNAL_COL]][
                s.ACT_DUR_SECONDS_COL].mean()
            if pd.isna(activity_time):
                logger.debug(f"Person {person_id} has no similar persons with the same last activity. ")
                activity_time = 3600  # 1 hour default

            # Create home_leg with the calculated duration
            home_leg = last_leg.copy()
            home_leg[s.LEG_NON_UNIQUE_ID_COL] = last_leg[s.LEG_NON_UNIQUE_ID_COL] + 1
            home_leg[s.UNIQUE_LEG_ID_COL] = f"{home_leg['unique_person_id']}_{home_leg[s.LEG_ID_COL]}"
            home_leg[s.LEG_START_TIME_COL] = last_leg[s.LEG_END_TIME_COL] + pd.Timedelta(seconds=activity_time)
            home_leg[s.LEG_END_TIME_COL] = home_leg[s.LEG_START_TIME_COL] + pd.Timedelta(minutes=home_leg_duration)
            home_leg[s.ACT_TO_INTERNAL_COL] = s.ACT_HOME
            home_leg[s.LEG_DURATION_MINUTES_COL] = home_leg_duration
            # home_leg[s.LEG_DISTANCE_KM_COL] = None
            home_leg[s.IS_MAIN_ACTIVITY_COL] = 0
            home_leg[s.IS_IMPUTED_LEG_COL] = 1

            new_rows.append(home_leg)

        new_rows_df = pd.DataFrame(new_rows)
        logger.info(
            f"Adding {len(new_rows_df)} return home legs for {len(self.df[s.UNIQUE_P_ID_COL].unique())} persons.")

        # Sorting by person_id and leg_id_col will insert the new rows in the correct place
        self.df = pd.concat([self.df, new_rows_df]).sort_values([s.UNIQUE_P_ID_COL, s.LEG_ID_COL]).reset_index(
            drop=True)
        logger.info(f"Added return home legs.")

    def mark_bad_times_as_nan(self):
        """
        Identifies and marks "bad" times in the dataframe for each person's legs as NaN.
        """
        # Check for bad start times
        bad_start_time_indices = self.df[self.df[s.LEG_START_TIME_COL].isna()].index
        # Check for bad end times
        bad_end_time_indices = self.df[self.df[s.LEG_END_TIME_COL].isna() |
                                       (self.df[s.ACT_DUR_SECONDS_COL] < 0) |
                                       (self.df["wegmin"] == 9994) |
                                       (self.df["wegmin"] == 9995)
                                       ].index
        # Check for bad duration times
        # Note: NaN activity durations aren't bad in the last leg (and the correction function will skip them)
        bad_duration_indices = self.df[(self.df[s.ACT_DUR_SECONDS_COL].isna()) |
                                       (self.df[s.ACT_DUR_SECONDS_COL] < 0) |
                                       (self.df[s.ACT_DUR_SECONDS_COL] > 86400) |
                                       (self.df["wegmin"] == 9994) |
                                       (self.df["wegmin"] == 9995)
                                       ].index

        # Mark bad times as NaN
        self.df.loc[bad_start_time_indices, s.LEG_START_TIME_COL] = pd.NA
        self.df.loc[bad_end_time_indices, s.LEG_END_TIME_COL] = pd.NA
        self.df.loc[bad_duration_indices, s.ACT_DUR_SECONDS_COL] = pd.NA

        # Log information about bad times marked as NaN
        if len(bad_start_time_indices) > 0 or len(bad_end_time_indices) > 0 or len(bad_duration_indices) > 0:
            logger.info(f"Number of bad times:")
            logger.info(f"Bad start times: {len(bad_start_time_indices)}")
            logger.info(f"Bad end times: {len(bad_end_time_indices)}")
            logger.info(f"Bad duration times (incl. last leg): {len(bad_duration_indices)}")
            logger.debug(f"Marked bad times as NaN at indices: "
                         f"start_time: {bad_start_time_indices.tolist()}, "
                         f"end_time: {bad_end_time_indices.tolist()}, "
                         f"duration: {bad_duration_indices.tolist()}")

    def correct_times(self):
        """
        Corrects the times in the dataframe for each person's legs.
        Identifies and fixes any NaN times by finding similar persons' activities.
        """
        # use tdqm for progress bar
        for person in tqdm(self.df[s.UNIQUE_P_ID_COL].unique(), desc="Correcting times"):
            person_legs = self.df[self.df[s.UNIQUE_P_ID_COL] == person]

            # Skip person with no mobility
            if pd.isna(person_legs.at[person_legs.index[0], s.LEG_NON_UNIQUE_ID_COL]):
                if len(person_legs) == 1:
                    continue
                else:
                    # A person with several legs must have leg_ids (or sth is seriously wrong)
                    raise ValueError(f"Person {person} has mobility, but no leg_id.")

            if len(person_legs) > 1:
                legs_except_last = person_legs.iloc[:-1]
                last_leg = person_legs.iloc[-1:]

                bad_time_indices_except_last = legs_except_last[
                    (legs_except_last[s.LEG_START_TIME_COL].isna()) |
                    (legs_except_last[s.LEG_END_TIME_COL].isna()) |
                    (legs_except_last[s.ACT_DUR_SECONDS_COL].isna())
                    ].index

                # For the last leg, missing activity time is ok
                bad_time_indices_last = last_leg[
                    (last_leg[s.LEG_START_TIME_COL].isna()) |
                    (last_leg[s.LEG_END_TIME_COL].isna())
                    ].index

                # Merge the indices
                bad_time_indices = bad_time_indices_except_last.union(bad_time_indices_last)
                if len(bad_time_indices) == 0:
                    continue

                if (legs_except_last[s.ACT_DUR_SECONDS_COL].isna()).any() or (
                        person_legs[s.LEG_START_TIME_COL].isna()).any():
                    similar_persons = self.find_similar_persons_with_activities(
                        person_legs.iloc[0], person_legs[s.ACT_TO_INTERNAL_COL].tolist(), cols_to_check_nan=[
                            s.LEG_START_TIME_COL, s.LEG_END_TIME_COL, s.ACT_DUR_SECONDS_COL])

            else:
                # Only one leg which thus is the last leg, missing activity time is ok
                bad_time_indices = person_legs[
                    (person_legs[s.LEG_START_TIME_COL].isna()) |
                    (person_legs[s.LEG_END_TIME_COL].isna())
                    ].index
                if len(bad_time_indices) == 0:
                    continue

                if person_legs[s.LEG_START_TIME_COL].isna().any():
                    similar_persons = self.find_similar_persons_with_activities(
                        person_legs.iloc[0], person_legs[s.ACT_TO_INTERNAL_COL].tolist(), cols_to_check_nan=[
                            s.LEG_START_TIME_COL, s.LEG_END_TIME_COL, s.ACT_DUR_SECONDS_COL])

            start_index = person_legs.index.get_loc(bad_time_indices[0])
            old_times = person_legs[
                [s.LEG_START_TIME_COL, s.LEG_END_TIME_COL, s.ACT_DUR_SECONDS_COL, s.ACT_TO_INTERNAL_COL]].to_string(
                index=False)
            logger.debug(f"Processing person_id {person} starting at leg index {start_index}")
            logger.debug(f"Old times: \n"
                         f" {old_times}")

            for i in range(start_index, len(person_legs)):
                current_leg_index = person_legs.index[i]
                current_leg = person_legs.loc[current_leg_index]

                if i == 0:
                    if pd.isna(current_leg[s.LEG_START_TIME_COL]):
                        # Try to get a similar person, where the day's first activity is our first activity
                        # Similar persons will only be referenced before assignment if things are really wrong ->
                        # Then we want an error anyway
                        try:
                            matching_similar_persons = [sp for _, sp in similar_persons.iterrows() if
                                                        self.df[
                                                            self.df[s.UNIQUE_P_ID_COL] == sp[s.UNIQUE_P_ID_COL]].iloc[
                                                            0][s.ACT_TO_INTERNAL_COL] ==
                                                        current_leg[s.ACT_TO_INTERNAL_COL]]
                            similar_person = matching_similar_persons[0]
                        except IndexError:
                            similar_person = similar_persons.iloc[0]

                        similar_person_legs = self.df[self.df[s.UNIQUE_P_ID_COL] == similar_person[s.UNIQUE_P_ID_COL]]
                        self.df.loc[current_leg_index, s.LEG_START_TIME_COL] = similar_person_legs.iloc[0][
                            s.LEG_START_TIME_COL]

                    self.df.loc[current_leg_index, s.LEG_END_TIME_COL] = self.df.loc[
                                                                             current_leg_index, s.LEG_START_TIME_COL] + pd.Timedelta(
                        seconds=current_leg[
                            s.LEG_DURATION_SECONDS_COL])
                else:
                    previous_leg_index = person_legs.index[i - 1]
                    previous_end_time = self.df.loc[previous_leg_index, s.LEG_END_TIME_COL]
                    previous_activity_time = self.df.loc[previous_leg_index, s.ACT_DUR_SECONDS_COL]
                    previous_activity_type = self.df.loc[previous_leg_index, s.ACT_TO_INTERNAL_COL]

                    if pd.isna(previous_activity_time):
                        # If the activity time would still be plausible (which here just means longer than 600s),
                        # use the existing times (this may now work if the previous end time was missing)
                        try:
                            previous_activity_time = round((self.df.loc[current_leg_index, s.LEG_START_TIME_COL] -
                                                            previous_end_time).total_seconds())
                        except ValueError:
                            pass
                        # Get time from similar person with same activity
                        if pd.isna(previous_activity_time) or previous_activity_time < 600:
                            previous_activity_time = round(
                                similar_persons[similar_persons[s.ACT_TO_INTERNAL_COL] == previous_activity_type][
                                    s.ACT_DUR_SECONDS_COL].mean())
                        if pd.isna(previous_activity_time):
                            raise ValueError(f"Could not find activity time for person {person} with activity "
                                             f"{previous_activity_type}."
                                             f"This should not be possible given similar persons are selected"
                                             f" to have all relevant activity types.")

                    self.df.loc[current_leg_index, s.LEG_START_TIME_COL] = previous_end_time + pd.Timedelta(
                        seconds=previous_activity_time)
                    self.df.loc[current_leg_index, s.LEG_END_TIME_COL] = self.df.loc[
                                                                             current_leg_index, s.LEG_START_TIME_COL] + pd.Timedelta(
                        seconds=current_leg[s.LEG_DURATION_SECONDS_COL])
                stats_tracker.increment("legs_with_corrected_times")

            new_times = self.df[self.df[s.UNIQUE_P_ID_COL] == person][[
                s.LEG_START_TIME_COL, s.LEG_END_TIME_COL, s.ACT_DUR_SECONDS_COL, s.ACT_TO_INTERNAL_COL
            ]].to_string(index=False)

            logger.debug(f"New times: "
                         f"\n {new_times}")
            stats_tracker.increment("persons_with_corrected_times")

    def find_similar_persons_with_activities(self, person, activity_types: list, attributes: list = None,
                                             cols_to_check_nan: list = None):
        """
        Find similar persons (or other entries) based on a dynamic number of matching attributes.
        Ensure there is at least one person with the specified activity types.

        :param person: DataFrame or Series, the person to find similar persons for.
        :param activity_types: List of strings, the required activity types.
        :param attributes: List of strings, the attributes to match on. If None, uses default attributes.
        :param cols_to_check_nan: List of strings, the columns to check for NaN values and remove such persons.
        """
        if isinstance(person, pd.DataFrame):
            person = person.iloc[0]
        if attributes is None:
            attributes = [s.H_REGION_TYPE_COL, s.NUMBER_OF_LEGS_COL, s.HAS_LICENSE_COL, s.H_CAR_IN_HH_COL]

        activity_types = set(activity_types)

        logger.debug(f"Finding similar persons for {person[s.UNIQUE_P_ID_COL]} with activities {activity_types}...")
        for min_matches in range(len(attributes), 0, -1):  # Decrease criteria to a minimum of 1 attribute
            attribute_combinations = itertools.combinations(attributes, min_matches)
            for combination in attribute_combinations:
                condition = (self.df[list(combination)] == person[list(combination)]).all(axis=1)
                similar_persons = self.df[condition]

                # Remove legs which have any missing entries in the specified columns
                if cols_to_check_nan:
                    similar_persons = similar_persons.dropna(subset=cols_to_check_nan)
                similar_persons = similar_persons[similar_persons[s.UNIQUE_P_ID_COL] != person[s.UNIQUE_P_ID_COL]]

                if similar_persons.empty:
                    continue

                # Check if similar persons have at least one person with each required activity type
                if all(activity_type in similar_persons[s.ACT_TO_INTERNAL_COL].values for activity_type in
                       activity_types):
                    number_of_similar_persons = len(similar_persons[s.UNIQUE_P_ID_COL].unique())
                    logger.debug(
                        f"Found {number_of_similar_persons} similar persons for {person[s.UNIQUE_P_ID_COL]} based on {combination}.")
                    return similar_persons
        logger.warning(f"Found no similar persons for {person[s.UNIQUE_P_ID_COL]} with activities {activity_types}.")
        return pd.DataFrame()  # Return an empty DataFrame if no similar persons found with required activity types

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

                if similar_persons.empty:
                    continue

                similar_persons = similar_persons[similar_persons[s.UNIQUE_P_ID_COL] != person[s.UNIQUE_P_ID_COL]]
                if len(similar_persons) >= min_similar:
                    logger.debug(f"Found {len(similar_persons)} similar persons for {person[s.UNIQUE_P_ID_COL]} "
                                 f"based on {combination}.")
                    # Drop the person itself from the similar persons
                    return similar_persons

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
            (self.df[s.HAS_LICENSE_COL] == s.LICENSE_YES) & (
                    self.df[s.PERSON_AGE_COL] < 17), 'imputed_license'] = s.LICENSE_NO
        logger.info(
            f"Changed {self.df['imputed_license'].eq(s.LICENSE_NO).sum()} license status to no license based on age.")

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
            [s.LICENSE_NO, s.LICENSE_YES], size=num_rows,
            p=[1 - licence_likelihood_with_car, licence_likelihood_with_car]
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

                    connected = set(
                        self.df.loc[self.df[s.UNIQUE_LEG_ID_COL] == current_leg, s.CONNECTED_LEGS_COL].iloc[0])
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
        Writes to a new column.
        """
        logger.info("Updating activity for protagonist legs...")

        # Make a copy of the activity column that will be updated
        self.df[s.TO_ACTIVITY_WITH_CONNECTED_COL] = self.df[s.ACT_TO_INTERNAL_COL]

        # if s.IS_PROTAGONIST_COL in self.df.columns:  # Just for debugging
        #     logger.debug("Protagonist column already exists.")
        prot_legs = self.df[self.df[s.IS_PROTAGONIST_COL] == 1]

        for row in prot_legs.itertuples():
            protagonist_activity = getattr(row, s.ACT_TO_INTERNAL_COL)
            protagonist_leg_id = getattr(row, s.UNIQUE_LEG_ID_COL)
            connected_legs_list = getattr(row, s.CONNECTED_LEGS_COL)

            if not isinstance(connected_legs_list, list):
                logger.error(
                    f"Protagonist leg {protagonist_leg_id} has no connected legs. This shouldn't happen. Skipping...")
                continue

            connected_legs = set(connected_legs_list)
            connected_legs.discard(protagonist_leg_id)

            # Assign the protagonist's activity to all connected legs
            self.df.loc[
                self.df[s.UNIQUE_LEG_ID_COL].isin(
                    connected_legs), s.TO_ACTIVITY_WITH_CONNECTED_COL] = protagonist_activity

        logger.info("Updated activity for protagonist legs.")

    def add_from_activity(self):
        """
        Add a 'from_activity' column to the DataFrame, which is the to_activity of the previous leg.
        For the first leg of each person, set 'from_activity' based on 'starts_at_home' (-> home or unspecified).
        :return:
        """
        logger.info("Adding from_activity column...")
        # Sort the DataFrame by person ID and leg number (the df should usually already be sorted this way)
        self.df.sort_values(by=[s.PERSON_ID_COL, s.LEG_NON_UNIQUE_ID_COL], inplace=True)

        # Shift the 'to_activity' down to create 'from_activity' for each group
        self.df[s.ACT_FROM_INTERNAL_COL] = self.df.groupby(s.PERSON_ID_COL)[s.ACT_TO_INTERNAL_COL].shift(1)

        # For the first leg of each person, set 'from_activity' based on 'starts_at_home'
        self.df.loc[(self.df[s.LEG_NON_UNIQUE_ID_COL] == 1) & (
                self.df[
                    s.FIRST_LEG_STARTS_AT_HOME_COL] == s.FIRST_LEG_STARTS_AT_HOME), s.ACT_FROM_INTERNAL_COL] = s.ACT_HOME
        self.df.loc[(self.df[s.LEG_NON_UNIQUE_ID_COL] == 1) & (
                self.df[
                    s.FIRST_LEG_STARTS_AT_HOME_COL] != s.FIRST_LEG_STARTS_AT_HOME), s.ACT_FROM_INTERNAL_COL] = s.ACT_UNSPECIFIED

        # Handle cases with no legs (NA in leg_id)
        self.df.loc[self.df[s.LEG_NON_UNIQUE_ID_COL].isna(), s.ACT_FROM_INTERNAL_COL] = None

        logger.info("Added from_activity column.")

    def calculate_slack_factors(self):
        slack_factors = []

        df = self.df[self.df[s.LEG_DISTANCE_KM_COL] < 500]

        for person_id, person_trips in df.groupby(s.PERSON_ID_COL):
            logger.debug(f"Searching sf at person {person_id}...")
            # Sort by ordered_id to ensure sequence
            person_trips = person_trips.sort_values(by=s.LEG_NON_UNIQUE_ID_COL)

            # Find indirect routes by checking consecutive legs
            for i in range(len(person_trips) - 1):
                first_leg = person_trips.iloc[i]
                second_leg = person_trips.iloc[i + 1]

                # This should always be true, except for missing data
                if first_leg[s.ACT_TO_INTERNAL_COL] == second_leg[s.ACT_FROM_INTERNAL_COL]:

                    direct_trip = self.df[
                        (self.df[s.PERSON_ID_COL] == person_id) &
                        # Exclude the two legs we're checking
                        (self.df[s.LEG_NON_UNIQUE_ID_COL] != first_leg[s.LEG_NON_UNIQUE_ID_COL]) &
                        (self.df[s.LEG_NON_UNIQUE_ID_COL] != second_leg[s.LEG_NON_UNIQUE_ID_COL]) &
                        # Find direct trip in both directions
                        ((self.df[s.ACT_FROM_INTERNAL_COL] == first_leg[s.ACT_FROM_INTERNAL_COL]) &
                         (self.df[s.ACT_TO_INTERNAL_COL] == second_leg[s.ACT_TO_INTERNAL_COL]) |
                         (self.df[s.ACT_FROM_INTERNAL_COL] == second_leg[s.ACT_TO_INTERNAL_COL]) &
                         (self.df[s.ACT_TO_INTERNAL_COL] == first_leg[s.ACT_FROM_INTERNAL_COL]))
                        ]

                    if not direct_trip.empty:
                        direct_distance = direct_trip.iloc[0][s.LEG_DISTANCE_KM_COL]
                        indirect_distance = first_leg[s.LEG_DISTANCE_KM_COL] + second_leg[s.LEG_DISTANCE_KM_COL]
                        slack_factor = indirect_distance / direct_distance
                        slack_factors.append((person_id,
                                              first_leg[s.H_REGION_TYPE_COL],
                                              first_leg[s.PERSON_AGE_COL],
                                              first_leg[s.ACT_FROM_INTERNAL_COL],
                                              first_leg[s.ACT_TO_INTERNAL_COL],
                                              second_leg[s.ACT_TO_INTERNAL_COL],
                                              first_leg[s.MODE_INTERNAL_COL],
                                              second_leg[s.MODE_INTERNAL_COL],
                                              direct_trip.iloc[0][s.MODE_INTERNAL_COL],
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

    def list_cars_in_household(self):
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

        logger.info(
            f"Listed {total_cars} cars in {len(hhs)} households for {self.df[s.UNIQUE_P_ID_COL].nunique()} persons, "
            f"meaning {total_cars / self.df[s.UNIQUE_P_ID_COL].nunique()} cars per person on average.")

    def impute_cars_in_household(self):  # TODO Unfinished.
        """
        Impute the number of cars in a household if unknown, based on the number of cars in similar households.
        """
        self.df.loc[self.df[s.H_NUMBER_OF_CARS_COL] == 99, s.H_NUMBER_OF_CARS_COL] = None
        logger.info(f"Imputing cars in household for {self.df[s.H_NUMBER_OF_CARS_COL].isna().sum()} of "
                    f"{len(self.df)} rows...")

        # Set all other unknown values to 0
        self.df.loc[self.df[s.H_NUMBER_OF_CARS_COL].isna, s.H_NUMBER_OF_CARS_COL] = 0

    def mark_mirroring_main_activities(self, duration_threshold_seconds=7200):
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
        self.df['next_next_activity'] = self.df[s.ACT_TO_INTERNAL_COL].shift(-2)
        self.df['next_next_person_id'] = self.df[s.UNIQUE_P_ID_COL].shift(-2)
        self.df['next_leg_distance'] = self.df[s.LEG_DISTANCE_KM_COL].shift(-1)
        self.df['next_next_leg_distance'] = self.df[s.LEG_DISTANCE_KM_COL].shift(-2)

        # Make sure we are checking the same person, and based on main activity
        person_id_condition = (self.df['next_person_id'] == self.df[s.UNIQUE_P_ID_COL]) & \
                              (self.df['next_next_person_id'] == self.df[s.UNIQUE_P_ID_COL]) & \
                              (self.df[s.IS_MAIN_ACTIVITY_COL] == 1)

        # Time threshold for the in-between activity
        short_duration_condition = (self.df['next_act_dur'] < duration_threshold_seconds)

        # Candidate activity must be the same as the main activity
        same_activity_condition = (self.df['next_next_activity'] == self.df[s.ACT_TO_INTERNAL_COL])

        # Leg distance to the in-between activity and from it to the candidate activity must be the same
        same_leg_distance_condition = (self.df['next_leg_distance'] == self.df['next_next_leg_distance'])

        self.df[s.MIRRORS_MAIN_ACTIVITY_COL] = (
            (
                    person_id_condition & short_duration_condition & same_activity_condition & same_leg_distance_condition).shift(
                2).fillna(False)).astype(int)

        # Drop temporary columns
        self.df.drop(
            ['next_person_id', 'next_act_dur', 'next_next_activity', 'next_next_person_id', 'next_leg_distance',
             'next_next_leg_distance'], axis=1, inplace=True)

        logger.info("Marked mirroring main.")

    def find_home_to_main_time_and_distance(self):
        """
        Determines the time (and distance) between home and main activity for each person in the DataFrame.
        It may not look it, but this was a pain to write.
        Main directly after home: Leg distance of to-main leg
        Main directly before home: Leg distance of to-home leg
        :return: Adds column with the distance to the DataFrame.
        """

        logger.info("Determining home to main activity times/distances...")

        persons = self.df.groupby(s.UNIQUE_P_ID_COL)
        home_to_main_distances = {}
        home_to_main_times = {}
        home_to_main_time_estimated = {}
        home_to_main_distance_estimated = {}

        for pid, person in persons:
            # Extract the indices for main activity, mirroring main and home activities
            main_activity_idx = np.where(person[s.IS_MAIN_ACTIVITY_COL])[0]  # where returns a tuple. Legs to main
            mirroring_main_idx = np.where(person[s.MIRRORS_MAIN_ACTIVITY_COL])[0]  # Legs to mirrored main
            home_indices = np.where(person[s.ACT_TO_INTERNAL_COL] == s.ACT_HOME)[0]  # legs to home

            if main_activity_idx.size == 0:
                rows = len(person)
                if rows == 0:
                    raise ValueError(f"Person {pid} has zero rows. This should be impossible.")
                elif rows == 1:
                    logger.debug(f"Person {pid} has no activities. Skipping.")
                    continue
                elif rows > 1:
                    if home_indices.size == len(person):
                        logger.info(f"Person {pid} has several legs, but only home activities. Skipping.")
                        continue
                    raise ValueError(f"Person {pid} has activities, but no main activity.")
                else:
                    raise ValueError(f"Person {pid} has {rows} rows. This should be impossible.")

            if main_activity_idx.size > 1:  # should not happen (would still work but is sign of error)
                raise ValueError(f"Person {pid} has more than one main activity. This should not happen.")

            # Determine the home activity closest to the main (or mirroring main) activity
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
            if closest_home_idx == main_activity_idx[0]:
                raise ValueError(
                    f"Person {pid} has a home activity marked as main activity. This should never be the case.")
            elif closest_home_idx - main_activity_idx[0] == 1:  # Main to home
                logger.debug(f"Person {pid} has a home activity directly after main. ")
                home_to_main_distance = person.at[closest_home_row, s.LEG_DISTANCE_METERS_COL]
                home_to_main_time = person.at[closest_home_row, s.LEG_DURATION_SECONDS_COL]
                time_is_estimated = 0
                distance_is_estimated = 0

            elif closest_home_idx - main_activity_idx[0] == -1:  # Home to main
                logger.debug(f"Person {pid} has a home activity directly before main. ")
                home_to_main_distance = person.at[main_activity_row, s.LEG_DISTANCE_METERS_COL]
                home_to_main_time = person.at[main_activity_row, s.LEG_DURATION_SECONDS_COL]
                time_is_estimated = 0
                distance_is_estimated = 0

            else:
                logger.debug(f"Person {pid} has a main activity and home activity more than one leg apart. "
                             f"Time and distance will be estimated.")

                # Get all legs between home and main activity (thus exclude leg towards first activity)
                if closest_home_row < main_activity_row:  # Home to main
                    legs = person.loc[closest_home_row + 1:main_activity_row]

                else:  # Main to home
                    legs = person.loc[main_activity_row + 1:closest_home_row]

                distances = legs[s.LEG_DISTANCE_METERS_COL].tolist()
                dist_estimation_tree = h.build_estimation_tree(distances)
                home_to_main_distance = dist_estimation_tree[-1][0][2]  # highest lvl, leg 0, estimated value

                times = legs[s.LEG_DURATION_SECONDS_COL].tolist()
                time_estimation_tree = h.build_estimation_tree(times)
                home_to_main_time = time_estimation_tree[-1][0][2]  # highest lvl, leg 0, estimated value

                time_is_estimated = 1
                distance_is_estimated = 1

            home_to_main_distances[pid] = home_to_main_distance
            home_to_main_times[pid] = home_to_main_time
            home_to_main_time_estimated[pid] = time_is_estimated
            home_to_main_distance_estimated[pid] = distance_is_estimated

        self.df[s.HOME_TO_MAIN_METERS_COL] = self.df[s.UNIQUE_P_ID_COL].map(home_to_main_distances)
        self.df[s.HOME_TO_MAIN_SECONDS_COL] = self.df[s.UNIQUE_P_ID_COL].map(home_to_main_times)
        self.df[s.HOME_TO_MAIN_TIME_IS_ESTIMATED_COL] = self.df[s.UNIQUE_P_ID_COL].map(home_to_main_time_estimated)
        self.df[s.HOME_TO_MAIN_DIST_IS_ESTIMATED_COL] = self.df[s.UNIQUE_P_ID_COL].map(home_to_main_distance_estimated)

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
                if person[s.ACT_TO_INTERNAL_COL].iloc[0] == s.ACT_HOME:
                    # If the person has only one leg, and it's home, there is no main activity
                    logger.debug(f"Person {pid} has only one leg and it's home. Skipping...")
                    continue

            # Find the main activity
            try:
                main_activity_idx = person[person[s.IS_MAIN_ACTIVITY_COL] == 1].index[0]
            except IndexError:
                if person[s.ACT_TO_INTERNAL_COL].eq(s.ACT_HOME).all():
                    # If all activities are home, there is no main activity
                    logger.debug(f"Person {pid} has only home activities. Skipping...")
                    continue
                logger.warning(f"Person {pid} has no main activity for unknown reasons. Skipping this person.")
                continue

            # Find the closest previous home activity or the start of the day. FROM_activity so the trip to home is excluded.
            home_indices = person[person[s.ACT_FROM_INTERNAL_COL] == s.ACT_HOME].index
            start_idx = home_indices[home_indices < main_activity_idx].max() if not home_indices.empty else \
                person.index[0]
            if pd.isna(start_idx):
                start_idx = person.index[0]

            # Calculate the total use time for each mode. Slicing is inclusive of both start and end index.
            mode_times = person.loc[start_idx:main_activity_idx].groupby(s.MODE_INTERNAL_COL)[
                s.LEG_DURATION_MINUTES_COL].sum()
            mode_distances = person.loc[start_idx:main_activity_idx].groupby(s.MODE_INTERNAL_COL)[
                s.LEG_DISTANCE_KM_COL].sum()

            main_mode_time_base = mode_times.idxmax()
            main_mode_dist_base = mode_distances.idxmax()

            # Store the main mode for the person
            main_modes_time[pid] = main_mode_time_base
            main_modes_dist[pid] = main_mode_dist_base

        # Add a new column to the dataframe with the main mode for each person
        self.df[s.MAIN_MODE_TO_MAIN_ACT_TIMEBASED_COL] = self.df[s.UNIQUE_P_ID_COL].map(main_modes_time)
        self.df[s.MAIN_MODE_TO_MAIN_ACT_DISTBASED_COL] = self.df[s.UNIQUE_P_ID_COL].map(main_modes_dist)
        logger.info("Determined main mode to main activity.")

    def update_number_of_legs(self, col_to_write_to=s.NUMBER_OF_LEGS_COL):
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
        logger.info("Finding connected legs...")

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
        file_loc = path.join(pipeline_setup.OUTPUT_DIR, 'leg_connections_logs.csv')
        checks_df.to_csv(file_loc, index=False)

        # Add connections as a new column to self.df
        self.df[s.CONNECTED_LEGS_COL] = connections

    def mark_connected_persons_and_hhs(self):
        logger.info("Marking connected persons and households...")
        self.df[s.HH_HAS_CONNECTIONS_COL] = 0
        self.df[s.P_HAS_CONNECTIONS_COL] = 0

        for person_id in self.df[s.PERSON_ID_COL].unique():
            if any(self.df[self.df[s.PERSON_ID_COL] == person_id][s.CONNECTED_LEGS_COL].apply(
                    lambda x: isinstance(x, list))):
                self.df.loc[self.df[s.PERSON_ID_COL] == person_id, s.P_HAS_CONNECTIONS_COL] = 1
                logger.debug(f"Person {person_id} has connections.")

        for hh_id in self.df[s.UNIQUE_HH_ID_COL].unique():
            if any(self.df[self.df[s.UNIQUE_HH_ID_COL] == hh_id][s.CONNECTED_LEGS_COL].apply(
                    lambda x: isinstance(x, list))):
                self.df.loc[self.df[s.UNIQUE_HH_ID_COL] == hh_id, s.HH_HAS_CONNECTIONS_COL] = 1
                logger.debug(f"Household {hh_id} has connections.")

    def count_connected_legs_per_person(self):
        logger.info("Counting connected legs per person...")
        self.df[s.NUM_CONNECTED_LEGS_COL] = 0

        for person_id in self.df[s.PERSON_ID_COL].unique():
            person_rows = self.df[self.df[s.PERSON_ID_COL] == person_id]
            num_connections = person_rows[s.CONNECTED_LEGS_COL].apply(
                lambda x: len(x) if isinstance(x, list) else 0).sum()
            self.df.loc[self.df[s.PERSON_ID_COL] == person_id, s.NUM_CONNECTED_LEGS_COL] = num_connections
            logger.debug(f"Person {person_id} has {num_connections} connected legs.")

    def mark_main_activity(self):
        """
        Check if the leg is travelling to the main activity of the day.
        Requires calculate_activity_time() to be run first.
        :return: Updates self.df with a new column indicating if each leg is the main activity (1) or not (0)
        """
        person_col = s.UNIQUE_P_ID_COL
        act_to_internal_col = s.ACT_TO_INTERNAL_COL
        leg_non_unique_id_col = s.LEG_NON_UNIQUE_ID_COL
        act_dur_seconds_col = s.ACT_DUR_SECONDS_COL

        def find_main_activity(person):
            is_main_activity_series = pd.Series(0, index=person.index)  # Initialize all values to 0

            # TODO: REMOVE
            if person[s.UNIQUE_P_ID_COL].iloc[0] == "69808010_817_69808012":
                print("HI")

            # Filter out home activities (home must not be the main activity)
            group = person[person[act_to_internal_col] != s.ACT_HOME]

            if group.empty:
                logger.debug(f"Person {person[person_col].iloc[0]} has no legs outside home. No main activity.")
                return is_main_activity_series

            if len(group) == 1:
                # If the person has no legs, there is no main activity
                if group[leg_non_unique_id_col].isna().all():
                    logger.debug(f"Person {group[person_col].iloc[0]} has no legs. No main activity.")
                    return is_main_activity_series

                # If the person has only one activity outside home, it is the main activity
                main_activity_index = group.index[0]
                is_main_activity_series.at[main_activity_index] = 1
                assert is_main_activity_series.sum() == 1
                return is_main_activity_series

            # If the person has more than one activity, the main activity is the first work activity
            work_activity_rows = group[group[act_to_internal_col] == s.ACT_WORK]
            if not work_activity_rows.empty:
                is_main_activity_series[work_activity_rows.index[0]] = 1
                assert is_main_activity_series.sum() == 1
                logger.debug(f"Person {group[person_col].iloc[0]} has a work activity. Main activity is work.")
                return is_main_activity_series

            # If the person has no work activity, the main activity is the first education activity
            education_activity_rows = group[
                group[act_to_internal_col].isin([s.ACT_EDUCATION, s.ACT_EARLY_EDUCATION, s.ACT_DAYCARE])]
            if not education_activity_rows.empty:
                is_main_activity_series[education_activity_rows.index[0]] = 1
                assert is_main_activity_series.sum() == 1
                logger.debug(
                    f"Person {group[person_col].iloc[0]} has an education activity. Main activity is education.")
                return is_main_activity_series

            # If the person has no work or education activity, the main activity is the longest activity
            if group[act_dur_seconds_col].isna().all():
                # If all activities have no duration, pick the middle one
                main_act_idx = group.index[
                    len(group) // 2]  # Integer division (within the filtered group so we don't pick home)
                is_main_activity_series[main_act_idx] = 1
                assert is_main_activity_series.sum() == 1
                logger.debug(
                    f"Person {group[person_col].iloc[0]} has no activities with duration. Main activity is middle.")
                return is_main_activity_series
            max_duration_index = group[act_dur_seconds_col].idxmax()
            is_main_activity_series[max_duration_index] = 1
            assert is_main_activity_series.sum() == 1
            logger.debug(f"Person {group[person_col].iloc[0]} has no work or education activity. "
                         f"Main activity is longest activity.")
            return is_main_activity_series

        self.df[s.IS_MAIN_ACTIVITY_COL] = self.df.groupby(person_col).apply(find_main_activity).reset_index(level=0,
                                                                                                            drop=True)

    def mark_protagonist_leg(self):
        """
        Identify the 'protagonist' leg among connected legs for each household.
        The leg with the highest-ranked activity in each group of connected legs is considered the protagonist.
        :return: Updates self.df with a new column indicating if each leg is a protagonist (1) or not (0)
        """
        household_col = s.UNIQUE_HH_ID_COL
        connected_legs_col = s.CONNECTED_LEGS_COL
        unique_leg_id_col = s.UNIQUE_LEG_ID_COL
        act_to_internal_col = s.ACT_TO_INTERNAL_COL
        activities_ranked = [
            s.ACT_ERRANDS,
            s.ACT_LEISURE,
            s.ACT_MEETUP,
            s.ACT_SHOPPING,
            s.ACT_EDUCATION,
            s.ACT_LESSONS,
            s.ACT_SPORTS,
            s.ACT_EARLY_EDUCATION,
            s.ACT_DAYCARE,  # Likely to be dropped off/picked up
            s.ACT_BUSINESS,  # Likely to be accompanied
            s.ACT_HOME]  # Home must stay home

        def find_protagonist(household_group):
            prot_series = pd.Series(0, index=household_group.index)
            if household_group[connected_legs_col].isna().all():
                logger.debug(f"No connections exist for household {household_group[household_col].iloc[0]}.")
                return prot_series
            else:
                logger.debug(f"Finding protagonist for household {household_group[household_col].iloc[0]}")

            checked_legs = []
            for idx, row in household_group.iterrows():
                if not isinstance(row[connected_legs_col], list):
                    continue
                if idx in checked_legs:
                    continue
                connected_legs = set(row[connected_legs_col]).union({row[unique_leg_id_col]})
                leg_data = []
                for leg_id in connected_legs:
                    if not all(elem in connected_legs for elem in
                               household_group.loc[
                                   household_group[unique_leg_id_col] == leg_id, connected_legs_col].iloc[0]):
                        logger.warning(
                            f"Leg {leg_id} has inconsistent connections. This might lead to unexpected results.")

                    checked_legs.append(household_group.loc[household_group[unique_leg_id_col] == leg_id].index[0])
                    leg_activity = \
                        household_group.loc[household_group[unique_leg_id_col] == leg_id, act_to_internal_col].iloc[0]
                    activity_rank = activities_ranked.index(leg_activity) if leg_activity in activities_ranked else -1
                    leg_data.append({'leg_id': leg_id, 'activity': leg_activity, 'activity_rank': activity_rank})

                connected_legs_df = pd.DataFrame(leg_data)
                if not connected_legs_df.empty:
                    connected_legs_df.sort_values(by='activity_rank', ascending=False, inplace=True)
                    protagonist_leg_id = connected_legs_df.iloc[0]['leg_id']
                    prot_series.loc[household_group[unique_leg_id_col] == protagonist_leg_id] = 1

            return prot_series

        self.df[s.IS_PROTAGONIST_COL] = self.df.groupby(household_col).apply(find_protagonist).reset_index(level=0,
                                                                                                           drop=True)

    def replace_rbw_with_work_activity(self):
        """
        For each person, remove all legs where s.IS_RBW is 1 and replace with a single work activity.
        """
        logger.info("Replacing RBW legs with work activity...")

        # Iterate over each person
        unique_person_ids = self.df[s.UNIQUE_P_ID_COL].unique()
        new_activities = []

        for person_id in unique_person_ids:
            person_df = self.df[self.df[s.UNIQUE_P_ID_COL] == person_id]
            rbw_legs = person_df[person_df[s.IS_RBW] == 1]

            if not rbw_legs.empty:
                # Calculate the total duration of RBW legs
                total_rbw_duration = rbw_legs[s.LEG_DURATION_SECONDS_COL].sum()

                # Create a new work activity based on the first RBW leg
                first_rbw_leg = rbw_legs.iloc[0]
                new_activity = first_rbw_leg.copy()
                new_activity[s.ACT_TO_INTERNAL_COL] = s.ACT_WORK
                new_activity[s.ACT_DUR_SECONDS_COL] = total_rbw_duration
                new_activity[s.LEG_END_TIME_COL] = new_activity[s.LEG_START_TIME_COL] + pd.Timedelta(
                    seconds=total_rbw_duration)

                new_activities.append(new_activity)

                # Remove RBW legs from the original dataframe
                self.df = self.df.drop(rbw_legs.index)

        # Append the new work activities to the dataframe
        new_activities_df = pd.DataFrame(new_activities)
        self.df = pd.concat([self.df, new_activities_df], ignore_index=True)

        logger.info(f"Replaced RBW legs with {len(new_activities)} work activities.")

    def guess_rbw_activity_duration(self):  # TODO: Medium term, this should get more sophisticated
        """
        Guess the activity duration for RBW activities.
        To be run before the main time correction methods - they will use what we estimate here, but after the
        activity duration calculation (otherwise it will be overwritten).
        """
        logger.info("Guessing RBW activity durations...")

        unique_person_ids = self.df[s.UNIQUE_P_ID_COL].unique()

        for person_id in unique_person_ids:
            person_df = self.df[self.df[s.UNIQUE_P_ID_COL] == person_id]
            rbw_legs = person_df[person_df[s.LEG_IS_RBW_COL] == 1]

            if rbw_legs.empty:
                continue

            # Get the indices of the first and last RBW leg
            first_rbw_index = rbw_legs.index[0]
            last_rbw_index = rbw_legs.index[-1]

            # Get the previous leg (if any)
            previous_leg_end_time = None
            if first_rbw_index > 0:
                previous_leg_end_time = person_df.loc[first_rbw_index - 1, s.LEG_END_TIME_COL]

            # Get the next leg (if any)
            next_leg_start_time = None
            if last_rbw_index < person_df.index.max():
                next_leg_start_time = person_df.loc[last_rbw_index + 1, s.LEG_START_TIME_COL]

            # Calculate total RBW leg duration
            total_rbw_leg_duration = rbw_legs[s.LEG_DURATION_SECONDS_COL].sum()

            if previous_leg_end_time and next_leg_start_time:
                # Calculate duration based on the gap between previous and next legs
                total_activity_duration = (
                                                  next_leg_start_time - previous_leg_end_time).total_seconds() - total_rbw_leg_duration
            elif previous_leg_end_time:
                # Calculate duration based on the gap after the previous leg
                total_activity_duration = (person_df[
                                               s.LEG_END_TIME_COL].max() - previous_leg_end_time).total_seconds() - total_rbw_leg_duration
            elif next_leg_start_time:
                # Calculate duration based on the gap before the next leg
                total_activity_duration = (next_leg_start_time - person_df[
                    s.LEG_START_TIME_COL].min()).total_seconds() - total_rbw_leg_duration
            else:
                # Default to the sum of leg durations if no other data is available
                total_activity_duration = total_rbw_leg_duration

            # Distribute the guessed activity duration across RBW legs
            average_activity_duration = total_activity_duration / len(rbw_legs)
            self.df.loc[rbw_legs.index, s.ACT_DUR_SECONDS_COL] = average_activity_duration

        logger.info("Guessed RBW activity durations for all relevant legs.")
