import os
import random

import pandas as pd

from utils import matsim_pipeline_setup
from utils.logger import logging

logger = logging.getLogger(__name__)

os.chdir(matsim_pipeline_setup.PROJECT_ROOT)
settings = matsim_pipeline_setup.load_yaml_config('settings.yaml')

# We commit the sin of using global (module-wide) variables because rules would have to declare them many times anyway, it's fine
ID_COLUMNS = settings['id_columns']
HH_COLUMNS = settings['hh_columns']
P_COLUMNS = settings['person_columns']
L_COLUMNS = settings['leg_columns']

AVERAGE_ACTIVITY_TIMES_MINUTES = settings['average_activity_times_minutes']

# Household-related constants
HOUSEHOLD_MID_ID_COL = ID_COLUMNS['household_mid_id_column']
HOUSEHOLD_POPSIM_ID_COL = ID_COLUMNS['household_popsim_id_column']

# Person-related constants
PERSON_ID_COL = ID_COLUMNS['person_id_column']
PERSON_AGE_COL = P_COLUMNS['person_age']
CAR_AVAIL_COL = P_COLUMNS['car_avail']
HAS_LICENSE_COL = P_COLUMNS['has_license']

# Leg-related constants
LEG_ID_COL = ID_COLUMNS['leg_id_column']
LEG_ACTIVITY_COL = L_COLUMNS['leg_target_activity']
LEG_MAIN_MODE_COL = L_COLUMNS['leg_main_mode']
LEG_START_TIME_COL = L_COLUMNS['leg_start_time']
LEG_END_TIME_COL = L_COLUMNS['leg_end_time']
LEG_DURATION_MINUTES_COL = L_COLUMNS['leg_duration_minutes']
LEG_DISTANCE_COL = L_COLUMNS['leg_distance']

# Value maps
VALUE_MAPS = settings['value_maps']

ACTIVITY_HOME = VALUE_MAPS['activities']['home']
ACTIVITIES_EDUCATION = set(VALUE_MAPS['activities']['education'])
ACTIVITY_WORK = VALUE_MAPS['activities']['work']

CAR_NEVER = VALUE_MAPS['car_availability']['never']

LICENSE_YES = VALUE_MAPS['license']['yes']
LICENSE_NO = VALUE_MAPS['license']['no']
LICENSE_UNKNOWN = VALUE_MAPS['license']['unknown']
ADULT_OVER_16_PROXY = VALUE_MAPS['license']['adult_over_16_proxy']
PERSON_UNDER_16 = VALUE_MAPS['license']['person_under_16']

MODE_CAR = VALUE_MAPS['modes']['car']
MODE_PT = VALUE_MAPS['modes']['pt']
MODE_RIDE = VALUE_MAPS['modes']['ride']
MODE_BIKE = VALUE_MAPS['modes']['bike']
MODE_WALK = VALUE_MAPS['modes']['walk']
MODE_UNDEFINED = VALUE_MAPS['modes']['undefined']


def unique_household_id(row):
    """
    Household_ids are unique in MiD (or other dataset), but not in the expanded population.
    PopSim provides a unique expanded household ID, we use that to create a unique household ID
    including the MiD-identifier.
    """
    return f"{row[HOUSEHOLD_MID_ID_COL]}_{row[HOUSEHOLD_POPSIM_ID_COL]}"


def unique_person_id(row):
    """
    Person_ids are unique in MiD (or other dataset), but not in the expanded population.
    Returns a unique person ID from the unique household ID and the person ID.
    """
    return f"{row['unique_household_id']}_{row[PERSON_ID_COL]}"


def unique_leg_id(row):
    """
    Returns a unique leg ID from the unique person ID and the leg ID.
    Usually not necessary because rules don't group by unique leg ID, as this would usually be a group of size 1.
    """
    return f"{row['unique_person_id']}_{row[LEG_ID_COL]}"


def has_license_imputed(row):
    """
    Impute license status based on age and statistical probabilities.
    :param row: Row of the population frame
    :return: 0 if no license, 1 if license
    """
    if row[HAS_LICENSE_COL] == LICENSE_NO:
        logger.debug(f"Person {row[PERSON_ID_COL]} has no license.")
        return 0
    elif row[HAS_LICENSE_COL] == LICENSE_YES:
        if row[PERSON_AGE_COL] < 17:
            logger.debug(
                f"Person {row[PERSON_ID_COL]} has a car driving license but is under 17 years old. Assuming no license.")
            return 0
        logger.debug(f"Person {row[PERSON_ID_COL]} has a license.")
        return 1
    elif row[HAS_LICENSE_COL] == LICENSE_UNKNOWN or ADULT_OVER_16_PROXY:
        r = random.random()
        if row[PERSON_AGE_COL] >= 17:
            if r <= 0.94:
                logger.debug(f"Adult {row[PERSON_ID_COL]} has unknown license status. Assuming license.")
                return 1
            else:
                logger.debug(f"Adult {row[PERSON_ID_COL]} has unknown license status. Assuming no license.")
                return 0
        logger.debug(f"Child {row[PERSON_ID_COL]} has unknown license status. Assuming no license.")
        return 0
    elif row[HAS_LICENSE_COL] == PERSON_UNDER_16:
        logger.debug(f"Child {row[PERSON_ID_COL]} has unknown license status. Assuming no license.")
        return 0
    else:
        logger.warning(f"Person {row[PERSON_ID_COL]} has no license status entry, not even unknown. Assuming no license.")
        return 0


def main_mode_imputed(row):
    """
    Impute main mode based on the distance of the leg, the availability of a car and statistical probabilities.
    Note: Imputed main mode is available in MiD, so this isn't needed, but it's here for reference.
    :param row:
    :return: mode
    """
    r = random.random()

    boundaries_never_or_no_license = [
        (0.5, [0.89], ["walk", "bike"]),
        (1, [0.74], ["walk", "bike"]),
        (2, [0.48, 0.71], ["walk", "bike", "ride"]),
        (5, [0.25, 0.52, 0.81], ["walk", "bike", "ride", "pt"]),
        (float('inf'), [0.53], ["ride", "pt"])
    ]

    boundaries_otherwise = [
        (0.5, [0.89], ["walk", "bike"]),
        (1, [0.56, 0.76], ["walk", "bike", "car"]),
        (2, [0.31, 0.52, 0.65], ["walk", "bike", "ride", "car"]),
        (5, [0.14, 0.29, 0.45, 0.90], ["walk", "bike", "ride", "car", "pt"]),
        (float('inf'), [0.26, 0.77], ["ride", "car", "pt"])
    ]

    boundaries = boundaries_never_or_no_license if row[CAR_AVAIL_COL] == CAR_NEVER or row[
        has_license_imputed] == 0 else boundaries_otherwise

    for distance, probabilities, modes in boundaries:
        if row[LEG_DISTANCE_COL] < distance:
            for prob, mode in zip(probabilities, modes):
                if r <= prob:
                    return mode
            return modes[-1]  # If the random number is greater than all probabilities, return the last mode in the list

    return None


def collapse_person_trip(group):
    """
    Process a person's trip and represent it as a list of activity legs (intermediary for further processing).
    :param group: Population frame grouped by person_id
    :return: summary_df
    """

    trip_representation = [{'activity': row['activity'], 'duration': row['duration']}
                           for _, row in group.iterrows()]
    summary_df = pd.DataFrame({
        'trip': [trip_representation] * len(group)
    }, index=group.index)
    return summary_df


# def add_return_home_leg(group):
#     """
#     Add a home leg at the end of the day if the person doesn't already have one.
#     The length of the activity and the leg duration are estimated.
#     Requires is_main_activity() to be run first.
#     :param group: Population frame grouped by person_id
#     :return: edited group
#     """
#
#     main_activity_index = group[group['is_main_activity'] == 1].index[
#         0]  # There should only be one main activity but [0] just in case
#     sum_durations_before_main = group.loc[:main_activity_index, LEG_DURATION_MINUTES_COL].sum()  # Minutes
#     sum_durations_after_main = group.loc[main_activity_index:, LEG_DURATION_MINUTES_COL].sum()
#
#     # Estimate leg duration:
#     average_leg_duration = group[LEG_DURATION_MINUTES_COL].mean()  # Minutes
#     average_leg_duration_after_main = group.loc[main_activity_index:, LEG_DURATION_MINUTES_COL].mean()
#     if average_leg_duration_after_main:
#         home_leg_duration = average_leg_duration_after_main
#     else:
#         home_leg_duration = average_leg_duration
#     # We assume the trip home is equal or longer than the trip to the main activity
#     if sum_durations_before_main > sum_durations_after_main + home_leg_duration:
#         # max() so the trip doesn't get crazy short
#         home_leg_duration = max(sum_durations_before_main - sum_durations_after_main, average_leg_duration / 4)
#
#     # Estimate activity duration:
#     last_leg = group.iloc[-1]
#     try:
#         activity_time = AVERAGE_ACTIVITY_TIMES_MINUTES[last_leg[LEG_ACTIVITY_COL]]
#     except KeyError:
#         activity_time = 60  # 1 hour default
#
#     # Create home_leg with the calculated duration
#     home_leg = last_leg.copy()
#     home_leg['LEG_ID'] = last_leg['LEG_ID'] + 1
#     home_leg['LEG_START_TIME'] = last_leg['LEG_END_TIME'] + pd.Timedelta(minutes=activity_time)
#     home_leg['LEG_END_TIME'] = home_leg['LEG_START_TIME'] + pd.Timedelta(minutes=home_leg_duration)
#     home_leg['LEG_ACTIVITY'] = ACTIVITY_HOME
#     home_leg['LEG_DURATION'] = home_leg_duration
#     home_leg['LEG_DISTANCE'] = None
#
#     # Append the home leg to the group
#     group = group.append(home_leg, ignore_index=True)
#     return group

def add_return_home_leg(df):
    """
    Add a home leg at the end of the day, if it doesn't exist. Alternative to change_last_leg_target_to_home().
    The length of the activity and the leg duration are estimated.
    Requires is_main_activity() to be run first.
    :param df: Population frame
    :return: DataFrame with added home legs
    """
    new_rows = []

    for person_id, group in df.groupby(PERSON_ID_COL):
        main_activity_index = group[group['is_main_activity'] == 1].index[0]  # There should only be one main activity
        sum_durations_before_main = group.loc[:main_activity_index, LEG_DURATION_MINUTES_COL].sum()
        sum_durations_after_main = group.loc[main_activity_index:, LEG_DURATION_MINUTES_COL].sum()

        # Estimate leg duration:
        average_leg_duration = group[LEG_DURATION_MINUTES_COL].mean()
        average_leg_duration_after_main = group.loc[main_activity_index:, LEG_DURATION_MINUTES_COL].mean()
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
            activity_time = AVERAGE_ACTIVITY_TIMES_MINUTES[last_leg[LEG_ACTIVITY_COL]]
        except KeyError:
            activity_time = 60  # 1 hour default

        # Create home_leg with the calculated duration
        home_leg = last_leg.copy()
        home_leg[LEG_ID_COL] = last_leg['LEG_ID'] + 1
        home_leg[LEG_START_TIME_COL] = last_leg[LEG_END_TIME_COL] + pd.Timedelta(minutes=activity_time)
        home_leg[LEG_END_TIME_COL] = home_leg[LEG_START_TIME_COL] + pd.Timedelta(minutes=home_leg_duration)
        home_leg[LEG_ACTIVITY_COL] = ACTIVITY_HOME
        home_leg[LEG_DURATION_MINUTES_COL] = home_leg_duration
        home_leg[LEG_DISTANCE_COL] = None  # Could also be estimated, but isn't necessary for the current use case

        new_rows.append(home_leg)

    new_rows_df = pd.DataFrame(new_rows)

    # Sorting by person_id and leg_start_time will insert the new rows in the correct place
    return pd.concat([df, new_rows_df]).sort_values([PERSON_ID_COL, LEG_START_TIME_COL]).reset_index(drop=True)


def is_main_activity(group):
    """
    Check if the leg is travelling to the main activity of the day.
    Requires activity_duration_in_seconds() to be run first.
    :param group: Population frame grouped by person_id
    :return: Series indicating if each row is the main activity: 1 if main activity, 0 if not
    """
    is_main_activity_series = pd.Series(0, index=group.index)  # Initialize all values to 0

    # If the person has only one activity, it is the main activity
    if len(group) == 1:
        is_main_activity_series.iloc[0] = 1
        assert is_main_activity_series.shape[0] == group.shape[0]
        return is_main_activity_series

    # If the person has more than one activity, the main activity is the first work activity
    work_activity_rows = group[group[LEG_ACTIVITY_COL] == ACTIVITY_WORK]
    if not work_activity_rows.empty:
        is_main_activity_series[work_activity_rows.index[0]] = 1
        assert is_main_activity_series.shape[0] == group.shape[0]
        return is_main_activity_series

    # If the person has no work activity, the main activity is the first education activity
    education_activity_rows = group[group[LEG_ACTIVITY_COL].isin(ACTIVITIES_EDUCATION)]
    if not education_activity_rows.empty:
        is_main_activity_series[education_activity_rows.index[0]] = 1
        assert is_main_activity_series.shape[0] == group.shape[0]
        return is_main_activity_series

    # If the person has no work or education activity, the main activity is the longest activity
    if group["activity_duration_in_minutes"].isna().all():
        # If all activities have no duration, pick the middle one
        is_main_activity_series.iloc[len(group) // 2] = 1
        assert is_main_activity_series.shape[0] == group.shape[0]
        return is_main_activity_series
    max_duration_index = group["activity_duration_in_minutes"].idxmax()
    is_main_activity_series[max_duration_index] = 1
    assert is_main_activity_series.shape[0] == group.shape[0]
    return is_main_activity_series


# def activity_duration_in_minutes(group):
#     """
#     Calculate the duration of each activity between the end of the previous leg and the start of the next leg.
#     :param group: Population frame grouped by person_id
#     :return: edited group with column "activity_duration"
#     """
#     previous_end_time = None  # Initialize previous end time
#     logger.debug(f"Calculating activity durations for person {group[PERSON_ID_COL].iloc[0]}...")
#     for idx, row in group.iterrows():
#         if previous_end_time is None:
#             # For the first row in each group, there's no previous leg's end time
#             group.at[idx, "activity_duration_in_minutes"] = None
#         else:
#             # Convert string to datetime
#             start_time = pd.to_datetime(row[LEG_START_TIME_COL], errors='coerce')
#             end_time = pd.to_datetime(previous_end_time, errors='coerce')
#
#             if pd.isna(start_time) or pd.isna(end_time):
#                 group.at[idx, "activity_duration_in_minutes"] = None
#             else:
#                 duration = (start_time - end_time).total_seconds() / 60
#                 group.at[idx, "activity_duration_in_minutes"] = int(duration)
#
#         # Update previous_end_time for the next iteration
#         previous_end_time = row.get(LEG_END_TIME_COL, None)
#
#     return group

def activity_duration_in_minutes(group):
    """
    Calculate the duration of each activity between the end of the previous leg and the start of the next leg.
    :param group: Population frame grouped by person_id
    :return: Series with activity_duration in minutes
    """
    previous_end_time = None  # Initialize previous end time
    activity_durations = []  # Initialize a list to store activity durations

    logger.debug(f"Calculating activity durations for person {group[PERSON_ID_COL].iloc[0]}...")
    for idx, row in group.iterrows():
        if previous_end_time is None:
            # For the first row in each group, there's no previous leg's end time
            activity_durations.append(None)
        else:
            # Convert string to datetime
            start_time = pd.to_datetime(row[LEG_START_TIME_COL], errors='coerce')
            end_time = pd.to_datetime(previous_end_time, errors='coerce')

            if pd.isna(start_time) or pd.isna(end_time):
                activity_durations.append(None)
            else:
                duration = (start_time - end_time).total_seconds() / 60
                activity_durations.append(int(duration))

        # Update previous_end_time for the next iteration
        previous_end_time = row.get(LEG_END_TIME_COL, None)

    return pd.Series(activity_durations, index=group.index)


def connected_activities(household_group):
    """
    Find connections between activities in a household.
    Assumes leg_id is unique.
    :param household_group:
    :return:
    """

    connections = pd.Series(index=household_group.index, dtype='object')
    if household_group[PERSON_ID_COL].nunique() == 1:
        logger.debug(f"Household {household_group[HOUSEHOLD_MID_ID_COL].iloc[0]} has only one person. No connections.")
        return connections

    for idx_a, person_a_leg in household_group.iterrows():
        connections.at[idx_a] = []
        for idx_b, person_b_leg in household_group.iterrows():
            if person_a_leg[PERSON_ID_COL] == person_b_leg[PERSON_ID_COL]:
                continue
            dist_match = check_distance(person_a_leg, person_b_leg)
            time_match = check_time(person_a_leg, person_b_leg)
            mode_match = check_mode(person_a_leg, person_b_leg)
            activity_match = check_activity(person_a_leg, person_b_leg)
            logger.debug(f"Person {person_a_leg[PERSON_ID_COL]} and {person_b_leg[PERSON_ID_COL]}: "
                         f"distance {dist_match}, time {time_match}, mode {mode_match}, activity {activity_match}")
            if dist_match and time_match and mode_match and activity_match:
                connections.at[idx_a].append(person_b_leg[LEG_ID_COL])
                if connections.at[idx_b] is None:
                    connections.at[idx_b] = []
                connections.at[idx_b].append(person_a_leg[LEG_ID_COL])
    # Find if all lists in connections are empty
    if all(len(lst) == 0 for lst in connections):
        logger.debug(f"No connections found for household {household_group[HOUSEHOLD_MID_ID_COL].iloc[0]}.")
    else:
        logger.info(f"Connections found for household {household_group[HOUSEHOLD_MID_ID_COL].iloc[0]}: {connections}")
    return connections


def check_distance(activity1, activity2):
    distance_to_find = activity1[LEG_DISTANCE_COL]
    distance_to_compare = activity2[LEG_DISTANCE_COL]

    if pd.isnull(distance_to_find) or pd.isnull(distance_to_compare):
        return False

    difference = abs(distance_to_find - distance_to_compare)
    range_tolerance = distance_to_find * 0.05

    return difference <= range_tolerance


def check_time(activity1, activity2):
    # Using constant variables instead of strings
    leg_begin_to_find = activity1[LEG_START_TIME_COL]
    leg_end_to_find = activity1[LEG_END_TIME_COL]
    leg_begin_to_compare = activity2[LEG_START_TIME_COL]
    leg_end_to_compare = activity2[LEG_END_TIME_COL]

    time_range = pd.Timedelta(minutes=5)

    if pd.isnull([leg_begin_to_find, leg_end_to_find, leg_begin_to_compare, leg_end_to_compare]).any():
        return False

    begin_difference = abs(leg_begin_to_find - leg_begin_to_compare)
    end_difference = abs(leg_end_to_find - leg_end_to_compare)

    return (begin_difference <= time_range) and (end_difference <= time_range)


def check_mode(leg_row_to_find, leg_row_to_compare):
    """
    Check if the modes of two legs are compatible.
    Note: Adjusting the mode "car" to "ride" based on age is now its own function.
    :param leg_row_to_find:
    :param leg_row_to_compare:
    :return:
    """
    mode_to_find = leg_row_to_find[LEG_MAIN_MODE_COL]
    mode_to_compare = leg_row_to_compare[LEG_MAIN_MODE_COL]

    if mode_to_find == mode_to_compare:
        return True

    mode_pairs = {(MODE_CAR, MODE_RIDE), (MODE_RIDE, MODE_CAR),
                  (MODE_WALK, MODE_BIKE), (MODE_BIKE, MODE_WALK)}
    if (mode_to_find, mode_to_compare) in mode_pairs:
        return True

    if MODE_UNDEFINED in [mode_to_find, mode_to_compare]:
        # Assuming if one mode is undefined and the other is car, they pair as ride
        return MODE_CAR in [mode_to_find, mode_to_compare]

    return False


# Define an enumeration for ActivityType for clarity
class ActivityType:
    accompany = "accompany"
    shopping = "shopping"
    errands = "errands"
    leisure = "leisure"
    undefined = "undefined"
    home = "home"
    work = "work"


def check_activity(leg_row_to_find, leg_row_to_compare):
    type_to_find = leg_row_to_find['activity_type']
    type_to_compare = leg_row_to_compare['activity_type']

    is_pair = False

    # If activity types are the same or one of them is 'accompany', they are a pair
    if type_to_find == type_to_compare or type_to_find == ActivityType.accompany or type_to_compare == ActivityType.accompany:
        is_pair = True
    # If activity types are different but compatible, they are also a pair
    elif type_to_find in [ActivityType.shopping, ActivityType.errands, ActivityType.leisure]:
        compatible_activities = {
            ActivityType.shopping: [ActivityType.errands],
            ActivityType.errands: [ActivityType.shopping, ActivityType.leisure],
            ActivityType.leisure: [ActivityType.errands, ActivityType.shopping]
        }
        if type_to_compare in compatible_activities[type_to_find]:
            is_pair = True
    # Undefined activities require special handling or logging
    elif type_to_find == ActivityType.undefined or type_to_compare == ActivityType.undefined:
        print("Activity Type Undefined")
    # Special case for 'home' and 'work' activities
    elif type_to_find == ActivityType.home and type_to_compare != ActivityType.work:
        # Assuming trip home, potentially adapt activity type here
        pass
    elif type_to_compare == ActivityType.home and type_to_find != ActivityType.work:
        # Assuming trip home, potentially adapt activity type here
        pass

    return is_pair
