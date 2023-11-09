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
LED_ID_COL = ID_COLUMNS['leg_id_column']
LEG_ACTIVITY_COL = L_COLUMNS['leg_target_activity']
LEG_MAIN_MODE_COL = L_COLUMNS['leg_main_mode']
LEG_START_TIME_COL = L_COLUMNS['leg_start_time']
LEG_END_TIME_COL = L_COLUMNS['leg_end_time']
LEG_DURATION_COL = L_COLUMNS['leg_duration']
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


def unique_household_id(row):
    """
    We need a unique household ID for later group-based rules. Popsim provides just that.
    """
    return f"{row[HOUSEHOLD_MID_ID_COL]}_{row[HOUSEHOLD_POPSIM_ID_COL]}"


def unique_person_id(row):
    """
    Returns a unique person ID from the unique household ID and the person ID.
    """
    return f"{row['unique_household_id']}_{row[PERSON_ID_COL]}"


def unique_leg_id(row):
    """
    Returns a unique leg ID from the unique person ID and the leg ID.
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


def add_return_home_leg(group):
    """
    Add a home leg at the end of the day if the person doesn't already have one.
    The length of the activity and the leg duration are estimated.
    Requires is_main_activity() to be run first.
    :param group: Population frame grouped by person_id
    :return: edited group
    """

    main_activity_index = group[group['is_main_activity'] == 1].index[
        0]  # There should only be one main activity but [0] just in case
    sum_durations_before_main = group.loc[:main_activity_index, LEG_DURATION_COL].sum()  # Minutes
    sum_durations_after_main = group.loc[main_activity_index:, LEG_DURATION_COL].sum()

    # Estimate leg duration:
    average_leg_duration = group[LEG_DURATION_COL].mean()  # Minutes
    average_leg_duration_after_main = group.loc[main_activity_index:, LEG_DURATION_COL].mean()
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
    home_leg['LEG_ID'] = last_leg['LEG_ID'] + 1
    home_leg['LEG_START_TIME'] = last_leg['LEG_END_TIME'] + pd.Timedelta(minutes=activity_time)
    home_leg['LEG_END_TIME'] = home_leg['LEG_START_TIME'] + pd.Timedelta(minutes=home_leg_duration)
    home_leg['LEG_ACTIVITY'] = ACTIVITY_HOME
    home_leg['LEG_DURATION'] = home_leg_duration
    home_leg['LEG_DISTANCE'] = None

    # Append the home leg to the group
    group = group.append(home_leg, ignore_index=True)
    return group


def is_main_activity(group):
    """
    Check if the leg is travelling to the main activity of the day.
    Requires activity_duration_in_seconds() to be run first.
    :param group: Population frame grouped by person_id
    :return: edited group with column "is_main_activity": 1 if main activity, 0 if not
    """
    # If the person has only one activity, it is the main activity
    if len(group) == 1:
        group["is_main_activity"] = 1
        return group

    else:
        group["is_main_activity"] = 0

        # If the person has more than one activity, the main activity is the first work activity
        for i, row in group.iterrows():
            if row[LEG_ACTIVITY_COL] == ACTIVITY_WORK:
                row["is_main_activity"] = 1
                return group

        # If the person has no work activity, the main activity is the first education activity
        for i, row in group.iterrows():
            if row[LEG_ACTIVITY_COL] in ACTIVITIES_EDUCATION:
                row["is_main_activity"] = 1
                return group

        # If the person has no work or education activity, the main activity is the longest activity
        max_duration = group["activity_duration_in_minutes"].idxmax()
        group.loc[max_duration, "is_main_activity"] = 1
        return group


def activity_duration_in_minutes(group):
    """
    Calculate the duration of each activity between the end of the previous leg and the start of the next leg.
    :param group: Population frame grouped by person_id
    :return: edited group with column "activity_duration"
    """
    for i, row in group.iterrows():
        if not row[LEG_START_TIME_COL] or not row[LEG_END_TIME_COL]:
            row["activity_duration_in_minutes"] = None
        else:
            row["activity_duration_in_minutes"] = int(
                (row[LEG_START_TIME_COL] - group.iloc[i - 1][LEG_END_TIME_COL]).total_seconds() / 60)
    return group
