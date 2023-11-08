import os
import random

import pandas as pd

from utils import matsim_pipeline_setup
from utils.logger import logging

logger = logging.getLogger(__name__)

# Column names that are used in several places are defined in settings.yaml
os.chdir(matsim_pipeline_setup.PROJECT_ROOT)
settings = matsim_pipeline_setup.load_yaml_config('settings.yaml')
HOUSEHOLD_ID_COLUMN = settings['household_id_column']
PERSON_ID_COLUMN = settings['person_id_column']
LEG_ID_COLUMN = settings['leg_id_column']

AVERAGE_ACTIVITY_TIMES = settings['average_activity_times']  # in seconds

# Column names that are just used by rules are defined here
LEG_ACTIVITY = 'leg_target_activity'
LEG_MAIN_MODE = 'leg_main_mode'
LEG_START_TIME = 'leg_start_time'
LEG_END_TIME = 'leg_end_time'
LEG_DURATION = 'leg_duration'
LEG_DISTANCE = 'leg_distance'

PERSON_AGE = 'HP_ALTER'

CAR_AVAIL = 'P_AUTO'
HAS_LICENSE = 'P_FS_PKW'

# Values in the MiD-columns
ACTIVITY_HOME = '8'
ACTIVITIES_EDUCATION = {'2', '3', '4', '5', '6', '7'}
ACTIVITY_WORK = '1'

CAR_NEVER = '3'

LICENSE_YES = '1'
LICENSE_NO = '2'
LICENSE_UNKNOWN = '9'
ADULT_OVER_16_PROXY = '206'
PERSON_UNDER_16 = '403'

rule_required_hh_columns = {
    PERSON_ID_COLUMN,
    LEG_ACTIVITY,
    LEG_MAIN_MODE,
    LEG_START_TIME,
    LEG_END_TIME,
    LEG_DURATION,
    LEG_DISTANCE,
    PERSON_AGE
}
rule_required_person_columns = {
    PERSON_ID_COLUMN,
    LEG_ACTIVITY,
    LEG_MAIN_MODE,
    LEG_START_TIME,
    LEG_END_TIME,
    LEG_DURATION,
    LEG_DISTANCE,
    PERSON_AGE
}
rule_required_leg_columns = {
    LEG_ID_COLUMN,
    LEG_ACTIVITY,
    LEG_MAIN_MODE,
    LEG_START_TIME,
    LEG_END_TIME,
    LEG_DURATION,
    LEG_DISTANCE,
    PERSON_AGE
}


def unique_leg_id(row):
    return row[PERSON_ID_COLUMN] + '_' + row[LEG_ID_COLUMN]


def has_license_imputed(row):
    if row[HAS_LICENSE] == LICENSE_NO:
        return 0
    elif row[HAS_LICENSE] == LICENSE_YES:
        if row[PERSON_AGE] < 17:
            logger.debug(f"Person {row[PERSON_ID_COLUMN]} has a car driving license but is under 17 years old. Assuming no license.")
            return 0
        return 1
    elif row[HAS_LICENSE] == LICENSE_UNKNOWN or ADULT_OVER_16_PROXY or pd.isnull(row[HAS_LICENSE]):
        r = random.random()
        if row[PERSON_AGE] >= 17:
            if r <= 0.94:
                logger.debug(f"Adult {row[PERSON_ID_COLUMN]} has unknown license status. Assuming license.")
                return 1
            else:
                logger.debug(f"Adult {row[PERSON_ID_COLUMN]} has unknown license status. Assuming no license.")
                return 0
        return 0
    elif row[HAS_LICENSE] == PERSON_UNDER_16:
        return 0
    else:
        logger.warning(f"Person {row[PERSON_ID_COLUMN]} has no license status entry, not even unknown. Assuming no license.")
        return 0


def main_mode_imputed(row):
    """
    Impute main mode based on the distance of the leg, the availability of a car and statistical probabilities.
    Note: Imputed main mode is available in MiD, so this isn't needed, but it's here for reference
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

    boundaries = boundaries_never_or_no_license if row[CAR_AVAIL] == CAR_NEVER or row[
        has_license_imputed] == 0 else boundaries_otherwise

    for distance, probabilities, modes in boundaries:
        if row[LEG_DISTANCE] < distance:
            for prob, mode in zip(probabilities, modes):
                if r <= prob:
                    return mode
            return modes[-1]  # If the random number is greater than all probabilities, return the last mode in the list

    return None


def collapse_person_trip(group):
    """
    Process a person's trip and represent it as a list of activity legs (intermediary for further processing).

    Parameters:
    - group (pd.DataFrame): A DataFrame group representing a person's trip.

    Returns:
    - tuple: (processed_trip_representation, missing_columns)
    """

    trip_representation = [{'activity': row['activity'], 'duration': row['duration']}
                           for _, row in group.iterrows()]
    summary_df = pd.DataFrame({
        'trip': [trip_representation] * len(group)
    }, index=group.index)
    return summary_df, []


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
    sum_durations_before_main = group.loc[:main_activity_index, LEG_DURATION].sum()  # Minutes
    sum_durations_after_main = group.loc[main_activity_index:, LEG_DURATION].sum()

    # Estimate leg duration:
    average_leg_duration = group[LEG_DURATION].mean()  # Minutes
    average_leg_duration_after_main = group.loc[main_activity_index:, LEG_DURATION].mean()
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
        activity_time = AVERAGE_ACTIVITY_TIMES[last_leg[LEG_ACTIVITY]]
    except KeyError:
        activity_time = 3600  # 1 hour default

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
            if row[LEG_ACTIVITY] == ACTIVITY_WORK:
                row["is_main_activity"] = 1
                return group

        # If the person has no work activity, the main activity is the first education activity
        for i, row in group.iterrows():
            if row[LEG_ACTIVITY] in ACTIVITIES_EDUCATION:
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
        if not row[LEG_START_TIME] or not row[LEG_END_TIME]:
            row["activity_duration_in_minutes"] = None
        else:
            row["activity_duration_in_minutes"] = int(
                (row[LEG_START_TIME] - group.iloc[i - 1][LEG_END_TIME]).total_seconds() / 60)
    return group
