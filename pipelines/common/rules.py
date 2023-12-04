import random

import pandas as pd

from pipelines.common import helpers as h
from utils import settings_values as s
from utils.logger import logging

logger = logging.getLogger(__name__)


def unique_household_id(row):
    """
    Household_ids are unique in MiD (or other dataset), but not in the expanded population.
    We simply use the MiD household id and the row index to create a unique household id.
    """
    return f"{row[s.HOUSEHOLD_MID_ID_COL]}_{row.name}"


def unique_person_id(row):
    """
    Person_ids are unique in MiD (or other dataset), but not in the expanded population.
    Returns a unique person ID from the unique household ID and the person ID.
    """
    return f"{row['unique_household_id']}_{row[s.PERSON_ID_COL]}"


def unique_leg_id(row):
    """
    Returns a unique leg ID from the unique person ID and the leg ID.
    """
    return f"{row['unique_person_id']}_{row[s.LEG_ID_COL]}"


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

    boundaries = boundaries_never_or_no_license if row[s.CAR_AVAIL_COL] == s.CAR_NEVER or row[
        "imputed_license"] == s.LICENSE_NO else boundaries_otherwise

    for distance, probabilities, modes in boundaries:
        if row[s.LEG_DISTANCE_COL] < distance:
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


def is_main_activity(group):
    """
    Check if the leg is travelling to the main activity of the day.
    Requires calculate_activity_time() to be run first.
    :param group: Population frame grouped by person_id
    :return: Series indicating if each row is the main activity: 1 if main activity, 0 if not
    """
    is_main_activity_series = pd.Series(0, index=group.index)  # Initialize all values to 0

    # If the person has only one activity, it is the main activity
    if len(group) == 1:
        is_main_activity_series.iloc[0] = 1
        assert is_main_activity_series.shape[0] == group.shape[0]
        assert is_main_activity_series.sum() == 1
        return is_main_activity_series

    # If the person has more than one activity, the main activity is the first work activity
    work_activity_rows = group[group[s.LEG_ACTIVITY_COL] == s.ACTIVITY_WORK]
    if not work_activity_rows.empty:
        is_main_activity_series[work_activity_rows.index[0]] = 1
        assert is_main_activity_series.shape[0] == group.shape[0]
        assert is_main_activity_series.sum() == 1
        return is_main_activity_series

    # If the person has no work activity, the main activity is the first education activity
    education_activity_rows = group[
        group[s.LEG_ACTIVITY_COL].isin([s.ACTIVITY_EDUCATION, s.ACTIVITY_EARLY_EDUCATION, s.ACTIVITY_DAYCARE])]
    if not education_activity_rows.empty:
        is_main_activity_series[education_activity_rows.index[0]] = 1
        assert is_main_activity_series.shape[0] == group.shape[0]
        assert is_main_activity_series.sum() == 1
        return is_main_activity_series

    # If the person has no work or education activity, the main activity is the longest activity
    if group["activity_duration_seconds"].isna().all():
        # If all activities have no duration, pick the middle one
        is_main_activity_series.iloc[len(group) // 2] = 1
        assert is_main_activity_series.shape[0] == group.shape[0]
        assert is_main_activity_series.sum() == 1
        return is_main_activity_series
    max_duration_index = group["activity_duration_seconds"].idxmax()
    is_main_activity_series[max_duration_index] = 1
    assert is_main_activity_series.shape[0] == group.shape[0]
    assert is_main_activity_series.sum() == 1
    return is_main_activity_series


def connected_legs(household_group):
    """
    Find connections between trip legs in a household.
    Uses unique_leg_id; lists all legs that are connected to each leg.
    :param household_group:
    :return: Series: Each item a list of all connected legs, NaN if no connections
    """

    connections = pd.Series(index=household_group.index, dtype='object')
    if household_group[s.PERSON_ID_COL].nunique() == 1:
        logger.debug(f"Household {household_group[s.HOUSEHOLD_MID_ID_COL].iloc[0]} has only one person. No connections.")
        return connections

    for idx_a, person_a_leg in household_group.iterrows():
        for idx_b, person_b_leg in household_group.iterrows():
            if person_a_leg[s.PERSON_ID_COL] == person_b_leg[s.PERSON_ID_COL] or idx_b <= idx_a:
                continue  # So we don't compare a leg to itself or to a leg it's already been compared to

            dist_match = check_distance(person_a_leg, person_b_leg)
            time_match = check_time(person_a_leg, person_b_leg)
            mode_match = check_mode(person_a_leg, person_b_leg)
            activity_match = check_activity(person_a_leg, person_b_leg)
            logger.debug(f"Legs {person_a_leg['unique_leg_id']} and {person_b_leg['unique_leg_id']}: "
                         f"distance {dist_match}, time {time_match}, mode {mode_match}, activity {activity_match}")
            if dist_match and time_match and mode_match and activity_match:
                if not isinstance(connections.at[idx_a], list):  # Checking for NaN doesn't work here
                    connections.at[idx_a] = []
                if not isinstance(connections.at[idx_b], list):
                    connections.at[idx_b] = []
                connections.at[idx_a].append(person_b_leg['unique_leg_id'])
                connections.at[idx_b].append(person_a_leg['unique_leg_id'])

    if connections.isna().all():
        logger.debug(f"No connections found for household {household_group[s.HOUSEHOLD_MID_ID_COL].iloc[0]}.")
    else:
        logger.debug(f"Connections found for household {household_group[s.HOUSEHOLD_MID_ID_COL].iloc[0]}")
        logger.debug(f"{connections}")
    return connections


def check_distance(leg_to_find, leg_to_compare):
    distance_to_find = leg_to_find[s.LEG_DISTANCE_COL]
    distance_to_compare = leg_to_compare[s.LEG_DISTANCE_COL]

    if pd.isnull(distance_to_find) or pd.isnull(distance_to_compare):
        return False

    difference = abs(distance_to_find - distance_to_compare)
    range_tolerance = distance_to_find * 0.05

    return difference <= range_tolerance


def check_time(leg_to_find, leg_to_compare):
    # Using constant variables instead of strings
    leg_begin_to_find = leg_to_find[s.LEG_START_TIME_COL]
    leg_end_to_find = leg_to_find[s.LEG_END_TIME_COL]
    leg_begin_to_compare = leg_to_compare[s.LEG_START_TIME_COL]
    leg_end_to_compare = leg_to_compare[s.LEG_END_TIME_COL]

    # Reduce the time range for short legs to avoid false positives (NaN evaluates to False)
    time_range = pd.Timedelta(minutes=5) if leg_to_find[s.LEG_DURATION_MINUTES_COL] > 5 and leg_to_compare[
        s.LEG_DURATION_MINUTES_COL] > 5 else pd.Timedelta(minutes=3)

    if pd.isnull([leg_begin_to_find, leg_end_to_find, leg_begin_to_compare, leg_end_to_compare]).any():
        return False

    begin_difference = abs(leg_begin_to_find - leg_begin_to_compare)
    end_difference = abs(leg_end_to_find - leg_end_to_compare)

    return (begin_difference <= time_range) and (end_difference <= time_range)


def check_mode(leg_to_find, leg_to_compare):
    """
    Check if the modes of two legs are compatible.
    Note: Adjusting the mode "car" to "ride" based on age is now its own function.
    :param leg_to_find:
    :param leg_to_compare:
    :return:
    """
    mode_to_find = leg_to_find[s.LEG_MAIN_MODE_COL]
    mode_to_compare = leg_to_compare[s.LEG_MAIN_MODE_COL]

    if mode_to_find == mode_to_compare:
        return True

    mode_pairs = {(s.MODE_CAR, s.MODE_RIDE), (s.MODE_RIDE, s.MODE_CAR),
                  (s.MODE_WALK, s.MODE_BIKE), (s.MODE_BIKE, s.MODE_WALK)}
    if (mode_to_find, mode_to_compare) in mode_pairs:
        return True

    if s.MODE_UNDEFINED in [mode_to_find, mode_to_compare]:
        # Assuming if one mode is undefined and the other is car, they pair as ride
        return s.MODE_CAR in [mode_to_find, mode_to_compare]

    return False


def check_activity(leg_to_find, leg_to_compare):  # TODO: Possibly create a matrix of compatible activities
    compatible_activities = {
        s.ACTIVITY_SHOPPING: [s.ACTIVITY_ERRANDS],
        s.ACTIVITY_ERRANDS: [s.ACTIVITY_SHOPPING, s.ACTIVITY_LEISURE],
        s.ACTIVITY_LEISURE: [s.ACTIVITY_ERRANDS, s.ACTIVITY_SHOPPING, s.ACTIVITY_MEETUP],
        s.ACTIVITY_MEETUP: [s.ACTIVITY_LEISURE]}

    type_to_find = leg_to_find[s.LEG_ACTIVITY_COL]
    type_to_compare = leg_to_compare[s.LEG_ACTIVITY_COL]

    if (type_to_find == type_to_compare or
            s.ACTIVITY_ACCOMPANY_ADULT in [type_to_find, type_to_compare] or
            s.ACTIVITY_PICK_UP_DROP_OFF in [type_to_find, type_to_compare]):
        return True
    elif s.ACTIVITY_UNSPECIFIED in [type_to_find, type_to_compare] or pd.isnull([type_to_find, type_to_compare]).any():
        logger.debug("Activity Type Undefined or Null (which usually means person has no legs).")
        return False
    # Assuming trip home
    elif (type_to_find == s.ACTIVITY_HOME and type_to_compare != s.ACTIVITY_WORK) or \
            (type_to_compare == s.ACTIVITY_HOME and type_to_find != s.ACTIVITY_WORK):
        return True

    return type_to_compare in compatible_activities.get(type_to_find, [])


def is_protagonist(household_group):
    """
    Identify the 'protagonist' leg among connected legs.
    The leg with the highest-ranked activity in each group of connected legs is considered the protagonist.
    Allows to easily add other ranking criteria.
    :return: pandas Series indicating if each leg is a protagonist (1) or not (0)
    """
    prot_series = pd.Series(0, index=household_group.index)

    if household_group['connected_legs'].isna().all():
        logger.debug(f"No connections exist for household {household_group[s.HOUSEHOLD_MID_ID_COL].iloc[0]}.")
        return prot_series

    # Ranked activities
    activities_ranked = [s.ACTIVITY_WORK, s.ACTIVITY_SHOPPING, s.ACTIVITY_LEISURE]

    # Collecting connected legs and their activities in a DataFrame
    checked_legs = []
    for idx, row in household_group.iterrows():
        if not isinstance(row['connected_legs'], list):  # Checking for NaN doesn't work here
            continue
        if idx in checked_legs:
            continue

        connected_legs = set(row['connected_legs']).union({row["unique_leg_id"]})  # Including the current leg
        leg_data = []
        for leg_id in connected_legs:
            if not all(elem in connected_legs for elem in
                       household_group.loc[household_group["unique_leg_id"] == leg_id, 'connected_legs'].iloc[0]):
                logger.warning(f"Leg {leg_id} has inconsistent connections."
                               f"This might lead to unexpected results.")

            checked_legs.append(household_group.loc[household_group["unique_leg_id"] == leg_id].index[0])
            leg_activity = household_group.loc[household_group["unique_leg_id"] == leg_id, s.LEG_ACTIVITY_COL].iloc[0]
            activity_rank = activities_ranked.index(leg_activity) if leg_activity in activities_ranked else -1
            leg_data.append({'leg_id': leg_id, 'activity': leg_activity, 'activity_rank': activity_rank})

        # Identifying protagonist leg
        connected_legs_df = pd.DataFrame(leg_data)
        if not connected_legs_df.empty:
            connected_legs_df.sort_values(by='activity_rank', ascending=False, inplace=True)
            protagonist_leg_id = connected_legs_df.iloc[0]['leg_id']
            prot_series.loc[household_group["unique_leg_id"] == protagonist_leg_id] = 1

    return prot_series


def home_loc(row):
    """
    Convert the home location string to a Shapely Point.
    :param row: Row of the household frame
    :return: Shapely Point
    """
    return h.string_to_shapely_point(row['home_loc'])
