# TODO: Refactor and remove obsolete functions.
# TODO: Medium term goal: Remove. We're trying to get away from this architecture
#  (running code on individual rows through this setup, where possible we should use vectorized operations, where not
#  this architecture is still not needed).

import random

import pandas as pd

from utils import settings as s, helpers as h
from utils.logger import logging

logger = logging.getLogger(__name__)


# def unique_household_id(row):
#     """
#     Household_ids are unique in MiD (or other dataset), but not in the expanded population.
#     We simply use the MiD household id and the row index to create a unique household id.
#     """
#     return f"{row[s.HOUSEHOLD_MID_ID_COL]}_{row.name}"


# def unique_person_id(row):
#     """
#     Person_ids are unique in MiD (or other dataset), but not in the expanded population.
#     Returns a unique person ID from the unique household ID and the person ID.
#     """
#     return f"{row['unique_household_id']}_{row[s.PERSON_ID_COL]}"
#
#
# def unique_leg_id(row):
#     """
#     Returns a unique leg ID from the unique person ID and the leg ID.
#     """
#     return f"{row['unique_person_id']}_{row[s.LEG_ID_COL]}"

#
# def main_mode_imputed(row):
#     """
#     Impute main mode based on the distance of the leg, the availability of a car and statistical probabilities.
#     Note: Imputed main mode is available in MiD, so this isn't needed, but it's here for reference.
#     :param row:
#     :return: mode
#     """
#     r = random.random()
#
#     boundaries_never_or_no_license = [
#         (0.5, [0.89], ["walk", "bike"]),
#         (1, [0.74], ["walk", "bike"]),
#         (2, [0.48, 0.71], ["walk", "bike", "ride"]),
#         (5, [0.25, 0.52, 0.81], ["walk", "bike", "ride", "pt"]),
#         (float('inf'), [0.53], ["ride", "pt"])
#     ]
#
#     boundaries_otherwise = [
#         (0.5, [0.89], ["walk", "bike"]),
#         (1, [0.56, 0.76], ["walk", "bike", "car"]),
#         (2, [0.31, 0.52, 0.65], ["walk", "bike", "ride", "car"]),
#         (5, [0.14, 0.29, 0.45, 0.90], ["walk", "bike", "ride", "car", "pt"]),
#         (float('inf'), [0.26, 0.77], ["ride", "car", "pt"])
#     ]
#
#     boundaries = boundaries_never_or_no_license if row[s.CAR_AVAIL_COL] == s.CAR_NEVER or row[
#         "imputed_license"] == s.LICENSE_NO else boundaries_otherwise
#
#     for distance, probabilities, modes in boundaries:
#         if row[s.LEG_DISTANCE_KM_COL] < distance:
#             for prob, mode in zip(probabilities, modes):
#                 if r <= prob:
#                     return mode
#             return modes[-1]  # If the random number is greater than all probabilities, return the last mode in the list
#
#     return None

#
# def is_main_activity(person):
#     """
#     Check if the leg is travelling to the main activity of the day.
#     Requires calculate_activity_time() to be run first.
#     :param person: Population frame grouped by person_id
#     :return: Series indicating if each row is the main activity: 1 if main activity, 0 if not
#     """
#     is_main_activity_series = pd.Series(0, index=person.index)  # Initialize all values to 0
#
#     # Filter out home activities (home must not be the main activity)
#     group = person[person[s.ACT_TO_INTERNAL_COL] != s.ACT_HOME]
#
#     if group.empty:
#         logger.debug(f"Person {person[s.PERSON_ID_COL].iloc[0]} has no legs outside home. No main activity.")
#         return is_main_activity_series
#
#     if len(group) == 1:
#         # If the person has no legs, there is no main activity
#         if group[s.LEG_NON_UNIQUE_ID_COL].isna().all():
#             logger.debug(f"Person {group[s.PERSON_ID_COL].iloc[0]} has no legs. No main activity.")
#             return is_main_activity_series
#
#         # If the person has only one activity outside home, it is the main activity
#         main_activity_index = group.index[0]
#         is_main_activity_series.at[main_activity_index] = 1
#         assert is_main_activity_series.sum() == 1
#         return is_main_activity_series
#
#     # If the person has more than one activity, the main activity is the first work activity
#     work_activity_rows = group[group[s.ACT_TO_INTERNAL_COL] == s.ACT_WORK]
#     if not work_activity_rows.empty:
#         is_main_activity_series[work_activity_rows.index[0]] = 1
#         assert is_main_activity_series.sum() == 1
#         logger.debug(f"Person {group[s.PERSON_ID_COL].iloc[0]} has a work activity. Main activity is work.")
#         return is_main_activity_series
#
#     # If the person has no work activity, the main activity is the first education activity
#     education_activity_rows = group[
#         group[s.ACT_TO_INTERNAL_COL].isin([s.ACT_EDUCATION, s.ACT_EARLY_EDUCATION, s.ACT_DAYCARE])]
#     if not education_activity_rows.empty:
#         is_main_activity_series[education_activity_rows.index[0]] = 1
#         assert is_main_activity_series.sum() == 1
#         logger.debug(f"Person {group[s.PERSON_ID_COL].iloc[0]} has an education activity. Main activity is education.")
#         return is_main_activity_series
#
#     # If the person has no work or education activity, the main activity is the longest activity
#     if group[s.ACT_DUR_SECONDS_COL].isna().all():
#         # If all activities have no duration, pick the middle one
#         is_main_activity_series.iloc[len(group) // 2] = 1  # Integer division
#         assert is_main_activity_series.sum() == 1
#         logger.debug(f"Person {group[s.PERSON_ID_COL].iloc[0]} has no activities with duration. Main activity is middle.")
#         return is_main_activity_series
#     max_duration_index = group[s.ACT_DUR_SECONDS_COL].idxmax()
#     is_main_activity_series[max_duration_index] = 1
#     assert is_main_activity_series.sum() == 1
#     logger.debug(f"Person {group[s.PERSON_ID_COL].iloc[0]} has no work or education activity. "
#                  f"Main activity is longest activity.")
#     return is_main_activity_series
#
#
# def is_protagonist(household_group):
#     """
#     Identify the 'protagonist' leg among connected legs.
#     The leg with the highest-ranked activity in each group of connected legs is considered the protagonist.
#     Allows to easily add other ranking criteria.
#     :return: pandas Series indicating if each leg is a protagonist (1) or not (0)
#     """
#     prot_series = pd.Series(0, index=household_group.index)
#
#     if household_group[s.CONNECTED_LEGS_COL].isna().all():
#         logger.debug(f"No connections exist for household {household_group[s.HOUSEHOLD_MID_ID_COL].iloc[0]}.")
#         return prot_series
#     else:
#         logger.debug(f"Finding protagonist for household {household_group[s.HOUSEHOLD_MID_ID_COL].iloc[0]}")
#
#     # Ranked activities (lowest to highest probability of being protagonist). Not an exact science.
#     # Activities not in this list are ranked lowest.
#     activities_ranked = [
#         s.ACT_ERRANDS,
#         s.ACT_LEISURE,
#         s.ACT_MEETUP,
#         s.ACT_SHOPPING,
#         s.ACT_EDUCATION,
#         s.ACT_LESSONS,
#         s.ACT_SPORTS,
#         s.ACT_EARLY_EDUCATION,
#         s.ACT_DAYCARE,  # Likely to be dropped off/picked up
#         s.ACT_BUSINESS,  # Likely to be accompanied
#         s.ACT_HOME]  # Home must stay home
#
#     # Keep track of checked legs, so we don't waste time checking them again
#     checked_legs = []
#     for idx, row in household_group.iterrows():
#         if not isinstance(row[s.CONNECTED_LEGS_COL], list):  # Checking for NaN doesn't work here
#             continue
#         if idx in checked_legs:
#             continue
#         connected_legs = set(row[s.CONNECTED_LEGS_COL]).union({row[s.UNIQUE_LEG_ID_COL]})  # Including the current leg
#         leg_data = []
#         for leg_id in connected_legs:
#             if not all(elem in connected_legs for elem in
#                        household_group.loc[household_group[s.UNIQUE_LEG_ID_COL] == leg_id, s.CONNECTED_LEGS_COL].iloc[0]):
#                 logger.warning(f"Leg {leg_id} has inconsistent connections."
#                                f"This might lead to unexpected results.")
#
#             checked_legs.append(household_group.loc[household_group[s.UNIQUE_LEG_ID_COL] == leg_id].index[0])
#             leg_activity = household_group.loc[household_group[s.UNIQUE_LEG_ID_COL] == leg_id, s.ACT_TO_INTERNAL_COL].iloc[0]
#             activity_rank = activities_ranked.index(leg_activity) if leg_activity in activities_ranked else -1
#             leg_data.append({'leg_id': leg_id, 'activity': leg_activity, 'activity_rank': activity_rank})
#
#         # Identifying protagonist leg
#         connected_legs_df = pd.DataFrame(leg_data)
#         if not connected_legs_df.empty:
#             connected_legs_df.sort_values(by='activity_rank', ascending=False, inplace=True)
#             protagonist_leg_id = connected_legs_df.iloc[0]['leg_id']
#             prot_series.loc[household_group[s.UNIQUE_LEG_ID_COL] == protagonist_leg_id] = 1
#
#     return prot_series

#
# def home_loc(row):
#     """
#     Convert the home location string to a Shapely Point.
#     :param row: Row of the household frame
#     :return: Shapely Point
#     """
#     return h.string_to_shapely_point(row['home_loc'])
