import random

import pandas as pd

from utils.logger import logging

logger = logging.getLogger(__name__)


def example_rule1(row):
    """
    Example of a row rule.

    Parameters:
    - row: A row from a DataFrame.

    Returns:
    - tuple:
        - result (any): The result of the rule for this row, any type that fits in a single DataFrame cell,
          e.g. a string, a number, a list, a dictionary, etc.
        - missing_columns (set): A set of strings indicating which necessary columns are missing.
    """
    missing_columns = set()

    # Check for missing columns
    has_license = row.get("hasLicense")  # Doesn't throw an error if the column is missing
    if has_license is None:
        missing_columns.add('hasLicense')

    has_pt_card = row.get("hasPTCard")
    if has_pt_card is None:
        missing_columns.add('hasPTCard')

    if missing_columns:
        return None, missing_columns

    mode = "car" if "true" in has_license else "pt" if "true" in has_pt_card else "walk"
    return mode, missing_columns


def example_rule2(group):
    """
    Example of a group rule.

    Parameters:
    - group (DataFrame)

    Returns:
    - tuple:
        - summary_df (DataFrame): Index and number of rows must be the same as the input group. Any number of columns.
          This allows for very flexible output.
        - missing_columns (set): A set of strings indicating which necessary columns are missing.
    """
    missing_columns = set()

    # A different way to check for missing columns, you can use either
    required_columns = ["activity", "duration"]
    missing_columns.update([col for col in required_columns if col not in group.columns])

    if missing_columns:
        return None, missing_columns

    total_shopping_duration = group.loc[group['activity'] == 'shopping', 'duration'].sum()

    # Create a summary DataFrame with the same number of rows and same index as the group (in this case, just duplicated rows)
    summary_df = pd.DataFrame({
        'total_shopping_duration': [total_shopping_duration] * len(group)
    }, index=group.index)

    return summary_df, missing_columns


def rulebased_main_mode(row):
    missing_columns = set()

    # Extract attributes from the row
    has_license = row.get("hasLicense")
    if has_license is None:
        missing_columns.add('hasLicense')

    has_pt_card = row.get("hasPTCard")
    if has_pt_card is None:
        missing_columns.add('hasPTCard')

    car_avail = row.get("carAvail")
    if car_avail is None:
        missing_columns.add('carAvail')

    hh_num_cars = row.get("hh_num_cars")
    if hh_num_cars is None:
        missing_columns.add('hh_num_cars')

    detailed_distance = row.get("detailed_distance")
    if detailed_distance is None:
        missing_columns.add('detailed_distance')

    if missing_columns:
        return None, missing_columns

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

    boundaries = boundaries_never_or_no_license if "never" in car_avail or "false" in has_license else boundaries_otherwise

    for distance, probabilities, modes in boundaries:
        if detailed_distance < distance:
            for prob, mode in zip(probabilities, modes):
                if r <= prob:
                    return mode, missing_columns
            return modes[
                -1], missing_columns  # If the random number is greater than all probabilities, return the last mode in the list

    return None, missing_columns


def collapse_person_trip(group):
    """
    Process a person's trip and represent it as a list of activity legs (intermediary for further processing).

    Parameters:
    - group (pd.DataFrame): A DataFrame group representing a person's trip.

    Returns:
    - tuple: (processed_trip_representation, missing_columns)
    """

    # Check for required columns
    required_columns = ['activity', 'duration']
    missing_columns = [col for col in required_columns if col not in group.columns]

    if missing_columns:
        return None, missing_columns

    trip_representation = [{'activity': row['activity'], 'duration': row['duration']}
                           for _, row in group.iterrows()]
    summary_df = pd.DataFrame({
        'trip': [trip_representation] * len(group)
    }, index=group.index)
    return summary_df, []


# Example usage
data = {
    'person': [1, 1, 1, 2, 2],
    'activity': ['home', 'work', 'home', 'home', 'school'],
    'duration': [0, 120, 40, 0, 100]
}

df = pd.DataFrame(data)

# Applying the rule
result = df.groupby('person').apply(collapse_person_trip)
print(result)
