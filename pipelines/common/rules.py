import random

from utils.logger import logging

logger = logging.getLogger(__name__)


def rule1(row):
    missing_columns = set()
    try:
        if row['col1'] == 1 and row['col2'] > 11 and row['col3'] == 'single':
            return True, missing_columns
    except KeyError as e:
        missing_columns.add(e.args[0])  # Add the missing column to the set
    return None, missing_columns


def raw_plan(row): # Concatenate plan attributes into a string
    missing_columns = set()

    has_license = row.get("hasLicense")  # Doesn't throw an error if the column is missing
    if has_license is None:
        missing_columns.add('hasLicense')

def rule2(row):
    missing_columns = set()
    try:
        if row['col4'] == 'A':
            return "Type A", missing_columns
        elif row['col4'] == 'B':
            return "Type B", missing_columns
        elif row['col4'] == 'C':
            return "Type C", missing_columns
        else:
            return "Unknown Type", missing_columns
    except KeyError as e:
        missing_columns.add(e.args[0])
        return None, missing_columns


def rulebased_main_mode(row):  # Function name will be used as the column name
    missing_columns = set()

    # Extract attributes from the row
    has_license = row.get("hasLicense")  # Doesn't throw an error if the column is missing
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


rules = [rule1, rule2, rule3, rule4, rule5]

updated_df = safe_apply_rules(df, rules)
