import random

import add_data_from_MiD
from utils.logger import logging

logger = logging.getLogger(__name__)


def safe_apply_rules(df, rules):
    """
    Applies a set of custom rules to a DataFrame. If a rule references missing columns,
    those columns are fetched from a secondary data source (MiD) and the rules are reapplied.

    Parameters:
    - df (pd.DataFrame): The DataFrame to which the rules will be applied.
    - rules (list of functions): A list of rule functions. Each rule function must return a tuple of (result, missing_columns).

    Returns:
    - pd.DataFrame: The DataFrame with the rules applied as new columns.

    Notes:
    - Could run at different places in the pipeline and might have different rule sets.
    - Will only add columns, never alter existing columns.
    """
    all_missing_columns = set()

    # First pass: identify all missing columns
    for rule_func in rules:
        _, missing_columns_list = zip(*df.apply(rule_func, axis=1))
        rule_missing_columns = set().union(*missing_columns_list)
        all_missing_columns.update(rule_missing_columns)

        if rule_missing_columns:
            logger.info(f"Rule '{rule_func.__name__}' identified missing columns: {', '.join(rule_missing_columns)}")

    # Fetch all missing columns at once
    if all_missing_columns:
        logger.info(f"Fetching missing columns: {', '.join(all_missing_columns)}")
        df = add_data_from_MiD(df, list(all_missing_columns))

    # Second pass: apply the rules now that all columns are present
    for rule_func in rules:
        column_name = rule_func.__name__
        results, missing_columns_list = zip(*df.apply(rule_func, axis=1))
        rule_missing_columns = set().union(*missing_columns_list)

        if rule_missing_columns:
            logger.error(
                f"Rule '{rule_func.__name__}' identified missing columns in second pass and was skipped: {', '.join(rule_missing_columns)}")
        else:
            df[column_name] = results
            null_mask = df[column_name].isnull()
            if null_mask.all():
                logger.warning(f"The rule '{rule_func.__name__}' returned None for all rows.")
            elif null_mask.any():
                logger.warning(f"The rule '{rule_func.__name__}' returned None for {null_mask.sum()} rows.")

    return df


def rule1(row):
    missing_columns = set()
    try:
        if row['col1'] == 1 and row['col2'] > 11 and row['col3'] == 'single':
            return True, missing_columns
    except KeyError as e:
        missing_columns.add(e.args[0])  # Add the missing column to the set
    return None, missing_columns


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

    hh_num_cars = row.get("hh_num_cars")  # Assuming this is needed and present in the DataFrame
    if hh_num_cars is None:
        missing_columns.add('hh_num_cars')

    detailed_distance = row.get("detailed_distance")  # Assuming this is needed and present in the DataFrame
    if detailed_distance is None:
        missing_columns.add('detailed_distance')

    # If any columns are missing, return immediately
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
