import os
import shutil
import pandas as pd
import pytest
from datetime import datetime, timedelta

from synthesis.enhanced_mid.run_enhanced_mid import enhance_travel_survey
from utils import settings as s


@pytest.fixture(scope='module')
def setup_test_environment():
    test_project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )  # Assuming test is two levels down from the project root
    test_input_dir = test_project_root + '/tests/test_data'
    test_output_dir = test_project_root + '/tests/output'
    test_mid_hh_folder = os.path.join(test_input_dir, s.MiD_HH_FOLDER)
    test_mid_persons_folder = os.path.join(test_input_dir, s.MiD_PERSONS_FOLDER)
    test_mid_trips_folder = os.path.join(test_input_dir, s.MiD_TRIPS_FOLDER)

    # Create output directory
    os.makedirs(test_output_dir, exist_ok=True)

    yield {
        'test_mid_hh_folder': test_mid_hh_folder,
        'test_mid_persons_folder': test_mid_persons_folder,
        'test_mid_trips_folder': test_mid_trips_folder,
        'test_output_dir': test_output_dir
    }

    # Clean up after tests
    shutil.rmtree(test_output_dir)


def is_float_or_nan(value):
    return isinstance(value, float) or pd.isna(value)


def is_convertible_to_datetime(series):
    """Check if all elements in a series are convertible to pd.Timestamp."""
    try:
        pd.to_datetime(series)
        return True
    except ValueError:
        return False


def test_enhance_travel_survey(setup_test_environment):
    paths = setup_test_environment

    # Run the function
    enhance_travel_survey(
        paths['test_mid_hh_folder'],
        paths['test_mid_persons_folder'],
        paths['test_mid_trips_folder'],
        paths['test_output_dir']
    )

    # Check if the output file is created
    output_file = os.path.join(paths['test_output_dir'], s.ENHANCED_MID_FILE)
    assert os.path.exists(output_file), "Output file was not created."

    # Load the output file and perform content verification
    output_df = pd.read_csv(output_file)
    assert not output_df.empty, "Output DataFrame is empty."

    # Load the hh file to compare row counts
    hh_df = pd.read_csv(paths['test_mid_hh_folder'] + '/households.csv')
    num_hhs = len(hh_df)
    num_output_rows = len(output_df)

    # Ensure the number of rows in the output is between 2 and 10 times the number of households
    assert num_hhs * 2 <= num_output_rows <= num_hhs * 10, (
        f"The number of rows in the output file ({num_output_rows}) is not within the expected range "
        f"relative to the number of households ({num_hhs})."
    )

    # Ensure there are no merge suffixes in the column names
    merge_suffixes = ['_x', '_y']
    for col in output_df.columns:
        assert not any(suffix in col for suffix in merge_suffixes), (
            f"Column '{col}' contains a merge suffix."
        )

    # Verify NON_UNIQUE_LEG_ID and other conditions for each person
    grouped = output_df.groupby(s.PERSON_ID_COL)

    for person_id, group in grouped:
        if len(group) > 1:
            # Ensure start_time and end_time columns are in datetime format
            assert group[s.LEG_START_TIME_COL].apply(is_convertible_to_datetime).all()
            assert group[s.LEG_END_TIME_COL].apply(is_convertible_to_datetime).all()

            # Define the base date and the maximum allowed datetime
            base_date = pd.to_datetime(s.BASE_DATE)  # Replace with your actual base date
            max_allowed_date = base_date + timedelta(days=2)

            group[s.LEG_START_TIME_COL] = pd.to_datetime(group[s.LEG_START_TIME_COL])
            group[s.LEG_END_TIME_COL] = pd.to_datetime(group[s.LEG_END_TIME_COL])

            # Ensure all times are within the allowed range
            assert (group[s.LEG_START_TIME_COL] <= max_allowed_date).all(), (
                f"One or more 'start_time' values exceed the allowed maximum date of {max_allowed_date} for person {person_id}."
            )
            assert (group[s.LEG_END_TIME_COL] <= max_allowed_date).all(), (
                f"One or more 'end_time' values exceed the allowed maximum date of {max_allowed_date} for person {person_id}."
            )

            # Check that NON_UNIQUE_LEG_ID is a float or string with dot (just for better NaN handling)
            assert group[s.LEG_NON_UNIQUE_ID_COL].apply(is_float_or_nan).all(), (
                f"Person {person_id} has NON_UNIQUE_LEG_ID values that are not whole numbers."
            )

            # Check that NON_UNIQUE_LEG_ID matches the last part of unique_leg_id
            non_unique_leg_ids = group[s.LEG_NON_UNIQUE_ID_COL].values
            unique_leg_ids = group['unique_leg_id'].values
            for i, unique_leg_id in enumerate(unique_leg_ids):
                last_part = unique_leg_id.split('_')[-1]
                assert str(last_part) == str(non_unique_leg_ids[i]), (
                    f"Person {person_id} has a mismatch between NON_UNIQUE_LEG_ID and the last part of unique_leg_id "
                    f"at index {i}. Found: {last_part}, Expected: {non_unique_leg_ids[i]}"
                )

            # Check the sequence of NON_UNIQUE_LEG_ID
            expected_leg_ids = list(range(1, len(non_unique_leg_ids) + 1))
            assert list(map(int, non_unique_leg_ids)) == expected_leg_ids, (
                f"NON_UNIQUE_LEG_ID for person {person_id} does not start at 1 "
                f"and increment by 1 without gaps. Found: {list(non_unique_leg_ids)}, "
                f"Expected: {expected_leg_ids}"
            )

            # Check that each person has a main activity (unless they only have home-to-home legs)
            if not group[s.ACT_TO_INTERNAL_COL].eq(s.ACT_HOME).all():
                assert sum(group[s.IS_MAIN_ACTIVITY_COL]) == 1, (
                    f"Person {person_id} does not have exactly one main activity."
                )
            else:
                assert not group[s.IS_MAIN_ACTIVITY_COL].any(), (
                    f"Person {person_id} has only home-to-home legs but also has a main activity."
                )

    # Additional content verification can be added here
