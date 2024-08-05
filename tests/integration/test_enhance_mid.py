import os
import shutil
import pytest
import pandas as pd

from synthesis.enhanced_mid.run_enhance_mid import enhance_travel_survey
from utils import settings as s

@pytest.fixture(scope='module')
def setup_test_environment():
    test_input_dir = 'tests/test_data'
    test_output_dir = 'tests/output'
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

    # Add more assertions to verify the correctness of the output
    # For example, check specific columns, values, etc.
