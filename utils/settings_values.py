"""
Load settings from settings.yaml file and define constants for use in the pipelines.
"""

import os

import utils.matsim_pipeline_setup

os.chdir(utils.matsim_pipeline_setup.PROJECT_ROOT)
settings = utils.matsim_pipeline_setup.load_yaml_config('settings.yaml')

# Files
INPUT_FILES = settings['input_files']
SHAPE_BOUNDARY_FILE = INPUT_FILES['shape_boundary_file']
EXPANDED_HOUSEHOLDS_FILES: list = INPUT_FILES['expanded_households_files']
MiD_HH_FILE = INPUT_FILES['mid_hh_file']
MiD_PERSONS_FILE = INPUT_FILES['mid_persons_file']
MiD_TRIPS_FILE = INPUT_FILES['mid_trips_file']
BUILDINGS_IN_LOWEST_GEOGRAPHY_WITH_WEIGHTS_FILE = INPUT_FILES['buildings_in_lowest_geography_with_weights_file']

OUTPUT_FILES = settings['output_files']
POPULATION_ANALYSIS_OUTPUT_FILE = OUTPUT_FILES['population_analysis_output_file']

# Columns
ID_COLUMNS = settings['id_columns']
HH_COLUMNS = settings['hh_columns']
P_COLUMNS = settings['person_columns']
L_COLUMNS = settings['leg_columns']
GEO_COLUMNS = settings['geography_columns']

# Household-related columns
HOUSEHOLD_MID_ID_COL = ID_COLUMNS['household_mid_id_column']
HOUSEHOLD_POPSIM_ID_COL = ID_COLUMNS['household_popsim_id_column']

# Person-related columns
PERSON_ID_COL = ID_COLUMNS['person_id_column']
PERSON_AGE_COL = P_COLUMNS['person_age']
CAR_AVAIL_COL = P_COLUMNS['car_avail']
HAS_LICENSE_COL = P_COLUMNS['has_license']

# Leg-related columns
LEG_ID_COL = ID_COLUMNS['leg_id_column']
LEG_NON_UNIQUE_ID_COL = ID_COLUMNS['leg_non_unique_id_column']
LEG_ACTIVITY_COL = L_COLUMNS['leg_target_activity']
LEG_MAIN_MODE_COL = L_COLUMNS['leg_main_mode']
LEG_START_TIME_COL = L_COLUMNS['leg_start_time']
LEG_END_TIME_COL = L_COLUMNS['leg_end_time']
LEG_DURATION_MINUTES_COL = L_COLUMNS['leg_duration_minutes']
LEG_DISTANCE_COL = L_COLUMNS['leg_distance']

# Geography-related columns
TT_MATRIX_CELL_ID_COL = ID_COLUMNS['tt_matrix_cell_id_column']

# Value maps
VALUE_MAPS = settings['value_maps']

ACTIVITY_WORK = VALUE_MAPS['activities']['work']
ACTIVITY_BUSINESS = VALUE_MAPS['activities']['business']
ACTIVITY_EDUCATION = VALUE_MAPS['activities']['education']
ACTIVITY_SHOPPING = VALUE_MAPS['activities']['shopping']
ACTIVITY_ERRANDS = VALUE_MAPS['activities']['errands']
ACTIVITY_PICK_UP_DROP_OFF = VALUE_MAPS['activities']['pick_up_drop_off']
ACTIVITY_LEISURE = VALUE_MAPS['activities']['leisure']
ACTIVITY_HOME = VALUE_MAPS['activities']['home']
ACTIVITY_RETURN_JOURNEY = VALUE_MAPS['activities']['return_journey']
ACTIVITY_OTHER = VALUE_MAPS['activities']['other']
ACTIVITY_EARLY_EDUCATION = VALUE_MAPS['activities']['early_education']
ACTIVITY_DAYCARE = VALUE_MAPS['activities']['daycare']
ACTIVITY_ACCOMPANY_ADULT = VALUE_MAPS['activities']['accompany_adult']
ACTIVITY_SPORTS = VALUE_MAPS['activities']['sports']
ACTIVITY_MEETUP = VALUE_MAPS['activities']['meetup']
ACTIVITY_LESSONS = VALUE_MAPS['activities']['lessons']
ACTIVITY_UNSPECIFIED = VALUE_MAPS['activities']['unspecified']

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

AVERAGE_ACTIVITY_TIMES_MINUTES = settings['average_activity_times_minutes']

# Misc
LOWEST_LEVEL_GEOGRAPHY = settings['lowest_level_geography']
DUN_DUN_DUUUN: bool = settings['misc']['dun_dun_duuun']  # Play sound on error
BASE_DATE = "2020-01-01"  # Arbitrary date for converting times to datetime objects

SAMPLE_SIZE = settings['sample_size']

N_CLOSEST_CELLS = settings['n_closest_cells']