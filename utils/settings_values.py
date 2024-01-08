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

TT_MATRIX_CAR_FILES: list = INPUT_FILES['tt_matrix_car_files']
TT_MATRIX_PT_FILES: list = INPUT_FILES['tt_matrix_pt_files']
TT_MATRIX_WALK_FILE = INPUT_FILES['tt_matrix_walk_file']
TT_MATRIX_BIKE_FILE = INPUT_FILES['tt_matrix_bike_file']

SLACK_FACTORS_FILE = INPUT_FILES['slack_factors_file']

OUTPUT_FILES = settings['output_files']
POPULATION_ANALYSIS_OUTPUT_FILE = OUTPUT_FILES['population_analysis_output_file']

# Columns
ID_COLUMNS: dict = settings['id_columns']
HH_COLUMNS: dict = settings['hh_columns']
P_COLUMNS: dict = settings['person_columns']
L_COLUMNS: dict = settings['leg_columns']
GEO_COLUMNS: dict = settings['geography_columns']

# Household-related columns
HOUSEHOLD_MID_ID_COL = ID_COLUMNS['household_mid_id_column']
HOUSEHOLD_POPSIM_ID_COL = ID_COLUMNS['household_popsim_id_column']
H_CAR_IN_HH_COL = HH_COLUMNS['car_in_hh_column']
H_REGION_TYPE_COL = HH_COLUMNS['region_type_column']
H_NUMBER_OF_CARS_COL = HH_COLUMNS['number_of_cars_column']

# Person-related columns
PERSON_ID_COL = ID_COLUMNS['person_id_column']
PERSON_AGE_COL = P_COLUMNS['person_age']
CAR_AVAIL_COL = P_COLUMNS['car_avail']
HAS_LICENSE_COL = P_COLUMNS['has_license']
NUMBER_OF_LEGS_COL = P_COLUMNS['number_of_legs']

# Leg-related columns
LEG_ID_COL = ID_COLUMNS['leg_id_column']
LEG_NON_UNIQUE_ID_COL = ID_COLUMNS['leg_non_unique_id_column']
LEG_TO_ACTIVITY_COL = L_COLUMNS['leg_target_activity']
LEG_MAIN_MODE_COL = L_COLUMNS['leg_main_mode']
LEG_START_TIME_COL = L_COLUMNS['leg_start_time']
LEG_END_TIME_COL = L_COLUMNS['leg_end_time']
LEG_DURATION_MINUTES_COL = L_COLUMNS['leg_duration_minutes']
LEG_DISTANCE_COL = L_COLUMNS['leg_distance']
FIRST_LEG_STARTS_AT_HOME_COL = L_COLUMNS['first_leg_starts_at_home']

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

CAR_IN_HH_NO = VALUE_MAPS['car_in_hh']['no']
CAR_IN_HH_YES = VALUE_MAPS['car_in_hh']['yes']

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

FIRST_LEG_STARTS_AT_HOME = VALUE_MAPS['misc']['first_leg_starts_at_home']

# Misc
LOWEST_LEVEL_GEOGRAPHY = settings['lowest_level_geography']
DUN_DUN_DUUUN: bool = settings['misc']['dun_dun_duuun']  # Play sound on error
BASE_DATE = "2020-01-01"  # Arbitrary date for converting times to datetime objects

SAMPLE_SIZE = settings['sample_size']

N_CLOSEST_CELLS = settings['n_closest_cells']
DEFAULT_SLACK_FACTOR = settings['default_slack_factor']


# Columns that are created by the enhancement pipeline
ENHANCEMENT_COLUMNS = settings['enhancement_columns']
RANDOM_LOCATION_COL = ENHANCEMENT_COLUMNS['random_location']
ACT_DUR_SECONDS_COL = ENHANCEMENT_COLUMNS['activity_duration_seconds']
NUMBER_OF_LEGS_INCL_IMPUTED_COL = ENHANCEMENT_COLUMNS['number_of_legs_incl_imputed']
IMPUTED_TIME_COL = ENHANCEMENT_COLUMNS['imputed_time']
IMPUTED_LEG_COL = ENHANCEMENT_COLUMNS['imputed_leg']
LIST_OF_CARS_COL = ENHANCEMENT_COLUMNS['list_of_cars']
LEG_FROM_ACTIVITY_COL = ENHANCEMENT_COLUMNS['leg_from_activity']

# Column names that are set at runtime
PROCESSING_COLUMNS = settings['processing_columns']

UNIQUE_LEG_ID_COL = PROCESSING_COLUMNS['unique_leg_id']
UNIQUE_HH_ID_COL = PROCESSING_COLUMNS['unique_household_id']
UNIQUE_P_ID_COL = PROCESSING_COLUMNS['unique_person_id']

# These names aren't exposed in the yaml because they'll probably never change and just clutter the file
FACILITY_ID_COL = "facility_id"
FACILITY_X_COL = "facility_x"
FACILITY_Y_COL = "facility_y"
FACILITY_ACTIVITIES_COL = "facility_activities"

TO_ACTIVITY_WITH_CONNECTED_COL = "to_activity_with_connected"  # Leg_to_activity with activity overwritten by connected_legs

IS_PROTAGONIST_COL = "is_protagonist"
IS_MAIN_ACTIVITY_COL = "is_main_activity"

CONNECTED_LEGS_COL = "connected_legs"

HOME_TO_MAIN_DIST_COL = "home_to_main_distance"  # Same distance type as leg distances
HOME_TO_MAIN_TIME_COL = "home_to_main_time"  # Same time type as leg times (here:minutes)

MAIN_MODE_TO_MAIN_ACT_TIMEBASED_COL = "main_mode_to_main_act_timebased"
MAIN_MODE_TO_MAIN_ACT_DISTBASED_COL = "main_mode_to_main_act_distbased"
HOME_TO_MAIN_TIME_ESTIMATED_COL = "home_to_main_time_is_estimated"

SIGMOID_BETA = settings['sigmoid_beta']
SIGMOID_DELTA_T = settings['sigmoid_delta_t']

MIRRORS_MAIN_ACTIVITY_COL = "mirrors_main_activity"

HH_HAS_CONNECTIONS_COL = "hh_has_connections"
P_HAS_CONNECTIONS_COL = "p_has_connections"
NUM_CONNECTED_LEGS_COL = "num_connected_legs"


