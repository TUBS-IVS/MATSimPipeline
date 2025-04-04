"""
MiD and custom column names and values
"""

# # --------------------------------------------------------------------------------------------------------------
# TODO: Remove/move
# Global Settings
SAMPLE_SIZE = 0.25
N_CLOSEST_CELLS = 10
DEFAULT_SLACK_FACTOR = 1.5
SIGMOID_BETA = -0.15
SIGMOID_DELTA_T = 1200

# # Input Files
# EXPANDED_HOUSEHOLDS_FILES = [
#     "data/synthetic_households_city.csv",
#     "data/synthetic_households_region.csv"
# ]
# ENHANCED_MID_FOLDER = "data/mid/enhanced"
# MID_HH_FOLDER = "data/mid/households"
# MID_PERSONS_FOLDER = "data/mid/persons"
# MID_TRIPS_FOLDER = "data/mid/trips"
# BUILDINGS_IN_LOWEST_GEOGRAPHY_WITH_WEIGHTS_FILE = "data/houses_with_weights.csv"
# CAPA_CELLS_CSV_PATH = "data/region_hanover_potentials.csv"
# CAPA_CELLS_SHP_PATH = "data/shapes/RH_useful__zone.SHP"
# REGION_WITHOUT_CITY_GPKG_FILE = "data/shapes/RegionOhneStadtGitter100m.gpkg"
# SHAPE_BOUNDARY_FILE = "data/shapes/region_hanover.shp"
# SLACK_FACTORS_FILE = "data/Slack_Factors.csv"
# 
# # Output Files
# POPULATION_ANALYSIS_OUTPUT_FILE = "full_population_frame.csv"
# MATSIM_PLANS_FILE = "population.xml"
# STATS_FILE = "stats.txt"
# ENHANCED_MID_FILE = "enhanced_mid.csv"

# Geography
GEOGRAPHY_COLUMNS = ["WELT", "STAAT", "STADTTLNR", "BAUBLOCKNR"]
LOWEST_LEVEL_GEOGRAPHY = "BAUBLOCKNR"

# --------------------------------------------------------------------------------------------------------------

# ID Columns
HOUSEHOLD_MID_ID_COL = "H_ID"
HOUSEHOLD_POPSIM_ID_COL = "household_id"
PERSON_MID_ID_COL = "HP_ID"
LEG_NUMBER_COL = "W_ID" # MiD ID, 1,2,3... for each leg
LEG_ID_COL = "HPW_ID" # MiD Person ID and leg number concatenated
TT_MATRIX_CELL_ID_COL = "cell_id"

# Unique ids added after population expansion
UNIQUE_LEG_ID_COL = "unique_leg_id"
UNIQUE_HH_ID_COL = "unique_household_id"
UNIQUE_P_ID_COL = "unique_person_id"

# Household Columns
H_CAR_IN_HH_COL = "auto"
H_NUMBER_OF_CARS_COL = "H_ANZAUTO"
H_REGION_TYPE_COL = "RegioStaR7"

# Person Columns
PERSON_AGE_COL = "HP_ALTER"
CAR_AVAIL_COL = "P_VAUTO"
HAS_LICENSE_COL = "P_FS_PKW"
NUMBER_OF_LEGS_COL = "anzwege1"

# Leg Columns
LEG_TARGET_ACTIVITY_COL = "W_ZWECK"
LEG_MAIN_MODE_COL = "hvm_imp"
LEG_START_TIME_COL = "W_SZ"
LEG_END_TIME_COL = "W_AZ"
LEG_DURATION_MINUTES_COL = "wegmin_imp1"
LEG_DURATION_SECONDS_COL = "leg_duration_seconds"
LEG_DISTANCE_KM_COL = "wegkm_imp"
LEG_DISTANCE_METERS_COL = "leg_distance_meters"
FIRST_LEG_STARTS_AT_HOME_COL = "W_SO1"
LEG_IS_RBW_COL = "W_RBW"
HOME_TO_MAIN_METERS_COL = "home_to_main_meters"
HOME_TO_MAIN_SECONDS_COL = "home_to_main_seconds"
HOME_TO_MAIN_TIME_IS_ESTIMATED_COL = "home_to_main_time_is_estimated"
HOME_TO_MAIN_DIST_IS_ESTIMATED_COL = "home_to_main_dist_is_estimated"

# Enhancement Columns
RANDOM_LOCATION_COL = "random_location"
ACT_DUR_SECONDS_COL = "activity_duration_seconds"
NUMBER_OF_LEGS_INCL_IMPUTED_COL = "number_of_legs_incl_imputed"
IS_IMPUTED_TIME_COL = "imputed_time"
IS_IMPUTED_LEG_COL = "imputed_leg"
LIST_OF_CARS_COL = "car_list"

FACILITY_ID_COL = "facility_id"
FACILITY_CENTROID_COL = "facility_centroid"
FACILITY_X_COL = "facility_x"
FACILITY_Y_COL = "facility_y"
FACILITY_ACTIVITIES_COL = "facility_activities"
FACILITY_NAME_COL = "facility_name"
FACILITY_POTENTIAL_COL = "facility_potential"

IS_MAIN_ACTIVITY_COL = "is_main_activity"
MIRRORS_MAIN_ACTIVITY_COL = "mirrors_main_activity"

CONNECTED_LEGS_COL = "connected_legs"
TO_ACTIVITY_WITH_CONNECTED_COL = "to_activity_with_connected"
IS_PROTAGONIST_COL = "is_protagonist"

HH_HAS_CONNECTIONS_COL = "hh_has_connections"
P_HAS_CONNECTIONS_COL = "p_has_connections"
NUM_CONNECTED_LEGS_COL = "num_connected_legs"
CELL_FROM_COL = "cell_from"
CELL_TO_COL = "cell_to"
#COORD_FROM_COL = "from_location"
COORD_TO_COL = "to_location"
FROM_X_COL = "from_x"
FROM_Y_COL = "from_y"
TO_X_COL = "to_x"
TO_Y_COL = "to_y"
HOME_CELL_COL = "home_cell"
HOME_LOC_COL = "home_location"
HOME_X_COL = "home_x"
HOME_Y_COL = "home_y"
MAIN_MODE_TO_MAIN_ACT_TIMEBASED_COL = "main_mode_to_main_act_timebased"
MAIN_MODE_TO_MAIN_ACT_DISTBASED_COL = "main_mode_to_main_act_distbased"

# Values
MODE_CAR = "car"
MODE_PT = "pt"
MODE_RIDE = "ride"
MODE_BIKE = "bike"
MODE_WALK = "walk"
MODE_UNDEFINED = "undefined"

ACT_WORK = "work"
ACT_BUSINESS = "business"
ACT_EDUCATION = "education"
ACT_SHOPPING = "shopping"
ACT_ERRANDS = "errands"
ACT_PICK_UP_DROP_OFF = "pick_up_drop_off"
ACT_LEISURE = "leisure"
ACT_HOME = "home"
ACT_RETURN_JOURNEY = "return_journey"
ACT_OTHER = "other"
ACT_EARLY_EDUCATION = "early_education"
ACT_DAYCARE = "daycare"
ACT_ACCOMPANY_ADULT = "accompany_adult"
ACT_SPORTS = "sports"
ACT_MEETUP = "meetup"
ACT_LESSONS = "lessons"
ACT_UNSPECIFIED = "unspecified"

MODE_INTERNAL_COL = "mode_internal"
MODE_MID_COL = "hvm_imp"
MODE_MATSIM_COL = "mode_matsim"

ACT_TO_INTERNAL_COL = "activity_to_internal"
ACT_FROM_INTERNAL_COL = "activity_from_internal"
ACT_MID_COL = "W_ZWECK"
ACT_MATSIM_COL = "activity_matsim"

CAR_NEVER = 3
CAR_IN_HH_NO = 0
CAR_IN_HH_YES = 1

LICENSE_YES = 1
LICENSE_NO = 2
LICENSE_UNKNOWN = 9
ADULT_OVER_16_PROXY = 206
PERSON_UNDER_16 = 403

FIRST_LEG_STARTS_AT_HOME = 1