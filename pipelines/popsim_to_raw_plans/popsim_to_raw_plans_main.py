import os

from pipelines.popsim_to_raw_plans.population_frame_processor import PopulationFrameProcessor
from utils import matsim_pipeline_setup
from utils.logger import logging

logger = logging.getLogger(__name__)

write_to_file = False

if __name__ == '__main__':
    output_dir = matsim_pipeline_setup.create_output_directory()
    write_to_file = True

# set working dir
os.chdir(matsim_pipeline_setup.PROJECT_ROOT)

settings = matsim_pipeline_setup.load_yaml_config('settings.yaml')

EXPANDED_HOUSEHOLDS_FILES: list = settings['expanded_households_files']
MiD_PERSONS_FILE = settings['mid_persons_file']
MiD_TRIPS_FILE = settings['mid_trips_file']

HOUSEHOLD_ID_COLUMN = settings['household_id_column']
PERSON_ID_COLUMN = settings['person_id_column']
LEG_ID_COLUMN = settings['leg_id_column']
LOWEST_LEVEL_GEOGRAPHY = settings['lowest_level_geography']

# Load data from PopSim, concat different PopSim results if necessary
# Lowest level of geography must be named the same in all input files, if there are multiple

population = PopulationFrameProcessor()
for csv_path in EXPANDED_HOUSEHOLDS_FILES:
    population.load_df_from_csv(csv_path, "concat")
population.id_column = HOUSEHOLD_ID_COLUMN

# Add household-specific rule-based attributes
# rq = rules.rule_required_columns
# rules = [rule1, rule2, rule3, rule4, rule5]
# population.safe_apply_rules(rules)

# Distribute buildings to households (if PopSim assigned them to a larger geography)
# buildings_in_lowest_geography_with_weights_df =
# population.distribute_by_weights(buildings_in_lowest_geography_with_weights_df, LOWEST_LEVEL_GEOGRAPHY)

# Add people to households
population.add_csv_data_on_id(MiD_PERSONS_FILE, [PERSON_ID_COLUMN], id_column=HOUSEHOLD_ID_COLUMN,
                              drop_duplicates_from_source=False)

# Add person-specific rule-based attributes



# Add MiD trips to people


# Add trip-specific rule-based attributes


# Write dataframe to csv file (if run as a single step)
# if write_to_file:
#     df.to_csv("output.csv", index=False)
