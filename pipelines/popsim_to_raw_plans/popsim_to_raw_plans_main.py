import os

from pipelines.common import rules
from pipelines.popsim_to_raw_plans.population_frame_processor import PopulationFrameProcessor
from utils import matsim_pipeline_setup
from utils.logger import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    output_dir = matsim_pipeline_setup.create_output_directory()
else:
    output_dir = matsim_pipeline_setup.OUTPUT_DIR

# Set working dir
os.chdir(matsim_pipeline_setup.PROJECT_ROOT)

# Load settings
settings = matsim_pipeline_setup.load_yaml_config('settings.yaml')

EXPANDED_HOUSEHOLDS_FILES: list = settings['expanded_households_files']
MiD_HH_FILE = settings['mid_hh_file']
MiD_PERSONS_FILE = settings['mid_persons_file']
MiD_TRIPS_FILE = settings['mid_trips_file']

HOUSEHOLD_ID_COLUMN = settings['household_id_column']
PERSON_ID_COLUMN = settings['person_id_column']
LEG_ID_COLUMN = settings['leg_id_column']
LOWEST_LEVEL_GEOGRAPHY = settings['lowest_level_geography']

POPULATION_ANALYSIS_OUTPUT_FILE = settings['population_analysis_output_file']
write_to_file = True if POPULATION_ANALYSIS_OUTPUT_FILE else False

# Load data from PopSim, concat different PopSim results if necessary
# Lowest level of geography must be named the same in all input files, if there are multiple
population = PopulationFrameProcessor()
for csv_path in EXPANDED_HOUSEHOLDS_FILES:
    population.load_df_from_csv(csv_path, "concat")
population.id_column = HOUSEHOLD_ID_COLUMN

# Add/edit household-specific rule-based attributes
if rules.rule_required_hh_columns:
    population.add_csv_data_on_id(MiD_HH_FILE, rules.rule_required_hh_columns, id_column=HOUSEHOLD_ID_COLUMN,
                                  drop_duplicates_from_source=True, delete_later=True)
# rq = rules.rule_required_columns
# rules = [rule1, rule2, rule3, rule4, rule5]
# population.safe_apply_rules(rules)

# Distribute buildings to households (if PopSim assigned them to a larger geography)
# buildings_in_lowest_geography_with_weights_df =
# population.distribute_by_weights(buildings_in_lowest_geography_with_weights_df, LOWEST_LEVEL_GEOGRAPHY)


# Add people to households
population.add_csv_data_on_id(MiD_PERSONS_FILE, [PERSON_ID_COLUMN], id_column=HOUSEHOLD_ID_COLUMN,
                              drop_duplicates_from_source=False)

# Add/edit person-specific rule-based attributes
if rules.rule_required_person_columns:
    population.add_csv_data_on_id(MiD_PERSONS_FILE, rules.rule_required_person_columns, id_column=PERSON_ID_COLUMN,
                                  drop_duplicates_from_source=True, delete_later=True)

apply_me = [rules.example_rule1, rules.example_rule2, rules.example_rule3, rules.example_rule4, rules.example_rule5]
population.apply_row_wise_rules(apply_me)

apply_me = [rules.example_rule6, rules.example_rule7, rules.example_rule8, rules.example_rule9, rules.example_rule10]
population.apply_group_wise_rules(apply_me, groupby_column=HOUSEHOLD_ID_COLUMN)

# Add MiD-trips to people
population.add_csv_data_on_id(MiD_TRIPS_FILE, [LEG_ID_COLUMN], id_column=PERSON_ID_COLUMN,
                              drop_duplicates_from_source=False)

# Add/edit trip-specific rule-based attributes
if rules.rule_required_leg_columns:
    population.add_csv_data_on_id(MiD_TRIPS_FILE, rules.rule_required_leg_columns, id_column=LEG_ID_COLUMN,
                                  drop_duplicates_from_source=True, delete_later=True)

apply_me = [rules.rulebased_main_mode, rules.example_rule2, rules.example_rule3, rules.example_rule4, rules.example_rule5, rules.]
population.apply_row_wise_rules(apply_me)

apply_me = [rules.example_rule6, rules.example_rule7, rules.example_rule8, rules.example_rule9, rules.example_rule10]
population.apply_group_wise_rules(apply_me, groupby_column=HOUSEHOLD_ID_COLUMN)

# Remove cols that were used by rules, to keep the df clean
population.remove_columns_marked_for_later_deletion()

#

# Write plans to MATSim XML format
rules = [rules.write_plans_to_matsim_xml]
population.apply_group_wise_rules(rules, groupby_column=HOUSEHOLD_ID_COLUMN)

# Write dataframe to csv file if desired
if write_to_file:
    population.df.to_csv(POPULATION_ANALYSIS_OUTPUT_FILE, index=False)
