import os

from pipelines.common import rules
from pipelines.popsim_to_matsim_plans.population_frame_processor import PopulationFrameProcessor
from utils import matsim_pipeline_setup
from utils.logger import logging


logger = logging.getLogger(__name__)

# Set working dir
os.chdir(matsim_pipeline_setup.PROJECT_ROOT)

# Load settings
settings = matsim_pipeline_setup.load_yaml_config('settings.yaml')

EXPANDED_HOUSEHOLDS_FILES: list = settings['expanded_households_files']
MiD_HH_FILE = settings['mid_hh_file']
MiD_PERSONS_FILE = settings['mid_persons_file']
MiD_TRIPS_FILE = settings['mid_trips_file']

HH_COLUMNS = settings['hh_columns']
P_COLUMNS = settings['person_columns']
L_COLUMNS = settings['leg_columns']

HOUSEHOLD_ID_COLUMN = HH_COLUMNS['household_id_column']
PERSON_ID_COLUMN = P_COLUMNS['person_id_column']
LEG_ID_COLUMN = L_COLUMNS['leg_id_column']

LOWEST_LEVEL_GEOGRAPHY = settings['lowest_level_geography']

POPULATION_ANALYSIS_OUTPUT_FILE = settings['population_analysis_output_file']
write_to_file = True if POPULATION_ANALYSIS_OUTPUT_FILE else False

def popsim_to_matsim_plans_main():
    """
    Main function for the popsim_to_matsim_plans pipeline.
    :return:
    Notes:
        - The ids of households, persons and trips must be unique within the population sample (e.g. MiD)
        (MiD: H_ID, HP_ID, and a previously added HPW_ID for legs)

    """
    # Load data from PopSim, concat different PopSim results if necessary
    # Lowest level of geography must be named the same in all input files, if there are multiple
    population = PopulationFrameProcessor()
    for csv_path in EXPANDED_HOUSEHOLDS_FILES:
        population.load_df_from_csv(csv_path, "concat")
    population.id_column = HOUSEHOLD_ID_COLUMN

    # Add household attributes from MiD
    if HH_COLUMNS:
        population.add_csv_data_on_id(MiD_HH_FILE, HH_COLUMNS, id_column=HOUSEHOLD_ID_COLUMN,
                                      drop_duplicates_from_source=True, delete_later=True)

    # Add/edit household-specific rule-based attributes
    apply_me = [rules.unique_household_id]
    population.apply_row_wise_rules(apply_me)

    # Distribute buildings to households (if PopSim assigned them to a larger geography)
    # buildings_in_lowest_geography_with_weights_df =
    # population.distribute_by_weights(buildings_in_lowest_geography_with_weights_df, LOWEST_LEVEL_GEOGRAPHY)


    # Add people to households (increases the number of rows)
    population.add_csv_data_on_id(MiD_PERSONS_FILE, [PERSON_ID_COLUMN], id_column=HOUSEHOLD_ID_COLUMN,
                                  drop_duplicates_from_source=False)

    # Add person attributes from MiD
    if P_COLUMNS:
        population.add_csv_data_on_id(MiD_PERSONS_FILE, P_COLUMNS, id_column=PERSON_ID_COLUMN,
                                      drop_duplicates_from_source=True, delete_later=True)

    # Add/edit person-specific rule-based attributes
    apply_me = [rules.unique_person_id]
    population.apply_row_wise_rules(apply_me)

    apply_me = [rules.]
    population.apply_group_wise_rules(apply_me, groupby_column="unique_household_id")

    # Add MiD-trips to people (increases the number of rows)
    population.add_csv_data_on_id(MiD_TRIPS_FILE, [LEG_ID_COLUMN], id_column=PERSON_ID_COLUMN,
                                  drop_duplicates_from_source=False)

    # Add trip attributes from MiD
    if L_COLUMNS:
        population.add_csv_data_on_id(MiD_TRIPS_FILE, L_COLUMNS, id_column=LEG_ID_COLUMN,
                                      drop_duplicates_from_source=True, delete_later=True)

    # Add/edit trip-specific rule-based attributes
    apply_me = [rules.rulebased_main_mode, rules.example_rule2, rules.example_rule3, rules.example_rule4, rules.example_rule5, rules.]
    population.apply_row_wise_rules(apply_me)

    apply_me = [rules.example_rule6, rules.example_rule7, rules.example_rule8, rules.example_rule9, rules.example_rule10]
    population.apply_group_wise_rules(apply_me, groupby_column="unique_person_id")

    # Remove cols that were used by rules, to keep the df clean
    population.remove_columns_marked_for_later_deletion()

    #

    # Write plans to MATSim XML format
    population.write_plans_to_matsim_xml()

    # Write dataframe to csv file if desired
    if write_to_file:
        population.df.to_csv(POPULATION_ANALYSIS_OUTPUT_FILE, index=False)

#
# if __name__ == '__main__':
#     output_dir = matsim_pipeline_setup.create_output_directory()
#     popsim_to_raw_plans_main()
# else:
#     output_dir = matsim_pipeline_setup.OUTPUT_DIR