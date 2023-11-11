import matsim.writers
import pandas as pd

from pipelines.common.data_frame_processor import DataFrameProcessor
from utils import matsim_pipeline_setup
from utils.logger import logging

logger = logging.getLogger(__name__)


class PopulationFrameProcessor(DataFrameProcessor):
    """
    A class to process population dataframes.
    Contains methods that are specific to population dataframes and need full access to the dataframe and/or
    are too complex to be implemented as rules.
    """

    def __init__(self, df: pd.DataFrame = None, id_column: str = None):
        super().__init__(df, id_column)

    settings = matsim_pipeline_setup.load_yaml_config('settings.yaml')  # shared across all instances, set at module import
    #  Column names
    PERSON_ID_COL = settings['id_columns']['person_id_column']
    LEG_ID_COL = settings['id_columns']['leg_id_column']
    LEG_ACTIVITY_COL = settings['leg_columns']['leg_target_activity']
    MODE_COL = settings['leg_columns']['leg_main_mode']
    PERSON_AGE_COL = settings['person_columns']['person_age']
    #  Value_maps
    ACTIVITY_HOME = settings['value_maps']['activities']['home']
    MODE_CAR = settings['value_maps']['mode']['car']
    MODE_RIDE = settings['value_maps']['mode']['ride']

    def distribute_by_weights(self, weights_df, external_id_column):
        """
        Distribute data points from `weights_df` across the population dataframe based on weights (e.g. assign buildings to households).

        The function modifies the internal dataframe by appending the point IDs from the weights dataframe
        based on their weights and the count of each ID in the population dataframe.

        Args:
            weights_df (pd.DataFrame): DataFrame containing the ID, point IDs, and their weights.
            external_id_column (str): The column name of the ID in the weights dataframe (e.g. 'BLOCK_NR').

        Returns:
            None. The internal dataframe is modified in place.
        """
        logger.info("Starting distribution by weights...")

        # Count of each ID in population_df
        id_counts = self.df[external_id_column].value_counts().reset_index()
        id_counts.columns = [external_id_column, '_processing_count']
        logger.info(f"Computed ID counts for {len(id_counts)} unique IDs.")

        # Merge with weights_df
        weights_df = pd.merge(weights_df, id_counts, on=external_id_column, how='left')

        def distribute_rows(group):
            total_count = group['_processing_count'].iloc[0]

            # Compute distribution
            group['_processing_repeat_count'] = (group['weight'] / group['weight'].sum()) * total_count
            group['_processing_int_part'] = group['_processing_repeat_count'].astype(int)
            group['_processing_frac_part'] = group['_processing_repeat_count'] - group['_processing_int_part']

            # Distribute remainder
            remainder = total_count - group['_processing_int_part'].sum()
            top_indices = group['_processing_frac_part'].nlargest(remainder).index
            group.loc[top_indices, '_processing_int_part'] += 1

            # Expand rows based on int_part
            expanded = []
            for _, row in group.iterrows():
                expanded.extend([row.to_dict()] * int(row['_processing_int_part']))
            return expanded

        expanded_rows = []
        groups = weights_df.groupby(external_id_column)
        for _, group in groups:
            expanded_rows.extend(distribute_rows(group))
        logger.info("Finished row distribution based on weights.")

        expanded_weights_df = pd.DataFrame(expanded_rows).drop(
            columns=['_processing_count', '_processing_repeat_count', '_processing_int_part', '_processing_frac_part'])
        logger.info(f"Generated expanded weights DataFrame with {len(expanded_weights_df)} rows.")
        if len(expanded_weights_df) != self.df.shape[0]:
            logger.error(
                f"Expanded weights DataFrame has {len(expanded_weights_df)} rows, but the population DataFrame has {self.df.shape[0]} rows.")

        # Add a sequence column to both dataframes to prevent cartesian product on merge
        self.df['_processing_seq'] = self.df.groupby(external_id_column).cumcount()
        expanded_weights_df['_processing_seq'] = expanded_weights_df.groupby(external_id_column).cumcount()

        # Merge using the ID column and the sequence
        self.df = pd.merge(self.df, expanded_weights_df, on=[external_id_column, '_processing_seq'],
                           how='left').drop(columns='_processing_seq')

        logger.info("Completed distribution by weights.")

    # Dummy function ---------------------------------------------------------------
    #     def generate_raw_plans(self):
    #         """
    #         Generates raw plans from the population frame.
    #         """
    #         logger.info("Generating raw plans...")
    #         # Create a copy of the population frame
    #         raw_plans = self.population_frame.copy()
    #
    #         # Rename columns
    #         raw_plans.rename(columns={"personID": "person_id", "householdID": "household_id"}, inplace=True)
    #
    #         # Add attributes
    #         raw_plans["selected"] = 1
    #         raw_plans["score"] = 1
    #         raw_plans["plan_type"] = "initial"
    #         raw_plans["plan_mode"] = raw_plans.apply(self.get_plan_mode, axis=1)
    #         raw_plans["plan_score"] = 1
    #         raw_plans["plan_selected"] = 1
    #
    #         # Reorder columns
    #         raw_plans = raw_plans[["person_id", "household_id", "selected", "score", "plan_type", "plan_mode", "plan_score",
    #                                "plan_selected"]]
    #
    #         # Write to CSV
    #         raw_plans.to_csv("raw_plans.csv", index=False)
    #         logger.info("Raw plans generated.")

    # test -------------------------------------------------------------------------

    # # Rule Functions:
    # def double_value(row):
    #     return row['Value'] * 2, []
    #
    #
    # def missing_column(row):
    #     return [], ['MissingColumn']
    #
    #
    # def mean_by_category(group_df):
    #     # Ensure that group_df is indeed a DataFrame
    #     if isinstance(group_df, pd.DataFrame):
    #         mean_val = group_df['Value'].mean()
    #         return (mean_val, [])
    #     else:
    #         # Just for debugging
    #         logger.error(f"Unexpected input type: {type(group_df)}")
    #         return (None, [])
    #
    #
    # # Test Data:
    # data = {
    #     'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
    #     'Value': [10, 15, 20, 25, 30, 35]
    # }
    # df = pd.DataFrame(data)
    #
    # # Using the PopulationFrameProcessor:
    # processor = PopulationFrameProcessor(df, 'Category')
    #
    # processor.safe_apply_rules([missing_column, double_value])
    # print(processor.df)
    #
    # processor.safe_apply_rules([missing_column], groupby_column='Category')
    # print(processor.df)
    #
    # # Applying group-wise rule with grouping:
    # processor.safe_apply_rules([mean_by_category], groupby_column='Category')
    # print(processor.df)
    #
    def write_plans_to_matsim_xml(self):
        """
        Write the population frame to MATSim XML format.
        """
        logger.info("Writing plans to MATSim XML format...")

        # Create a copy of the population frame
        raw_plans = self.df.copy()

        # Rename columns
        raw_plans.rename(columns={"personID": "person_id", "householdID": "household_id"}, inplace=True)

        # Add attributes
        raw_plans["selected"] = 1
        raw_plans["score"] = 1
        raw_plans["plan_type"] = "initial"
        raw_plans["plan_mode"] = raw_plans.apply(self.get_plan_mode, axis=1)
        raw_plans["plan_score"] = 1
        raw_plans["plan_selected"] = 1

        # Reorder columns
        raw_plans = raw_plans[["person_id", "household_id", "selected", "score", "plan_type", "plan_mode", "plan_score",
                               "plan_selected"]]

        # Write to CSV
        raw_plans.to_csv("raw_plans.csv", index=False)
        logger.info("Raw plans generated.")

        with open("plans.xml", 'wb+') as f_write:
            writer = matsim.writers.PopulationWriter(f_write)

            writer.start_population()
            writer.start_person("person_id_123")
            writer.start_plan(selected=True)

        for _, group in self.df.groupby(self.PERSON_ID_COLUMN):
            writer.add_activity(type='home', x=0.0, y=0.0, end_time=8 * 3600)
            writer.add_leg(mode='walk')
            writer.add_activity(type='work', x=10.0, y=0.0, end_time=18 * 3600)
            writer.add_leg(mode='pt')
            writer.add_activity(type='home', x=0.0, y=0.0)

            writer.end_plan()
            writer.end_person()

            writer.end_population()

        grouped = []
        writer.start_population()
        for (person_id, plan_id), group in grouped:
            writer.start_person(person_id)
            writer.start_plan(selected=True)

            # Iterate over each activity/leg in the plan
            for index, row in group.iterrows():
                # Add activity
                writer.add_activity(
                    type=row['ACTIVITY_TYPE'],
                    x=row['X'],
                    y=row['Y'],
                    end_time=row['END_TIME']
                )
                # If there's a leg mode, add a leg
                if pd.notnull(row['MODE']):
                    writer.add_leg(mode=row['MODE'])

            writer.end_plan()
            writer.end_person()

        writer.end_population()

    def change_last_leg_activity_to_home(self) -> None:
        """
        Change the target activity of the last leg to home. Alternative to add_return_home_leg().
        Assumes LEG_ID is ascending in order of legs (which it is in MiD and should be in other datasets).
        """
        self.df = self.df.sort_values(by=[self.PERSON_ID_COL, self.LEG_ID_COL])

        is_last_leg = self.df['person_id'].ne(self.df['person_id'].shift(-1))

        self.df.loc[is_last_leg, self.LEG_ACTIVITY_COL] = self.ACTIVITY_HOME

    def adjust_mode_based_on_age(self):
        """
        Change the mode of transportation from car to ride if age < 17.
        """
        conditions = (self.df[self.MODE_COL] == self.MODE_CAR) & (self.df[self.PERSON_AGE_COL] < 17)
        self.df.loc[conditions, self.MODE_COL] = self.MODE_RIDE
