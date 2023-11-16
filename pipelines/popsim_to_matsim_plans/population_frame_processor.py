import os.path
import random

import geopandas as gpd
import matsim.writers
import pandas as pd
from shapely.geometry import Point

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
    HOUSEHOLD_ID_COL = settings['id_columns']['household_mid_id_column']
    PERSON_ID_COL = settings['id_columns']['person_id_column']
    LEG_NON_UNIQUE_ID_COL = settings['id_columns']['leg_non_unique_id_column']
    LEG_ID_COL = settings['id_columns']['leg_id_column']
    LEG_ACTIVITY_COL = settings['leg_columns']['leg_target_activity']
    LEG_START_TIME_COL = settings['leg_columns']['leg_start_time']
    LEG_END_TIME_COL = settings['leg_columns']['leg_end_time']
    MODE_COL = settings['leg_columns']['leg_main_mode']
    PERSON_AGE_COL = settings['person_columns']['person_age']
    SHAPE_BOUNDARY_FILE = settings['shape_boundary_file']
    #  Value_maps
    ACTIVITY_HOME = (settings['value_maps']['activities']['home'])
    MODE_CAR = settings['value_maps']['modes']['car']
    MODE_RIDE = settings['value_maps']['modes']['ride']

    def distribute_by_weights(self, weights_df, external_id_column, cut_missing_ids=False):
        """
        Distribute data points from `weights_df` across the population dataframe based on weights (e.g. assign buildings to households).

        The function modifies the internal population dataframe by appending the point IDs from the weights dataframe
        based on their weights and the count of each ID in the population dataframe.

        Args:
            weights_df (pd.DataFrame): DataFrame containing the ID of the geography, point IDs, and their weights.
            Must contain all geography IDs in the population dataframe. Is allowed to contain more (they will be skipped).
            external_id_column (str): The column name of the ID in the weights dataframe (e.g. 'BLOCK_NR').
            cut_missing_ids (bool): If True, IDs in the population dataframe that are not in the weights dataframe are cut from the population dataframe.
        """
        logger.info("Starting distribution by weights...")

        if not self.df[external_id_column].isin(weights_df[external_id_column]).all():
            if cut_missing_ids:
                logger.warning(f"Not all geography IDs in the population dataframe are in the weights dataframe. "
                               f"Cutting missing IDs: {set(self.df[external_id_column]) - set(weights_df[external_id_column])}")
                self.df = self.df[self.df[external_id_column].isin(weights_df[external_id_column])]
            else:
                raise ValueError(f"Not all geography IDs in the population dataframe are in the weights dataframe. "
                                 f"Missing IDs: {set(self.df[external_id_column]) - set(weights_df[external_id_column])}")

        # Count of each ID in population_df
        id_counts = self.df[external_id_column].value_counts().reset_index()
        id_counts.columns = [external_id_column, '_processing_count']
        logger.info(f"Computed ID counts for {len(id_counts)} unique IDs.")

        # Merge with weights_df
        weights_df = pd.merge(weights_df, id_counts, on=external_id_column, how='left')

        def distribute_rows(group):
            total_count = group['_processing_count'].iloc[0]
            if total_count == 0 or pd.isna(total_count):
                logger.debug(f"Geography ID {group[external_id_column].iloc[0]} is not in the given dataframe, "
                             f"likely because no person/activity etc. exists there. Skipping distribution for this ID.")
                return []
            # Compute distribution
            group['_processing_repeat_count'] = (group['ewzahl'] / group['ewzahl'].sum()) * total_count
            group['_processing_int_part'] = group['_processing_repeat_count'].astype(int)
            group['_processing_frac_part'] = group['_processing_repeat_count'] - group['_processing_int_part']

            # Distribute remainder
            remainder = total_count - group['_processing_int_part'].sum()
            assert remainder >= 0 and remainder % 1 == 0, f"Remainder is {remainder}, should be a positive integer."
            remainder = int(remainder)
            top_indices = group['_processing_frac_part'].nlargest(remainder).index
            group.loc[top_indices, '_processing_int_part'] += 1

            # Expand rows based on int_part
            expanded = []
            for _, row in group.iterrows():
                expanded.extend([row.to_dict()] * int(row['_processing_int_part']))
            return expanded

        expanded_rows = []
        for _, group in weights_df.groupby(external_id_column):
            expanded_rows.extend(distribute_rows(group))

        expanded_weights_df = pd.DataFrame(expanded_rows).drop(
            columns=['_processing_count', '_processing_repeat_count', '_processing_int_part', '_processing_frac_part'])
        logger.info(f"Generated expanded weights DataFrame with {len(expanded_weights_df)} rows.")
        if len(expanded_weights_df) != self.df.shape[0]:
            raise ValueError(f"Expanded weights DataFrame has {len(expanded_weights_df)} rows, "
                             f"but the population DataFrame has {self.df.shape[0]} rows.")

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
        Write to MATSim xml.gz directly from the dataframe.
        The design of this method decides which data from the population frame is written and which is not.
        """
        logger.info("Writing plans to MATSim xml.gz...")

        output_file = os.path.join(matsim_pipeline_setup.OUTPUT_DIR, "population.xml.gz")
        with open(output_file, 'wb+') as f_write:
            writer = matsim.writers.PopulationWriter(f_write)

            writer.start_population(attributes={"coordinateReferenceSystem": "UTM-32N"})  # TODO: verify CRS everywhere

            for _, group in self.df.groupby(['unique_person_id']):
                writer.start_person(group['unique_person_id'].iloc[0])
                writer.start_plan(selected=True)
                # One row in the df contains the leg and the following activity
                # All trips are assumed to start at home
                if group[self.LEG_ACTIVITY_COL].iloc[0] != self.ACTIVITY_HOME:
                    logger.info(
                        f"First activity of person {group[self.PERSON_ID_COL].iloc[0]} is not home. Assuming home anyway.")
                # All trips should end at home. If not, we warn the user but use the given activity.
                if group[self.LEG_ACTIVITY_COL].iloc[-1] != self.ACTIVITY_HOME:
                    logger.warning(f"Last activity of person {group[self.PERSON_ID_COL].iloc[0]} is not home.")
                writer.add_activity(
                    type="home",
                    x=group['home_loc'].iloc[0].x, y=group['home_loc'].iloc[0].y,
                    end_time=(group[self.LEG_START_TIME_COL].iloc[0]))
                for idx, row in group.iterrows():
                    writer.add_leg(mode=row[self.MODE_COL])
                    writer.add_activity(
                        type=row[self.LEG_ACTIVITY_COL],
                        x=row["random_point"].x, y=row["random_point"].y,
                        # The writer expects seconds. Also, we mean max_dur here, but the writer doesn't have that yet.
                        end_time=row["activity_duration_seconds"])

                writer.end_plan()
                writer.end_person()

            writer.end_population()

    def change_last_leg_activity_to_home(self) -> None:
        """
        Change the target activity of the last leg to home. Alternative to add_return_home_leg().
        Assumes LEG_ID is ascending in order of legs (which it is in MiD and should be in other datasets).
        """
        logger.info("Changing last leg activity to home...")
        self.df = self.df.sort_values(by=[self.HOUSEHOLD_ID_COL, self.PERSON_ID_COL, self.LEG_ID_COL])

        is_last_leg = self.df['person_id'].ne(self.df['person_id'].shift(-1))

        number_of_rows_to_change = len(self.df[is_last_leg & (self.df[self.LEG_ACTIVITY_COL] != self.ACTIVITY_HOME)])

        self.df.loc[is_last_leg, self.LEG_ACTIVITY_COL] = self.ACTIVITY_HOME
        logger.info(f"Changed last leg activity to home for {number_of_rows_to_change} of {len(self.df)} rows.")

    def adjust_mode_based_on_age(self):
        """
        Change the mode of transportation from car to ride if age < 17.
        """
        logger.info("Adjusting mode based on age...")
        conditions = (self.df[self.MODE_COL] == self.MODE_CAR) & (self.df[self.PERSON_AGE_COL] < 17)
        self.df.loc[conditions, self.MODE_COL] = self.MODE_RIDE
        logger.info(f"Adjusted mode based on age for {conditions.sum()} of {len(self.df)} rows.")

    def calculate_activity_time(self):
        """
        Calculate the time between the end of one leg and the start of the next leg in minutes and seconds.
        :return:
        """
        self.df.sort_values(by=[self.HOUSEHOLD_ID_COL, self.PERSON_ID_COL, self.LEG_NON_UNIQUE_ID_COL], inplace=True,
                            ignore_index=True)

        # Group by person and calculate the time difference within each group
        self.df['activity_time_seconds'] = self.df.groupby(self.PERSON_ID_COL)[self.LEG_START_TIME_COL].shift(-1) - self.df[
            self.LEG_END_TIME_COL]

        self.df['activity_time_seconds'] = self.df['activity_time_seconds'].dt.total_seconds()
        self.df['activity_time_seconds'] = pd.to_numeric(self.df['activity_time_seconds'], downcast='integer', errors='coerce')

        # Set the activity time of the last leg to None
        is_last_leg = self.df[self.PERSON_ID_COL] != self.df[self.PERSON_ID_COL].shift(-1)
        self.df.loc[is_last_leg, 'activity_time_in_seconds'] = None

    def assign_random_location(self):
        """
        Assign a random location to each activity.
        :return:
        """
        polygon = self.find_outer_boundary()
        self.df['random_point'] = self.df.apply(lambda row: random_point_in_polygon(polygon), axis=1)

        def random_point_in_polygon(polygon):
            if not polygon.is_valid or polygon.is_empty:
                raise ValueError("Invalid polygon")

            min_x, min_y, max_x, max_y = polygon.bounds

            while True:
                random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
                if polygon.contains(random_point):
                    return random_point

    def find_outer_boundary(self, method='convex_hull'):
        # Read the shapefile
        gdf = gpd.read_file(self.SHAPE_BOUNDARY_FILE)

        # Combine all geometries in the GeoDataFrame
        combined = gdf.geometry.unary_union

        # Calculate the convex hull or envelope
        if method == 'convex_hull':
            outer_boundary = combined.convex_hull
        elif method == 'envelope':
            outer_boundary = combined.envelope
        else:
            raise ValueError("Method must be 'convex_hull' or 'envelope'")

        return outer_boundary
