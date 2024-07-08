import os.path
from datetime import timedelta

import geopandas as gpd
import matsim.writers
import numpy as np
import pandas as pd

from utils.data_frame_processor import DataFrameProcessor
from utils import helpers as h, settings as s, pipeline_setup
from utils.stats_tracker import stats_tracker
from utils.logger import logging

logger = logging.getLogger(__name__)


class PopulationProcessor(DataFrameProcessor):
    def distribute_by_weights(self, weights_df: pd.DataFrame, cell_id_col: str, cut_missing_ids: bool = False):
        result = h.distribute_by_weights(self.df, weights_df, cell_id_col, cut_missing_ids)
        self.df = self.df.merge(result[[s.UNIQUE_HH_ID_COL, 'home_loc']], on=s.UNIQUE_HH_ID_COL, how='left')

    def write_plans_to_matsim_xml(self):
        """
        Write to MATSim xml.gz directly from the dataframe.
        The design of this method decides which data from the population frame is written and which is not.
        All trips are assumed to start at home.
        All trips should end at home. If not, we warn the user but use the given activity.
        """
        logger.info("Writing plans to MATSim xml...")
        logger.info("All trips are assumed to start at home.")

        self.df.reset_index(drop=True, inplace=True)

        output_file = os.path.join(pipeline_setup.OUTPUT_DIR, "population.xml")
        with open(output_file, 'wb+') as f_write:
            writer = matsim.writers.PopulationWriter(f_write)

            writer.start_population()  # attributes={"coordinateReferenceSystem": "UTM-32N"}

            for _, group in self.df.groupby([s.UNIQUE_P_ID_COL]):
                writer.start_person(group[s.UNIQUE_P_ID_COL].iloc[0])
                writer.start_plan(selected=True)

                # Add home activity
                writer.add_activity(
                    type="home",
                    x=group['home_loc'].iloc[0].x, y=group['home_loc'].iloc[0].y,
                    end_time=abs(h.seconds_from_datetime(group[s.LEG_START_TIME_COL].iloc[0])))
                # One row in the df contains the leg and the following activity
                for idx, row in group.iterrows():
                    writer.add_leg(mode=row['mode_translated_string'])
                    if not pd.isna(row[s.ACT_DUR_SECONDS_COL]):
                        # Create an own activity type for each duration (for correct matsim scoring)
                        # Rounding must fit the matsim config
                        max_dur: int = round(row[s.ACT_DUR_SECONDS_COL] / 600) * 600
                        writer.add_activity(
                            type=f"{row['activity_translated_string']}_{max_dur}",
                            x=row[s.COORD_TO_COL].x, y=row[s.COORD_TO_COL].y,
                            # The writer expects seconds. Also, we mean max_dur here, but the writer doesn't have that yet.
                            start_time=max_dur)
                    elif idx == group.index[-1]:
                        # No time for the last activity
                        writer.add_activity(
                            type=row['activity_translated_string'],
                            x=row[s.COORD_TO_COL].x, y=row[s.COORD_TO_COL].y)
                    else:
                        # If for some reason we have no time
                        max_dur: int = round(3600 / 600) * 600
                        writer.add_activity(
                            type=f"{row['activity_translated_string']}_{max_dur}",
                            x=row[s.COORD_TO_COL].x, y=row[s.COORD_TO_COL].y,
                            # The writer expects seconds. Also, we mean max_dur here, but the writer doesn't have that yet.
                            start_time=max_dur)
                writer.end_plan()
                writer.end_person()

            writer.end_population()
        logger.info(f"Wrote plans to MATSim xml: {output_file}")
        return output_file

    def write_households_to_matsim_xml(self):
        logger.info("Writing households to MATSim xml...")
        output_file = os.path.join(pipeline_setup.OUTPUT_DIR, "households.xml")
        with open(output_file, 'wb+') as f_write:
            households_writer = matsim.writers.HouseholdsWriter(f_write)
            households_writer.start_households()

            for _, hh in self.df.groupby([s.UNIQUE_HH_ID_COL]):
                household_id = hh[s.UNIQUE_HH_ID_COL].iloc[0]
                person_ids = hh[s.UNIQUE_P_ID_COL].unique().tolist()
                vehicle_ids = hh[s.LIST_OF_CARS_COL].iloc[0]

                households_writer.start_household(household_id)
                households_writer.add_members(person_ids)
                # if vehicle_ids:
                #     households_writer.add_vehicles(vehicle_ids)
                households_writer.end_household()

            households_writer.end_households()

    def write_facilities_to_matsim_xml(self, facilities_df: pd.DataFrame):
        with open(os.path.join(pipeline_setup.OUTPUT_DIR, "facilities.xml"), 'wb+') as f_write:
            facilities_writer = matsim.writers.FacilitiesWriter(f_write)
            facilities_writer.start_facilities()

            for row in facilities_df.itertuples():
                # Using itertuples() with getattr() is much faster than iterrows(), even if it's a bit uglier
                facility_id = getattr(row, s.FACILITY_ID_COL)
                x = getattr(row, s.FACILITY_X_COL)
                y = getattr(row, s.FACILITY_Y_COL)
                activities = list(getattr(row, s.FACILITY_ACTIVITIES_COL))

                facilities_writer.start_facility(facility_id, x, y)

                for activity in activities:
                    facilities_writer.add_activity(activity)

                facilities_writer.end_facility()

            facilities_writer.end_facilities()

    def write_vehicles_to_matsim_xml(self):
        logger.info("Writing vehicles to MATSim xml...")
        output_file = os.path.join(pipeline_setup.OUTPUT_DIR, "vehicles.xml")
        with open(output_file, 'wb+') as f_write:
            vehicle_writer = matsim.writers.VehiclesWriter(f_write)
            vehicle_writer.start_vehicle_definitions()

            vehicle_id = "car"
            length = 7.5
            width = 1.0
            pce = 1.0
            network_mode = "car"

            vehicle_writer.add_vehicle_type(vehicle_id=vehicle_id, length=length, width=width, pce=pce,
                                            network_mode=network_mode)

            for _, hh in self.df.groupby([s.UNIQUE_HH_ID_COL]):
                vehicle_ids: list = hh[s.LIST_OF_CARS_COL].iloc[0]
                if vehicle_ids:
                    for vehicle_id in vehicle_ids:
                        vehicle_writer.add_vehicle(vehicle_id=vehicle_id, vehicle_type="car")

            vehicle_writer.end_vehicle_definitions()

        logger.info(f"Wrote vehicles to MATSim xml: {output_file}")

    def change_last_leg_activity_to_home(self) -> None:
        """
        Change the target activity of the last leg to home. Alternative to add_return_home_leg().
        Assumes LEG_ID is ascending in order of legs (which it is in MiD and should be in other datasets).
        """
        logger.info("Changing last leg activity to home...")
        self.df = self.df.sort_values(by=[s.UNIQUE_LEG_ID_COL])

        is_last_leg = self.df[s.PERSON_ID_COL].ne(self.df[s.PERSON_ID_COL].shift(-1))

        number_of_rows_to_change = len(self.df[is_last_leg & (self.df[s.ACT_TO_INTERNAL_COL] != s.ACT_HOME)])

        self.df.loc[is_last_leg, s.ACT_TO_INTERNAL_COL] = s.ACT_HOME
        self.df.loc[is_last_leg, 'activity_translated_string'] = "home"
        # We also need to remove markers for main or mirroring main activities, because home is never main
        self.df.loc[is_last_leg, s.IS_MAIN_ACTIVITY_COL] = 0
        self.df.loc[is_last_leg, s.MIRRORS_MAIN_ACTIVITY_COL] = 0
        # This means there might be some persons with no main activity. This is not a problem for the current model.
        logger.info(f"Changed last leg activity to home for {number_of_rows_to_change} of {len(self.df)} rows.")

    def assign_random_location(self):
        """
        Assign a random location to each activity.
        :return:
        """
        gdf = gpd.read_file(s.SHAPE_BOUNDARY_FILE)
        polygon = h.find_outer_boundary(gdf)
        self.df['random_point'] = self.df.apply(lambda row: h.random_point_in_polygon(polygon), axis=1)





    # def translate_modes(self):
    #     """
    #     Translate the modes from the MiD codes to the MATSim strings.
    #     Recommended to do this just before writing to MATSim xml.
    #     :return:
    #     """
    #     logger.info(f"Translating modes...")
    #     defined_modes = [s.MODE_CAR, s.MODE_PT, s.MODE_RIDE, s.MODE_BIKE, s.MODE_WALK, s.MODE_UNDEFINED]
    #     count_non_matching = (~self.df[s.LEG_MAIN_MODE_COL].isin(defined_modes)).sum()
    #     count_nan = self.df[s.LEG_MAIN_MODE_COL].isna().sum()
    #     count_non_leg = self.df[s.LEG_ID_COL].isna().sum()
    #     if count_non_matching > 0:
    #         logger.warning(f"{count_non_matching} rows have a mode that is not in the defined modes."
    #                        f"{count_nan} rows have no mode."
    #                        f"{count_non_leg} rows have no leg.")
    #     mode_translation = {
    #         s.MODE_CAR: "car",
    #         s.MODE_PT: "pt",
    #         s.MODE_RIDE: "ride",
    #         s.MODE_BIKE: "bike",
    #         s.MODE_WALK: "walk",
    #     }
    #     self.df['mode_translated_string'] = self.df[s.LEG_MAIN_MODE_COL].map(mode_translation)
    #     logger.info(f"Translated modes.")

    # def translate_activities(self):
    #     """
    #     Translate the activities from the MiD codes to the MATSim strings.
    #     Recommended to do this just before writing to MATSim xml.
    #     :return:
    #     """
    #     logger.info(f"Translating activities...")
    #     activity_translation = {
    #         s.ACT_WORK: "work",
    #         s.ACT_BUSINESS: "work",
    #         s.ACT_EDUCATION: "education",
    #         s.ACT_SHOPPING: "shopping",
    #         s.ACT_ERRANDS: "leisure",
    #         s.ACT_PICK_UP_DROP_OFF: "other",
    #         s.ACT_LEISURE: "leisure",
    #         s.ACT_HOME: "home",
    #         s.ACT_RETURN_JOURNEY: "other",
    #         s.ACT_OTHER: "other",
    #         s.ACT_EARLY_EDUCATION: "education",
    #         s.ACT_DAYCARE: "education",
    #         s.ACT_ACCOMPANY_ADULT: "other",
    #         s.ACT_SPORTS: "leisure",
    #         s.ACT_MEETUP: "leisure",
    #         s.ACT_LESSONS: "leisure",
    #         s.ACT_UNSPECIFIED: "other",
    #     }
    #     self.df['activity_translated_string'] = self.df[s.TO_ACTIVITY_WITH_CONNECTED_COL].map(activity_translation)
    #     logger.info(f"Translated activities.")

    def vary_times_by_household(self, hh_id_col, time_cols, max_shift_minutes=15):
        """
        Varies times in the DataFrame by the same random amount (±max_shift_minutes) for each household.

        :param hh_id_col: String, the column name for the unique hh identifier.
        :param time_cols: List of strings, the names of the columns containing time data.
        :param max_shift_minutes: Integer, the maximum number of minutes for the time shift.
        :return: pandas DataFrame with varied times.
        """

        logger.info("Varying times by household...")

        def apply_time_shift(group):
            time_shift = timedelta(minutes=np.random.randint(-max_shift_minutes, max_shift_minutes + 1))

            # Apply this time shift to all time columns
            for col in time_cols:
                group[col] = group[col].apply(lambda x: x + time_shift if pd.notnull(x) else x)
            return group

        self.df = self.df.groupby(hh_id_col).apply(apply_time_shift)
        logger.info("Times varied by person.")

    def downsample_population(self, sample_percentage):
        """
        Downsample the population to a given sample percentage size of the original population.
        Recommended to sample households (e.g. at the point when only households are loaded),
        not persons, to keep the household structure intact.
        """
        logger.info("Downsampling population...")
        self.df = self.df.sample(frac=sample_percentage)
        stats_tracker.log("downsampled_population", sample_percentage)
        logger.info(f"Downsampled population to {sample_percentage * 100}% of the original population.")


