
"""
Post-processor is part of writing because 1) there is not much to do 2) it is very related to the writing step
"""

from datetime import timedelta
import utils.column_names as s
from utils.data_frame_processor import DataFrameProcessor

class PopulationPostProcessor(DataFrameProcessor):
    def __init__(self, stats_tracker):
        super().__init__(stats_tracker)

    def change_last_leg_activity_to_home(self) -> None:
        """
        Change the target activity of the last leg to home. Alternative to add_return_home_leg().
        Assumes LEG_ID is ascending in order of legs (which it is in MiD and should be in other datasets).
        """
        self.logger.info("Changing last leg activity to home...")
        self.df = self.df.sort_values(by=[s.UNIQUE_LEG_ID_COL])

        is_last_leg = self.df[s.PERSON_MID_ID_COL].ne(self.df[s.PERSON_MID_ID_COL].shift(-1))

        number_of_rows_to_change = len(self.df[is_last_leg & (self.df[s.ACT_TO_INTERNAL_COL] != s.ACT_HOME)])

        self.df.loc[is_last_leg, s.ACT_TO_INTERNAL_COL] = s.ACT_HOME
        self.df.loc[is_last_leg, 'activity_translated_string'] = "home"
        # We also need to remove markers for main or mirroring main activities, because home is never main
        self.df.loc[is_last_leg, s.IS_MAIN_ACTIVITY_COL] = 0
        self.df.loc[is_last_leg, s.MIRRORS_MAIN_ACTIVITY_COL] = 0
        # This means there might be some persons with no main activity. This is not a problem for the current model.
        self.logger.info(f"Changed last leg activity to home for {number_of_rows_to_change} of {len(self.df)} rows.")

    def vary_times_by_household(self, hh_id_col, time_cols, max_shift_minutes=15):
        """
        Varies times in the DataFrame by the same random amount (±max_shift_minutes) for each household.

        :param hh_id_col: String, the column name for the unique hh identifier.
        :param time_cols: List of strings, the names of the columns containing time data.
        :param max_shift_minutes: Integer, the maximum number of minutes for the time shift.
        :return: pandas DataFrame with varied times.
        """

        self.logger.info("Varying times by household...")

        def apply_time_shift(group):
            time_shift = timedelta(minutes=np.random.randint(-max_shift_minutes, max_shift_minutes + 1))

            # Apply this time shift to all time columns
            for col in time_cols:
                group[col] = group[col].apply(lambda x: x + time_shift if pd.notnull(x) else x)
            return group

        self.df = self.df.groupby(hh_id_col).apply(apply_time_shift)
        self.logger.info("Times varied by person.")
