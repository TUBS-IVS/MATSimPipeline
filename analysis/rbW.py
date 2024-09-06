# Get all households with rwB for analysis
import pandas as pd

from utils import helpers as h, settings as s
from synthesis.data_prep.mid_data_enhancer import MiDDataEnhancer
from utils.logger import logging
from utils.stats_tracker import stats_tracker
import os
from utils.pipeline_setup import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

logger = logging.getLogger(__name__)


# All of this stuff is just merging the files
population = MiDDataEnhancer()
population.load_df_from_csv(h.get_files(s.MiD_HH_FOLDER), test_col=s.HOUSEHOLD_MID_ID_COL)

logger.info(f"Population df after adding HH attributes: \n{population.df.head()}")
population.check_for_merge_suffixes()

population.df = h.generate_unique_household_id(population.df)

# Add persons to households (increases the number of rows)
population.add_csv_data_on_id(h.get_files(s.MiD_PERSONS_FOLDER), [s.PERSON_ID_COL],
                              id_column=s.HOUSEHOLD_MID_ID_COL,
                              drop_duplicates_from_source=False)
logger.info(f"Population df after adding persons: \n{population.df.head()}")
population.check_for_merge_suffixes()

# Add person attributes from MiD
population.add_csv_data_on_id(h.get_files(s.MiD_PERSONS_FOLDER), id_column=s.PERSON_ID_COL,
                              drop_duplicates_from_source=True, delete_later=True)
logger.info(f"Population df after adding P attributes: \n{population.df.head()}")
population.check_for_merge_suffixes()

population.df = h.generate_unique_person_id(population.df)

# Add MiD-trips to people (increases the number of rows)
population.add_csv_data_on_id(h.get_files(s.MiD_TRIPS_FOLDER), [s.LEG_ID_COL], id_column=s.PERSON_ID_COL,
                              drop_duplicates_from_source=False)
logger.info(f"Population df after adding trips: \n{population.df.head()}")
population.check_for_merge_suffixes()

# Add trip attributes from MiD
population.add_csv_data_on_id(h.get_files(s.MiD_TRIPS_FOLDER), id_column=s.LEG_ID_COL,
                              drop_duplicates_from_source=True)
logger.info(f"Population df after adding L attributes: \n{population.df.head()}")
population.check_for_merge_suffixes()

rbw_filter = population.df[s.LEG_IS_RBW_COL] == 1

logger.info(f"Number of rows total: {population.df.shape[0]}")
logger.info(f"Number of rows marked as rbW: {population.df[rbw_filter].shape[0]}")

hhs_with_rbw = population.df[rbw_filter][s.HOUSEHOLD_MID_ID_COL].unique()
logger.info(f"Number of households with rbW: {len(hhs_with_rbw)}")

filtered_df = population.df[population.df[s.HOUSEHOLD_MID_ID_COL].isin(hhs_with_rbw)]
filtered_df = filtered_df[[s.HOUSEHOLD_MID_ID_COL, s.PERSON_ID_COL, s.LEG_NON_UNIQUE_ID_COL, s.LEG_IS_RBW_COL,
                           s.LEG_START_TIME_COL, s.LEG_END_TIME_COL, s.LEG_DURATION_MINUTES_COL, s.LEG_DISTANCE_KM_COL,
                           s.ACT_MID_COL, s.MODE_MID_COL]]
# Print the filtered DataFrame
print(filtered_df)

filtered_df.to_csv('analysis/analysis_data/households_with_rbW.csv', index=False)
