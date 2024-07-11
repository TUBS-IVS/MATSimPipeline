import os
import pandas as pd
import numpy as np
from utils import settings as s
from utils import helpers as h
from utils import pipeline_setup


def create_test_data(root, test_data_dir, hh_dir, person_dir, leg_dir):
    # Paths to main data files
    hh_path = os.path.join(root, hh_dir)
    person_path = os.path.join(root, person_dir)
    leg_path = os.path.join(root, leg_dir)

    # Paths to test data files
    test_hh_path = os.path.join(root, test_data_dir, hh_dir, 'households.csv')
    test_person_path = os.path.join(root, test_data_dir, person_dir, 'persons.csv')
    test_leg_path = os.path.join(root, test_data_dir, leg_dir, 'trips.csv')

    # Load data
    hh_df = h.read_csv(h.get_files(hh_path), test_col=s.HOUSEHOLD_MID_ID_COL)
    person_df = h.read_csv(h.get_files(person_path), test_col=s.HOUSEHOLD_MID_ID_COL)
    leg_df = h.read_csv(h.get_files(leg_path), test_col=s.HOUSEHOLD_MID_ID_COL)

    # Identify "difficult" households
    # 1. Households with more than 10 legs
    households_with_many_legs = leg_df.groupby(s.HOUSEHOLD_MID_ID_COL).size()
    difficult_hhs_many_legs = households_with_many_legs[households_with_many_legs > 10].index

    # 2. Households with missing values in start or end times
    difficult_hhs_missing_times = leg_df[(leg_df["wegmin"].astype(str) == "9994") | (leg_df["wegmin"].astype(str) == "9995")][
        s.HOUSEHOLD_MID_ID_COL].unique()

    # 3. Households with persons having only one leg
    legs_per_person = leg_df.groupby(s.PERSON_ID_COL).size()
    persons_with_one_leg = legs_per_person[legs_per_person == 1].index
    difficult_hhs_one_leg_person = person_df[person_df[s.PERSON_ID_COL].isin(persons_with_one_leg)][
        s.HOUSEHOLD_MID_ID_COL].unique()

    # 4. More to be added here

    # Combine all difficult households
    difficult_hhs = pd.Index(difficult_hhs_many_legs).union(difficult_hhs_missing_times).union(
        difficult_hhs_one_leg_person)

    # Sample households
    difficult_hh_sample = np.random.choice(difficult_hhs, 1000, replace=False)
    remaining_hhs = set(hh_df[s.HOUSEHOLD_MID_ID_COL]) - set(difficult_hhs)
    random_hh_sample = np.random.choice(list(remaining_hhs), 1000, replace=False)

    sampled_hhs = np.concatenate([difficult_hh_sample, random_hh_sample])

    # Filter data based on sampled households
    hh_sample_df = hh_df[hh_df[s.HOUSEHOLD_MID_ID_COL].isin(sampled_hhs)]
    person_sample_df = person_df[person_df[s.HOUSEHOLD_MID_ID_COL].isin(sampled_hhs)]
    leg_sample_df = leg_df[leg_df[s.HOUSEHOLD_MID_ID_COL].isin(sampled_hhs)]

    # Save test data
    hh_sample_df.to_csv(test_hh_path, index=False)
    # person_sample_df.to_csv(test_person_path, index=False)  # Don't sample those, to test if sampling works in the actual function
    # leg_sample_df.to_csv(test_leg_path, index=False)

    print(f"Test data created in {test_data_dir}.")


root = pipeline_setup.PROJECT_ROOT
create_test_data(
    root=root,
    test_data_dir='tests/',
    hh_dir=s.MiD_HH_FOLDER,
    person_dir=s.MiD_PERSONS_FOLDER,
    leg_dir=s.MiD_TRIPS_FOLDER
)
