import pandas as pd

from utils.logger import logging

logger = logging.getLogger(__name__)


def merge_dataframes_on_column(primary_df, secondary_df, merge_column, columns_to_add,
                               drop_duplicates_from_secondary=True):
    """
    Merges data from secondary_df into primary_df based on the specified merge column.

    Parameters:
    - primary_df: DataFrame, the primary dataframe.
    - secondary_df: DataFrame, the dataframe from which selected columns will be added to primary_df.
    - merge_column: str, the name of the column on which the merge is based.
    - columns_to_add: list, the columns in secondary_df that you want to add to primary_df.
    - drop_duplicates: bool, whether to drop duplicates based on merge_column in secondary_df.

    Returns:
    - Merged DataFrame.
    """

    if drop_duplicates_from_secondary:
        # Identify duplicates
        duplicate_mask = secondary_df.duplicated(subset=merge_column, keep='first')

        if duplicate_mask.any():
            duplicate_ids = secondary_df.loc[duplicate_mask, merge_column].tolist()
            logger.warning(f"Duplicate {merge_column} values found and dropped in secondary_df: {duplicate_ids}")

        secondary_df = secondary_df.drop_duplicates(subset=merge_column, keep='first')

    # Ensure merge_column is in columns_to_add
    if merge_column not in columns_to_add:
        columns_to_add.append(merge_column)

    # Merge the dataframes on the specified merge_column
    merged_df = pd.merge(primary_df, secondary_df[columns_to_add], on=merge_column, how='left')

    return merged_df


# Example of usage:
# merged_df = merge_dataframes_on_column(df1, df2, 'ID', ['column_name_1', 'column_name_2'], log_duplicates=True)

GLOBAL_ID_COLUMN = "my_global_id"  # Example global setting for the identifier column's name
MID_CSV_PATH = "path_to_MiD_data.csv"  # Path to the CSV containing the MiD data


def add_data_from_MiD(df, columns_to_add):
    """
    Fetches specified columns from MiD and adds them to the DataFrame.

    Parameters:
    - df: DataFrame, the input dataframe.
    - columns_to_add: list of str, the columns to fetch from MiD and add to df.

    Returns:
    - DataFrame with the added columns.
    """

    # Validation
    if not isinstance(columns_to_add, list):
        logger.error("columns_to_add should be a list. Returning the original DataFrame.")
        return df

    if GLOBAL_ID_COLUMN not in df.columns:
        logger.error(
            f"The DataFrame is missing the global ID column: {GLOBAL_ID_COLUMN}. Returning the original DataFrame.")
        return df

    existing_columns = set(df.columns)
    for col in columns_to_add:
        if col in existing_columns:
            logger.info(f"Column {col} already exists in the DataFrame. Overwriting.")
            df = df.drop(columns=[col])

    # Load MiD data from CSV
    MiD_df = pd.read_csv(MID_CSV_PATH, usecols=[GLOBAL_ID_COLUMN] + columns_to_add)

    # Ensure GLOBAL_ID_COLUMN is not part of columns_to_add for clarity
    columns_to_add = [col for col in columns_to_add if col != GLOBAL_ID_COLUMN]

    # Merge the data using the provided merge function
    augmented_df = merge_dataframes_on_column(df, MiD_df, GLOBAL_ID_COLUMN, columns_to_add)

    return augmented_df
