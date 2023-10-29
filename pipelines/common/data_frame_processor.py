import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataFrameProcessor:
    def __init__(self, df, id_column):
        self.df = df
        self.id_column = id_column

    @property
    def id_column(self):
        return self._id_column

    @id_column.setter
    def id_column(self, value):
        if value not in self.df.columns:
            logger.warning(f"ID column {value} not found in DataFrame.")
        self._id_column = value

    def add_data_on_id(self, source_df, columns_to_add, overwrite_existing=False, drop_duplicates_from_source=True):
        """
        Adds specified columns from a source DataFrame to the current DataFrame based on an ID.

        Parameters:
        - source_df: DataFrame, the source dataframe.
        - columns_to_add: list of str, the columns to fetch and add to df.
        - overwrite_existing: bool, whether to overwrite columns that already exist. Default is True.
        - drop_duplicates_from_source: bool, whether to drop duplicates based on id_column in source_df. Default is True.
        """

        # Validate that columns_to_add is a list
        if not isinstance(columns_to_add, list):
            logger.error("columns_to_add should be a list. Not updating the DataFrame.")
            return

        # Validation for ID columns in both dataframes
        if self.id_column not in self.df.columns or self.id_column not in source_df.columns:
            logger.error(
                f"Either the primary or source DataFrame is missing the ID column: {self.id_column}. Not updating the DataFrame.")
            return

        # Handle duplicates
        if drop_duplicates_from_source:
            duplicate_mask = source_df.duplicated(subset=self.id_column, keep='first')
            if duplicate_mask.any():
                duplicate_ids = source_df.loc[duplicate_mask, self.id_column].tolist()
                logger.warning(f"Duplicate {self.id_column} values found and dropped in source_df: {duplicate_ids}")
                source_df = source_df.drop_duplicates(subset=self.id_column, keep='first')

        # Handle columns
        for col in columns_to_add:
            if col in self.df.columns:
                if overwrite_existing:
                    logger.info(f"Column {col} already exists in the DataFrame. Overwriting.")
                    self.df.drop(columns=[col], inplace=True)
                else:
                    logger.info(f"Column {col} already exists in the DataFrame. Skipping.")
                    columns_to_add.remove(col)

        # Ensure id_column is in columns_to_add for merging
        if self.id_column not in columns_to_add:
            columns_to_add.append(self.id_column)

        # Merge the dataframes on the specified id_column
        self.df = pd.merge(self.df, source_df[columns_to_add], on=self.id_column, how='left')

    def add_csv_data(self, csv_path, columns_to_add, overwrite_existing=False, id_column=None):
        """
        Load specified columns from a given CSV file and add them to the primary DataFrame.

        Parameters:
        - csv_path: str, path to the CSV file.
        - columns_to_add: list of str, the columns to fetch from the CSV.
        - overwrite_existing: bool, whether to overwrite columns that already exist. Default is False.
        - id_column: str, the column to use as the ID. Default is the id_column specified in the constructor.
        """
        if id_column is None:
            id_column = self.id_column
        try:
            source_df = pd.read_csv(csv_path, usecols=[id_column] + columns_to_add)
            self.add_data_on_id(source_df, columns_to_add, overwrite_existing)
        except Exception as e:
            logger.error(f"Failed to load and add CSV data from {csv_path}: {e}")

    # Note: The implementation for merge_dataframes_on_column is assumed to exist elsewhere.

    def safe_apply_rules(self, csv_path, rules):
        """
        Applies a set of custom rules to the DataFrame stored in this instance. Will only add columns, never alter existing columns.
        If a rule references missing columns, those columns are fetched from a secondary data source (e.g. MiD) and the rules are reapplied.

        Parameters:
        - rules (list of functions): A list of rule functions. Each rule function must return a tuple of (result, missing_columns).
        - csv_path (str): Path to the CSV file containing the secondary data source (e.g. MiD).

        Notes:
        - Could run at different places in the pipeline and might have different rule sets.
        """
        all_missing_columns = set()

        # First pass: identify all missing columns
        for rule_func in rules:
            logger.debug(f"Collecting missing columns for rule '{rule_func.__name__}'")
            try:
                _, missing_columns_list = zip(*self.df.apply(rule_func, axis=1))
            except Exception as e:
                logger.error(f"Failed to apply rule '{rule_func.__name__}': {e}")
                continue
            rule_missing_columns = set().union(*missing_columns_list)
            all_missing_columns.update(rule_missing_columns)

            if rule_missing_columns:
                logger.info(
                    f"Rule '{rule_func.__name__}' identified missing columns: {', '.join(rule_missing_columns)}")

        # Fetch all missing columns at once
        if all_missing_columns:
            logger.info(f"Fetching missing columns: {', '.join(all_missing_columns)}")
            self.add_csv_data(csv_path, list(all_missing_columns), overwrite_existing=False)

        # Second pass: apply the rules now that all columns are present
        for rule_func in rules:
            logger.debug(f"Applying rule '{rule_func.__name__}'")
            column_name = rule_func.__name__
            try:
                results, missing_columns_list = zip(*self.df.apply(rule_func, axis=1))
            except Exception as e:
                logger.error(f"Failed to apply rule '{rule_func.__name__}': {e}")
                continue
            rule_missing_columns = set().union(*missing_columns_list)

            if rule_missing_columns:
                logger.error(
                    f"Rule '{rule_func.__name__}' identified missing columns in second pass and was skipped: {', '.join(rule_missing_columns)}")
            else:
                self.df[column_name] = results
                null_mask = self.df[column_name].isnull()
                if null_mask.all():
                    logger.warning(f"The rule '{rule_func.__name__}' returned None for all rows.")
                elif null_mask.any():
                    logger.warning(f"The rule '{rule_func.__name__}' returned None for {null_mask.sum()} rows.")
