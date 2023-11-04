import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataFrameProcessor:
    def __init__(self, df: pd.DataFrame = None, id_column: str = None):
        self.df = df
        self.id_column = id_column
        self.added_missing_columns = set()  # Columns added by the safe_apply_rules method; input for rules, not results

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            logger.error("DataFrame is not yet initialized.")
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame) and value is not None:
            raise ValueError("The df attribute must be a Pandas DataFrame or None.")
        self._df = value

    @property
    def id_column(self) -> str:
        return self._id_column

    @id_column.setter
    def id_column(self, value: str) -> None:
        if self._df is not None and value is not None and value not in self._df.columns:
            logger.warning(f"ID column '{value}' not found in DataFrame, setting anyway.")

        self._id_column = value

    def load_df_from_csv(self, csv_path: str, id_column: str = None) -> None:
        """
        Initializes the DataFrame from a CSV file.

        Parameters:
        - csv_path: str, path to the CSV file.
        - id_column: str, optional, column to use as the ID if provided.
        """
        try:
            self.df = pd.read_csv(csv_path)
            if id_column:
                self.id_column = id_column
            logger.info("DataFrame loaded successfully from CSV.")
        except Exception as e:
            logger.error(f"Failed to load DataFrame from {csv_path}: {e}")

    def add_data_on_id(self, source_df, columns_to_add, overwrite_existing=False, drop_duplicates_from_source=True):
        """
        Adds specified columns from a source DataFrame to the current DataFrame based on an ID.

        Parameters:
        - source_df: DataFrame, the source dataframe.
        - columns_to_add: list of str, the columns to fetch and add to df.
        - overwrite_existing: bool, whether to overwrite columns that already exist. Default is True.
        - drop_duplicates_from_source: bool, whether to drop duplicates based on id_column in source_df. Default is True.
            If False, additional rows will be added to the DataFrame if source_df contains duplicate IDs.
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

    def safe_apply_rules(self, rules, csv_path=None, groupby_column=None):
        """
        Applies a set of custom rules to the DataFrame stored in this instance. Will only add columns, never alter existing columns.
        If a rule references missing columns, those columns are fetched from a secondary data source (e.g. MiD) and the rules are reapplied.

        Parameters:
        - rules (list of functions): A list of rule functions. Each rule function must return a tuple of (result, missing_columns).
        - csv_path (str): Path to the CSV file containing the secondary data source (e.g. MiD).
        - groupby_column (str, optional): Column to group by before applying rules. If not provided, rules are applied per row.

        Notes:
        - Could run at different places in the pipeline and with different rule sets.
        """

        def apply_rules(data, groupby_column, rule_func):
            try:
                if groupby_column:
                    results = []
                    missing_columns_list = []
                    skip_more_logs = False

                    for _, group in data.groupby(groupby_column):
                        result, missing_columns = rule_func(group)
                        if result is not None:  # Allow None to be returned e.g. when just checking for missing columns
                            if len(result) != len(group):
                                if not skip_more_logs:
                                    logger.error(
                                        f"Rule '{rule_func.__name__}' returned DataFrame with incorrect number of rows. Skipping its results."
                                    )
                                skip_more_logs = True
                                continue
                            if not result.index.equals(group.index):
                                if not skip_more_logs:
                                    logger.error(
                                        f"Rule '{rule_func.__name__}' returned DataFrame with incorrect index. Skipping its results."
                                    )
                                skip_more_logs = True
                                continue
                        results.append(result)
                        missing_columns_list.extend(missing_columns)
                else:
                    results, missing_columns_list = zip(*data.apply(rule_func, axis=1))
            except Exception as e:
                logger.error(f"Failed to apply rule '{rule_func.__name__}': {e}")
                return None, []

            return results, list(set().union(*missing_columns_list))

        all_missing_columns = set()

        if csv_path:
            # First pass: identify all missing columns
            for rule_func in rules:
                logger.info(f"Collecting missing columns for rule '{rule_func.__name__}'")
                _, rule_missing_columns = apply_rules(self.df, groupby_column, rule_func)
                all_missing_columns.update(rule_missing_columns)

                if rule_missing_columns:
                    logger.info(
                        f"Rule '{rule_func.__name__}' identified missing columns: {', '.join(rule_missing_columns)}")

            # Fetch all missing columns at once
            if all_missing_columns:
                logger.info(f"Fetching missing columns: {', '.join(all_missing_columns)}")
                self.add_csv_data(csv_path, list(all_missing_columns), overwrite_existing=False)
                self.added_missing_columns.update(all_missing_columns)  # Keep track of added columns for optional removal later

        # Second pass: apply the rules now that all columns are present
        for rule_func in rules:
            logger.info(f"Applying rule '{rule_func.__name__}'")
            results, rule_missing_columns = apply_rules(self.df, groupby_column, rule_func)

            if not results:
                logger.error(f"Rule '{rule_func.__name__}' returned None for all rows and was skipped.")
                continue

            if rule_missing_columns:
                logger.error(
                    f"Rule '{rule_func.__name__}' identified missing columns in second pass and was skipped: {', '.join(rule_missing_columns)}")
                continue

            if groupby_column:
                for result_df in results:
                    result_df.columns = [f"{col}_{rule_func.__name__}" for col in result_df.columns]
                    self.df = self.df.join(result_df)
                null_mask = self.df[[col for col in self.df.columns if rule_func.__name__ in col]].isnull().any(axis=1)
            else:
                self.df[rule_func.__name__] = results
                null_mask = self.df[rule_func.__name__].isnull()

            if null_mask.any():
                logger.warning(f"The rule '{rule_func.__name__}' returned None for {null_mask.sum()} rows.")

    def remove_added_columns(self):
        """Removes the columns added by the safe_apply_rules method."""
        for col in self.added_missing_columns:
            if col in self.df.columns:
                self.df.drop(columns=col, inplace=True)
                logger.info(f"Removed column: {col}")
        self.added_missing_columns = []
