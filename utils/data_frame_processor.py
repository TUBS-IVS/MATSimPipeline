from typing import Literal
import pandas as pd
import logging
import gzip

class DataFrameProcessor:
    """
    Base class for processing a Pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame = None, stats_tracker=None, logger=None):
        """
        Parameters should be set as many functions require them, but for some quick scripting they can be omitted.
        """
        self.stats_tracker = stats_tracker
        self.logger = logger if logger is not None else logging.getLogger("data_frame_processor")
        self.df = df
        self.columns_to_delete = set()

    def read_csv(self, csv_path: str, test_col=None, use_cols=None) -> pd.DataFrame:
        """
        Read a csv file with unknown separator and return a dataframe.
        Also works for gzipped files.

        :param csv_path: Path to the CSV file.
        :param test_col: Column name that should be present in the file for validation.
        :param use_cols: List of columns to use from the file. Defaults to all columns.
        :return: DataFrame with the contents of the CSV file.
        :raises: KeyError, ValueError if `test_col` is not found after attempting to read the file.
        """
        try:
            if csv_path.endswith('.gz'):
                with gzip.open(csv_path, 'rt') as f:
                    df = pd.read_csv(f, sep=',', usecols=use_cols)
            else:
                df = pd.read_csv(csv_path, sep=',', usecols=use_cols)
            if test_col is not None:
                test = df[test_col]
        except (KeyError,
                ValueError):  # Sometimes also throws without test_col, when the file is not comma-separated. This is good.
            self.logger.info(f"ID column '{test_col}' not found in {csv_path}, trying to read as ';' separated file...")
            if csv_path.endswith('.gz'):
                with gzip.open(csv_path, 'rt') as f:
                    df = pd.read_csv(f, sep=';', usecols=use_cols)
            else:
                df = pd.read_csv(csv_path, sep=';', usecols=use_cols)
            try:
                if test_col is not None:
                    test = df[test_col]
            except (KeyError, ValueError):
                self.logger.error(
                    f"ID column '{test_col}' still not found in {csv_path}, verify column name and try again.")
                raise
            self.logger.info("Success.")
        return df

    def load_df_from_csv(self, csv_path, if_df_exists: Literal['replace', 'concat'] = 'concat', use_cols=None,
                         test_col=None) -> None:
        """
        Initializes the DataFrame from a CSV file.

        Parameters:
        - csv_path: str, path to the CSV file.
        - if_df_exists: str, whether to replace the existing DataFrame or concatenate to it.
        """
        try:
            new_df = self.read_csv(csv_path, test_col=test_col, use_cols=use_cols)

            if if_df_exists == 'replace':
                if self.df:
                    self.logger.info("DataFrame loaded and replaced successfully from CSV.")
                else:
                    self.logger.info("DataFrame loaded successfully from CSV.")
                self.df = new_df
            elif if_df_exists == 'concat':
                self.df = pd.concat([self.df, new_df], ignore_index=True) if self.df is not None else new_df
                self.logger.info("DataFrame loaded and concatenated successfully from CSV.")
        except Exception as e:
            self.logger.error(f"Failed to load DataFrame from {csv_path}: {e}")
            raise

    def add_df_on_id(self, source_df, columns_to_add=None, overwrite_existing=False, id_column=None,
                     drop_duplicates_from_source=True) -> None:
        """
        Adds specified columns from a source DataFrame to the current DataFrame based on an ID.

        Parameters:
        - source_df: DataFrame, the source dataframe.
        - columns_to_add: list of str, the columns to fetch and add to df. If None, all columns are fetched.
        - overwrite_existing: bool, whether to overwrite columns that already exist. Default is False.
        - drop_duplicates_from_source: bool, whether to drop duplicates based on id_column in source_df. Default is True.
            If False, additional rows will be added to the DataFrame if source_df contains duplicate IDs.
        """
        if columns_to_add is None:
            columns_to_add = source_df.columns.tolist()

        if not isinstance(columns_to_add, list):
            try:
                columns_to_add = list(columns_to_add.values())
            except AttributeError:
                raise ValueError("columns_to_add should be a list or convertible to a list.")

        if id_column is None:
            if self.id_column is None:
                raise ValueError("id_column is None. Please specify an ID column to use.")
            else:
                id_column = self.id_column
                self.logger.info(f"No id_column provided, using id_column from DataFrame: {id_column}")

        # Validation for ID columns in both dataframes
        if id_column not in self.df.columns or id_column not in source_df.columns:
            raise ValueError(f"Either the primary or source DataFrame is missing the ID column: {id_column}")

        # Handle duplicates
        if drop_duplicates_from_source:
            duplicate_mask = source_df.duplicated(subset=id_column, keep='first')
            if duplicate_mask.any():
                duplicate_ids = source_df.loc[duplicate_mask, id_column].tolist()
                self.logger.info(f"Duplicate {id_column} values found and dropped in source_df: {duplicate_ids}")
                source_df = source_df.drop_duplicates(subset=id_column, keep='first')

        # Handle columns
        columns_to_iterate = columns_to_add.copy()
        for col in columns_to_iterate:
            if col in self.df.columns:
                if overwrite_existing:
                    self.logger.info(f"Column {col} already exists in the DataFrame. Overwriting.")
                    self.df.drop(columns=[col], inplace=True)
                else:
                    self.logger.info(f"Column {col} already exists in the DataFrame. Skipping.")
                    columns_to_add.remove(col)

        # Ensure id_column is in columns_to_add for merging
        if id_column not in columns_to_add:
            columns_to_add.append(id_column)

        # Merge the dataframes on the specified id_column
        self.df = pd.merge(self.df, source_df[columns_to_add], on=id_column, how='left')

    def add_csv_data_on_id(self, csv_path, columns_to_add=None, overwrite_existing=False, id_column=None,
                           drop_duplicates_from_source=True, delete_later=False) -> None:
        """
        Load specified columns from a given CSV file and add them to the primary DataFrame.

        Parameters:
        - csv_path: str, path to the CSV file.
        - columns_to_add: list of str, the columns to fetch from the CSV. If None, all columns are fetched.
        - overwrite_existing: bool, whether to overwrite columns that already exist. Default is False.
        - id_column: str, the column to use as the ID. Default is the id_column specified in the constructor.
        - add_prefix: str, optional, prefix to add to the column names from the CSV.
        - drop_duplicates_from_source: bool, whether to drop duplicates based on id_column in the CSV. Default is True.
        """
        if columns_to_add is not None:
            if not isinstance(columns_to_add, list):
                try:
                    columns_to_add = list(columns_to_add.values())
                except AttributeError:
                    raise ValueError("columns_to_add should be a list or convertible to a list.")

            if id_column is None:
                id_column = self.id_column

                if id_column is None:  # If id_column is still None, raise an error
                    raise ValueError("id_column is None. Please specify an ID column to use.")

            if id_column not in columns_to_add:
                columns_to_add.append(id_column)
        try:
            source_df = self.read_csv(csv_path, id_column, use_cols=columns_to_add)

            if delete_later:
                if columns_to_add is None:
                    columns_to_add = source_df.columns.tolist()
                self.columns_to_delete.update(columns_to_add)

            self.add_df_on_id(source_df, columns_to_add, overwrite_existing, id_column, drop_duplicates_from_source)
        except Exception as e:
            self.logger.error(f"Failed to load and add CSV data from {csv_path}: {e}")
            raise

    def remove_columns_startswith(self, startswith) -> None:
        """
        Removes columns from the DataFrame that start with the specified string.
        """
        # Use a list comprehension to find columns starting with the specified string
        cols_to_remove = [col for col in self.df.columns if col.startswith(startswith)]

        # Drop the columns from the DataFrame
        self.df.drop(cols_to_remove, axis=1, inplace=True)

        if cols_to_remove:
            self.logger.info(f"Removed columns: {cols_to_remove}")
        else:
            self.logger.info("No columns were removed.")

    def remove_columns_marked_for_later_deletion(self) -> None:
        """
        Removes columns from the DataFrame that were marked for later deletion.
        """
        cols_to_remove = [col for col in self.df.columns if col in self.columns_to_delete]

        cols_not_found = set(self.columns_to_delete) - set(cols_to_remove)
        if cols_not_found:
            self.logger.warning(f"Columns marked for deletion were not found: {cols_not_found}")

        self.df.drop(cols_to_remove, axis=1, inplace=True)

        self.columns_to_delete.clear()

        if cols_to_remove:
            self.logger.info(f"Removed columns: {cols_to_remove}")
        else:
            self.logger.info("No columns were removed.")

    def convert_time_to_datetime(self, column_names: list) -> None:
        """
        Convert specified columns to datetime; as time alone would not support arithmetic operations.
        The date part is set to a default date, otherwise the date part would be today's date,
        which might break stuff if the pipeline runs across midnight.
        :param column_names: The names of the columns to convert.
        :return: The DataFrame with the specified columns converted to datetime.
        """
        default_date = "1970-01-01"

        for column_name in column_names:
            self.logger.info(f"Converting column '{column_name}' with {len(self.df)} total rows to datetime.")

            # Convert to datetime with a fixed date
            self.df[column_name] = pd.to_datetime(
                self.df[column_name].astype(str).apply(lambda x: f"{default_date} {x}"),
                errors='coerce')

            null_count = self.df[column_name].isnull().sum()  # Count NaT values, which are considered null
            self.logger.info(f"Number of failed conversions (NaT): {null_count}")

    def convert_datetime_to_seconds(self, column_names: list) -> None:
        """
        Calculates the seconds from midnight of the reference day (2020-01-01).
        Adds a new column with the column name + '_seconds' to the DataFrame.
        :param column_names: The names of the columns to convert.
        :return: The DataFrame with new columns containing the seconds from midnight.
        """
        default_date = '2020-01-01'
        for column_name in column_names:
            if pd.api.types.is_datetime64_any_dtype(self.df[column_name]):
                self.logger.info(f"Converting column '{column_name}' with {len(self.df)} total rows to seconds.")
                self.df[f"{column_name}_seconds"] = (
                            self.df[column_name] - pd.Timestamp(default_date)).dt.total_seconds()
            else:
                self.logger.warning(f"Column '{column_name}' is not a datetime column.")

    def filter_out_rows(self, column_name, values_to_filter: list):
        """
        Filter out rows from a DataFrame based on specific values in a given column.
        """
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
        self.logger.info(f"Filtering out rows where column '{column_name}' has values: {values_to_filter}")
        self.logger.info(f"Number of rows before filtering: {len(self.df)}")
        self.df = self.df[~self.df[column_name].isin(values_to_filter)]
        self.logger.info(f"Number of rows after filtering: {len(self.df)}")

    def check_for_merge_suffixes(self, suffixes=('_x', '_y'), throw_error=True):
        """
        Check for columns in a DataFrame that have specified merge suffixes.

        :param df: Pandas DataFrame to check.
        :param suffixes: Tuple of suffixes to look for. Defaults to ('_x', '_y').
        :return: Dictionary with suffixes as keys and list of columns with these suffixes as values.
        """
        columns_with_suffixes = {suffix: [] for suffix in suffixes}
        for col in self.df.columns:
            for suffix in suffixes:
                if col.endswith(suffix):
                    columns_with_suffixes[suffix].append(col)
        if throw_error and any(columns_with_suffixes.values()):
            raise ValueError(f"Columns with merge suffixes found: {columns_with_suffixes}")
        self.logger.info(f"Columns with merge suffixes: {columns_with_suffixes}")
        return columns_with_suffixes

    def set_column_type(self, columns, dtype):
        """
        Set the data types of specified columns in the DataFrame, and raise an error if conversion fails
        or if any column is not found.

        :param columns: List of column names to be converted.
        :param dtype: The desired data type to convert all specified columns to.
        """
        for col in columns:
            if col not in self.df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        for col in columns:

            try:
                self.df[col] = self.df[col].astype(dtype)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Error converting column '{col}' to {dtype}: {e}")

    def transpose(self):
        """
        Transpose the DataFrame.
        """
        self.df = self.df.T
