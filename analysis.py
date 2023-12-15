import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm


from pipelines.common import helpers as h
from utils import settings_values as s
from utils.logger import logging

logger = logging.getLogger(__name__)


class DataframeAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def find_upper_cutoff(self, column, jump_multiplier=10):
        # Remove NaN values and sort the column
        sorted_values = column.dropna().sort_values().reset_index(drop=True)

        # Calculate differences between consecutive values, skipping the first NaN
        differences = sorted_values.diff().iloc[1:]
        average_jump = differences.mean()

        # Identify points where the jump is significantly larger than the average jump
        significant_jumps = differences[differences > jump_multiplier * average_jump]

        # Find the smallest value after these significant jumps
        if not significant_jumps.empty:
            smallest_significant_jump_index = significant_jumps.idxmin()
            return sorted_values.iloc[smallest_significant_jump_index + 1]

        # If no significant jump is found, return the maximum value
        return sorted_values.max()

    def find_upper_cutoff_by_median(self, column):
        return 3 * column.median()

    def count_anomalous_values(self, col_name):
        column = self.df[col_name]

        # Count NaN values and values greater than the upper cutoff
        nan_count = column.isna().sum()
        upper_bound = self.find_upper_cutoff(column)
        sig_diff_count = (column > upper_bound).sum()

        # Return the total count of anomalous values
        return nan_count + sig_diff_count

    def percentage_of_anomalous_values(self, col_name):
        # Extract the specified column
        column = self.df[col_name]

        # Count NaN values and values greater than the upper cutoff
        nan_count = column.isna().sum()
        upper_bound = self.find_upper_cutoff(column)
        sig_diff_count = (column > upper_bound).sum()

        # Calculate total non-NaN values
        total_non_nan = len(column) - nan_count

        # Calculate and return the percentage of anomalous values
        if total_non_nan == 0:  # Avoid division by zero
            return 0
        return (sig_diff_count / total_non_nan) * 100

    def plot_all_columns(self, bins=100):
        for column_name in self.df.columns:
            self.plot_column(column_name, bins=bins)

    def plot_valid_vs_anomalous(self, col_names=None):
        """
        Plot columns from DataFrame categorizing values into valid and non-valid.

        :param df: DataFrame to plot data from.
        :param col_names: List of column names to be plotted.
        """
        if col_names is None:
            col_names = self.df.columns
        for col_name in col_names:
            logger.info(f"Plotting '{col_name}'")
            # Calculate the cutoff for the column
            upper_bound = self.find_upper_cutoff_by_median(self.df[col_name])

            # Prepare data for plotting
            valid = self.df[col_name][self.df[col_name] <= upper_bound]
            non_valid = self.df[col_name][self.df[col_name] > upper_bound]

            # Plotting
            plt.figure(figsize=(10, 6))
            sns.histplot(valid, color="green", label='Valid', kde=True)
            sns.histplot(non_valid, color="red", label='Non-valid', kde=True)

            plt.title(f"Distribution of Valid vs Non-valid Values in '{col_name}'")
            plt.xlabel(col_name)
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()


def analyze_influence_on_slack(df):
    logger.info(f"Analyzing influence on slack factor, {len(df)} rows.")
    df = df[(df['slack_factor'] > 1) & (df['slack_factor'] < 50)].reset_index(drop=True)
    logger.info(f"Dropped outliers and false positives, {len(df)} rows remaining.")

    # One-hot encode the categorical variables
    encoder = OneHotEncoder(drop='first')  # Drop first column to avoid multicollinearity
    categorical_columns = [s.H_REGION_TYPE_COL, 'start_activity', 'via_activity', 'end_activity']
    encoded_vars = encoder.fit_transform(df[categorical_columns])
    encoded_vars_df = pd.DataFrame(encoded_vars.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

    # Combine with the original DataFrame
    X = pd.concat([df[[s.PERSON_AGE_COL]].reset_index(drop=True), encoded_vars_df], axis=1)
    X = sm.add_constant(X)  # Add a constant to the model
    y = df['slack_factor']

    # Fit the linear regression model
    model = sm.OLS(y, X).fit()

    # Return the summary of the regression
    return model.summary()


# df = h.read_csv("data/MiD2017_Wege_edited.csv")
# # analysis = DataframeAnalysis(df)
# columns_to_plot = []
# # columns_to_plot.extend(s.L_COLUMNS.values())
# columns_to_plot.extend(s.P_COLUMNS.values())
# columns_to_plot.extend(s.HH_COLUMNS.values())
# # analysis.plot_valid_vs_anomalous(columns_to_plot)
# for col in s.L_COLUMNS.values():
#     h.plot_column(df, col)

slack_df = h.read_csv("output/20231215_001306/slack_factors.csv")
print(analyze_influence_on_slack(slack_df))
