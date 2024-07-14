import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os

from synthesis.enhanced_mid_data.mid_data_enhancer import MiDDataEnhancer
from utils import helpers as h
from utils import matsim_pipeline_setup as m
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

    def plot_all_columns(self):
        for column_name in self.df.columns:
            self.plot_column(column_name)

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

    def df_value_counts(self):
        transformed_dfs = []

        for col in self.df.columns:
            # Counting unique values in the column and creating a new DataFrame
            counts = self.df[col].value_counts(dropna=False).reset_index()
            counts.columns = [f'{col}_value', f'{col}_count']

            # Calculate the total count of values in the column
            total_count = counts[f'{col}_count'].sum()

            # Calculate the percentage of each unique value
            counts[f'{col}_percentage'] = (counts[f'{col}_count'] / total_count) * 100

            # Creating a new index for each DataFrame to ensure proper alignment
            counts.index = range(len(counts))
            transformed_dfs.append(counts)

        # Combining all transformed DataFrames side-by-side
        combined_df = pd.concat(transformed_dfs, axis=1)

        return combined_df

    def plot_column(self, column, title=None, xlabel=None, ylabel='Frequency', plot_type=None, figsize=(10, 6), save_name=None):
        """
        Plots a column from a pandas DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to plot.
        title (str, optional): Title of the plot. Defaults to None, which will use the column name.
        xlabel (str, optional): Label for the x-axis. Defaults to None, which will use the column name.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Frequency'.
        plot_type (str, optional): Type of plot (hist, bar, box, violin, strip, swarm, point). If None, the plot type is inferred from the column type. Defaults to None.
        figsize (tuple, optional): Size of the figure (width, height). Defaults to (10, 6).
        save_name (str, optional): Name with file extension to save the figure. If None, the figure is not saved. Defaults to None.
        """
        # Infer plot type if not specified
        if plot_type is None:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                plot_type = 'hist'
            elif pd.api.types.is_datetime64_any_dtype(self.df[column]):
                plot_type = 'hist'
            else:
                plot_type = 'bar'

        # Set plot title and labels
        if title is None:
            title = f'Distribution of {column}'
        if xlabel is None:
            xlabel = column

        # Create the plot
        plt.figure(figsize=figsize)
        if plot_type == 'hist':
            sns.histplot(self.df[column].dropna(), kde=True)  # KDE for numeric and datetime
        elif plot_type == 'bar':
            sns.countplot(x=column, data=self.df)
        elif plot_type == 'box':
            sns.boxplot(x=column, data=self.df)
        elif plot_type == 'violin':
            sns.violinplot(x=column, data=self.df)
        elif plot_type == 'strip':
            sns.stripplot(x=column, data=self.df)
        elif plot_type == 'swarm':
            sns.swarmplot(x=column, data=self.df)
        elif plot_type == 'point':
            sns.pointplot(x=column, data=self.df)
        else:
            raise ValueError("Unsupported plot type. Use 'hist' or 'bar'.")

        # Set title and labels
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Save the plot if a save path is provided
        if save_name:
            save_name = os.path.join(m.OUTPUT_DIR, save_name)
            plt.savefig(save_name, bbox_inches='tight')

        else:
            plt.show()


def analyze_influence_on_slack(df):
    logger.info(f"Analyzing influence on slack factor, {len(df)} rows.")
    df = df[(df['slack_factor'] > 1) & (df['slack_factor'] < 50)].reset_index(drop=True)
    logger.info(f"Dropped outliers and false positives, {len(df)} rows remaining.")

    # One-hot encode the categorical variables
    encoder = OneHotEncoder(drop='first')  # Drop first column to avoid multicollinearity
    categorical_columns = ['direct_mode'] # ,'RegioStaR7_x', 'start_activity', 'via_activity', 'end_activity', "pergrup1", "start_mode", "end_mode"]
    encoded_vars = encoder.fit_transform(df[categorical_columns])
    encoded_vars_df = pd.DataFrame(encoded_vars.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

    # X = pd.concat([df[[s.PERSON_AGE_COL]].reset_index(drop=True), encoded_vars_df], axis=1)
    X = encoded_vars_df
    X = sm.add_constant(X)
    y = df['slack_factor']

    model = sm.OLS(y, X).fit()

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

# slack_df = h.read_csv("data/slack_factors_raw.csv")
# persons_df = h.read_csv(s.MiD_PERSONS_FILE)
# slack_df = slack_df.merge(persons_df, on=s.PERSON_ID_COL, how='left')
# print(analyze_influence_on_slack(slack_df))

# ttdf1 = h.read_csv("data/TTmatrices/hour_Min_0_Max 1car_travel_times.csv" )
# ttdf2 = h.read_csv("data/TTmatrices/hour_Min_4_Max 5car_distances.csv", "FROM")
# ttdf3= h.read_csv("data/TTmatrices/hour_Min_4_Max 5car_travel_times.csv", "FROM")
# connections_df = h.read_csv("output/20240103_020348/leg_connections_logs.csv")  # TODO:repeat and check.
enhanced_mid_df = h.read_csv("output/enhanced_frame_final.csv")
#  mid_df = h.read_csv(s.MiD_TRIPS_FILE)  # TODO: check num of rows (enh: 999803)
logger.info("Loaded DataFrame")
# Filter out where time and distance are False
# connections_df = connections_df[(connections_df['mode_match'] != False) & (connections_df["activity_match"] != False)]
# Add a col that is true when all four comparisons are true
# connections_df['all_match'] = connections_df[['mode_match', 'activity_match', 'time_match', 'dist_match']].all(axis=1)

pop = MiDDataEnhancer(enhanced_mid_df)
pop.check_for_merge_suffixes()

ana = DataframeAnalysis(enhanced_mid_df)
logger.info(f"Rows: {len(enhanced_mid_df)} Columns: {len(enhanced_mid_df.columns)}")
vc_df = ana.df_value_counts()
vc_df.to_csv("testdata/analyze/enhanced_final_mid_analysis.csv", index=False)
logger.info("Done")






def plot_sigmoid():
    delta_T = 20
    time_diff_range = np.arange(0, 60, 0.1)
    beta_values = [-0.25, -0.2, -0.15, -0.1]
    colors = ['blue', 'green', 'red', 'grey']  # 'blue', 'green', 'red', 'grey'

    # Plotting
    plt.figure(figsize=(10, 6))
    for beta_val, color in zip(beta_values, colors):
        sigmoid_values = 1 / (1 + np.exp(-beta_val * (time_diff_range - delta_T)))
        plt.plot(time_diff_range, sigmoid_values, label=f'β = {beta_val}', color=color)

    plt.xlabel('Time Differential (Minutes)')
    plt.ylabel('Sigmoid Value')
    # plt.title('Sigmoid Function with ΔT = 30 Minutes')
    # plt.axvline(x=10, color='blue', linestyle='--', label='10 min')
    # plt.axvline(x=20, color='green', linestyle='--', label='20 min')
    # plt.axvline(x=30, color='red', linestyle='--', label='30 min (ΔT Adjusted)')
    # plt.axvline(x=40, color='grey', linestyle='--', label='40 min')
    plt.grid(True)
    plt.legend()
    plt.show()

