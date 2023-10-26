import pandas as pd

from utils.logger import logger


def distribute_by_weights(df, total_col, weight_col):
    logger.info(f"Distributing totals for cell_id: {df['cell_id'].iloc[0]}")

    # Check for non-zero sum of weights
    weight_sum = df[weight_col].sum()
    if weight_sum == 0:
        logger.warning(f"No weights for cell_id: {df['cell_id'].iloc[0]}. Assigning 0 to distributed values.")
        return df.assign(distributed=0)

    # Calculate distributed values
    df['distributed'] = (df[weight_col] / weight_sum * df[total_col].iloc[0]).astype(int)

    # Calculate and distribute the remainder
    remainder = (df[weight_col] / weight_sum * df[total_col].iloc[0]) - df['distributed']
    diff = df[total_col].iloc[0] - df['distributed'].sum()
    top_indices = remainder.nlargest(diff).index
    df.loc[top_indices, 'distributed'] += 1

    return df[['distributed']]


def distribute_cell_totals(cell_df, point_df):
    """
    Distribute the total values of cells to points inside the cell based on their weights.

    Parameters:
    - cell_df (DataFrame): A dataframe with columns ['cell_id', 'total'].
    - point_df (DataFrame): A dataframe with columns ['point_id', 'cell_id', 'weight'].

    Returns:
    - DataFrame: A merged dataframe with columns ['point_id', 'cell_id', 'weight', 'total', 'distributed'].
    """

    merged_df = point_df.merge(cell_df, on="cell_id")

    distributed_values_df = merged_df.groupby('cell_id').apply(distribute_by_weights, total_col='total',
                                                               weight_col='weight').reset_index(drop=True)

    final_result = pd.concat([merged_df, distributed_values_df], axis=1)

    logger.info("Distribution completed successfully!")

    return final_result

# Sample usage
# cell_df_sample = pd.DataFrame({
#     'cell_id': [1, 2, 3],
#     'total': [100, 200, 150]
# })
#
# point_df_sample = pd.DataFrame({
#     'point_id': [101, 102, 103, 104, 105, 106],
#     'cell_id': [1, 1, 2, 2, 3, 3],
#     'weight': [2, 3, 1, 4, 5, 2]
# })
#
# result_df = distribute_totals_by_weights(cell_df_sample, point_df_sample)
# print(result_df)
