import pandas as pd
import numpy as np
from openpyxl.styles.builtins import total
from tqdm import tqdm
from utils import settings as s
from utils import helpers as h
from utils.logger import logging
import pickle
logger = logging.getLogger(__name__)

file_path = r'C:\Users\petre\Documents\GitHub\MATSimPipeline\data\raw_data\krpend-k-0-202306-xlsx.xlsx'  # Update this with the actual path to your Excel file
sheet_name_auspendler = 'Auspendler Kreise'
sheet_name_einpendler = 'Einpendler Kreise'

# We lose a very small amount of commuters (thousand out of 13 mill) by ignoring "Übrige Bundesländer" and "Übrige Regierungsbezirke"

def get_od_matrix(file_path: str, sheet_name: str, value_col: int, all_commuters_col: int):
    # Starting from line 9 (index 8 in zero-indexed Python)
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=8, dtype={0: str, 1: str, 2: str, 3: str}, header=None)

    # Calculate missing * values in Übrige Kreise (in the df)
    if value_col == all_commuters_col: # Means it's not split by gender etc

        five_digit_sum = 0
        three_digit_sum = 0
        prefix = None

        for index, row in df.iterrows():
            code = str(row[2])  # Column C has the code
            value = row[value_col] if pd.notna(row[value_col]) else 0

            if len(code) == 5:
                if prefix is None:
                    prefix = code[:2]  # Set the prefix to compare with 3-digit codes
                five_digit_sum += value

            elif len(code) == 3 and prefix and code.startswith(prefix):  # Also skips any following 3-digit entries (prefix=None)
                if "Übrig" in str(row[3]):
                    next_index = index + 1
                    if next_index < len(df):
                        next_row = df.iloc[next_index]
                        next_code = str(next_row[2])
                        if next_code.startswith(prefix):
                            three_digit_sum = next_row[value_col]

                    if row[value_col] == "*":
                        calculated_value = three_digit_sum - five_digit_sum
                        logger.info(f"Calculated missing value for {code}: {calculated_value}")
                        assert calculated_value >= 0, f"Calculated value is negative for {code}"
                        assert calculated_value < 10, f"Calculated value is too large for {code}"
                        df.at[index, value_col] = calculated_value
                        five_digit_sum += calculated_value
                    else:
                        five_digit_sum += row[value_col]  # Add the "Übrig" value if not a "*"
                else:
                    three_digit_sum = value

                assert np.isclose(five_digit_sum, three_digit_sum, atol=1e-5), (
                    f"Sum mismatch for prefix {prefix}: "
                    f"five-digit sum {five_digit_sum} != three-digit sum {three_digit_sum}"
                )
                five_digit_sum = 0
                three_digit_sum = 0
                prefix = None

    # Add an "Übrige" entry to three-digit locations where there are no 5-digit ones before (in the df)
    inserted_rows = 0
    new_df = df.copy()
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Adding missing Übrige entries"):
        code = str(row[2])
        if code == "nan":
            continue
        if len(code) == 3:
            prev_code = str(df.iloc[index - 1, 2])
            next_code = str(df.iloc[index + 1, 2]) if index + 1 < len(df) else "nan"

            if ((not prev_code == code and len(prev_code) == 3) or
                (len(prev_code) == 2 and not next_code == code)):
                # Insert a new row with the same values but the Übrige Kreise name
                new_row = row.copy()
                new_row[3] = "Übrige Kreise (Regierungsbezirk)"
                new_row = pd.DataFrame([new_row])
                new_df = pd.concat([new_df.iloc[:index+inserted_rows], new_row, new_df.iloc[index+inserted_rows:]], ignore_index=True)
                inserted_rows += 1
    logger.info(f"Inserted {inserted_rows} missing rows.")

    # # Rename existing ones so they will be sorted correctly
    # df.iloc[:,3] = df.iloc[:,3].replace("Übrige Kreise (Regierungsbezirk)", "_Übrige Kreise (Regierungsbezirk)")
    # df.sort_values(by=[2, 3], inplace=True)

    concated_names = df.iloc[:,2].astype(str) + '_' + df.iloc[:,3].astype(str)
    gemeinde_names = concated_names.unique()
    gemeinde_names = list(gemeinde_names)
    gemeinde_names = [name for name in gemeinde_names if
                      name not in ['ZZ_Auspendler in das Bundesgebiet', 'Z_Auspendler insgesamt', 'nan_nan', 'ZZ_Übrige Bundesländer']]
    gemeinde_names = sorted(gemeinde_names)
    od_matrix = pd.DataFrame(index=gemeinde_names, columns=gemeinde_names).fillna(0)

    # Fill OD matrix from the df
    current_origin = None
    for index, row in tqdm(new_df.iterrows(), total=len(new_df), desc="Filling OD matrix"):
        # Check if there's a value in column A, indicating a new origin
        if pd.notna(row[0]) and pd.notna(row[1]):
            current_origin = f"{row[0]}_{row[1]}"  # Concatenate column A and B with an underscore

        # Process rows that belong to the current origin group
        if current_origin and pd.notna(row[2]) and pd.notna(row[3]) and pd.notna(row[4]):
            dest_name = f"{row[2]}_{row[3]}"  # Concatenate column C and D with an underscore
            value = row[value_col]  # col E = 4, F = 5, G = 6, etc.

            # Update the matrix at the correct position
            if current_origin in od_matrix.index and dest_name in od_matrix.columns:
                od_matrix.loc[current_origin, dest_name] = value

    assert od_matrix.shape[0] == od_matrix.shape[1], "The matrix is not square (n*n)."
    assert np.all(np.diag(od_matrix) == 0), "The diagonal contains non-zero values."

    # Split off international origins
    international_origins = None
    international_origins_sum = 0
    try:
        prefixes = [
            'A_', 'BG_', 'BIH_', 'B_', 'CDN_', 'CH_', 'CZ_', 'DK_', 'EST_', 'ES_',
            'E_', 'F_', 'GB_', 'GEO_', 'GR_', 'HR_', 'H_', 'IND_', 'IRL_', 'IS_',
            'I_', 'KIS_', 'LT_', 'LV_', 'L_', 'MD_', 'NL_', 'N_', 'PL_', 'P_',
            'RO_', 'RP_', 'RUS_', 'SGP_', 'SK_', 'SLO_', 'SRB_', 'S_', 'TJ_',
            'TR_', 'UA_', 'USA_'
        ]
        mask = od_matrix.index.str.startswith(tuple(prefixes))
        international_origins = od_matrix.T.loc[mask].copy()
        international_origins_sum = international_origins.values.sum()
        od_matrix = od_matrix.drop(index=od_matrix.index[mask], columns=od_matrix.columns[mask])
    except:
        logger.info("No international origins found.")

    # Remove one and two-digit (summary) entries from both rows and columns
    two_digit_names = [name for name in gemeinde_names if len(name.split('_')[0]) <= 2]
    od_matrix = od_matrix.drop(index=two_digit_names, columns=two_digit_names, errors='ignore')

    # Remove three-digit (summary) entries except Übrige Kreise from both rows and columns
    three_digit_names = [name for name in gemeinde_names if len(name.split('_')[0]) == 3 and not name.split('_')[1].startswith("Übrig")]
    od_matrix = od_matrix.drop(index=three_digit_names, columns=three_digit_names, errors='ignore')

    if value_col == all_commuters_col:
        # Compare the original total number of commuters
        df_sum_rows = df[df.iloc[:, 2] == "Z"].copy().reset_index(drop=True)
        df_row_sums = df_sum_rows.iloc[:, 4]
        df_row_sums = np.array(df_row_sums)
        df_total_commuters = df_row_sums.sum()

        od_row_sums = od_matrix[~od_matrix.index.str.split('_').str[1].str.startswith("Übrig")].sum(axis=1)
        od_row_sums = np.array(od_row_sums)

        od_row_deviations = od_row_sums - df_row_sums
        od_total_commuters = od_matrix.values.sum() + international_origins_sum

        assert abs(df_total_commuters - od_total_commuters) < 5000, "The total number of commuters does not match."
        assert np.max(np.abs(od_row_deviations)) < 100, "The row sums do not match."

    return od_matrix, international_origins


def distribute_ubrige_values(matrix_df):
    """
    Distributes the values from 'Übrige' rows proportionally across zero entries in the target rows
    based on row totals.

    Args:
        matrix_df (pd.DataFrame): The DataFrame containing the data with 'Übrige' rows and target columns.

    Returns:
        pd.DataFrame: The updated DataFrame with values distributed from 'Übrige' rows.
    """
    # Iterate through the DataFrame to process "Übrige" rows
    for index, row in tqdm(matrix_df.iterrows(),total=len(matrix_df), desc="Distributing Übrige values"):
        if "Übrige" in index:
            # Extract the prefix before "Übrige"
            prefix = index.split('_')[0]

            # Find target rows that start with the five-digit code using the three-digit prefix
            target_rows = [idx for idx in matrix_df.index if
                           idx.startswith(prefix) and len(idx.split('_')[0]) == 5]

            # If no target rows found, shorten prefix to the first two digits and search again
            if not target_rows:
                shortened_prefix = prefix[:2]  # Use the first two digits
                target_rows = [idx for idx in matrix_df.index if
                               idx.startswith(shortened_prefix) and len(idx.split('_')[0]) == 5]

            # Distribute each value in the Übrige row across zero entries in the target rows
            for col in matrix_df.columns:
                ubrige_value = row[col]
                if ubrige_value == 0:
                    continue

                # Find indices of target rows where the current column has zero entries
                zero_indices = [idx for idx in target_rows if matrix_df.at[idx, col] == 0 and idx != col]

                # If there are zero entries and a positive Übrige value, proceed with distribution
                if len(zero_indices) > 0:
                    # Calculate row sums for the rows with zero entries in the target rows
                    row_sums = matrix_df.loc[zero_indices, :].sum(axis=1).replace(0,
                                                                                  np.nan)  # Replace 0 to avoid division by zero

                    # Calculate proportional weights for distribution
                    weights = row_sums / row_sums.sum()

                    # Distribute the Übrige value proportionally and ensure integer distribution
                    distributed_values = (ubrige_value * weights).round().fillna(0).astype(int)

                    # Correct for rounding errors to ensure the sum matches the ubrige_value
                    rounding_difference = ubrige_value - distributed_values.sum()

                    # Adjust the largest distributed value to match the total ubrige_value
                    if rounding_difference != 0:
                        adjustment_index = distributed_values.idxmax() if rounding_difference > 0 else distributed_values.idxmin()
                        distributed_values[adjustment_index] += rounding_difference

                    # Spread the distributed values across the zero entries of the target rows
                    for zero_idx in zero_indices:
                        matrix_df.at[zero_idx, col] += distributed_values[zero_idx]

    return matrix_df

# raw_auspendler_matrix, _ = get_od_matrix(file_path, sheet_name_auspendler, 4, 4)
# transposed_auspendler_matrix = raw_auspendler_matrix.T
# # raw_einpendler_matrix, _ = get_od_matrix(file_path, sheet_name_einpendler, 4, 4)
# # transposed_einpendler_matrix = raw_einpendler_matrix.T
# # ein_sum = raw_einpendler_matrix.values.sum()
# # aus_sum = raw_auspendler_matrix.values.sum()
# pickle.dump(transposed_auspendler_matrix, open(r'C:\Users\petre\Documents\GitHub\MATSimPipeline\data\raw_data\transposed_auspendler_matrix.pkl', 'wb'))
transposed_auspendler_matrix = pickle.load(open(r'C:\Users\petre\Documents\GitHub\MATSimPipeline\data\raw_data\transposed_auspendler_matrix.pkl', 'rb'))
auspendler_matrix = distribute_ubrige_values(transposed_auspendler_matrix)
auspendler_matrix = auspendler_matrix.drop(index=auspendler_matrix.index[auspendler_matrix.index.str.split('_').str[1].str.startswith("Übrig")],
    columns=auspendler_matrix.columns[auspendler_matrix.index.str.split('_').str[1].str.startswith("Übrig")])
logger.info(f"Matrix size: {auspendler_matrix.shape}")
logger.info(f"Matrix sum: {auspendler_matrix.values.sum()}")
commuter_matrix = auspendler_matrix.T
commuter_matrix.to_csv(r'C:\Users\petre\Documents\GitHub\MATSimPipeline\data\commuter_matrix\commuter_matrix.csv')

