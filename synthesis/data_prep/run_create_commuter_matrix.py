import pandas as pd
import numpy as np
from utils import settings as s
from utils import helpers as h
from utils.logger import logging
logger = logging.getLogger(__name__)

file_path = r'C:\Users\petre\Documents\GitHub\MATSimPipeline\data\raw_data\krpend-k-0-202306-xlsx.xlsx'  # Update this with the actual path to your Excel file
sheet_name_auspendler = 'Auspendler Kreise'
sheet_name_einpendler = 'Einpendler Kreise'

# We lose a very small amount of commuters (thousand out of 13 mill) by ignoring "Übrige Bundesländer" and "Übrige Regierungsbezirke"

def get_od_matrix(file_path: str, sheet_name: str, value_col: int, all_commuters_col: int) -> pd.DataFrame:
    # Starting from line 9 (index 8 in zero-indexed Python)
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=8, dtype={0: str, 1: str, 2: str, 3: str}, header=None)

    concated_names = df.iloc[:,2].astype(str) + '_' + df.iloc[:,3].astype(str)
    gemeinde_names = concated_names.unique()
    gemeinde_names = list(gemeinde_names)
    gemeinde_names = [name for name in gemeinde_names if
                      name not in ['ZZ_Auspendler in das Bundesgebiet', 'Z_Auspendler insgesamt', 'nan_nan', 'ZZ_Übrige Bundesländer']]
    gemeinde_names = sorted(gemeinde_names)
    od_matrix = pd.DataFrame(index=gemeinde_names, columns=gemeinde_names).fillna(0)


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
    for index, row in df.iterrows():
        code = str(row[2])
        if len(code) == 3:
            prev_code = str(df.iloc[index - 1, 2])
            if not prev_code == code or len(prev_code) == 5:
                # Insert a new row with the same values but the Übrige Kreise name
                new_row = row.copy()
                new_row[3] = "Übrige Kreise (Regierungsbezirk)"
                df = df.append(new_row, ignore_index=True)
        # TODO: Finish this part

    # Fill OD matrix from the df
    current_origin = None
    for index, row in df.iterrows():
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

    # Remove two-digit entries from both rows and columns
    two_digit_names = [name for name in gemeinde_names if len(name.split('_')[0]) == 2]
    od_matrix = od_matrix.drop(index=two_digit_names, columns=two_digit_names, errors='ignore')

    # Remove three-digit entries except Übrige Kreise from both rows and columns
    three_digit_names = [name for name in gemeinde_names if len(name.split('_')[0]) == 3 and not name.split('_')[1].startswith("Übrig")]
    od_matrix = od_matrix.drop(index=three_digit_names, columns=three_digit_names, errors='ignore')

    if value_col == all_commuters_col:
        # Compare the original total number of commuters
        sum_rows = df[df.iloc[:, 2] == "Z"]
        total_commuters = sum_rows.iloc[:, 4].sum()
        total_commuters_matrix = od_matrix.values.sum()
        assert abs(total_commuters - total_commuters_matrix) < 2000, "The total number of commuters does not match."

    return od_matrix

raw_auspendler_matrix = get_od_matrix(file_path, sheet_name_auspendler, 4, 4)
raw_einpendler_matrix = get_od_matrix(file_path, sheet_name_einpendler, 4, 4)
transposed_einpendler_matrix = raw_einpendler_matrix.T
print("hi")


