import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any, Union
import numpy as np
import time
import random

class DataFrameConverter:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.nested_dict = defaultdict(list)

    def convert_to_nested_dict(self):
        current_identifier = None
        for i, row in self.df.iterrows():
            identifier = row['identifier']
            if identifier != current_identifier:
                # Start with an 'activity' entry for the new identifier
                entry_type = 'activity'
                current_identifier = identifier
            else:
                # Alternate the entry type
                entry_type = 'leg' if entry_type == 'activity' else 'activity'
                
            if entry_type == 'activity':
                info = {
                    'type': 'activity',
                    'location': np.array(row['location']),
                    'purpose': row['purpose']
                }
            else:  # entry_type == 'leg'
                info = {
                    'type': 'leg',
                    'leg_id': row['leg_id'],
                    'distance': row['distance']
                }
            self.nested_dict[identifier].append(info)
    
    def get_nested_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        # Convert defaultdict to dict for cleaner output
        return dict(self.nested_dict)
    
    def test_access_time(self, num_accesses: int = 1000) -> float:
        identifiers = list(self.nested_dict.keys())
        start_time = time.time()
        
        for _ in range(num_accesses):
            id_to_access = random.choice(identifiers)
            _ = self.nested_dict[id_to_access]
        
        end_time = time.time()
        total_time = end_time - start_time
        return total_time

# Sample DataFrame with alternating activity and leg entries
data = {
    'identifier': ['id1', 'id1', 'id1', 'id1', 'id2', 'id2', 'id2', 'id2'],
    'location': [[1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [13,14], [15,16]],
    'purpose': ['work', None, 'home', None, 'gym', None, 'store', None],
    'leg_id': [None, 'leg1', None, 'leg2', None, 'leg3', None, 'leg4'],
    'distance': [None, 'dist1', None, 'dist2', None, 'dist3', None, 'dist4']
}

df = pd.DataFrame(data)

# Instantiate the converter with the DataFrame
converter = DataFrameConverter(df)

# Convert to nested dictionary
converter.convert_to_nested_dict()

# Retrieve the nested dictionary
nested_dict = converter.get_nested_dict()

# Print the nested dictionary
print(nested_dict)

# Testing access time
elapsed_time = converter.test_access_time(1000)
print(f"Time taken to access the dictionary 1000 times: {elapsed_time:.4f} seconds")
