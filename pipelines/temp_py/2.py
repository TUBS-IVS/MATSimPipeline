# Concating arrays in np is very fast

# passing int or string doesnt matter

# nested dict is much faster for most operations

import time
import random
from collections import defaultdict

# Set up the dictionary with 100,000 identifiers, each having 10 entries
num_ids = 100000
entries_per_id = 10

nested_dict = defaultdict(list)

for i in range(num_ids):
    identifier = f'id{i}'
    for j in range(entries_per_id):
        leg_info = {
            'start_time': f'start{j}',
            'end_time': f'end{j}',
            'to_activity': f'activity{j}',
            'distance': f'dist{j}'
        }
        nested_dict[identifier].append(leg_info)

nested_dict = dict(nested_dict)

# Define a function to test access time
def test_access_time(nested_dict, num_accesses=1000):
    identifiers = list(nested_dict.keys())
    start_time = time.time()
    
    for _ in range(num_accesses):
        id_to_access = random.choice(identifiers)
        _ = nested_dict[id_to_access]
    
    end_time = time.time()
    total_time = end_time - start_time
    return total_time

# Run the test
num_accesses = 10000
elapsed_time = test_access_time(nested_dict, num_accesses)
print(f"Time taken to access the dictionary {num_accesses} times: {elapsed_time:.4f} seconds")
