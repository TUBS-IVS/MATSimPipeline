import numpy as np
import timeit

def setup_and_concatenate():
    # Mock data for testing
    candidate_identifiers = np.random.randint(0, 100, size=(5,))
    candidate_names = np.array(['name1', 'name2', 'name3', 'name4', 'name5'])
    candidate_coordinates = np.random.rand(5, 2)
    candidate_capacities = np.random.randint(1, 10, size=(5,))
    candidate_distances = np.random.rand(5)

    candidate_identifiers2 = np.random.randint(0, 100, size=(5,))
    candidate_names2 = np.array(['name6', 'name7', 'name8', 'name9', 'name10'])
    candidate_coordinates2 = np.random.rand(5, 2)
    candidate_capacities2 = np.random.randint(1, 10, size=(5,))
    candidate_distances2 = np.random.rand(5)

    # Concatenation process
    candidate_identifiers = np.concatenate((candidate_identifiers, candidate_identifiers2), axis=0)
    candidate_names = np.concatenate((candidate_names, candidate_names2), axis=0)
    candidate_coordinates = np.concatenate((candidate_coordinates, candidate_coordinates2), axis=0)
    candidate_capacities = np.concatenate((candidate_capacities, candidate_capacities2), axis=0)
    candidate_distances = np.concatenate((candidate_distances, candidate_distances2), axis=0)

# Measure the time it takes to run the test using timeit
execution_time = timeit.timeit("setup_and_concatenate()", setup="from __main__ import setup_and_concatenate", number=1000)

print(f"Average time for concatenation for 1000 runs: {execution_time:.8f} seconds")
