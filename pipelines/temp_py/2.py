# Using str or int as dict key makes no difference in the time taken to retrieve the value from the dictionary.

# Import the required module for timing
import timeit

# Setup dictionaries
string_dict = {
    "abcdefgh": "value1",
    "ijklmnopdghkjfghdfg": "value2",
    "qrstuvwx": "value3"
}

int_dict = {
    11: "value1",
    21: "value2",
    31: "value3"
}

# Define functions to retrieve values from the dictionaries
def string_function(val: str) -> str:
    return string_dict[val]

def int_function(val: int) -> str:
    return int_dict[val]

# Setup code to import these functions and dictionaries in the timing environment
setup_code = """
from __main__ import string_function, int_function, string_dict, int_dict
"""

# Timing the function with string key
string_time = timeit.timeit(stmt='string_function("ijklmnopdghkjfghdfg")', setup=setup_code, number=100000000)
print(f"Time for retrieving a value with an 8-character string key: {string_time:.6f} seconds")

# Timing the function with integer key
int_time = timeit.timeit(stmt='int_function(11)', setup=setup_code, number=100000000)
print(f"Time for retrieving a value with a short integer key: {int_time:.6f} seconds")
