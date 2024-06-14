import timeit

def generate_lists(n: int) -> list[list[int]]:
    original_list = list(range(1, n + 1))
    result = []
    current_list = original_list

    while len(current_list) > 0:
        result.append(current_list)
        # Create the new list by taking every second element
        next_list = current_list[1::2]
        # If the original list has an odd length and we removed the last element, we add it back
        if len(current_list) % 2 == 1:
            next_list.append(current_list[-1])
        current_list = next_list
            
    return result

# Wrapper function to call generate_lists with n=9
def test_generate_lists():
    generate_lists(9)

# Measure performance
number_of_runs = 1000
time_taken = timeit.timeit(test_generate_lists, number=number_of_runs)

print(f"Time taken to run generate_lists(9) {number_of_runs} times: {time_taken:.5f} seconds")
