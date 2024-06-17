# Concating arrays in np is very fast

# passing int or string doesnt matter

# nested dict is much faster than df for most operations

def build_position_on_segment_info(n: int) -> list[list[int]]:
    # The first list contains numbers from 0 to n-2
    original_list = list(range(n-1))
    result = []

    # Process lists until no more elements can be taken
    while original_list:
        result.append(original_list)
        # Generate the next list by taking every second element starting from index 1 (second element)
        original_list = original_list[1::2]
            
    return result

# Example
print(build_position_on_segment_info(9))

