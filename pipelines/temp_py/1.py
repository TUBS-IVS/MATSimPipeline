import numpy as np
from typing import Dict, Any

def reformat_locations(locations_data: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, np.ndarray]]:
    reformatted_data = {}

    for purpose, locations in locations_data.items():
        names = []
        coordinates = []
        capacities = []

        for location_id, location_details in locations.items():
            names.append(location_details['name'])
            coordinates.append(location_details['coordinates'])
            capacities.append(location_details['capacity'])

        reformatted_data[purpose] = {
            'names': np.array(names, dtype=object),
            'coordinates': np.array(coordinates, dtype=float),
            'capacities': np.array(capacities, dtype=float)
        }
    
    return reformatted_data

# Example usage
locations_data = {
    'shop': {
        'loc_1': {'coordinates': np.array([52.370216, 4.895168]), 'capacity': 150, 'name': 'Shop A'},
        'loc_2': {'coordinates': np.array([48.856613, 2.352222]), 'capacity': 200, 'name': 'Shop B'}
    },
    'warehouse': {
        'loc_3': {'coordinates': np.array([34.052235, -118.243683]), 'capacity': 300, 'name': 'Warehouse A'}
    }
}

reformatted_data = reformat_locations(locations_data)
print(reformatted_data)
