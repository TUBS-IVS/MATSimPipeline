import matplotlib.pyplot as plt
import numpy as np

# Example data
locations = np.array([
    [40.7128, -74.0060],  # New York
    [34.0522, -118.2437],  # Los Angeles
    [41.8781, -87.6298],   # Chicago
    [29.7604, -95.3698]    # Houston
])

target_locations = np.array([
    [37.7749, -122.4194],  # San Francisco
    [47.6062, -122.3321]   # Seattle
])

def plot_locations(locations, target_locations=None, show_lines=True):
    plt.figure(figsize=(10, 6))
    
    # Plot the locations
    plt.scatter(locations[:, 1], locations[:, 0], c='blue', marker='o', label='Locations')
    
    if target_locations is not None:
        # Plot the target locations
        plt.scatter(target_locations[:, 1], target_locations[:, 0], c='red', marker='x', label='Target Locations')
    
    if show_lines:
        # Draw lines between the locations
        for i in range(len(locations) - 1):
            plt.plot([locations[i, 1], locations[i + 1, 1]], [locations[i, 0], locations[i + 1, 0]], 'b-')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.title('Locations with Paths')
    plt.show()

# Plot locations with lines and target locations
plot_locations(locations, target_locations, show_lines=True)
