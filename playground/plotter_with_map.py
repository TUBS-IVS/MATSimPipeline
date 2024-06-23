import folium
import numpy as np

# Generate example data for locations in Hanover
# Hanover coordinates approximately (52.3759, 9.7320)
main_locations = [
    (52.3759, 9.7320),  # Example central point in Hanover
    (52.3791, 9.7390),  # Another point in Hanover
    (52.3730, 9.7190),  # Another point in Hanover
    (52.3720, 9.7430)   # Another point in Hanover
]

# Generate random target locations around Hanover
num_targets = 50
target_locations = [(52.3759 + np.random.uniform(-0.01, 0.01), 9.7320 + np.random.uniform(-0.01, 0.01)) for _ in range(num_targets)]

# Create a map centered around Hanover
m = folium.Map(location=(52.3759, 9.7320), zoom_start=13)

# Add main locations to the map as dots
for loc in main_locations:
    folium.CircleMarker(
        location=loc,
        radius=5,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.7,
        popup='Main Location'
    ).add_to(m)

# Add target locations to the map as dots
for target in target_locations:
    folium.CircleMarker(
        location=target,
        radius=3,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        popup='Target Location'
    ).add_to(m)

# Draw lines only between the main locations
for i in range(len(main_locations) - 1):
    folium.PolyLine(locations=[main_locations[i], main_locations[i + 1]], color='blue').add_to(m)

# Save the map to an HTML file
m.save('hanover_map.html')

# Display the map in a Jupyter notebook (if you're using one)
m
