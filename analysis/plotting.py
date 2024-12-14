import folium
from shapely.geometry import Point
from pyproj import Transformer

def plot_person_plan(person_id, hoerl_df=None, petre_df=None):
    if hoerl_df is None and petre_df is None:
        raise ValueError("At least one dataframe (hoerl_df or petre_df) must be provided.")

    # Set up a transformer to convert EPSG 25832 to EPSG 4326
    transformer = Transformer.from_crs("epsg:25832", "epsg:4326", always_xy=True)

    def transform_point(point):
        if isinstance(point, Point):
            lon, lat = transformer.transform(point.x, point.y)
            return [lat, lon]  # Folium expects [latitude, longitude]
        raise ValueError(f"Invalid location format: {point}")

    m = None

    # Helper function to create markers
    def add_marker(location, popup_content, color, icon):
        folium.Marker(
            location=location,
            popup=popup_content,
            icon=folium.Icon(color=color, icon=icon)
        ).add_to(m)

    # Plot Hoerl's plan
    if hoerl_df is not None:
        person_data = hoerl_df[hoerl_df['unique_person_id'] == person_id]
        if not person_data.empty:
            first_location = transform_point(person_data.iloc[0]['from_location'])
            last_location = transform_point(person_data.iloc[-1]['to_location'])
            m = folium.Map(location=first_location, zoom_start=12) if m is None else m

            # Add marker for start location
            add_marker(first_location, "Start Location", "blue", "play")

            for _, row in person_data.iterrows():
                from_loc = transform_point(row['from_location'])
                to_loc = transform_point(row['to_location'])

                # # Add marker for from_location
                # add_marker(
                #     from_loc,
                #     f"Leg ID: {row['unique_leg_id']}<br>Activity Type: {row['activity_to_internal']}<br>"
                #     f"From Activity: {row['activity_from_internal']}<br>Home Loc: {row['home_location']}<br>"
                #     f"From Location: {row['from_location']}<br>To Location: {row['to_location']}<br>"
                #     f"Coords: {from_loc}<br>Main Activity: {row['is_main_activity']}",
                #     "blue",
                #     "info-sign"
                # )

                # Add marker for to_location
                add_marker(
                    to_loc,
                    f"Leg ID: {row['unique_leg_id']}<br>Activity Type: {row['activity_to_internal']}<br>"
                    f"From Activity: {row['activity_from_internal']}<br>Home Loc: {row['home_location']}<br>"
                    f"From Location: {row['from_location']}<br>To Location: {row['to_location']}<br>"
                    f"Coords: {from_loc}<br>Main Activity: {row['is_main_activity']}"
                    f"Mirrors main activity: {row['mirrors_main_activity']}",
                    "blue",
                    "info-sign"
                )

                # Add a line between the locations
                folium.PolyLine([from_loc, to_loc], color="blue", weight=3.5, opacity=1).add_to(m)

            # Add marker for end location
            add_marker(last_location, "End Location", "blue", "stop")

    # Plot Petre's plan
    if petre_df is not None:
        person_data = petre_df[petre_df['unique_person_id'] == person_id]
        if not person_data.empty:
            if m is None:
                first_location = transform_point(person_data.iloc[0]['from_location'])
                last_location = transform_point(person_data.iloc[-1]['to_location'])
                m = folium.Map(location=first_location, zoom_start=12)

            # Add marker for start location
            add_marker(first_location, "Start Location", "green", "play")

            for _, row in person_data.iterrows():
                from_loc = transform_point(row['from_location'])
                to_loc = transform_point(row['to_location'])

                # # Add marker for from_location
                # add_marker(
                #     from_loc,
                #     f"Leg ID: {row['unique_leg_id']}<br>Activity Type: {row['activity_to_internal']}<br>"
                #     f"From Activity: {row['activity_from_internal']}<br>Home Loc: {row['home_location']}<br>"
                #     f"From Location: {row['from_location']}<br>To Location: {row['to_location']}<br>"
                #     f"Coords: {from_loc}<br>Main Activity: {row['is_main_activity']}",
                #     "green",
                #     "info-sign"
                # )

                # Add marker for to_location
                add_marker(
                    to_loc,
                    f"Leg ID: {row['unique_leg_id']}<br>Activity Type: {row['activity_to_internal']}<br>"
                    f"From Activity: {row['activity_from_internal']}<br>Home Loc: {row['home_location']}<br>"
                    f"From Location: {row['from_location']}<br>To Location: {row['to_location']}<br>"
                    f"Coords: {from_loc}<br>Main Activity: {row['is_main_activity']}"
                    f"Mirrors main activity: {row['mirrors_main_activity']}",
                    "green",
                    "info-sign"
                )

                # Add a line between the locations
                folium.PolyLine([from_loc, to_loc], color="green", weight=2.5, opacity=1).add_to(m)

            # Add marker for end location
            add_marker(last_location, "End Location", "green", "stop")

    return m