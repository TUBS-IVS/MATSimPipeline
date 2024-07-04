"""Produces target locations with potentials from pretty much any source."""

from utils.logger import logging
from pyproj import Transformer
from utils import settings as s

logger = logging.getLogger(__name__)
import requests
import numpy as np
import unicodedata
import pickle
import folium
import random as rnd

class OSMGetter:
    def __init__(self, query, overpass_url="http://overpass-api.de/api/interpreter"):
        self.overpass_url = overpass_url
        self.query = query
        self.data = None

    def get_data(self):
        data = None
        try:
            response = requests.get(self.overpass_url, params={"data": self.query})
            response.raise_for_status()  # raise an HTTPError on bad status
            data = response.json()
            logger.info(f"Overpass request successful: {response.status_code}")
            logger.debug(f"Overpass response: {data}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Overpass request failed: {e}")

        self.data = data
        return data
    
    def plot_locations(self, map_filename='osm_locations_map.html', show=True):
        if self.data is None:
            logger.error("No data to plot. Please call get_data first.")
            return

        # Extract the main locations (latitude, longitude) and names from the data
        main_locations = []
        if 'elements' in self.data:
            for element in self.data['elements']:
                if 'lat' in element and 'lon' in element:
                    loc_name = element['tags'].get('name', 'Unnamed Location')
                    main_locations.append((element['lat'], element['lon'], loc_name))

        if not main_locations:
            logger.warning("No locations found in data.")
            return

        # Create a map centered around the first location in the list
        first_location = main_locations[0] if main_locations else (52.3759, 9.7320)
        m = folium.Map(location=(first_location[0], first_location[1]), zoom_start=13)

        # Add main locations to the map as dots with popups showing location names
        for loc in main_locations:
            folium.CircleMarker(
                location=(loc[0], loc[1]),
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.7,
                popup=loc[2]  # Location name
            ).add_to(m)

        # Save the map to an HTML file
        m.save(map_filename)
        if show:
            import webbrowser
            webbrowser.open(map_filename)
            
class OSMDataProcessor:
    def __init__(self, data, purpose_remap):
        self.data = data
        self.purpose_remap = purpose_remap
        self.wgs_84_to_epsg25832_transformer = Transformer.from_crs("epsg:4326", "epsg:25832")
        self.epsg25832_to_wgs_84_transformer = Transformer.from_crs("epsg:25832", "epsg:4326")
        self.processed_locations_data = None

    def get_purposes(self, element):
        # Loop through possible keys to match elements with purposes
        for key in [
            "amenity",
            "building",
            "landuse",
            "leisure",
            "medicalcare",
            "healthcare",
            "office",
            "government",
            "shop",
        ]:
            if key in element["tags"]:
                value = element["tags"][key]
                if value in self.purpose_remap[key]:
                    return self.purpose_remap[key][value]
                elif "*" in self.purpose_remap[key]:
                    return self.purpose_remap[key]["*"]
        return [s.ACT_UNSPECIFIED]

    def format_name(self, name):
        # Remove diacritics and replace spaces with hyphens
        formatted_name = (
            unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
        )
        formatted_name = formatted_name.replace(" ", "-")
        # Truncate to first 20 characters
        return formatted_name[:20]

    def transform_coords(self, lon, lat):
        x, y = self.wgs_84_to_epsg25832_transformer.transform(lat, lon)
        return x, y

    def transform_elements_coords(self):
        # Transform the coordinates of all elements in the JSON data
        for element in self.data['elements']:
            if 'lon' in element and 'lat' in element:
                element['x'], element['y'] = self.transform_coords(element['lon'], element['lat'])
                del element['lon']
                del element['lat']

    def process_data(self):
        # Ensure coordinates are transformed before processing anything else
        self.transform_elements_coords()
        
        locations_data = {}
        
        for element in self.data["elements"]:
            if "tags" in element:
                name = element["tags"].get("name", "Unnamed")
                x = element.get("x", np.nan)
                y = element.get("y", np.nan)
                capacity = int(element["tags"].get("capacity", 0))  # Assuming capacity is an integer

                purposes = self.get_purposes(element)

                # Get the unique ID of the location from Overpass API
                location_id = str(element["id"])

                # Format and truncate the name
                formatted_name = self.format_name(name)

                # Add the location to each purpose in the purpose dictionary
                if not np.isnan(x) and not np.isnan(y):
                    for purpose in purposes:
                        if purpose not in locations_data:
                            locations_data[purpose] = {}
                        # Store the location data along with its capacity and formatted name
                        locations_data[purpose][location_id] = {
                            "coordinates": np.array([x, y]),
                            "capacity": capacity,
                            "name": formatted_name,
                        }

        self.processed_locations_data = locations_data
        return locations_data

# def get_city_bbox(city_name):
#     overpass_url = "http://overpass-api.de/api/interpreter"
#     query = f"""
#     [out:json];
#     area[name="{city_name}"];
#     (
#       node(area);
#       way(area);
#       relation(area);
#     );
#     out bb;
#     """
#     response = requests.get(overpass_url, params={'data': query})

#     if response.status_code != 200:
#         raise ValueError(f"Error fetching city bounding box: {response.status_code}")

#     data = response.json()

#     # Extract the bounding box from the response
#     if 'elements' in data and len(data['elements']) > 0:
#         element = data['elements'][0]
#         bbox = element['bounds']
#         return bbox
#     else:
#         raise ValueError("City bounding box not found.")


def expand_bbox(bbox, percentage):
    """Expands the bounding box by a percentage in all directions (so 10% will expand the box by 20% in total)."""
    min_lat = bbox["minlat"]
    min_lon = bbox["minlon"]
    max_lat = bbox["maxlat"]
    max_lon = bbox["maxlon"]

    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon

    expand_lat = lat_diff * (percentage / 100)
    expand_lon = lon_diff * (percentage / 100)

    new_bbox = {
        "minlat": min_lat - expand_lat,
        "minlon": min_lon - expand_lon,
        "maxlat": max_lat + expand_lat,
        "maxlon": max_lon + expand_lon,
    }

    return new_bbox


# Define the bounding box coordinates: [south, west, north, east]
# bounding_box = [52.3167, 9.6869, 52.4397, 9.8524]  # Hannover
bounding_box = [52.1786, 9.3082, 52.5551, 10.2145] # larger hanover
# bounding_box = [52.464, 13.207, 52.574, 13.527] # Berlin
# bounding_box = get_city_bbox("Hannover")
# bounding_box = expand_bbox(bounding_box, 20)
# print(bounding_box)

# Define the Overpass query with bounding box
query = f"""
[out:json][bbox:{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}];
(
  node["building"="hotel"];
  node["building"="commercial"];
  node["building"="retail"];
  node["building"="supermarket"];
  node["building"="industrial"];
  node["building"="office"];
  node["building"="warehouse"];
  node["building"="bakehouse"];
  node["building"="firestation"];
  node["building"="government"];
  node["building"="cathedral"];
  node["building"="chapel"];
  node["building"="church"];
  node["building"="mosque"];
  node["building"="religious"];
  node["building"="shrine"];
  node["building"="synagogue"];
  node["building"="temple"];
  node["building"="hospital"];
  node["building"="veterinary"];
  node["building"="kindergarten"];
  node["building"="school"];
  node["building"="university"];
  node["building"="college"];
  node["building"="sports_hall"];
  node["building"="stadium"];
  node["amenity"="bar"];
  node["amenity"="pub"];
  node["amenity"="cafe"];
  node["amenity"="fast_food"];
  node["amenity"="food_court"];
  node["amenity"="ice_cream"];
  node["amenity"="restaurant"];
  node["amenity"="college"];
  node["amenity"="kindergarten"];
  node["amenity"="language_school"];
  node["amenity"="library"];
  node["amenity"="music_school"];
  node["amenity"="school"];
  node["amenity"="university"];
  node["amenity"="bank"];
  node["amenity"="clinic"];
  node["amenity"="dentist"];
  node["amenity"="doctors"];
  node["amenity"="hospital"];
  node["amenity"="pharmacy"];
  node["amenity"="social_facility"];
  node["amenity"="veterinary"];
  node["amenity"="arts_centre"];
  node["amenity"="casino"];
  node["amenity"="cinema"];
  node["amenity"="community_centre"];
  node["amenity"="gambling"];
  node["amenity"="studio"];
  node["amenity"="theatre"];
  node["amenity"="courthouse"];
  node["amenity"="crematorium"];
  node["amenity"="embassy"];
  node["amenity"="fire_station"];
  node["amenity"="funeral_hall"];
  node["amenity"="internet_cafe"];
  node["amenity"="marketplace"];
  node["amenity"="place_of_worship"];
  node["amenity"="police"];
  node["amenity"="post_box"];
  node["amenity"="post_depot"];
  node["amenity"="post_office"];
  node["amenity"="prison"];
  node["amenity"="townhall"];
  node["landuse"="commercial"];
  node["landuse"="industrial"];
  node["landuse"="retail"];
  node["landuse"="depot"];
  node["landuse"="port"];
  node["landuse"="quarry"];
  node["landuse"="religious"];
  node["leisure"="adult_gaming_centre"];
  node["leisure"="amusement_arcade"];
  node["leisure"="beach_resort"];
  node["leisure"="dance"];
  node["leisure"="escape_game"];
  node["leisure"="fishing"];
  node["leisure"="fitness_centre"];
  node["leisure"="fitness_station"];
  node["leisure"="garden"];
  node["leisure"="horse_riding"];
  node["leisure"="ice_rink"];
  node["leisure"="marina"];
  node["leisure"="miniature_golf"];
  node["leisure"="nature_reserve"];
  node["leisure"="park"];
  node["leisure"="pitch"];
  node["leisure"="playground"];
  node["leisure"="sports_centre"];
  node["leisure"="stadium"];
  node["leisure"="swimming_pool"];
  node["leisure"="track"];
  node["leisure"="water_park"];
  node["tourism"="museum"];
  node["tourism"="gallery"];
  node["tourism"="zoo"];
  node["tourism"="aquarium"];
  node["tourism"="attraction"];
  node["tourism"="theme_park"];
  node["medicalcare"];
  node["healthcare"];
  node["office"];
  node["office"="tax_advisor"];
  node["office"="insurance"];
  node["government"];
  node["government"="tax"];
  node["government"="register_office"];
  node["shop"];
  node["shop"="supermarket"];
  node["shop"="convenience"];
  node["shop"="chemist"];
  node["shop"="bakery"];
  node["shop"="deli"];
  node["shop"="hairdresser"];
);
out body;
>;
out skel qt;
"""



    

# The purpose remapping dictionary
purpose_remap = {
    "building": {
        "hotel": [s.ACT_WORK, s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "commercial": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "retail": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "supermarket": [s.ACT_SHOPPING, s.ACT_OTHER, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "industrial": [s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "office": [s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "warehouse": [s.ACT_WORK, s.ACT_OTHER, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "bakehouse": [s.ACT_WORK, s.ACT_OTHER, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "firestation": [s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "government": [s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "cathedral": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "chapel": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "church": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "mosque": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "religious": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "shrine": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "synagogue": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "temple": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "hospital": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "veterinary": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "kindergarden": [s.ACT_EARLY_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_DAYCARE, s.ACT_ACCOMPANY_ADULT, s.ACT_UNSPECIFIED],
        "school": [s.ACT_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_DAYCARE, s.ACT_ACCOMPANY_ADULT, s.ACT_UNSPECIFIED],
        "university": [s.ACT_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "college": [s.ACT_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "sports_hall": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_SPORTS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "stadium": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_SPORTS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
    },
    "amenity": {
        "bar": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "pub": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "cafe": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "fast_food": [s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "food_court": [s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "ice_cream": [s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "restaurant": [s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "college": [s.ACT_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "kindergarten": [s.ACT_EARLY_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_DAYCARE, s.ACT_ACCOMPANY_ADULT, s.ACT_UNSPECIFIED],
        "language_school": [s.ACT_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "library": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "music_school": [s.ACT_EDUCATION, s.ACT_LESSONS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "school": [s.ACT_EDUCATION, s.ACT_LEISURE, s.ACT_WORK, s.ACT_LESSONS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_DAYCARE, s.ACT_ACCOMPANY_ADULT, s.ACT_UNSPECIFIED],
        "university": [s.ACT_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "bank": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "clinic": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "dentist": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "doctors": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "hospital": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "pharmacy": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "social_facility": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "vetinary": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "arts_centre": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "casino": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "cinema": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "community_centre": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "gambling": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "studio": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_LESSONS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "theatre": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "courthouse": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "crematorium": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "embassy": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "fire_station": [s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "funeral_hall": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "internet_cafe": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "marketplace": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "place_of_worship": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "police": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "post_box": [s.ACT_BUSINESS, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "post_depot": [s.ACT_BUSINESS, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "post_office": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "prison": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "townhall": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
    },
    "landuse": {
        "commercial": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "industrial": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "retail": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "depot": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "port": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "quary": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "religious": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
    },
    "leisure": {
        "adult_gaming_centre": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "amusement_arcade": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "beach_resort": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "dance": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "escape_game": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "fishing": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "fitness_centre": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "fitness_station": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "garden": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "horse_riding": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "ice_rink": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "marina": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "miniature_golf": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "nature_reserve": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "park": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "pitch": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "playground": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
        "sports_centre": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "stadium": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "swimming_pool": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "track": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "water_park": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
    },
    "medicalcare": {"*": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED]},
    "healthcare": {"*": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED]},
    "office": {
        "*": [s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "tax_advisor": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "insurance": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
    },
    "government": {
        "*": [s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
        "tax": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
        "register_office": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
    },
    "shop": {
        "*": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
        "supermarket": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
        "convenience": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
        "chemist": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
        "bakery": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
        "deli": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
    },
}






""" First, translate all locations with a total potential of 1 per loc. Then multiply this with any given cell or point potentials.
The potentials are later normalized to the demand.
Sum-preserve-round if needed."""


def assign_potentials_from_cell_data():
    """Assigns potentials to locations from cell data."""
    pass


def assign_potentials_from_point_data():
    """Assigns potentials to locations from point data."""
    raise NotImplementedError


def assign_random_capacities(locations_data):
    """Assigns random capacities to locations."""
    rng = np.random.default_rng(999)  # Creating a random number generator instance
    for purpose, locations in locations_data.items():
        for location_id, location in locations.items():
            location["capacity"] = rng.integers(1, 100)
    return locations_data
    
getter = OSMGetter(query)
data = getter.get_data()
getter.plot_locations()
logger.debug(f"OSM data: {data}")
locations_data = OSMDataProcessor(data, purpose_remap).process_data()
logger.debug(f"Processed OSM data: {locations_data}")

with open('locations_data.pkl', 'wb') as file:
    pickle.dump(locations_data, file)
    
with open('locations_data.pkl', 'rb') as file:
    locations_data = pickle.load(file)

locations_data = assign_random_capacities(locations_data)
logger.debug(f"Locations data with potentials: {locations_data}")

with open('locations_data_with_capacities.pkl', 'wb') as file:
    pickle.dump(locations_data, file)
    
print("Processed OSM Data:")
for purpose, locations in locations_data.items():
    print(f"\nPurpose: {purpose}")
    print(f"Number of locations: {len(locations)}")
