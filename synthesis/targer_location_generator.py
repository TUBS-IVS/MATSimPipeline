"""Produces target locations with potentials from pretty much any source."""

from utils.logger import logging

logger = logging.getLogger(__name__)
import requests
import numpy as np
import unicodedata


class OSMGetter:
    def __init__(self, query, overpass_url="http://overpass-api.de/api/interpreter"):
        self.overpass_url = overpass_url
        self.query = query

    def get_data(self):
        data = None
        try:
            response = requests.get(self.overpass_url, params={"data": query})
            response.raise_for_status()  # raise an HTTPError on bad status
            data = response.json()
            logger.info(f"Overpass request successful: {response.status_code}")
            logger.debug(f"Overpass response: {data}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Overpass request failed: {e}")

        return data
    
class OSMDataProcessor:
    def __init__(self, data):
        self.data = data

    def get_purposes(self, element):
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
                if value in purpose_remap[key]:
                    return purpose_remap[key][value]
                elif "*" in purpose_remap[key]:
                    return purpose_remap[key]["*"]
        return ["Unknown"]

    def format_name(self, name):
        # Remove diacritics and replace spaces with hyphens
        formatted_name = (
            unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
        )
        formatted_name = formatted_name.replace(" ", "-")
        # Truncate to first 12 characters
        return formatted_name[:12]

    def process_data(self):
        locations_data = {}
        for element in self.data["elements"]:
            if "tags" in element:
                name = element["tags"].get("name", "Unnamed")
                lat = float(element.get("lat", np.nan))
                lon = float(element.get("lon", np.nan))
                capacity = int(
                    element["tags"].get("capacity", 0)
                )  # Assuming capacity is an integer

                purposes = self.get_purposes(element)

                # Get the unique ID of the location from Overpass API
                location_id = str(element["id"])

                # Format and truncate the name
                formatted_name = self.format_name(name)

                # Add the location to each purpose in the purpose dictionary
                if lat and lon:
                    for purpose in purposes:
                        if purpose not in locations_data:
                            locations_data[purpose] = {}
                        # Store the location data along with its capacity and formatted name
                        locations_data[purpose][location_id] = {
                            "coordinates": np.array([lat, lon]),
                            "capacity": capacity,
                            "name": formatted_name,
                        }

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
bounding_box = [52.3167, 9.6869, 52.4397, 9.8524]  # Hannover
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
  node["building"="kindergarden"];
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
  node["amenity"="vetinary"];
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
  node["landuse"="quary"];
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
);
out body;
>;
out skel qt;
"""



# The purpose remapping dictionary
purpose_remap = {
    "building": {
        "hotel": ["work"],
        "commercial": ["shop", "work", "delivery"],
        "retail": ["shop", "work", "delivery"],
        "supermarket": ["shop", "shop_daily", "dining", "work", "delivery"],
        "industrial": ["work", "delivery"],
        "office": ["work"],
        "warehouse": ["work", "depot", "delivery"],
        "bakehouse": ["work", "depot", "delivery"],
        "firestation": ["work"],
        "government": ["work"],
        "cathedral": ["religious"],
        "chapel": ["religious"],
        "church": ["religious"],
        "mosque": ["religious"],
        "religious": ["religious"],
        "shrine": ["religious"],
        "synagogue": ["religious"],
        "temple": ["religious"],
        "hospital": ["medical", "work"],
        "veterinary": ["p_business", "work"],
        "kindergarden": ["edu_kiga", "work"],
        "school": ["edu_prim", "work"],
        "university": ["edu_higher", "work"],
        "college": ["edu_higher", "work"],
        "sports_hall": ["leisure", "work"],
        "stadium": ["leisure", "work"],
    },
    "amenity": {
        "bar": ["leisure", "work", "delivery", "dining"],
        "pub": ["leisure", "work", "delivery", "dining"],
        "cafe": ["leisure", "work", "delivery", "dining"],
        "fast_food": ["work", "delivery", "dining"],
        "food_court": ["work", "delivery", "dining"],
        "ice_cream": ["work", "delivery", "dining"],
        "restaurant": ["work", "delivery", "dining"],
        "college": ["edu_higher", "work"],
        "kindergarten": ["edu_kiga", "work"],
        "language_school": ["edu_other", "work"],
        "library": ["leisure", "work"],
        "music_school": ["edu_other", "work"],
        "school": ["edu_prim", "leisure", "work"],
        "university": ["edu_higher", "work"],
        "bank": ["p_business", "work"],
        "clinic": ["medical", "work"],
        "dentist": ["medical", "work"],
        "doctors": ["medical", "work"],
        "hospital": ["medical", "work"],
        "pharmacy": ["shop", "work"],
        "social_facility": ["medical", "work"],
        "vetinary": ["p_business", "work"],
        "arts_centre": ["leisure", "work"],
        "casino": ["leisure", "work"],
        "cinema": ["leisure", "work"],
        "community_centre": ["leisure", "work"],
        "gambling": ["leisure", "work"],
        "studio": ["leisure", "work"],
        "theatre": ["leisure", "work"],
        "courthouse": ["p_business", "work"],
        "crematorium": ["p_business", "work"],
        "embassy": ["p_business", "work"],
        "fire_station": ["work"],
        "funeral_hall": ["p_business", "work"],
        "internet_cafe": ["leisure", "work"],
        "marketplace": ["shop", "shop_daily", "work", "dining", "delivery"],
        "place_of_worship": ["religious"],
        "police": ["p_business", "work"],
        "post_box": ["p_business", "work"],
        "post_depot": ["p_business", "work"],
        "post_office": ["p_business", "work"],
        "prison": ["p_business", "work"],
        "townhall": ["p_business", "work"],
    },
    "landuse": {
        "commercial": ["shop", "work", "delivery"],
        "industrial": ["shop", "work", "delivery", "depot"],
        "retail": ["shop", "work", "delivery"],
        "depot": ["depot"],
        "port": ["depot"],
        "quary": ["depot"],
        "religious": ["religious"],
    },
    "leisure": {
        "adult_gaming_centre": ["leisure", "work"],
        "amusement_arcade": ["leisure", "work"],
        "beach_resort": ["leisure"],
        "dance": ["leisure", "work"],
        "escape_game": ["leisure", "work"],
        "fishing": ["leisure"],
        "fitness_centre": ["leisure", "work"],
        "fitness_station": ["leisure"],
        "garden": ["leisure"],
        "horse_riding": ["leisure", "work"],
        "ice_rink": ["leisure", "work"],
        "marina": ["leisure", "work"],
        "miniature_golf": ["leisure"],
        "nature_reserve": ["leisure"],
        "park": ["leisure"],
        "pitch": ["leisure"],
        "playground": ["leisure"],
        "sports_centre": ["leisure", "work"],
        "stadium": ["leisure", "work"],
        "swimming_pool": ["leisure", "work"],
        "track": ["leisure"],
        "water_park": ["leisure", "work"],
    },
    "medicalcare": {"*": ["medical", "work"]},
    "healthcare": {"*": ["medical", "work"]},
    "office": {
        "*": ["work"],
        "tax_advisor": ["p_business", "work"],
        "insurance": ["p_business", "work"],
    },
    "government": {
        "*": ["work"],
        "tax": ["p_business", "work"],
        "register_office": ["p_business", "work"],
    },
    "shop": {
        "*": ["shop", "work"],
        "supermarket": ["shop_daily", "work"],
        "convenience": ["shop_daily", "work"],
        "chemist": ["shop_daily", "work"],
        "bakery": ["shop_daily", "work"],
        "deli": ["shop_daily", "work"],
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


def assign_random_potentials():
    """Assigns random potentials to locations."""

data = OSMGetter(query).get_data()
locations_data = OSMDataProcessor(data).process_data()
print(locations_data)