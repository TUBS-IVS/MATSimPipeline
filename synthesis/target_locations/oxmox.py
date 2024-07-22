import json
import os
from utils import settings as s, pipeline_setup
from utils.logger import logging

logger = logging.getLogger(__name__)

oxmox_config = {
    "$schema": "../src/osmox/schema.json",
    "filter": {
        "building": [
            "apartments",
            "bungalow",
            "detached",
            "dormitory",
            "hotel",
            "residential",
            "semidetached_house",
            "terrace",
            "commercial",
            "retail",
            "supermarket",
            "industrial",
            "office",
            "warehouse",
            "bakehouse",
            "firestation",
            "government",
            "cathedral",
            "chapel",
            "church",
            "mosque",
            "religious",
            "shrine",
            "synagogue",
            "temple",
            "hospital",
            "kindergarden",
            "school",
            "university",
            "college",
            "sports_hall",
            "stadium",
            "veterinary",
            "yes"
        ],
        "amenity": [
            "bar",
            "pub",
            "cafe",
            "fast_food",
            "food_court",
            "ice_cream",
            "restaurant",
            "college",
            "kindergarten",
            "language_school",
            "library",
            "music_school",
            "school",
            "university",
            "bank",
            "clinic",
            "dentist",
            "doctors",
            "hospital",
            "pharmacy",
            "social_facility",
            "veterinary",
            "arts_centre",
            "casino",
            "cinema",
            "community_centre",
            "gambling",
            "studio",
            "theatre",
            "courthouse",
            "crematorium",
            "embassy",
            "fire_station",
            "funeral_hall",
            "internet_cafe",
            "marketplace",
            "place_of_worship",
            "police",
            "post_box",
            "post_depot",
            "post_office",
            "prison",
            "townhall"
        ],
        "landuse": [
            "commercial",
            "industrial",
            "residential",
            "retail",
            "depot",
            "port",
            "quary",
            "religious"
        ],
        "leisure": [
            "adult_gaming_centre",
            "amusement_arcade",
            "beach_resort",
            "dance",
            "escape_game",
            "fishing",
            "fitness_centre",
            "fitness_station",
            "garden",
            "horse_riding",
            "ice_rink",
            "marina",
            "miniature_golf",
            "nature_reserve",
            "park",
            "pitch",
            "playground",
            "sports_centre",
            "stadium",
            "swimming_pool",
            "track",
            "water_park"
        ],
        "office": [
            "accountant",
            "tax_advisor",
            "insurance"
        ],
        "tourism": [
            "museum",
            "gallery",
            "zoo",
            "aquarium",
            "attraction",
            "theme_park"
        ],
        "medicalcare": ["*"],
        "healthcare": ["*"],
        "public_transport": ["*"],
        "highway": ["bus_stop"]
    },

    "object_features": ["units", "levels", "area", "floor_area"],

    "default_tags": [["building", "residential"]],

    "activity_mapping": {
        "building": {
            "apartments": [s.ACT_HOME],
            "bungalow": [s.ACT_HOME],
            "detached": [s.ACT_HOME],
            "dormitory": [s.ACT_HOME],
            "hotel": [s.ACT_WORK, s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                      s.ACT_UNSPECIFIED],
            "residential": [s.ACT_HOME],
            "semidetached_house": [s.ACT_HOME],
            "terrace": [s.ACT_HOME],
            "commercial": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                           s.ACT_MEETUP, s.ACT_UNSPECIFIED],
            "retail": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                       s.ACT_UNSPECIFIED],
            "supermarket": [s.ACT_SHOPPING, s.ACT_OTHER, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF,
                            s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "industrial": [s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "office": [s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
            "warehouse": [s.ACT_WORK, s.ACT_OTHER, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                          s.ACT_UNSPECIFIED],
            "bakehouse": [s.ACT_WORK, s.ACT_OTHER, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                          s.ACT_UNSPECIFIED],
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
            "kindergarden": [s.ACT_EARLY_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                             s.ACT_DAYCARE, s.ACT_ACCOMPANY_ADULT, s.ACT_UNSPECIFIED],
            "school": [s.ACT_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_DAYCARE,
                       s.ACT_ACCOMPANY_ADULT, s.ACT_UNSPECIFIED],
            "university": [s.ACT_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                           s.ACT_UNSPECIFIED],
            "college": [s.ACT_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "sports_hall": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_SPORTS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                            s.ACT_UNSPECIFIED],
            "stadium": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_SPORTS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                        s.ACT_UNSPECIFIED]
        },
        "amenity": {
            "bar": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                    s.ACT_MEETUP, s.ACT_UNSPECIFIED],
            "pub": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                    s.ACT_MEETUP, s.ACT_UNSPECIFIED],
            "cafe": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF,
                     s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
            "fast_food": [s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                          s.ACT_UNSPECIFIED],
            "food_court": [s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                           s.ACT_UNSPECIFIED],
            "ice_cream": [s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                          s.ACT_UNSPECIFIED],
            "restaurant": [s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                           s.ACT_MEETUP, s.ACT_UNSPECIFIED],
            "college": [s.ACT_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "kindergarten": [s.ACT_EARLY_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                             s.ACT_DAYCARE, s.ACT_ACCOMPANY_ADULT, s.ACT_UNSPECIFIED],
            "language_school": [s.ACT_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                                s.ACT_UNSPECIFIED],
            "library": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                        s.ACT_UNSPECIFIED],
            "music_school": [s.ACT_EDUCATION, s.ACT_LESSONS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                             s.ACT_UNSPECIFIED],
            "school": [s.ACT_EDUCATION, s.ACT_LEISURE, s.ACT_WORK, s.ACT_LESSONS, s.ACT_PICK_UP_DROP_OFF,
                       s.ACT_RETURN_JOURNEY, s.ACT_DAYCARE, s.ACT_ACCOMPANY_ADULT, s.ACT_UNSPECIFIED],
            "university": [s.ACT_EDUCATION, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                           s.ACT_UNSPECIFIED],
            "bank": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED, s.ACT_ERRANDS, s.ACT_OTHER],
            "clinic": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED, s.ACT_ERRANDS, s.ACT_OTHER],
            "dentist": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED, s.ACT_ERRANDS, s.ACT_OTHER],
            "doctors": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED, s.ACT_ERRANDS, s.ACT_OTHER],
            "hospital": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED, s.ACT_ERRANDS,s.ACT_OTHER],
            "pharmacy": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED, s.ACT_ERRANDS,s.ACT_OTHER],
            "social_facility": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                                s.ACT_UNSPECIFIED],
            "vetinary": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "arts_centre": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                            s.ACT_UNSPECIFIED],
            "casino": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                       s.ACT_UNSPECIFIED],
            "cinema": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "community_centre": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                                 s.ACT_UNSPECIFIED],
            "gambling": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "studio": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_LESSONS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                       s.ACT_UNSPECIFIED],
            "theatre": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                        s.ACT_UNSPECIFIED],
            "courthouse": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "crematorium": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                            s.ACT_UNSPECIFIED],
            "embassy": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "fire_station": [s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "funeral_hall": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                             s.ACT_UNSPECIFIED],
            "internet_cafe": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                              s.ACT_UNSPECIFIED],
            "marketplace": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF,
                            s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
            "place_of_worship": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                                 s.ACT_UNSPECIFIED],
            "police": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "post_box": [s.ACT_BUSINESS, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                         s.ACT_UNSPECIFIED],
            "post_depot": [s.ACT_BUSINESS, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                           s.ACT_UNSPECIFIED],
            "post_office": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                            s.ACT_UNSPECIFIED],
            "prison": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "townhall": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                         s.ACT_UNSPECIFIED]
        },
        "landuse": {
            "commercial": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                           s.ACT_UNSPECIFIED],
            "industrial": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF,
                           s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "residential": [s.ACT_HOME],
            "retail": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_ERRANDS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                       s.ACT_UNSPECIFIED],
            "depot": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "port": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "quary": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "religious": [s.ACT_OTHER, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED]
        },
        "leisure": {
            "adult_gaming_centre": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                                    s.ACT_MEETUP, s.ACT_UNSPECIFIED],
            "amusement_arcade": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                                 s.ACT_UNSPECIFIED],
            "beach_resort": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                             s.ACT_UNSPECIFIED],
            "dance": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                      s.ACT_UNSPECIFIED],
            "escape_game": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "fishing": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "fitness_centre": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                               s.ACT_UNSPECIFIED],
            "fitness_station": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "garden": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "horse_riding": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                             s.ACT_UNSPECIFIED],
            "ice_rink": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                         s.ACT_UNSPECIFIED],
            "marina": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "miniature_golf": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                               s.ACT_UNSPECIFIED],
            "nature_reserve": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                               s.ACT_UNSPECIFIED],
            "park": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP, s.ACT_UNSPECIFIED],
            "pitch": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "playground": [s.ACT_LEISURE, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                           s.ACT_UNSPECIFIED],
            "sports_centre": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                              s.ACT_UNSPECIFIED],
            "stadium": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                        s.ACT_UNSPECIFIED],
            "swimming_pool": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                              s.ACT_UNSPECIFIED],
            "track": [s.ACT_LEISURE, s.ACT_SPORTS, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "water_park": [s.ACT_LEISURE, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_MEETUP,
                           s.ACT_UNSPECIFIED]
        },
        "medicalcare": {
            "*": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED]
        },
        "healthcare": {
            "*": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED]
        },
        "office": {
            "*": [s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "tax_advisor": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY,
                            s.ACT_UNSPECIFIED],
            "insurance": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED]
        },
        "government": {
            "*": [s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY, s.ACT_UNSPECIFIED],
            "tax": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
            "register_office": [s.ACT_BUSINESS, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY]
        },
        "shop": {
            "*": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
            "supermarket": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
            "convenience": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
            "chemist": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
            "bakery": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY],
            "deli": [s.ACT_SHOPPING, s.ACT_WORK, s.ACT_PICK_UP_DROP_OFF, s.ACT_RETURN_JOURNEY]
        }
    }
}

def write_config_to_json(config, file_path):
    def replace_vars(obj):
        if isinstance(obj, dict):
            return {key: replace_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [replace_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("s."):
            var_name = obj.split(".")[1]
            return getattr(s, var_name, obj)
        else:
            return obj

    updated_config = replace_vars(config)

    with open(file_path, 'w') as json_file:
        json.dump(updated_config, json_file, indent=4)

os.chdir(pipeline_setup.PROJECT_ROOT)
output_dir = os.path.join(pipeline_setup.OUTPUT_DIR, 'config.json')
write_config_to_json(oxmox_config, output_dir)


def run_oxmox(config_file):
    logger.info(f"Running OXMox with config file: {config_file}")
    # Run OXMox here
    pass


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