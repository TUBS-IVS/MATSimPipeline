import os
import geopandas as gpd
import pandas as pd
from utils import pipeline_setup

# Install osmox

# Run osmox with some geofabrik file and the config below

# Run_target_locations to convert the output geopackage to the expected csv format



## Currently:
# osmox.py makes a osmox config. The config is used to run osmox with a geofabrik file.
# The output is a geopackage file.
# The geopackage file is then converted to a json file with the target_locations.py class.
# That json file is then used to assign locations to the population.


def run_target_locations(**kwargs):
    raise NotImplementedError("This function is not implemented yet.")
