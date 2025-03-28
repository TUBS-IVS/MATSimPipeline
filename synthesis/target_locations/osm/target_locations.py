import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
import logging
import json
from typing import Dict, Any

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class TargetLocations:
    """
    Spatial index of activity locations split by type.
    This class is used to quickly find the nearest activity locations for a given location.
    """

    def __init__(self, geopackage_path: str = None, shape_path: str = None, data: Dict[str, Dict[str, np.ndarray]] = None):
        if geopackage_path and shape_path:
            # Read and process the GeoPackage
            df = self.read_geopackage(geopackage_path, shape_path)
            self.data: Dict[str, Dict[str, np.ndarray]] = self.reformat_locations_from_df(df)
        elif data:
            self.data = data
        else:
            raise ValueError("Either geopackage_path and shape_path or data must be provided.")

        self.indices: Dict[str, KDTree] = {}

        for type, pdata in self.data.items():
            logger.debug(f"Constructing spatial index for {type} ...")
            self.indices[type] = KDTree(pdata["coordinates"])

    @staticmethod
    def read_geopackage(geopackage_path: str, shape_path: str) -> pd.DataFrame:
        # Read the GeoPackage
        gdf = gpd.read_file(geopackage_path)

        # Ensure the GeoDataFrame is in EPSG:25832
        if gdf.crs != "EPSG:25832":
            gdf = gdf.to_crs("EPSG:25832")

        # Read the shapefile to limit the region
        shape = gpd.read_file(shape_path)

        # Ensure the shapefile GeoDataFrame is in EPSG:25832
        if shape.crs != "EPSG:25832":
            shape = shape.to_crs("EPSG:25832")

        # Clip the GeoDataFrame to the shapefile region
        gdf = gpd.clip(gdf, shape)

        # Extract necessary columns and handle missing values efficiently
        gdf['x'] = gdf.geometry.x
        gdf['y'] = gdf.geometry.y

        # Exploding 'tags' column if it contains a dictionary
        if 'tags' in gdf.columns:
            tags_df = pd.json_normalize(gdf['tags'])
            gdf = pd.concat([gdf.drop(columns=['tags']), tags_df], axis=1)

        # Selecting relevant columns and renaming them
        columns = ['id', 'name', 'x', 'y', 'capa', 'activities']
        missing_columns = set(columns) - set(gdf.columns)
        for col in missing_columns:
            gdf[col] = np.nan

        df = gdf[columns].copy()
        df['capa'] = df['capa'].fillna(0).astype(int)
        df = df.rename(columns={'id': 'identifier', 'name': 'name', 'capa': 'capacity'})

        return df

    @staticmethod
    def reformat_locations_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        reformatted_data = {}

        # Process each row to extract purposes from activities using itertuples
        for row in df.itertuples(index=False):
            activities = row.activities.split(',')
            for purpose in activities:
                if purpose not in reformatted_data:
                    reformatted_data[purpose] = {
                        'identifiers': [],
                        'names': [],
                        'coordinates': [],
                        'potentials': []
                    }

                reformatted_data[purpose]['identifiers'].append(row.identifier)
                reformatted_data[purpose]['names'].append(row.name)
                reformatted_data[purpose]['coordinates'].append(np.array([row.x, row.y]))
                reformatted_data[purpose]['potentials'].append(row.capacity)

        # Convert lists to numpy arrays
        for purpose in reformatted_data:
            reformatted_data[purpose]['identifiers'] = np.array(reformatted_data[purpose]['identifiers'], dtype=object)
            reformatted_data[purpose]['names'] = np.array(reformatted_data[purpose]['names'], dtype=str)
            reformatted_data[purpose]['coordinates'] = np.array(reformatted_data[purpose]['coordinates'], dtype=float)
            reformatted_data[purpose]['potentials'] = np.array(reformatted_data[purpose]['potentials'], dtype=float)

        return reformatted_data

    def save_reformatted_data(self, file_path: str):
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            purpose: {
                key: value.tolist() for key, value in data.items()
            } for purpose, data in self.data.items()
        }
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f)
        logger.debug(f"Reformatted data saved to {file_path}")

    @classmethod
    def load_reformatted_data(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Convert lists back to numpy arrays
        for purpose in data:
            for key in data[purpose]:
                data[purpose][key] = np.array(data[purpose][key])
        return cls(data=data)

import os
import time

os.chdir(r"/")
# Example usage
start_time = time.time()
geopackage_path = r"C:\Users\petre\Documents\GitHub\osmox\LOWER_SAXONY\niedersachsen_epsg_25832.gpkg"
shape_path = r"C:\Users\petre\Documents\GitHub\MATSimPipeline\data\shapes\boundaries_hannover.shp"
target_locations = TargetLocations(geopackage_path, shape_path)
print(target_locations.data)
print(target_locations.indices)
edit_end_time = time.time()
print(f"Time taken to load and reformat data: {edit_end_time - start_time:.2f} seconds")


# Save reformatted data
target_locations.save_reformatted_data(r"C:\Users\petre\Documents\GitHub\MATSimPipeline\playground\reformatted_data2.json")
save_time = time.time()
print(f"Time taken to save reformatted data: {save_time - edit_end_time:.2f} seconds")

# Load reformatted data
loaded_target_locations = TargetLocations.load_reformatted_data(r"/playground/reformatted_data2.json")
print(loaded_target_locations.data)
print(loaded_target_locations.indices)
end_time = time.time()
print(f"Time taken to load reformatted data: {end_time - save_time:.2f} seconds")