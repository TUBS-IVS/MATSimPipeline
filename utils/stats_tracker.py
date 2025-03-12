import os
import yaml
import numpy as np

from utils.logger import logging

logger = logging.getLogger("stats_tracker")

class StatsTracker:
    def __init__(self, output_folder):
        """Initialize StatsTracker, loading existing stats if available."""
        self.output_folder = output_folder
        self.stats_file = os.path.join(output_folder, "stats.yaml")
        self.stats = self._load_stats()

    def _load_stats(self):
        """Load stats from the stats.yaml file if it exists."""
        if os.path.exists(self.stats_file):
            with open(self.stats_file, "r") as f:
                try:
                    return yaml.safe_load(f) or {}
                except yaml.YAMLError as e:
                    logger.error(f"Error reading stats file: {e}")
        return {}

    def _get_nested_dict(self, composite_key, create_missing=True):
        """
        Traverse the stats dictionary following a dot-separated key.
        If `create_missing` is True, missing keys are initialized.
        """
        keys = composite_key.split('.')
        d = self.stats
        for key in keys[:-1]:
            if key not in d:
                if create_missing:
                    d[key] = {}
                else:
                    return None  # If missing and no creation allowed, return None
            d = d[key]
        return d, keys[-1]

    def increment(self, stat):
        """Increase a counter stat in a nested structure."""
        d, last_key = self._get_nested_dict(stat)

        if d is not None:
            if last_key in d and isinstance(d[last_key], int):
                d[last_key] += 1
            else:
                logger.debug(f"Stat {stat} not found, initializing to 1.")
                d[last_key] = 1

    def log(self, stat, value):
        """Log a value under a stat (appends to a list) in a nested structure."""
        d, last_key = self._get_nested_dict(stat)

        if d is not None:
            if last_key in d and isinstance(d[last_key], list):
                d[last_key].append(value)
            else:
                logger.debug(f"Stat {stat} not found, creating a new list.")
                d[last_key] = [value]

    def get_stats(self):
        """Return the current nested stats dictionary."""
        return self.stats

    def reset(self):
        """Clear all stats."""
        self.stats = {}
        logger.debug("Stats reset.")

    def print_stats(self):
        """Print all collected stats to the log."""
        for stat, value in self.stats.items():
            logger.info(f"{stat}: {value}")

    def write_stats(self):
        """Writes the current stats to the stats.yaml file as a nested structure."""
        cleaned_stats = self._clean_dict(self.stats)

        with open(self.stats_file, "w") as f:
            yaml.safe_dump(cleaned_stats, f, sort_keys=False)

        logger.info(f"Stats updated in {self.stats_file}")

    def _clean_dict(self, data):
        """Recursively cleans dictionary values for YAML serialization."""
        if isinstance(data, dict):
            return {k: self._clean_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_dict(v) for v in data if v is not None]  # Remove None values
        elif isinstance(data, np.integer):  # Convert NumPy int types to standard int
            return int(data)
        elif isinstance(data, np.floating):  # Convert NumPy float types to standard float
            return float(data)
        elif isinstance(data, np.ndarray):  # Convert NumPy arrays to lists
            return data.tolist()
        return data
