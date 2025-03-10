import os
import yaml
from build.lib.ivs_helpers import stats_tracker

from utils.logger import logging

logger = logging.getLogger(__name__)

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

    def increment(self, stat):
        """Increase a counter stat."""
        if stat in self.stats and isinstance(self.stats[stat], int):
            self.stats[stat] += 1
        else:
            logger.debug(f"Stat {stat} not found, initializing to 1.")
            self.stats[stat] = 1
        self._write_stats()

    def log(self, stat, value):
        """Log a value under a stat (appends to a list)."""
        if stat in self.stats and isinstance(self.stats[stat], list):
            self.stats[stat].append(value)
        else:
            logger.debug(f"Stat {stat} not found, creating a new list.")
            self.stats[stat] = [value]
        self._write_stats()

    def get_stats(self):
        """Return the current stats dictionary."""
        return self.stats

    def reset(self):
        """Clear all stats."""
        self.stats = {}
        self._write_stats()
        logger.debug("Stats reset.")

    def print_stats(self):
        """Print all collected stats to the log."""
        for stat, value in self.stats.items():
            logger.info(f"{stat}: {value}")

    def _write_stats(self):
        """Writes the current stats to the stats.yaml file."""
        with open(self.stats_file, "w") as f:
            yaml.safe_dump(self.stats, f, sort_keys=False)
        logger.info(f"Stats updated in {self.stats_file}")

