from typing import Dict, Union, List
from utils.logger import logging

logger = logging.getLogger(__name__)


class StatsTracker:
    def __init__(self):
        self.stats: Dict[str, Union[int, List[Union[int, float, str]]]] = {}

    def increment(self, stat: str) -> None:
        if stat in self.stats and isinstance(self.stats[stat], int):
            self.stats[stat] += 1
            logging.debug(f"{stat} incremented to {self.stats[stat]}")
        else:
            logging.debug(f"Stat {stat} not found in stats, creating it or resetting to 1.")
            self.stats[stat] = 1

    def log(self, stat: str, value: Union[int, float, str]) -> None:
        if stat in self.stats and isinstance(self.stats[stat], list):
            self.stats[stat].append(value)
            logging.debug(f"Value {value} added to {stat}")
        else:
            logging.debug(f"Stat {stat} not found in stats, creating it as a list with the initial value.")
            self.stats[stat] = [value]

    def get_stats(self) -> Dict[str, Union[int, List[Union[int, float, str]]]]:
        return self.stats

    def reset(self) -> None:
        self.stats = {}
        logging.debug("Stats reset.")

    def print_stats(self) -> None:
        for stat, value in self.stats.items():
            logging.info(f"{stat}: {value}")


stats_tracker = StatsTracker()  # Global singleton instance

