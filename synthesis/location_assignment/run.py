import os
import sys
import logging
import yaml
from datetime import datetime
from utils.logger import setup_logging
from utils.config import Config
from utils.stats_tracker import StatsTracker

logger = logging.getLogger(__name__)


def main(output_folder, project_root):
    """Main function for the step."""
    # Initialize Config
    config = Config(output_folder, project_root)

    # Determine step name dynamically (assumes folder structure: steps/<step_name>/run.py)
    step_name = os.path.basename(os.path.dirname(__file__))

    # Initialize StatsTracker for this step
    stats_tracker = StatsTracker(output_folder)

    logging.info(f"▶ Running step: {step_name}...")

    time_start = datetime.now()

    # Simulate step execution (Replace this with actual logic)
    # Do actual processing...

    time_end = datetime.now()
    time_diff = (time_end - time_start).total_seconds()

    # Log execution time for this step
    stats_tracker.log(step_name, time_diff)

    logging.info(f"✅ {step_name} completed in {time_diff:.2f} seconds.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: run.py <output_folder> <project_root>")
        sys.exit(1)

    output_folder = sys.argv[1]
    project_root = sys.argv[2]
    main(output_folder, project_root)
