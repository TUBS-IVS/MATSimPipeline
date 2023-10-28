import os
from datetime import datetime
from utils.logger import logging
logger = logging.getLogger(__name__)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))  # Assuming matsim_pipeline_setup.py is one level down from the project root

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', current_time)


def create_output_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")
