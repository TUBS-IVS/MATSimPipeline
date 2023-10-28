import logging

import pandas as pd

from pipelines.common.data_frame_processor import DataFrameProcessor

logger = logging.getLogger(__name__)


class PopulationFrameProcessor(DataFrameProcessor):
    def __init__(self, population_frame, df: pd.DataFrame):
        super().__init__(df)
        self.population_frame = population_frame

    def process(self):
        return self.population_frame
