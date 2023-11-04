from utils import matsim_pipeline_setup
from pipelines.common.data_frame_processor import DataFrameProcessor as dfp
from utils.logger import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    matsim_pipeline_setup.create_output_directory()


rules = [rule1, rule2, rule3, rule4, rule5]

updated_df = dfp.safe_apply_rules(df, rules)