#!/usr/bin/env python

"""
Entry point for running the pipeline.
No changes are needed here. Use the config.yaml!
"""

import os
import logging
import subprocess
from datetime import datetime
from utils.config import Config
from utils.logger import setup_logging

class PipelineRunner:
    def __init__(self, config_yaml="config.yaml", project_root=None):
        """
        Initializes the pipeline, sets up logging, and prepares configuration files.
        """
        self.config_yaml = config_yaml
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = project_root

        self.output_folder = self.create_output_folder()

        self.config_obj = Config(self.output_folder, self.project_root, self.config_yaml)

        console_level = self.config_obj.get("settings.logging.console_level")
        file_level = self.config_obj.get("settings.logging.file_level")
        setup_logging(self.output_folder, console_level, file_level)

    def create_output_folder(self):
        """Creates a new output folder based on the current timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(self.project_root, "output", timestamp)
        os.makedirs(folder, exist_ok=False)
        return folder

    def run_step(self, step_name):
        """
        Runs a single pipeline step via subprocess.
        Instead of passing the full Config object, it passes the paths to the updated config file and used config file.
        """
        logging.info(f"Running step: {step_name}...")

        step_script = self.config_obj.get(f"{step_name}.script")

        command = [os.sys.executable, step_script, self.output_folder, self.project_root, self.config_yaml]

        result = subprocess.run(command, text=True)
        if result.returncode == 0:
            logging.info(f"{step_name} completed successfully.")
            return True
        else:
            logging.error(f"Error in {step_name}: {result.stderr}")
            return False

    def run(self):
        """
        Runs each pipeline step sequentially as specified in execution_order.
        After all steps complete, the used_config.yaml file will reflect settings used by all steps.
        """
        execution_order = self.config_obj.get("execution")
        for step_name in execution_order:
            if not self.run_step(step_name):
                logging.error(f"Step {step_name} failed. Stopping pipeline.")
                return
        logging.info("Pipeline completed successfully.")
        return self.output_folder

if __name__ == "__main__":
    runner = PipelineRunner()
    runner.run()
