#!/usr/bin/env python

"""
Entry point for running the pipeline.
No changes are needed here. Use the config.yaml!
"""

import cProfile
import pstats
import io
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
        Runs each pipeline step sequentially as specified in execution_order, with optional profiling.
        After all steps complete, the used_config.yaml file will reflect settings used by all steps.
        """
        profile_enabled = self.config_obj.get("settings.profiling.enabled", default=False)
        save_txt = self.config_obj.get("settings.profiling.save_txt", default=True)
        save_raw = self.config_obj.get("settings.profiling.save_raw", default=False)

        profiler = None
        if profile_enabled:
            logging.info("Profiling enabled — running with cProfile.")
            profiler = cProfile.Profile()
            profiler.enable()

        execution_order = self.config_obj.get("execution")
        for step_name in execution_order:
            if not self.run_step(step_name):
                logging.error(f"Step {step_name} failed. Stopping pipeline.")
                return

        if profile_enabled and profiler:
            profiler.disable()

            if save_txt:
                stats_path = os.path.join(self.output_folder, "profile_stats.txt")
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
                ps.print_stats()
                with open(stats_path, "w") as f:
                    f.write(s.getvalue())
                logging.info(f"Profiling summary saved to {stats_path}")

            if save_raw:
                raw_path = os.path.join(self.output_folder, "profile_stats.prof")
                profiler.dump_stats(raw_path)
                logging.info(f"Raw profiler data saved to {raw_path}")

        logging.info("Pipeline completed successfully.")
        return self.output_folder

if __name__ == "__main__":
    runner = PipelineRunner()
    runner.run()
