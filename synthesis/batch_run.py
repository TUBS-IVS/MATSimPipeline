
"""
Run multiple instances (after each other) of the MATSim pipeline with different configurations.
For structured evaluations.
"""

import os
import yaml
import copy
from datetime import datetime
from run import PipelineRunner

PROJECT_BASE_DIR = r"C:\Users\petre\Documents\GitHub\MATSimPipeline"
BATCHED_RUNS_FOLDER = "batched_runs"
BASE_CONFIG_FILE = "config.yaml"
TEMP_CONFIG_FILE = "temp_config.yaml"

def load_base_config(config_path):
    """Loads the base config from the given YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_variations(base_config, parameter_variations):
    """Generates multiple variations of the base config by modifying specific parameters."""
    variations = []
    for param, values in parameter_variations.items():
        for value in values:
            config = copy.deepcopy(base_config)
            keys = param.split(".")
            current = config
            for key in keys[:-1]:
                current = current.setdefault(key, {})  # Ensure structure exists
            current[keys[-1]] = value
            variations.append((param, value, config))
    return variations

def run_pipeline_with_config(config, run_id):
    """Runs a pipeline run using a given config and returns the output folder."""
    temp_config_path = os.path.join(PROJECT_BASE_DIR, TEMP_CONFIG_FILE)
    with open(temp_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Starting pipeline run {run_id}.")

    runner = PipelineRunner(config_yaml=TEMP_CONFIG_FILE, project_root=PROJECT_BASE_DIR)
    output_folder = runner.run()

    print(f"Completed pipeline run {run_id}, output in {output_folder}")
    return output_folder

def save_batched_runs_list(runs_list):
    """Saves the list of all batched runs in a uniquely named file."""
    batched_runs_dir = os.path.join(PROJECT_BASE_DIR, BATCHED_RUNS_FOLDER)
    os.makedirs(batched_runs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_file = os.path.join(batched_runs_dir, f"batched_runs_{timestamp}.txt")

    with open(runs_file, "w") as f:
        for run in runs_list:
            f.write(run + "\n")

    print(f"Batched runs list saved to {runs_file}")

def main():
    base_config_path = os.path.join(PROJECT_BASE_DIR, BASE_CONFIG_FILE)
    base_config = load_base_config(base_config_path)
    batched_runs = []

    ##################### Parameter variations for testing
    parameter_variations = {
        "settings.num_cores": [2, 4, 8],
        # "steps.Population.settings.sample_size": [0.1, 0.25, 0.5],
        # "steps.MID_Enhancement.settings.id_columns.household_id": ["H_ID", "HH_ID"],
    }

    generated_configs = generate_variations(base_config, parameter_variations)

    # Run pipelines with generated variations
    for idx, (param, value, config) in enumerate(generated_configs, start=1):
        output_folder = run_pipeline_with_config(config, f"generated_{idx}")
        batched_runs.append(output_folder)
    #
    # ###################### Predefined YAMLs
    # predefined_yaml_paths = [
    #     os.path.join(PROJECT_BASE_DIR, "configs/run1.yaml"),
    #     os.path.join(PROJECT_BASE_DIR, "configs/run2.yaml"),
    #     os.path.join(PROJECT_BASE_DIR, "configs/run3.yaml"),
    # ]
    # # Run pipelines with predefined YAMLs
    # for idx, yaml_path in enumerate(predefined_yaml_paths, start=1):
    #     with open(yaml_path, "r") as f:
    #         config = yaml.safe_load(f)
    #     output_folder = run_pipeline_with_config(config, f"predefined_{idx}")
    #     batched_runs.append(output_folder)

    # Save the list of batched runs
    save_batched_runs_list(batched_runs)

    print("All pipeline runs completed.")

if __name__ == "__main__":
    main()
