import os
import yaml
import pandas as pd
from datetime import datetime

PROJECT_BASE_DIR = r"C:\Users\petre\Documents\GitHub\MATSimPipeline"
BATCHED_RUNS_DIR = os.path.join(PROJECT_BASE_DIR, "batched_runs")

def load_yaml(file_path):
    """Safely loads a YAML file, returning an empty dictionary if missing or invalid."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"Error reading {file_path}: {e}")
    return {}


def get_batched_runs():
    """Reads the most recent batched run file and returns a list of run directories."""
    run_files = sorted(
        [f for f in os.listdir(BATCHED_RUNS_DIR) if f.startswith("batched_runs_")],
        reverse=True
    )
    if not run_files:
        print("No batched run lists found.")
        return []

    latest_run_file = os.path.join(BATCHED_RUNS_DIR, run_files[0])
    with open(latest_run_file, "r") as f:
        runs = [line.strip() for line in f.readlines() if line.strip()]

    return runs


def extract_config_differences(runs):
    """Identifies differences in configuration across runs."""
    all_configs = {}
    changed_keys = set()

    for run in runs:
        config_path = os.path.join(run, "used_config.yaml")
        used_config = load_yaml(config_path)
        all_configs[run] = used_config

    # Determine keys that have differences across runs
    base_keys = set(all_configs[runs[0]].keys()) if runs else set()
    for run, config in all_configs.items():
        for key in base_keys.union(config.keys()):
            values = {all_configs[r].get(key, None) for r in runs}
            if len(values) > 1:  # If there's more than one unique value, it's changed
                changed_keys.add(key)

    # Create a DataFrame showing only changed keys
    config_data = {run: {key: all_configs[run].get(key, None) for key in changed_keys} for run in runs}
    config_df = pd.DataFrame.from_dict(config_data, orient="index").reset_index()
    config_df.rename(columns={"index": "run"}, inplace=True)

    return config_df


def extract_runtime_data(runs):
    """Extracts all recorded runtimes from the 'runtimes' section of stats.yaml."""
    runtime_data = []

    for run in runs:
        stats_path = os.path.join(run, "stats.yaml")
        stats = load_yaml(stats_path)

        runtimes = stats.get("runtimes", {})  # Extract all runtimes
        runtime_entry = {"run": run, **runtimes}
        runtime_data.append(runtime_entry)

    return pd.DataFrame(runtime_data)


def generate_additional_analysis_df(runs):
    """
    Extracts additional analysis from each run (e.g., final result files).
    This is a placeholder function—extend it to pull specific metrics.
    """
    additional_data = []

    for run in runs:
        analysis_entry = {"run": run}

        # Example: Extracting the number of lines from a result CSV
        result_file = os.path.join(run, "mobile_population_load_intermediate.csv")
        if os.path.exists(result_file):
            try:
                df = pd.read_csv(result_file)
                analysis_entry["num_records_in_results"] = len(df)
            except Exception as e:
                print(f"Error reading {result_file}: {e}")
                analysis_entry["num_records_in_results"] = None
        else:
            analysis_entry["num_records_in_results"] = None

        additional_data.append(analysis_entry)

    return pd.DataFrame(additional_data)


def analyze_batched_runs():
    """Performs analysis of all batched runs and generates a final merged summary."""
    runs = get_batched_runs()
    if not runs:
        print("No batched runs found.")
        return

    print(f"Analyzing {len(runs)} runs...")

    config_df = extract_config_differences(runs)
    runtime_df = extract_runtime_data(runs)
    additional_analysis_df = generate_additional_analysis_df(runs)

    # Merge all three DataFrames on 'run'
    final_df = config_df.merge(runtime_df, on="run", how="outer")
    final_df = final_df.merge(additional_analysis_df, on="run", how="outer")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"batched_runs_overview_{timestamp}.csv"
    final_df.to_csv(output_name, index=False)

    print(f"Analysis completed. Results saved to {output_name}")


if __name__ == "__main__":
    analyze_batched_runs()
