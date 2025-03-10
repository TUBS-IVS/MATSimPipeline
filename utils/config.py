import os
import yaml
from utils.helpers import nest_dict

class Config:
    def __init__(self, output_folder, project_root, config_file = "config.yaml"):
        self.output_folder = output_folder

        # Load the config file
        with open(os.path.join(project_root, config_file), "r") as f:
            self.full_config = yaml.safe_load(f)
            if not isinstance(self.full_config, dict):
                raise ValueError("Invalid YAML format: Expected a dictionary at the top level.")

        # Load the used configuration (for documentation), which will be updated as the pipeline runs.
        used_config_path = os.path.join(output_folder, "used_config.yaml")
        if not os.path.exists(used_config_path):
            with open(used_config_path, "w") as f:
                yaml.safe_dump({}, f)  # Writes an empty dictionary to the file
        with open(os.path.join(output_folder, "used_config.yaml"), "r") as f:
            self.used_config = yaml.safe_load(f) or {}
            if not isinstance(self.used_config, dict):
                raise ValueError("Invalid YAML format: Expected a dictionary at the top level.")

    def resolve_paths(self):
        """
        Recursively replace any "${CURRENT_OUTPUT}" placeholders in the entire configuration.
        Also ensures the global output folder is set in settings.
        """

        def replace_placeholders(obj):
            """Recursively traverse dicts and lists to replace "${CURRENT_OUTPUT}"."""
            if isinstance(obj, dict):
                return {k: replace_placeholders(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_placeholders(v) for v in obj]
            elif isinstance(obj, str) and "${CURRENT_OUTPUT}" in obj:
                return obj.replace("${CURRENT_OUTPUT}", self.output_folder)
            return obj

        self.full_config = replace_placeholders(self.full_config)

    def get(self, key, default="Stop_when_missing", use_used=False):
        """
        Retrieve a configuration value using a dot-separated key.
        - If `use_used` is True, return the value from `used_config`.
        - Otherwise, traverse `full_config`, record the accessed value, and return it.
        - If `default` is not specified, the method raises an error if the key is missing.
        """

        def traverse(config, key_path):
            keys = key_path.split('.')
            value = config
            for k in keys:
                if isinstance(value, dict):
                    if k in value:
                        value = value[k]
                    else:
                        raise KeyError(f"Key '{key}' not found in {'used_config' if use_used else 'full_config'}.")
                else:
                    raise KeyError(f"Key '{key}' not found in {'used_config' if use_used else 'full_config'}.")
            return value

        if use_used:
            return traverse(self.used_config, key)
        value = traverse(self.full_config, key)

        if value is None:
            if default == "Stop_when_missing":
                raise KeyError(f"Key '{key}' is required but not found in full_config.")
            else:
                value = default

        # Store the accessed value in used_config
        if key not in self.used_config:
            self.used_config[key] = value

        return value

    def write_used_config(self, used_config_file):
        """
        Convert the flat used_config into a nested dictionary (via nest_dict)
        and write it to the provided used_config_file.
        """
        nested_config = nest_dict(self.used_config)
        with open(used_config_file, "w") as f:
            yaml.safe_dump(nested_config, f, sort_keys=False)
