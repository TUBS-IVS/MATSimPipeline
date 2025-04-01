import os
import yaml


class Config:
    def __init__(self, output_folder, project_root, config_file="config.yaml"):
        self.output_folder = output_folder

        # Load the config file
        with open(os.path.join(project_root, config_file), "r") as f:
            self.full_config = yaml.safe_load(f)
            if not isinstance(self.full_config, dict):
                raise ValueError("Invalid YAML format: Expected a dictionary at the top level.")

        # Load the used configuration (for documentation), which will be updated as the pipeline runs.
        used_config_path = os.path.join(self.output_folder, "used_config.yaml")
        if not os.path.exists(used_config_path):
            with open(used_config_path, "w") as f:
                yaml.safe_dump({}, f)  # Writes an empty dictionary to the file
        with open(os.path.join(self.output_folder, "used_config.yaml"), "r") as f:
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
        - If the key is missing entirely, return `default` (or raise KeyError if `default` is "Stop_when_missing").
        """

        def traverse(config, key_path):
            """Traverses a nested dictionary using dot-separated keys."""
            keys = key_path.split('.')
            value = config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    if default == "Stop_when_missing":
                        raise KeyError(f"Key '{key}' not found in {'used_config' if use_used else 'full_config'}.")
                    return default
            return value  # Return the actual value, even if None

        value = traverse(self.used_config if use_used else self.full_config, key)

        # Store the accessed value in used_config (only if it wasn't fetched from there)
        if not use_used and key not in self.used_config:
            self.used_config[key] = value

        return value

    def nest_dict(self, flat_dict):
        """
        Convert a flat dictionary with dot-separated keys into a nested dictionary.

        Example:
          {
              "settings.num_cores": 8,
              "settings.log_level": "INFO",
              "steps.Population.script": "steps/population/run.py"
          }
        becomes:
          {
              "settings": {
                  "num_cores": 8,
                  "log_level": "INFO"
              },
              "steps": {
                  "Population": {
                      "script": "steps/population/run.py"
                  }
              }
          }
        """
        nested = {}
        for composite_key, value in flat_dict.items():
            keys = composite_key.split('.')
            d = nested
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = value
        return nested

    def write_used_config(self):
        """
        Convert the flat used_config into a nested dictionary (via nest_dict)
        and write it to the provided used_config_file.
        """
        nested_config = self.nest_dict(self.used_config)
        with open(os.path.join(self.output_folder, "used_config.yaml"), "w") as f:
            yaml.safe_dump(nested_config, f, sort_keys=False)
