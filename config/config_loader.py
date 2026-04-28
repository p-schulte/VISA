import yaml
import os
from pprint import pprint as pp

# Configuration file paths (env vars override defaults)
AC_CONFIG_FILE = os.environ.get("AC_CONFIG_FILE", "../config/yaml_files/ac_config.yaml")
DSG_CONFIG_FILE = os.environ.get("DSG_CONFIG_FILE", "../config/yaml_files/dsg_config.yaml")
DSET_CONFIG_FILE = os.environ.get("DSET_CONFIG_FILE", "../config/yaml_files/dset_config.yaml")


class ConfigLoader:
    _instance = None  

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)

            # Load the configuration files
            cls._instance.ac = cls._instance._load_config(AC_CONFIG_FILE)
            cls._instance.dsg = cls._instance._load_config(DSG_CONFIG_FILE)
            cls._instance.dset = cls._instance._load_config(DSET_CONFIG_FILE)
            
            # Print the loaded configuration
            print('#'*80)
            print("LOADED AC CONFIG:")
            pp(cls._instance.ac)
            print("\nLOADED DSG CONFIG:")
            pp(cls._instance.dsg)
            print("\nLOADED DSET CONFIG:")
            pp(cls._instance.dset)
            print('#'*80)
            print('\n')
        return cls._instance

    def _load_config(self, yaml_file):
        """Loads the YAML configuration and performs string interpolation."""
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(f"Config file {yaml_file} not found!")
        
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)

        # Perform string interpolation
        return {k: v.format(**config) if isinstance(v, str) else v for k, v in config.items()}

    def get_ac(self, key, default=None):
        """Retrieve a configuration value from the AC config."""
        return self._get_nested_value(self.ac, key, default)

    def get_dsg(self, key, default=None):
        """Retrieve a configuration value from the DSG config."""
        return self._get_nested_value(self.dsg, key, default)

    def get_dset(self, key, default=None):
        """Retrieve a configuration value from the DSET config."""
        return self._get_nested_value(self.dset, key, default)

    def _get_nested_value(self, config_dict, key, default):
        """Helper function to retrieve nested configuration values."""
        keys = key.split(".")
        value = config_dict
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

# Create a global instance of ConfigLoader that gets initialized once
CONFIG = ConfigLoader()
