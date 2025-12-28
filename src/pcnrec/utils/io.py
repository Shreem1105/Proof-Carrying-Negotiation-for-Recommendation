import os
import pickle
import pandas as pd
import yaml
from pathlib import Path

def ensure_dir(path):
    """Ensures that the directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_pickle(obj, path):
    """Saves an object to a pickle file."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    """Loads an object from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_parquet(df, path):
    """Saves a DataFrame to a parquet file."""
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, index=False)

def load_parquet(path):
    """Loads a DataFrame from a parquet file."""
    return pd.read_parquet(path)

def save_yaml(data, path):
    """Saves data to a YAML file."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
