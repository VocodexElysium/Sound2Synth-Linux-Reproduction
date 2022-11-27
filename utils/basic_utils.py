"""Library of basic helper functions."""

import mido
import torchaudio as tau
import librosa as lbrs
from pyheaven import *

DEFAULT_CONFIG_PATH = "config.json"

def get_config(key, default=None, config_path=DEFAULT_CONFIG_PATH):
    """Return the config."""
    config = LoadJson(config_path)
    if key in config:
        return config[key]
    else:
        return default
    
def set_config(key, value, override=True, config_path=DEFAULT_CONFIG_PATH):
    """Set the config."""
    config = LoadJson(config_path)
    if override == True or (key not in config):
        config[key] = value
    SaveJson(config, config_path, indent=4)