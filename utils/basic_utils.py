"""Library of basic helper functions."""

import mido
import torchaudio as tau
import librosa as lbrs
from pyheaven import *

DEFAULT_CONFIG_PATH = "config.json"

def get_config(key, default=None, config_path=DEFAULT_CONFIG_PATH):
    """Return the config.
    
    Example
        >>> get_config("default_metric_args")['n_ffts']
            [64, 128, 256, 512, 1024, 2048]
    """
    config = LoadJson(config_path)
    if key in config:
        return config[key]
    else:
        return default
    
def set_config(key, value, override=True, config_path=DEFAULT_CONFIG_PATH):
    """Set the config.
    
    Example
        >>> set_config("default_metric_args", "test")
        >>> get_config("default_metric_args")
        'test'
    """
    config = LoadJson(config_path)
    if override == True or (key not in config):
        config[key] = value
    SaveJson(config, config_path, indent=4)