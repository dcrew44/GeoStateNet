"""Configuration utilities for state classification model."""
import yaml
from types import SimpleNamespace


def dict_to_namespace(d):
    """
    Recursively convert a dictionary into a SimpleNamespace.
    
    Args:
        d (dict): Dictionary to convert
        
    Returns:
        SimpleNamespace: Namespace representation of the dictionary
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)


def load_config(path):
    """
    Load configuration from a YAML file as a SimpleNamespace.
    
    Args:
        path (str): Path to the YAML configuration file
        
    Returns:
        SimpleNamespace: Namespace representation of the configuration
    """
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)


def override_config_with_wandb(config):
    """
    Update config with values from wandb.config if present.
    
    Args:
        config (SimpleNamespace): Original configuration
        
    Returns:
        SimpleNamespace: Updated configuration
    """
    try:
        import wandb
        if not hasattr(wandb, 'config'):
            return config
            
        # Training parameters
        if hasattr(wandb.config, "batch_size"):
            config.batch_size = wandb.config.batch_size
            
        # Hyperparameters
        if hasattr(wandb.config, "lr"):
            config.hyperparameters.lr = wandb.config.lr
            
        if hasattr(wandb.config, "weight_decay"):
            config.hyperparameters.weight_decay = wandb.config.weight_decay
            
        if hasattr(wandb.config, "optimizer"):
            config.hyperparameters.optimizer = wandb.config.optimizer
            
        if hasattr(wandb.config, "max_lr"):
            config.hyperparameters.max_lr = wandb.config.max_lr
            
    except ImportError:
        pass
        
    return config
