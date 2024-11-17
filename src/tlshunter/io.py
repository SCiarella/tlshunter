import pandas as pd
import yaml


def load_config(config_file_path: str) -> dict:
    """Load the configuration file from a given path.

    This function reads a YAML configuration file and returns its contents as a dictionary.

    Parameters
    ----------
    config_file_path : str
        Path to the .yaml configuration file

    Returns:
    -------
    config : dict
        Dictionary containing the configuration
    """
    with open(config_file_path) as file:
        config = yaml.safe_load(file)
    return config


def load_data(file_path: str, file_type: str = "feather") -> pd.DataFrame:
    """Load data from a file.

    This function reads data from a specified file and returns it as a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the file.
    file_type : str, optional
        Type of the file ('feather' or 'csv'). Default is 'feather'.

    Returns:
    -------
    pd.DataFrame
        Loaded data.
    """
    try:
        if file_type == "feather":
            return pd.read_feather(file_path)
        if file_type == "csv":
            return pd.read_csv(file_path, index_col=0)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()
