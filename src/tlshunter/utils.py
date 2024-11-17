import os


def create_directory(directory: str) -> None:
    """Create a directory if it does not exist.

    This function creates a directory at the specified path if it does not already exist.

    Parameters
    ----------
    directory : str
        Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
