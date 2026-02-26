# Import libraries
from pathlib import Path
import joblib

def save_artifact(artifact, path):
    """
    Save a trained model artifact to disk.

    Parameters
    ----------
    artifact : dict
        Dictionary containing the trained model, preprocessing objects,
        feature list and metadata.
    path : str
        Path where the artifact will be saved.
    """
    path= Path(path)
    path.parent.mkdir(parents=True, exist_ok= True)

    joblib.dump(artifact, path)

def load_artifact(path):
    """
    Load a trained model artifact from disk.

    Parameters
    ----------
    path : str
        Path to the saved artifact.

    Returns
    -------
    dict
        Loaded artifact containing model and metadata.
    """
    path= Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at path: {path}")
    
    artifact= joblib.load(path)
    return artifact