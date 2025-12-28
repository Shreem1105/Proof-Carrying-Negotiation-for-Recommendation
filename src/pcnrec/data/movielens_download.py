import os
import requests
import zipfile
import io
from pcnrec.utils.logging import setup_logger
from pcnrec.utils.io import ensure_dir

logger = setup_logger(__name__)

MOVIELENS_URLS = {
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
}

EXPECTED_FILES = {
    "ml-100k": ["u.data", "u.item"],
    "ml-1m": ["ratings.dat", "movies.dat"]
}

def download_movielens(variant, data_dir):
    """
    Downloads and extracts the MovieLens dataset.
    """
    if variant not in MOVIELENS_URLS:
        raise ValueError(f"Unknown variant: {variant}")
    
    url = MOVIELENS_URLS[variant]
    target_dir = os.path.join(data_dir, variant)
    
    # Check if already exists
    if os.path.exists(target_dir):
        # Basic check
        missing = [f for f in EXPECTED_FILES[variant] if not os.path.exists(os.path.join(target_dir, f))]
        if not missing:
            logger.info(f"Dataset {variant} already exists in {target_dir}")
            return target_dir
        else:
            logger.info(f"Dataset {variant} incomplete. Missing: {missing}. Re-downloading.")

    ensure_dir(data_dir)
    logger.info(f"Downloading {variant} from {url}...")
    
    try:
        r = requests.get(url)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(data_dir)
        logger.info(f"Extracted to {data_dir}")
        
        # Verify extraction
        # Note: zips often extract to a subdir like 'ml-100k/' so target_dir should match that
        # ml-100k zip contains a folder 'ml-100k'
        # ml-1m zip contains a folder 'ml-1m'
        if not os.path.exists(target_dir):
             raise RuntimeError(f"Extraction failed to create {target_dir}")
             
        logger.info(f"Successfully downloaded and extracted {variant}")
        return target_dir
        
    except Exception as e:
        logger.error(f"Failed to download {variant}: {e}")
        raise
