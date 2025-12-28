from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np
from pcnrec.utils.logging import setup_logger
from pcnrec.utils.io import ensure_dir, save_pickle

logger = setup_logger(__name__)

def build_interactions(train_df, num_users, num_items):
    """
    Builds LightFM interaction matrix.
    Assumes user_idx and item_idx are contiguous 0..N-1
    """
    # Create a Dataset object
    dataset = Dataset()
    dataset.fit(users=np.arange(num_users), items=np.arange(num_items))
    
    # Build interactions
    # (user_idx, item_idx) tuples
    interactions_zip = zip(train_df['user_idx'], train_df['item_idx'])
    (interactions, _) = dataset.build_interactions(interactions_zip)
    
    return interactions

def train_model(config, train_df, num_users, num_items):
    """
    Trains a LightFM model.
    """
    logger.info("Building interaction matrix...")
    interactions = build_interactions(train_df, num_users, num_items)
    
    params = config['candidates']
    model = LightFM(
        no_components=params['factors'],
        learning_rate=params['learning_rate'],
        loss=params['loss'],
        random_state=params['seed']
    )
    
    logger.info(f"Training LightFM with {params['epochs']} epochs...")
    model.fit(
        interactions,
        epochs=params['epochs'],
        num_threads=params['num_threads'],
        verbose=True
    )
    
    return model
