import numpy as np
import pandas as pd
from pcnrec.utils.logging import setup_logger
from tqdm import tqdm

logger = setup_logger(__name__)

def generate_candidates(model, train_df, num_users, num_items, top_k, num_threads=1, batch_size=1000, items_df=None):
    """
    Generates top-K candidates for each user.
    Excludes items seen in train.
    """
    logger.info("Generating candidates...")
    
    # Create a set of seen items per user for fast lookup
    seen_items = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    
    all_users = np.arange(num_users)
    all_items = np.arange(num_items)
    
    results = []
    
    # Process users in batches to manage memory
    for start in tqdm(range(0, num_users, batch_size)):
        end = min(start + batch_size, num_users)
        batch_users = all_users[start:end]
        
        # Predict scores for all items for these users
        # model.predict takes (user_ids, item_ids)
        # We need to predict for every user against every item
        # LightFM predict is efficient if we broadcast? No, it expects equal length arrays.
        # But for batch generation, we loop or repeat.
        # LightFM doesn't have a bulk 'predict_rank' easily accessible without dataset object quirks.
        # Efficient way: user_embedding * item_embedding + biases
        # But let's stick to model.predict for correctness with LightFM logic (biases etc)
        
        # ACTUALLY, model.predict runs for pairs.
        # To score all items for one user: model.predict(user_id, np.arange(num_items))
        
        for user_id in batch_users:
            # LightFM expects user_ids and item_ids to be same length arrays
            user_ids_repeated = np.full(len(all_items), user_id, dtype=np.int32)
            scores = model.predict(user_ids_repeated, all_items, num_threads=1)
            
            # Mask seen items
            if user_id in seen_items:
                seen = list(seen_items[user_id])
                scores[seen] = -np.inf
                
            # Top K
            # argpartition is faster than argsort
            if len(scores) > top_k:
                top_k_indices = np.argpartition(scores, -top_k)[-top_k:]
                # Sort the top k
                top_k_indices = top_k_indices[np.argsort(-scores[top_k_indices])]
            else:
                top_k_indices = np.argsort(-scores)
            
            top_scores = scores[top_k_indices]
            
            for rank, (item_idx, score) in enumerate(zip(top_k_indices, top_scores)):
                results.append({
                    'user_idx': user_id,
                    'item_idx': item_idx,
                    'cand_score': float(score),
                    'rank': rank + 1
                })
                
    results_df = pd.DataFrame(results)
    
    # Join with item info if provided
    if items_df is not None:
        # items_df has 'internal_id' which is 'item_idx'
        # We need to map item_idx -> popularity_count, popularity_bin, original_id
        # items_df columns: original_id, internal_id, title, genres, popularity_count, popularity_bin
        
        # Create a lookup
        items_info = items_df.set_index('internal_id')[['original_id', 'title', 'genres', 'popularity_count', 'popularity_bin']]
        
        results_df = results_df.join(items_info, on='item_idx')
        
    return results_df
