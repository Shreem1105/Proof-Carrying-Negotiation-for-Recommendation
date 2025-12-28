import pandas as pd

def compute_popularity(train_df, items_df, config):
    """
    Computes popularity counts and bins based on training data.
    """
    pop_counts = train_df['item_idx'].value_counts()
    
    # We map back to items_df
    # items_df must have 'internal_id' which corresponds to item_idx
    items_df = items_df.copy()
    items_df['popularity_count'] = items_df['internal_id'].map(pop_counts).fillna(0).astype(int)
    
    # Calculate thresholds from TRAIN set popularity of items that exist in train
    # or all items? Usually distribution is over all available items, 
    # but strictly we only know about train items.
    
    train_pop = items_df[items_df['popularity_count'] > 0]['popularity_count']
    
    head_thresh = train_pop.quantile(config['popularity']['head_quantile'])
    torso_thresh = train_pop.quantile(config['popularity']['torso_quantile'])
    
    def get_bin(count):
        if count >= head_thresh:
            return 'head'
        elif count >= torso_thresh:
            return 'torso'
        else:
            return 'tail'
            
    items_df['popularity_bin'] = items_df['popularity_count'].apply(get_bin)
    
    stats = {
        'head_threshold': float(head_thresh),
        'torso_threshold': float(torso_thresh),
        'head_items': int((items_df['popularity_bin'] == 'head').sum()),
        'torso_items': int((items_df['popularity_bin'] == 'torso').sum()),
        'tail_items': int((items_df['popularity_bin'] == 'tail').sum())
    }
    
    return items_df, stats
