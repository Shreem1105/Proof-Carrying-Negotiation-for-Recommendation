import pandas as pd
import numpy as np
import os
from pcnrec.utils.logging import setup_logger
from pcnrec.utils.io import save_parquet, save_yaml, ensure_dir
import json

logger = setup_logger(__name__)

def load_ml_100k(data_dir):
    # u.data: user id | item id | rating | timestamp
    # u.item: movie id | movie title | ... genres ...
    
    ratings_path = os.path.join(data_dir, "u.data")
    names = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(ratings_path, sep='\t', names=names, engine='python')
    
    items_path = os.path.join(data_dir, "u.item")
    # genres are columns 5-23
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", 
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", 
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    # u.item is pipe separated, might have encoding issues
    items = pd.read_csv(items_path, sep='|', encoding='latin-1', header=None, engine='python')
    items = items.iloc[:, :24] # keep relevant cols
    items.columns = ["item_id", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_names
    
    def get_genres(row):
        return "|".join([g for g in genre_names if row[g] == 1])
    
    items['genres'] = items.apply(get_genres, axis=1)
    items = items[['item_id', 'title', 'genres']]
    
    return df, items

def load_ml_1m(data_dir):
    # ratings.dat: UserID::MovieID::Rating::Timestamp
    # movies.dat: MovieID::Title::Genres
    
    ratings_path = os.path.join(data_dir, "ratings.dat")
    df = pd.read_csv(ratings_path, sep='::', names=["user_id", "item_id", "rating", "timestamp"], engine='python', encoding='latin-1')
    
    movies_path = os.path.join(data_dir, "movies.dat")
    items = pd.read_csv(movies_path, sep='::', names=["item_id", "title", "genres"], engine='python', encoding='latin-1')
    
    return df, items

from pcnrec.data.splits import time_aware_split
from pcnrec.data.popularity import compute_popularity

def prepare_data(config, raw_data_dir):
    variant = config['dataset']['variant']
    test_ratio = config['dataset']['test_ratio']
    min_interactions = config['dataset']['min_user_interactions']
    
    logger.info(f"Loading {variant} from {raw_data_dir}")
    if variant == 'ml-100k':
        df, items_df = load_ml_100k(raw_data_dir)
    elif variant == 'ml-1m':
        df, items_df = load_ml_1m(raw_data_dir)
    else:
        raise ValueError(f"Unsupported variant {variant}")
        
    logger.info(f"Original interactions: {len(df)}")
    
    # 1. Filter users
    user_counts = df.groupby('user_id').size()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df['user_id'].isin(valid_users)].copy()
    logger.info(f"Filtered interactions (min {min_interactions}): {len(df)}")
    
    # 2. Remap IDs
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()
    
    user_map = {uid: i for i, uid in enumerate(unique_users)}
    item_map = {iid: i for i, iid in enumerate(unique_items)}
    
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    
    # Map items metadata
    items_df = items_df[items_df['item_id'].isin(unique_items)].copy()
    items_df['item_idx'] = items_df['item_id'].map(item_map)
    items_df = items_df.sort_values('item_idx').reset_index(drop=True)
    
    # Save ID maps
    users_out = pd.DataFrame({'original_id': list(user_map.keys()), 'internal_id': list(user_map.values())})
    items_df = items_df[['item_id', 'item_idx', 'title', 'genres']].rename(columns={'item_id': 'original_id', 'item_idx': 'internal_id'})
    
    # 3. Time-aware Split
    logger.info("Splitting train/test...")
    train_df, test_df = time_aware_split(df, test_ratio)
    
    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # 4. Popularity
    logger.info("Computing popularity...")
    items_df, pop_stats = compute_popularity(train_df, items_df, config)
    
    stats = {
        'num_users': len(users_out),
        'num_items': len(items_df),
        'train_interactions': len(train_df),
        'test_interactions': len(test_df),
        **pop_stats
    }
    
    return train_df, test_df, users_out, items_df, stats

