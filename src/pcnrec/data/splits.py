import pandas as pd
from pcnrec.utils.logging import setup_logger

logger = setup_logger(__name__)

def time_aware_split(df, test_ratio):
    """
    Performs time-aware split per user.
    df must have 'user_idx' and 'timestamp'.
    """
    logger.info("Sorting by timestamp for splitting...")
    df = df.sort_values(['user_idx', 'timestamp'])
    
    train_list = []
    test_list = []
    
    for uid, group in df.groupby('user_idx'):
        n = len(group)
        n_test = int(n * test_ratio)
        if n_test < 1 and n > 1:
            n_test = 1
        
        if n_test >= n:
             n_test = n - 1
             
        split_idx = n - n_test
        
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])
        
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)
    
    return train_df, test_df
