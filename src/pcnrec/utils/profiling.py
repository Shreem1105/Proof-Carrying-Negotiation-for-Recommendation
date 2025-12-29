import pandas as pd
from collections import Counter

def get_user_profile(user_id, train_df, items_df, max_items=5, max_genres=3):
    """
    Constructs a text summary of user profile from training interactions.
    """
    history = train_df[train_df['user_id'] == user_id]
    if history.empty:
        return "No history available."
        
    # Get high rated items (assuming implicit feedback if rating not present, or filter by rating)
    # Checks if 'rating' exists
    if 'rating' in history.columns:
        liked = history[history['rating'] >= 4.0]
        if liked.empty:
            liked = history # Fallback to all if no high ratings
    else:
        liked = history
        
    # Get item details
    # Join with items
    # item_id in train_df -> index of items_df
    # Assuming user_id, item_id in train_df
    
    # We loop or vectorized? 
    # For single user, valid to be procedural
    
    liked_ids = liked['item_id'].values
    
    movies = []
    all_genres = []
    
    for iid in liked_ids:
        if iid in items_df.index:
            row = items_df.loc[iid]
            title = row.get('title', f"Item {iid}")
            # Limit length of title
            movies.append(title)
            
            gs = row.get('genres', '')
            if gs:
                all_genres.extend(gs.split('|'))
                
    # Top genres
    genre_counts = Counter(all_genres)
    top_genres = [g for g, c in genre_counts.most_common(max_genres)]
    
    # Recent items? Or random sample of liked?
    # Let's take last few if sorted by timestamp, otherwise head
    if 'timestamp' in liked.columns:
        liked = liked.sort_values('timestamp', ascending=False)
        
    # Get top movies text
    # Map back to titles
    # We already collected titles above, but order matters
    # We iterate again on sorted 'liked'
    top_movies = []
    count = 0
    for idx, row in liked.iterrows():
        iid = row['item_id']
        if iid in items_df.index:
            title = items_df.loc[iid].get('title', str(iid))
            top_movies.append(title)
            count += 1
            if count >= max_items:
                break
                
    profile_str = f"User likes genres associated with: {', '.join(top_genres)}.\n"
    profile_str += f"Recently liked items: {', '.join(top_movies)}."
    
    return profile_str
