from typing import List

def check_no_duplicates(selected_ids: List[int]) -> bool:
    return len(selected_ids) == len(set(selected_ids))

def check_max_head(head_count: int, limit: int) -> bool:
    return head_count <= limit

def check_min_tail(tail_count: int, limit: int) -> bool:
    return tail_count >= limit

def check_min_unique_genres(unique_count: int, limit: int) -> bool:
    return unique_count >= limit
