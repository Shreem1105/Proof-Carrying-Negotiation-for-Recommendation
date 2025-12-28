import argparse
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))

from pcnrec.utils.io import load_config, load_parquet, load_pickle, save_parquet
from pcnrec.utils.logging import setup_logger
from pcnrec.candidates.generate_candidates import generate_candidates

logger = setup_logger("step1_generate_candidates")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--max_users", type=int, default=None, help="Limit users for smoke test")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_id = args.run_id
    
    output_dir = os.path.join(config['dataset']['output_dir'], run_id)
    data_dir = os.path.join(output_dir, "data")
    model_path = os.path.join(output_dir, "models", "lightfm.pkl")
    cand_dir = os.path.join(output_dir, "candidates")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}. Run step1_train_candidates.py first.")
        sys.exit(1)
        
    logger.info("Loading data and model...")
    train_df = load_parquet(os.path.join(data_dir, "interactions_train.parquet"))
    items_df = load_parquet(os.path.join(data_dir, "items.parquet"))
    users_df = load_parquet(os.path.join(data_dir, "users.parquet"))
    model = load_pickle(model_path)
    
    num_users = len(users_df)
    num_items = len(items_df)
    
    if args.max_users:
        logger.info(f"Limiting to first {args.max_users} users.")
        num_users = min(num_users, args.max_users)
        
    top_k = config['candidates']['top_k']
    
    logger.info(f"Generating top-{top_k} candidates for {num_users} users...")
    
    candidates_df = generate_candidates(
        model, 
        train_df, 
        num_users, 
        num_items, 
        top_k, 
        items_df=items_df
    )
    
    out_path = os.path.join(cand_dir, "candidates_topk.parquet")
    logger.info(f"Saving candidates to {out_path}")
    save_parquet(candidates_df, out_path)
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
