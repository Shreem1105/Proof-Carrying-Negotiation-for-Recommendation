import argparse
import sys
import os
import datetime

sys.path.append(os.path.join(os.getcwd(), 'src'))

from pcnrec.utils.io import load_config, load_parquet, save_pickle, ensure_dir
from pcnrec.utils.logging import setup_logger
from pcnrec.utils.seed import set_seed
from pcnrec.candidates.train_lightfm import train_model

logger = setup_logger("step1_train_candidates")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run_id", required=True, help="Run ID of the data to use")
    args = parser.parse_args()
    
    # Load config from the run directory to ensure consistency? 
    # Or load provided config and override?
    # Better to load the resolved config from data step if possible, or just use the passed one.
    # Let's use the passed one but look for data in run_id.
    
    config = load_config(args.config)
    run_id = args.run_id
    
    output_dir = os.path.join(config['dataset']['output_dir'], run_id)
    data_dir = os.path.join(output_dir, "data")
    model_dir = os.path.join(output_dir, "models")
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}. Run step1_prepare_data.py first.")
        sys.exit(1)
        
    set_seed(config['candidates']['seed'])
    
    logger.info("Loading training data...")
    train_df = load_parquet(os.path.join(data_dir, "interactions_train.parquet"))
    users_df = load_parquet(os.path.join(data_dir, "users.parquet"))
    items_df = load_parquet(os.path.join(data_dir, "items.parquet"))
    
    num_users = len(users_df)
    num_items = len(items_df)
    
    logger.info(f"Training LightFM model (users={num_users}, items={num_items})...")
    model = train_model(config, train_df, num_users, num_items)
    
    logger.info(f"Saving model to {model_dir}")
    save_pickle(model, os.path.join(model_dir, "lightfm.pkl"))
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
