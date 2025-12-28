import argparse
import sys
import os
import datetime

# Add src to path if running from repo root
sys.path.append(os.path.join(os.getcwd(), 'src'))

from pcnrec.utils.io import load_config, save_yaml, save_parquet, save_pickle
from pcnrec.utils.logging import setup_logger
from pcnrec.utils.seed import set_seed
from pcnrec.data.movielens_download import download_movielens
from pcnrec.data.movielens_prepare import prepare_data

logger = setup_logger("step1_prepare_data")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to config")
    parser.add_argument("--run_id", default=None, help="Run ID (default: timestamp)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Set run_id
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config['run']['run_id'] = run_id
    
    # Setup outputs
    output_dir = os.path.join(config['dataset']['output_dir'], run_id)
    data_out_dir = os.path.join(output_dir, "data")
    
    logger.info(f"Starting Run: {run_id}")
    logger.info(f"Outputs will be in: {output_dir}")
    
    # 1. Download
    raw_dir = config['dataset']['data_dir']
    variant = config['dataset']['variant']
    downloaded_path = download_movielens(variant, raw_dir)
    
    # 2. Prepare
    train_df, test_df, users, items, stats = prepare_data(config, downloaded_path)
    
    # 3. Save
    logger.info("Saving outputs...")
    save_parquet(train_df, os.path.join(data_out_dir, "interactions_train.parquet"))
    save_parquet(test_df, os.path.join(data_out_dir, "interactions_test.parquet"))
    save_parquet(users, os.path.join(data_out_dir, "users.parquet"))
    save_parquet(items, os.path.join(data_out_dir, "items.parquet"))
    save_yaml(config, os.path.join(data_out_dir, "config_resolved.yaml"))
    
    # Save stats as json
    import json
    with open(os.path.join(data_out_dir, "stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
        
    logger.info("Done.")
    logger.info(f"Data ready at: {data_out_dir}")

if __name__ == "__main__":
    main()
