import argparse
import sys
import os
import pandas as pd
import json

sys.path.append(os.path.join(os.getcwd(), 'src'))

from pcnrec.utils.io import load_config, load_parquet
from pcnrec.eval.evaluate_runs import evaluate_run
from pcnrec.analysis.feasibility import check_feasibility

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--methods", default="mf_topn,mmr,constrained_greedy,single_llm,pcnrec")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_id = args.run_id
    
    output_dir = os.path.join(config['dataset']['output_dir'], run_id)
    candidates_path = os.path.join(output_dir, "candidates", "candidates_topk.parquet")
    items_path = os.path.join(output_dir, "data", "items.parquet")
    test_path = os.path.join(output_dir, "data", "interactions_test.parquet")
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"Loading data from {output_dir}...")
    items_df = load_parquet(items_path)
    # Ensure index is internal_id logic if needed, but feasibility usually uses lookup
    # Items df usually indexed by item_idx
    
    test_df = load_parquet(test_path)
    cands_df = load_parquet(candidates_path)
    if 'user_idx' in cands_df.columns:
        cands_df = cands_df.rename(columns={'user_idx': 'user_id'})
        
    print("Computing Feasible Users set...")
    constraints = config['constraints']
    cand_window = config['pcn']['candidate_window']
    
    feasible_users = set()
    total_users = 0
    
    user_groups = cands_df.groupby('user_id')
    for uid, group in user_groups:
        is_feasible, _ = check_feasibility(group, constraints, top_k=cand_window)
        if is_feasible:
            feasible_users.add(uid)
        total_users += 1
        
    feasible_rate = len(feasible_users) / total_users if total_users > 0 else 0
    print(f"Feasible Users: {len(feasible_users)}/{total_users} ({feasible_rate:.1%}) at Window={cand_window}")
    
    methods = args.methods.split(',')
    
    all_summaries = []
    feas_summaries = []
    
    for method in methods:
        run_dir = os.path.join(output_dir, "runs", method)
        if not os.path.exists(run_dir):
            print(f"Skipping {method} (not found)")
            continue
            
        print(f"Evaluating {method}...")
        
        # 1. All Users
        summary_all, _ = evaluate_run(run_dir, test_df, items_df, subset_users=None)
        if not summary_all: continue
        summary_all['method'] = method
        all_summaries.append(summary_all)
        
        # 2. Feasible Only
        summary_feas, _ = evaluate_run(run_dir, test_df, items_df, subset_users=feasible_users)
        summary_feas['method'] = method
        feas_summaries.append(summary_feas)
        
    # Save All
    df_all = pd.DataFrame(all_summaries)
    path_all = os.path.join(analysis_dir, "compare_methods_all_users.csv")
    df_all.to_csv(path_all, index=False)
    print(f"Saved {path_all}")
    print(df_all[['method','ndcg@10','recall@10','verifier_pass','repair_used']].to_string())
    
    # Save Feasible
    df_feas = pd.DataFrame(feas_summaries)
    path_feas = os.path.join(analysis_dir, "compare_methods_feasible_only.csv")
    df_feas.to_csv(path_feas, index=False)
    print(f"\nSaved {path_feas} (Feasible Subset)")
    if not df_feas.empty:
        print(df_feas[['method','ndcg@10','recall@10','verifier_pass','repair_used']].to_string())
        
    # Stats
    # Feasibility Rate
    stats = {
        'total_users': total_users,
        'feasible_users': len(feasible_users),
        'feasible_rate': feasible_rate,
        'window': cand_window
    }
    with open(os.path.join(analysis_dir, "feasibility_context.json"), 'w') as f:
        json.dump(stats, f, indent=2)

    # Statistics Step
    from pcnrec.eval.significance import bootstrap_paired_test
    
    stats_results = {}
    
    # Needs: dict of method -> user_id -> metric
    # We have df_feas which has rows for each method.
    # We need to pivot or re-organize.
    
    # Let's extract per-user metrics for feasible users
    # run dictionaries: method -> {uid: metric}
    
    methods_dfs = {} 
    # Re-read or just accumulate during loop?
    # We didn't save per-user details in memory except in summary loop, but discarded.
    # Let's re-read specifically for stats or Modify loop to return DF.
    # evaluate_run returns (summary, df_m). df_m has 'user_id'.
    
    # Re-run evaluate_run for specific methods to get DFs if needed, OR store them above.
    
    # Let's refactor loop above to store DFs
    
    # ... Refactor main loop ...
    
    # Store DFs for significance
    feas_dfs = {}
    
    for method in methods:
        run_dir = os.path.join(output_dir, "runs", method)
        if not os.path.exists(run_dir):
            continue
            
        # Feasible Only
        summary_feas, df_feas_m = evaluate_run(run_dir, test_df, items_df, subset_users=feasible_users)
        feas_dfs[method] = df_feas_m
        
    # Primary Comparison 1: PCN-Rec vs Single-LLM (Feasible)
    if 'pcnrec' in feas_dfs and 'single_llm' in feas_dfs:
        df_pcn = feas_dfs['pcnrec'].set_index('user_id')
        df_base = feas_dfs['single_llm'].set_index('user_id')
        
        # Align
        common = df_pcn.index.intersection(df_base.index)
        
        # Metrics: ndcg@10, tail_fraction@10? 
        # Metric names: 'ndcg@10', 'tail_prop@10' (from metrics.py)
        
        metrics_to_test = ['ndcg@10', 'tail_prop@10', 'recall@10']
        
        for m in metrics_to_test:
            if m not in df_pcn.columns:
                continue
                
            a = df_pcn.loc[common, m].values
            b = df_base.loc[common, m].values
            
            delta, p_val = bootstrap_paired_test(a, b)
            stats_results[f"pcn_vs_single_{m}"] = {
                "delta": delta,
                "p_value": p_val,
                "mean_pcn": float(a.mean()),
                "mean_single": float(b.mean())
            }

    # Primary Comparison 2: PCN-Rec vs mf_topn (Feasible)
    if 'pcnrec' in feas_dfs and 'mf_topn' in feas_dfs:
        df_pcn = feas_dfs['pcnrec'].set_index('user_id')
        df_base = feas_dfs['mf_topn'].set_index('user_id')
        common = df_pcn.index.intersection(df_base.index)
        
        for m in ['ndcg@10', 'recall@10']:
            a = df_pcn.loc[common, m].values
            b = df_base.loc[common, m].values
            delta, p_val = bootstrap_paired_test(a, b)
            stats_results[f"pcn_vs_mf_{m}"] = {
                 "delta": delta,
                 "p_value": p_val
            }
            
    # Save Stats
    stats_path = os.path.join(analysis_dir, "significance.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_results, f, indent=2)
        
    print(f"Saved significance tests to {stats_path}")
    
    # Snippet
    snippet_path = os.path.join(analysis_dir, "paper_stats_snippet.md")
    with open(snippet_path, 'w') as f:
        f.write("## Statistical Significance (Feasible Subset)\n")
        for k, v in stats_results.items():
            f.write(f"- **{k}**: Delta={v['delta']:.4f}, p={v['p_value']:.4f}\n")
    print(f"Saved stats snippet to {snippet_path}")

if __name__ == "__main__":
    main()
