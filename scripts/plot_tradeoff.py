import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--x_metric", default="verifier_pass") 
    parser.add_argument("--y_metric", default="ndcg@10")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"Input not found: {args.input_csv}")
        return
        
    df = pd.read_csv(args.input_csv)
    
    plt.figure(figsize=(8, 6))
    
    colors = {'single_llm': 'blue', 'pcnrec': 'green', 'mf_topn': 'gray', 'constrained_greedy': 'orange', 'mmr': 'purple'}
    markers = {'single_llm': 'o', 'pcnrec': '*', 'mf_topn': 's', 'constrained_greedy': '^', 'mmr': 'D'}
    
    for i, row in df.iterrows():
        method = row['method']
        c = colors.get(method, 'black')
        m = markers.get(method, 'o')
        
        plt.scatter(row[args.x_metric], row[args.y_metric], color=c, marker=m, s=150, label=method)
        plt.text(row[args.x_metric], row[args.y_metric]+0.01, method, fontsize=9, ha='center')
        
    plt.xlabel(args.x_metric)
    plt.ylabel(args.y_metric)
    plt.title(f'Tradeoff: {args.y_metric} vs {args.x_metric}')
    plt.grid(True, alpha=0.3)
    # plt.legend() 
    
    plt.savefig(args.output_png, dpi=300)
    print(f"Saved plot to {args.output_png}")

if __name__ == "__main__":
    main()
