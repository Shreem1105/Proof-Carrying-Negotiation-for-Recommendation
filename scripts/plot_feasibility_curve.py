import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_png", required=True)
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"Input not found: {args.input_csv}")
        return
        
    df = pd.read_csv(args.input_csv)
    
    plt.figure(figsize=(8, 5))
    plt.plot(df['window'], df['feasible_rate'], marker='o', label='Overall Feasible Rate', linewidth=2, color='green')
    
    # Plot failure causes
    if 'tail_shortage_rate' in df.columns:
        plt.plot(df['window'], df['tail_shortage_rate'], marker='x', linestyle='--', label='Tail Shortage', color='red', alpha=0.7)
    if 'genre_shortage_rate' in df.columns:
        plt.plot(df['window'], df['genre_shortage_rate'], marker='s', linestyle=':', label='Genre Shortage', color='blue', alpha=0.7)
        
    plt.title('Constraint Feasibility vs Candidate Window Size')
    plt.xlabel('Window Size (Top-k Candidates)')
    plt.ylabel('Proportion of Users')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.05, 1.05)
    
    plt.savefig(args.output_png, dpi=300)
    print(f"Saved plot to {args.output_png}")

if __name__ == "__main__":
    main()
