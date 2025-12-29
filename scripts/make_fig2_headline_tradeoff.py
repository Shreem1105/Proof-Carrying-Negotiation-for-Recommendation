import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from pcnrec.figures import style, fig_headline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="final_ml100k_w80_v2")
    args = parser.parse_args()
    
    # Setup paths
    base_dir = os.path.join(os.getcwd(), 'outputs')
    run_dir = os.path.join(base_dir, args.run_id)
    out_dir = os.path.join(run_dir, "paper_figures")
    os.makedirs(os.path.join(out_dir, "captions"), exist_ok=True)
    
    # File
    tradeoff_csv = os.path.join(run_dir, "analysis", "compare_methods_feasible_only.csv")
    
    # Set Style
    style.set_paper_style()
    
    # Generate
    fig_headline.generate_figure(tradeoff_csv, out_dir)
    
    print("Done.")

if __name__ == "__main__":
    main()
