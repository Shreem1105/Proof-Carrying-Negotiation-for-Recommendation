import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from pcnrec.figures import style, fig_feasibility, fig_tradeoff, fig_system_overview

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="final_ml100k_w80_v2")
    parser.add_argument("--feas_run_id", default="final_ml100k_feas_v2")
    args = parser.parse_args()
    
    # Setup paths
    base_dir = os.path.join(os.getcwd(), 'outputs')
    run_dir = os.path.join(base_dir, args.run_id)
    feas_dir = os.path.join(base_dir, args.feas_run_id)
    
    out_dir = os.path.join(run_dir, "paper_figures")
    os.makedirs(os.path.join(out_dir, "captions"), exist_ok=True)
    
    # Files
    feas_csv = os.path.join(feas_dir, "analysis", "feasibility.csv")
    tradeoff_csv = os.path.join(run_dir, "analysis", "compare_methods_feasible_only.csv")
    
    # Set Style
    style.set_paper_style()
    print("Generating figures...")
    
    # Figure 1
    fig_feasibility.generate_figure(feas_csv, out_dir)
    
    # Figure 2
    fig_tradeoff.generate_figure(tradeoff_csv, out_dir)
    
    # Figure 3
    fig_system_overview.generate_figure(out_dir)
    
    print("\n--------------------------")
    print(f"Figures generated in: {out_dir}")

if __name__ == "__main__":
    main()
