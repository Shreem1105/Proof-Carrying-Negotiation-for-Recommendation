import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns

def generate_figure(csv_path, output_dir):
    """
    Generates Figure 2: Headline Tradeoff (2 panels: Governance, Utility).
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df.set_index('method', inplace=True)
    
    # Extract Data
    methods = ['single_llm', 'pcnrec']
    labels = ['Single LLM', 'PCN-Rec']
    colors = ['#c0392b', '#27ae60'] # Red, Green
    
    # Check if methods exist
    if not all(m in df.index for m in methods):
        print(f"Error: Missing methods in CSV. Found: {df.index.tolist()}")
        return

    # Data for plotting
    pass_rates = [df.loc[m, 'verifier_pass'] for m in methods]
    ndcgs = [df.loc[m, 'ndcg@10'] for m in methods]
    
    greedy_ndcg = df.loc['constrained_greedy', 'ndcg@10'] if 'constrained_greedy' in df.index else None

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # --- Panel (a): Governance (Pass Rate) ---
    ax0 = axes[0]
    bars0 = ax0.bar(labels, pass_rates, color=colors, alpha=0.9, width=0.6)
    
    ax0.set_title('(a) Governance (Pass Rate)', fontweight='bold', fontsize=11)
    ax0.set_ylabel('Pass Rate among Feasible')
    ax0.set_ylim(0, 1.15) # More headroom for labels/title
    
    # Annotate bars
    for bar, val in zip(bars0, pass_rates):
        height = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    # Note
    ax0.text(0.5, 0.95, "Feasible Users (n=551, W=80)", transform=ax0.transAxes, 
             ha='center', fontsize=9, color='gray', style='italic')

    # --- Panel (b): Utility (NDCG@10) ---
    ax1 = axes[1]
    
    # Reference line first
    if greedy_ndcg is not None:
        ax1.axhline(y=greedy_ndcg, color='#34495e', linestyle='--', linewidth=1, alpha=0.6)
        # Move label up and away from potential bar overlap
        ax1.text(1.15, greedy_ndcg + 0.015, f'Greedy\n(deterministic bound)\n{greedy_ndcg:.3f}', 
                 ha='center', va='bottom', fontsize=8, color='#34495e', style='italic')

    bars1 = ax1.bar(labels, ndcgs, color=colors, alpha=0.9, width=0.6)
    
    ax1.set_title('(b) Utility (NDCG@10)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('NDCG@10')
    
    # Y-limit fixed for readability
    ax1.set_ylim(0, 0.6)
    
    # Annotate bars
    for bar, val in zip(bars1, ndcgs):
        height = bar.get_height()
        # Single LLM label might overlap with greedy line? 
        # Single LLM is 0.424, Greedy is 0.426. They are very close.
        # Let's put label inside the bar for clarity if it's high enough? 
        # Or just above. 0.424 vs 0.426 is tight.
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Delta annotation
    # Ensure delta is consistent with plotted values
    delta = ndcgs[1] - ndcgs[0] # (PCN - SingleLLM)
    mid_x = (bars1[0].get_x() + bars1[1].get_x() + bars1[0].get_width()) / 2
    max_h = max(ndcgs)
    
    # Simple Delta text high up
    ax1.annotate(f"Δ = {delta:.3f}", 
                 xy=(mid_x, max_h + 0.08), 
                 ha='center', color='#c0392b', fontweight='bold', fontsize=10)

    # Global styling
    sns.despine()
    plt.tight_layout()
    
    # Output paths
    png_path = os.path.join(output_dir, 'fig2_headline_tradeoff.png')
    pdf_path = os.path.join(output_dir, 'fig2_headline_tradeoff.pdf')
    
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    
    # Caption
    caption = (
        "Figure 2: Headline Results (Feasible Users, n=551, W=80). "
        "(a) PCN-Rec achieves near-perfect governance (98.6% pass rate) compared to the Single LLM baseline (0.0%). "
        "Pass rate measures constraint satisfaction; Single LLM often violates constraints even when feasible solutions exist. "
        "(b) This massive gain in compliance comes with only a minor utility cost (NDCG drop of 0.022), "
        "remaining competitive with the deterministic greedy bound."
    )
    with open(os.path.join(output_dir, 'captions', 'fig2_headline_tradeoff_caption.txt'), 'w') as f:
        f.write(caption)

    print(f"Generated Figure 2 Headline in {output_dir}")
