import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_figure(feasibility_csv, output_dir):
    """
    Generates Figure 1: Feasibility Curve.
    """
    if not os.path.exists(feasibility_csv):
        print(f"Warning: {feasibility_csv} not found.")
        return

    df = pd.read_csv(feasibility_csv)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Main Feasibility Line
    ax.plot(df['window'], df['feasible_rate'], marker='o', label='Feasibility Rate', color='#2c3e50', linewidth=2, zorder=3)
    
    # Secondary Lines
    if 'tail_shortage_rate' in df.columns:
        ax.plot(df['window'], df['tail_shortage_rate'], marker='x', linestyle=':', label='Tail Shortage', color='#e74c3c', alpha=0.7)
    
    # Only plot genre shortage if meaningful
    if 'genre_shortage_rate' in df.columns and df['genre_shortage_rate'].max() > 0.01:
        ax.plot(df['window'], df['genre_shortage_rate'], marker='^', linestyle=':', label='Genre Shortage', color='#f39c12', alpha=0.7)

    # Vertical line for W=80
    ax.axvline(x=80, color='#27ae60', linestyle='--', alpha=0.8, zorder=2)
    
    # Specific Annotation for W=80
    # Find rate at 80
    rate_80 = df.loc[df['window'] == 80, 'feasible_rate'].values[0] if 80 in df['window'].values else 0
    feas_users_80 = int(rate_80 * 943)
    
    ax.annotate(f"Selected W=80\n{feas_users_80}/943 ({rate_80:.1%})", 
                xy=(80, rate_80), xytext=(85, rate_80 - 0.15),
                arrowprops=dict(facecolor='#27ae60', arrowstyle='->', alpha=0.7),
                fontsize=10, color='#27ae60', fontweight='bold', ha='left')
    
    # Formatting
    ax.set_xlabel('Candidate Window Size (W)')
    ax.set_ylabel('Rate of Users')
    ax.set_title('Constraint Feasibility vs. Window Size')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='upper left')
    
    # Note
    ax.text(0.02, 0.05, f"n_total=943", transform=ax.transAxes, fontsize=9, color='gray', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    
    # Save
    fig.savefig(os.path.join(output_dir, 'fig1_feasibility_curve.png'))
    fig.savefig(os.path.join(output_dir, 'fig1_feasibility_curve.pdf'))
    plt.close(fig)
    
    # Caption
    caption = (
        "Figure 1: Feasibility Analysis. The percentage of users for whom a valid slate satisfying all "
        "constraints exists within the top-W candidates. We select W=80 to balance feasibility (~58%) "
        "with candidate relevance. Tail shortage is the primary bottleneck at lower W."
    )
    with open(os.path.join(output_dir, 'captions', 'fig1_caption.txt'), 'w') as f:
        f.write(caption)

    print(f"Generated Figure 1 in {output_dir}")
