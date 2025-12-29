import matplotlib.pyplot as plt
import seaborn as sns

def set_paper_style():
    """
    Sets a clean, academic style for matplotlib figures.
    """
    # Use seaborn style as base
    sns.set_theme(style="whitegrid", context="paper")
    
    # Custom overrides for ACM/WWW style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'pdf.fonttype': 42, # TrueType fonts for editing in Illustrator
        'ps.fonttype': 42,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })

def get_palette():
    return sns.color_palette("deep")
