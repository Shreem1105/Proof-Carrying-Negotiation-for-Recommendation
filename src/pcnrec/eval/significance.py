import numpy as np
from scipy import stats

def bootstrap_paired_test(scores_a, scores_b, n_bootstrap=1000):
    """
    Paired bootstrap test for difference in means.
    """
    diffs = np.array(scores_a) - np.array(scores_b)
    mean_diff = np.mean(diffs)
    
    # Bootstrap
    boot_means = []
    for _ in range(n_bootstrap):
        sample_diffs = np.random.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(np.mean(sample_diffs))
        
    # p-value: proportion of samples where sign opposes observed mean
    # Two-tailed
    if mean_diff > 0:
        p_val = np.mean(np.array(boot_means) <= 0) * 2
    else:
        p_val = np.mean(np.array(boot_means) >= 0) * 2
        
    return float(mean_diff), float(p_val)

def wilcoxon_test(scores_a, scores_b):
    """
    Wilcoxon signed-rank test.
    """
    try:
        stat, p_val = stats.wilcoxon(scores_a, scores_b)
        return float(stat), float(p_val)
    except Exception:
        # e.g. all diffs are zero
        return 0.0, 1.0
