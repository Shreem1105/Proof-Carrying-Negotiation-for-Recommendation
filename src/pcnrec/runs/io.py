import json
import os
from pcnrec.utils.io import ensure_dir

def append_result_row(output_dir, row_dict):
    """
    Appends a row to results.jsonl in output_dir.
    """
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "results.jsonl")
    with open(path, 'a') as f:
        f.write(json.dumps(row_dict) + "\n")

def read_results(output_dir):
    """
    Reads results.jsonl as list of dicts.
    """
    path = os.path.join(output_dir, "results.jsonl")
    if not os.path.exists(path):
        return []
    
    results = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def save_summary(output_dir, summary_dict):
    """
    Saves summary.json.
    """
    path = os.path.join(output_dir, "summary.json")
    with open(path, 'w') as f:
        json.dump(summary_dict, f, indent=2)

def save_manifest(output_dir, manifest_dict):
    """
    Saves run_manifest.json.
    """
    path = os.path.join(output_dir, "run_manifest.json")
    with open(path, 'w') as f:
        json.dump(manifest_dict, f, indent=2)
