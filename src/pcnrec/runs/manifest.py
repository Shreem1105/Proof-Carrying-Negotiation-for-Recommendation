import os
import datetime
import subprocess

def get_git_revision_short_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('ascii')
    except:
        return "unknown"

def create_manifest(config):
    """
    Creates a manifest dict for the run.
    """
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_revision_short_hash(),
        "run_id": config['run']['run_id'],
        "model": config['llm']['model'],
        "config": config
    }
