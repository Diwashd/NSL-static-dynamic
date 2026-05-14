import os

def get_repo_root():
    current = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(current))

REPO_ROOT = get_repo_root()
