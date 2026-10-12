import sys
from pathlib import Path

# make the project modules importable when running `pytest` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
