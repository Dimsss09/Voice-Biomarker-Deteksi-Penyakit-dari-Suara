"""Prepare acoustic feature tables for model training.

The current default dataset is UCI Parkinson's, which already ships as a
tabular acoustic-feature file. For raw-audio datasets, this module is the place
to add librosa/parselmouth extraction in a later phase.
"""

from __future__ import annotations

from src.prepare_features import main


if __name__ == "__main__":
    main()
