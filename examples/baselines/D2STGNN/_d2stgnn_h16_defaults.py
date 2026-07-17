"""Shared defaults for D2STGNN PeMS04 16-input horizon runs."""

INPUT_LEN = 16
DEFAULT_HORIZONS = [16, 32, 64]
DEFAULT_GAP = 4  # use 2 or 4 so F∈{16,32,64} aligns with labels (gap=3 truncates)
