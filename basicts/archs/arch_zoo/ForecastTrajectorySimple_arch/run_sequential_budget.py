"""Compatibility entry point for online sequential-budget training.

The former cached cost-to-go training implementation was intentionally removed.
Invoking this historical module name now runs the genuine on-policy forecasting
loop in :mod:`run_online_sequential_rl`.
"""

from .run_online_sequential_rl import main


if __name__ == "__main__":
    main()
