"""Backward-compatible entry point. Prefer running train.py directly."""

from train import cli

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    cli()
