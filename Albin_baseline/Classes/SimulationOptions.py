"""If we want to tweak small things without changing code. Using argparse to change options in the command line."""
import argparse
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "TempModels" / "unified"

def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number

def simulation_options(defense, steps, training_runs, training_steps):
    parser = argparse.ArgumentParser(description=f"{defense} defense with unified Random Forest Eve")
    parser.add_argument("--steps", type=positive_int, default=steps)
    parser.add_argument("--training-runs", type=positive_int, default=training_runs)
    parser.add_argument("--training-steps", type=positive_int, default=training_steps)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--model", type=Path, help="Load an existing model instead of training")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Hide slot progress")
    return parser.parse_args()
