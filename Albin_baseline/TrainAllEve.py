"""Train four independent models using the actual simulator configurations."""
import argparse
from pathlib import Path
import subprocess
import sys

from Classes.SimulationOptions import MODEL_DIR, positive_int


SIMULATORS = (
    "Baseline_Simulator.py",
    "Filip_Simulator.py",
    "SofiaRandom_Simulator.py",
    "SofiaKeyed_Simulator.py",
    "Alma_Simulator.py",
    "Emelie_Simulator.py",
    "Albin_Defense_Simulator.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-runs", type=positive_int, default=5)
    parser.add_argument("--training-steps", type=positive_int, default=20_000)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    for simulator in SIMULATORS:
        subprocess.run([
            sys.executable, str(Path(__file__).parent / simulator),
            #"--quiet",
            "--training-runs", str(args.training_runs),
            "--training-steps", str(args.training_steps),
            "--model-dir", str(args.model_dir.resolve()),
        ], check=True)
        print("----------------------------------------")


if __name__ == "__main__":
    main()
