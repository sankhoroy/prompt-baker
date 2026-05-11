from __future__ import annotations

import argparse

from prompt_baker.visualizer import create_scores_csv, plot_progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Prompt Baker optimization logs.")
    parser.add_argument("--run-dir", required=True, help="Path to run directory containing scores.jsonl")
    parser.add_argument("--output-file", default=None, help="Optional output image path")
    parser.add_argument("--csv-output-file", default=None, help="Optional output CSV path")
    args = parser.parse_args()
    output = plot_progress(run_dir=args.run_dir, output_file=args.output_file)
    csv_output = create_scores_csv(run_dir=args.run_dir, output_file=args.csv_output_file)
    print(f"Saved plot: {output}")
    print(f"Saved CSV: {csv_output}")


if __name__ == "__main__":
    main()
