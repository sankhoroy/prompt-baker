import argparse


def main() -> None:
    """Entry point for the package CLI."""
    parser = argparse.ArgumentParser(description="Prompt Baker package utilities.")
    parser.add_argument(
        "--about",
        action="store_true",
        help="Print package overview.",
    )
    args = parser.parse_args()
    if args.about:
        print("prompt-baker: genetic optimization for prompt/model combinations on benchmark datasets.")
        return
    print("prompt-baker is installed. Use Python API for optimization and scripts/visualize_logs.py for plots.")


if __name__ == "__main__":
    main()
