import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the overall average COMET score from comet_scores.json."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="data/CommonMT/output/comet_scores/comet_scores.json",
        help="Path to the comet_scores.json file (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.input)

    if not json_path.is_file():
        raise SystemExit(f"Input file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    mean_scores = []
    for sample in data.values():
        mean = sample.get("mean_score")
        if isinstance(mean, (int, float)):
            mean_scores.append(float(mean))

    if not mean_scores:
        raise SystemExit("No mean_score entries found in the JSON file.")

    overall_average = sum(mean_scores) / len(mean_scores)
    print(f"Entries counted: {len(mean_scores)}")
    print(f"Overall average COMET score: {overall_average:.6f}")


if __name__ == "__main__":
    main()
