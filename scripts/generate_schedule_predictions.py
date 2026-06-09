import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.prediction.schedule_model import OUTPUT_PATH, generate_schedule_predictions, save_schedule_predictions


def main():
    parser = argparse.ArgumentParser(description="Generate schedule-driven 2026 World Cup predictions")
    parser.add_argument("--n-sim", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    payload = generate_schedule_predictions(n_sim=args.n_sim, seed=args.seed)
    save_schedule_predictions(payload, args.output)
    print(f"saved {args.output} teams={len(payload['teams'])} matches={len(payload['matches'])}")


if __name__ == "__main__":
    main()
