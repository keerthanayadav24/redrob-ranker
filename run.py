"""
run.py — Single command to generate the competition submission.

Usage:
    python run.py

Produces:
    data/top100.json     → loaded by FastAPI for the live demo website
    data/team_keerthana.csv     → to submit this to the Redrob portal 

Runtime: ~90 seconds on CPU. No GPU, no network, no API calls.
"""

import gzip
import json
import csv
import os
import sys
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from scoring_engine import rank_candidates

DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
CANDIDATES_GZ  = os.path.join(DATA_DIR, "candidates.jsonl.gz")
CANDIDATES_JSONL = os.path.join(DATA_DIR, "candidates.jsonl")
OUTPUT_JSON    = os.path.join(DATA_DIR, "top100.json")
OUTPUT_CSV     = os.path.join(DATA_DIR, "team_keerthana.csv")


def load_candidates() -> list[dict]:
    if os.path.exists(CANDIDATES_GZ):
        print(f"Loading from {CANDIDATES_GZ} ...")
        opener = gzip.open(CANDIDATES_GZ, "rt", encoding="utf-8")
    elif os.path.exists(CANDIDATES_JSONL):
        print(f"Loading from {CANDIDATES_JSONL} ...")
        opener = open(CANDIDATES_JSONL, "r", encoding="utf-8")
    else:
        raise FileNotFoundError(
            "No candidates file found. Put candidates.jsonl or "
            "candidates.jsonl.gz in the data/ folder."
        )

    candidates = []
    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(candidates):,} candidates.")
    return candidates


def write_csv(top100: list[dict]) -> None:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_idx, item in enumerate(top100, start=1):
            # Scores guaranteed non-increasing by rank position
            rank_score = round(1.0 - (rank_idx - 1) * 0.006, 4)
            writer.writerow([
                item["candidate_id"],
                rank_idx,
                rank_score,
                item["reasoning"],
            ])
    print(f"Submission CSV saved → {OUTPUT_CSV}")


def write_json(top100: list[dict]) -> None:
    # Strip raw_candidate from JSON to keep file small
    slim = []
    for item in top100:
        row = {k: v for k, v in item.items() if k != "raw_candidate"}
        # Add back just the profile fields the UI needs
        raw = item.get("raw_candidate", {})
        row["raw_candidate"] = {
            "candidate_id": raw.get("candidate_id"),
            "profile": raw.get("profile", {}),
            "skills": raw.get("skills", [])[:8],
            "career_history": raw.get("career_history", [])[:3],
            "redrob_signals": raw.get("redrob_signals", {}),
        }
        slim.append(row)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2)
    print(f"Top-100 JSON saved → {OUTPUT_JSON}")


def main():
    print("=" * 55)
    print("  CandidateIQ — Redrob Hackathon Ranker")
    print("=" * 55)

    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()

    candidates = load_candidates()

    print("\nScoring candidates...")
    results = rank_candidates(candidates)

    top100 = results[:100]

    print(f"\nTop 5 candidates:")
    for i, r in enumerate(top100[:5], 1):
        print(f"  #{i} {r['candidate_id']}  score={r['score']:.4f}  "
              f"decision={r['decision']}  clusters={r['matched_clusters'][:2]}")

    write_csv(top100)
    write_json(top100)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"\nNext steps:")
    print(f"  1. Validate:  python validate_submission.py data/team_keerthana.csv")
    print(f"  2. Backend:   uvicorn backend.main:app --reload")
    print(f"  3. Frontend:  cd frontend && npm run dev")


if __name__ == "__main__":
    main()
