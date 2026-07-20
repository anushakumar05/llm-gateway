"""
Measure the hit-rate vs. false-hit-rate tradeoff across similarity thresholds.
Produces the data behind the README chart.
"""
import asyncio
import csv

from gateway.embeddings import embed_sync

# Hand-labeled pairs. TRUE = genuinely the same question (a hit is correct).
# FALSE = superficially similar but semantically different (a hit is a BUG).
PAIRS = [
    ("What is Python?", "Can you explain Python to me?", True),
    ("How do I reverse a list in Python?", "What's the way to reverse a Python list?", True),
    ("What is the capital of France?", "Which city is France's capital?", True),
    ("How do I install Docker?", "What are the steps to install Docker?", True),
    ("Explain recursion", "What is recursion?", True),
    ("How do I center a div?", "How do I center a div horizontally?", True),

    ("How do I install Python?", "How do I uninstall Python?", False),
    ("What is the capital of France?", "What is the capital of Germany?", False),
    ("How do I start the server?", "How do I stop the server?", False),
    ("Convert JSON to CSV", "Convert CSV to JSON", False),
    ("What is Python 2?", "What is Python 3?", False),
    ("How do I encrypt a file?", "How do I decrypt a file?", False),
    ("List files in a directory", "Delete files in a directory", False),
    ("What's the max of this array?", "What's the min of this array?", False),
]


def main():
    scored = []
    for a, b, same in PAIRS:
        sim = float(embed_sync(a) @ embed_sync(b))   # normalized -> dot = cosine
        scored.append((a, b, same, sim))
        print(f"{sim:.4f}  {'SAME ' if same else 'DIFF '}  {a[:40]!r} vs {b[:40]!r}")

    print("\nthreshold, hit_rate, false_hit_rate")
    rows = []
    for t in [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
        true_pairs = [s for *_, same, s in [(a, b, sm, s) for a, b, sm, s in scored] if same]
        false_pairs = [s for a, b, sm, s in scored if not sm]
        hit_rate = sum(1 for s in true_pairs if s >= t) / len(true_pairs)
        false_rate = sum(1 for s in false_pairs if s >= t) / len(false_pairs)
        rows.append((t, hit_rate, false_rate))
        print(f"{t:.2f}, {hit_rate:.2%}, {false_rate:.2%}")

    with open("threshold_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "hit_rate", "false_hit_rate"])
        w.writerows(rows)
    print("\nwrote threshold_sweep.csv")


if __name__ == "__main__":
    main()