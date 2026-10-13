"""
Compares VADER's sentiment labels with your manual labels
and reports how accurate VADER is.
"""

import csv
from collections import Counter

from src.nlp_analysis import get_sentiment

LABELS_FILE = "evaluation/manual_labels.csv"
REPORT_FILE = "evaluation/sentiment_report.txt"


def main():
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No labels found. Run python -m scripts.make_label_set first.")
        return

    correct = 0
    mistakes = []
    pair_counts = Counter()  # counts of (my label, VADER label)

    for row in rows:
        mine = row["my_label"]
        vader = get_sentiment(row["text"])
        pair_counts[(mine, vader)] += 1

        if mine == vader:
            correct += 1
        else:
            mistakes.append((mine, vader, row["text"]))

    total = len(rows)
    accuracy = correct / total * 100

    lines = []
    lines.append("SENTIMENT EVALUATION (VADER vs my manual labels)")
    lines.append("=" * 50)
    lines.append(f"Chunks checked: {total}")
    lines.append(f"Correct: {correct}")
    lines.append(f"Accuracy: {accuracy:.1f}%")
    lines.append("")
    lines.append("My label -> VADER label : how many")
    for (mine, vader), count in sorted(pair_counts.items()):
        lines.append(f"  {mine} -> {vader} : {count}")
    lines.append("")
    lines.append(f"Disagreements ({len(mistakes)}):")
    for mine, vader, text in mistakes:
        lines.append("-" * 50)
        lines.append(f"Me: {mine} | VADER: {vader}")
        lines.append(text[:300])

    report = "\n".join(lines)
    print(report)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()