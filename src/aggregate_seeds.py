"""
Aggregate per-seed closed-loop results into mean +/- std tables.

    python -m src.aggregate_seeds --glob "runs/seed*/eval_test.json" --out runs/summary_test
"""

import argparse
import glob
import json

import numpy as np

METRICS = [
    ("total_loss", "Total loss", "{:,.0f}"),
    ("damage", "Attack damage", "{:,.0f}"),
    ("disruption", "Benign disruption", "{:,.0f}"),
    ("damage_reduction", "Damage reduction", "{:.1%}"),
    ("benign_disrupted_rate", "Benign flows disrupted", "{:.2%}"),
    ("mean_latency_flows", "Mean latency (flows)", "{:.1f}"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True, help="output prefix; writes .md and .json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")
    runs = [json.load(open(f)) for f in files]
    systems = list(runs[0].keys())

    summary = {}
    for sysname in systems:
        summary[sysname] = {}
        for key, _, _ in METRICS:
            vals = [r[sysname][key] for r in runs if sysname in r and r[sysname][key] is not None]
            summary[sysname][key] = (
                {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals), "values": vals}
                if vals else None
            )

    lines = [f"Seeds aggregated: {len(files)} ({', '.join(files)})", "",
             "| System | " + " | ".join(m[1] for m in METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    for sysname in systems:
        cells = []
        for key, _, fmt in METRICS:
            s = summary[sysname][key]
            cells.append("n/a" if s is None else f"{fmt.format(s['mean'])} +/- {fmt.format(s['std'])}")
        lines.append(f"| {sysname} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    open(args.out + ".md", "w", encoding="utf-8").write(text + "\n")
    json.dump(summary, open(args.out + ".json", "w"), indent=2)


if __name__ == "__main__":
    main()
