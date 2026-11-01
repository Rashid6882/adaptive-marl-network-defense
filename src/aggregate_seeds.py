# Module docstring title: combines the per-seed results into mean plus or minus standard deviation tables.
# Example command.
# Ends the docstring.
"""
Aggregate per-seed closed-loop results into mean +/- std tables.

    python -m src.aggregate_seeds --glob "runs/seed*/eval_test.json" --out runs/summary_test
"""

# Imports argparse for command-line options.
import argparse
# Imports glob to find files by pattern.
import glob
# Imports json.
import json

# Imports numpy for mean and standard deviation.
import numpy as np

# Starts the list of metrics to summarize: (key in the results file, column title, number format).
METRICS = [
    # Total loss, shown with thousands separators and no decimals.
    ("total_loss", "Total loss", "{:,.0f}"),
    # Attack damage.
    ("damage", "Attack damage", "{:,.0f}"),
    # Benign disruption.
    ("disruption", "Benign disruption", "{:,.0f}"),
    # Damage reduction, shown as a percentage with one decimal.
    ("damage_reduction", "Damage reduction", "{:.1%}"),
    # Share of benign flows disrupted, shown as a percentage with two decimals.
    ("benign_disrupted_rate", "Benign flows disrupted", "{:.2%}"),
    # Mean containment latency in flows, one decimal.
    ("mean_latency_flows", "Mean latency (flows)", "{:.1f}"),
# Ends the list.
]


# Main function.
def main():
    # Creates the command-line parser.
    ap = argparse.ArgumentParser()
    # Required option: the file pattern to aggregate.
    ap.add_argument("--glob", required=True)
    # Required option: output file prefix (writes a .md and a .json file).
    ap.add_argument("--out", required=True, help="output prefix; writes .md and .json")
    # Reads the options.
    args = ap.parse_args()

    # Finds the matching files and sorts them.
    files = sorted(glob.glob(args.glob))
    # If no file matches...
    if not files:
        # ...stop with an error message.
        raise SystemExit(f"no files match {args.glob}")
    # Loads each seed's results file.
    runs = [json.load(open(f)) for f in files]
    # The system names (none, rule, rule_tuned, v2_marl, v2_single) from the first file.
    systems = list(runs[0].keys())

    # Dictionary that will hold the summary.
    summary = {}
    # Loops over the systems.
    for sysname in systems:
        # Creates an entry for this system.
        summary[sysname] = {}
        # Loops over the metrics.
        for key, _, _ in METRICS:
            # Collects the metric's value from every seed (skipping missing values).
            vals = [r[sysname][key] for r in runs if sysname in r and r[sysname][key] is not None]
            # Starts storing the summary for this system and metric.
            summary[sysname][key] = (
                # Stores the mean, the sample standard deviation (0 if only one seed), the number of seeds and
                # the raw values.
                {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals), "values": vals}
                # Stores None if there are no values.
                if vals else None
            # Ends the assignment.
            )

    # First text line: how many seeds and which files.
    lines = [f"Seeds aggregated: {len(files)} ({', '.join(files)})", "",
             # Table header: system name plus each metric title.
             "| System | " + " | ".join(m[1] for m in METRICS) + " |",
             # Table separator row.
             "|---|" + "---|" * len(METRICS)]
    # Loops over the systems again to build the table rows.
    for sysname in systems:
        # Empty list of cells for this row.
        cells = []
        # Loops over the metrics.
        for key, _, fmt in METRICS:
            # Gets the summary for this metric.
            s = summary[sysname][key]
            # Writes "n/a" if missing, otherwise "mean +/- std" in the metric's format.
            cells.append("n/a" if s is None else f"{fmt.format(s['mean'])} +/- {fmt.format(s['std'])}")
        # Adds the finished row to the table.
        lines.append(f"| {sysname} | " + " | ".join(cells) + " |")
    # Joins all lines into one text block.
    text = "\n".join(lines)
    # Prints the table.
    print(text)
    # Saves the table as a Markdown file.
    open(args.out + ".md", "w", encoding="utf-8").write(text + "\n")
    # Saves the summary numbers as a JSON file.
    json.dump(summary, open(args.out + ".json", "w"), indent=2)


# Runs main() only when the file is run directly.
if __name__ == "__main__":
    # Calls main().
    main()
