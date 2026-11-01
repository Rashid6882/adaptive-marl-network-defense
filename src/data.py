# Module docstring: this file loads and cleans the three CICIDS2017 CSV files (Tuesday brute force, Friday
# DDoS, Friday PortScan) and splits them in time order.
# Lists the known problems in the data that this file fixes: stray spaces in column names, infinite or missing
# values, duplicate rows and inconsistent label spelling.
# Explains an important limitation: this CSV release has no Source IP column, so a repeat count keyed on
# destination port is used as a stand-in for "repeat offender". The feature is named dport_repeat_count so
# nobody mistakes it for a real source-IP count.
"""
Load, clean and feature-engineer the three CICIDS2017 CSVs used by this project
(Tuesday brute-force, Friday-DDoS, Friday-PortScan), and split chronologically
into train/test.

Known dataset quirks handled here (see kickoff brief):
  - leading whitespace in column names
  - inf/-inf and NaN in Flow Bytes/s, Flow Packets/s
  - exact duplicate rows
  - inconsistent Label spacing/casing

IMPORTANT DATASET LIMITATION:
This CSV variant (the standard "MachineLearningCVE" release of CICIDS2017) does
NOT include a Source IP column -- it only has Destination Port onward. The
project spec calls for a "rolling count of flows from the same source IP" /
"source IP repeat count" as part of the Detection/Firewall/Response state.
Since source IP isn't available, we substitute a documented proxy: a rolling
repeat-count keyed on Destination Port (+ attack-signature-like grouping is not
possible either since Protocol/Flow ID are also absent). Repeated flows to the
same destination port in a short window is the same underlying signal the
original feature was meant to capture (bursty / repeated behavior from a single
actor), so the proxy preserves the intent of the feature even though it is not
literally "source IP". This is called out explicitly wherever the feature is
used (`dport_repeat_count` in the code, not `src_ip_repeat_count`).
"""

# Imports deque (a fast double-ended queue) for the rolling window.
from collections import deque

# Imports numpy for numeric arrays.
import numpy as np
# Imports pandas for tables.
import pandas as pd

# Folder that holds the CSV files.
DATASET_DIR = "DATASET"

# Starts the dictionary that links each CSV file to its main attack type.
FILES = {
    # Tuesday file contains the brute-force attacks.
    "Tuesday-WorkingHours.pcap_ISCX.csv": "BruteForce",
    # Friday afternoon DDoS file.
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "DDoS",
    # Friday afternoon PortScan file.
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "PortScan",
# Closes the dictionary.
}

# Canonical attack classes used by the Detection agent's action space.
# Starts the label-spelling map (raw label to our class name).
LABEL_MAP = {
    # BENIGN stays BENIGN.
    "BENIGN": "BENIGN",
    # DDOS becomes DDoS.
    "DDOS": "DDoS",
    # PORTSCAN becomes PortScan.
    "PORTSCAN": "PortScan",
    # FTP-Patator (FTP brute force) becomes BruteForce.
    "FTP-PATATOR": "BruteForce",
    # SSH-Patator (SSH brute force) also becomes BruteForce.
    "SSH-PATATOR": "BruteForce",
# Closes the map.
}

# The class list; position is the class number.
CLASSES = ["BENIGN", "DDoS", "PortScan", "BruteForce"]
# Dictionary from class name to number.
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# The repeat count looks back over the last 50 flows.
ROLLING_WINDOW = 50  # flows looked back when computing dport_repeat_count


# Function that loads and cleans one CSV file.
def _clean_one(path: str, source_file: str) -> pd.DataFrame:
    # Reads the CSV into a table (low_memory=False avoids mixed-type warnings).
    df = pd.read_csv(path, low_memory=False)
    # Removes leading/trailing spaces from every column name.
    df.columns = df.columns.str.strip()

    # Converts labels to text, strips spaces and makes them upper case so spelling is consistent.
    df["Label"] = df["Label"].astype(str).str.strip().str.upper()
    # Maps raw labels to our four class names; labels not in the map become missing.
    df["Label"] = df["Label"].map(LABEL_MAP)
    # Drops rows with an unexpected label.
    df = df.dropna(subset=["Label"])  # drop anything unexpected

    # inf/-inf -> NaN, then drop rows with NaN in the affected columns
    # Replaces +infinity and -infinity with NaN (missing).
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drops rows where Flow Bytes/s or Flow Packets/s is missing.
    df = df.dropna(subset=["Flow Bytes/s", "Flow Packets/s"])

    # Removes exact duplicate rows.
    df = df.drop_duplicates()

    # Records which file each row came from.
    df["source_file"] = source_file
    # Copies the label into an attack_type column.
    df["attack_type"] = df["Label"]

    # Preserve original row order (chronological within the capture) as the
    # engineered rolling feature and the chronological split both depend on it.
    # Renumbers the rows 0, 1, 2... while keeping their order.
    df = df.reset_index(drop=True)
    # Returns the cleaned table.
    return df


# Function that adds the rolling repeat-count feature.
def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # Docstring: computed per file, in the original row order.
    """Per-file rolling repeat-count proxy, computed in original row order."""
    # List that will collect the processed pieces.
    out = []
    # Loops over each source file's rows separately (sort=False keeps file order).
    for _, g in df.groupby("source_file", sort=False):
        # Copies the piece so the original is not modified.
        g = g.copy()
        # Gets the destination-port values as an array.
        dport = g["Destination Port"].values
        # Creates an array of zeros that will hold the count for each flow.
        counts = np.zeros(len(g), dtype=np.float32)
        # Queue holding the ports of the last 50 flows.
        buf = deque()
        # Dictionary counting how many times each port appears in that queue.
        freq = {}
        # Loops over the flows in order (i is the position, p is the port).
        for i, p in enumerate(dport):
            # Adds this flow's port to the window.
            buf.append(p)
            # Increases that port's count by 1.
            freq[p] = freq.get(p, 0) + 1
            # Checks whether the window now holds more than 50 flows.
            if len(buf) > ROLLING_WINDOW:
                # Removes the oldest port from the window.
                old = buf.popleft()
                # Decreases that old port's count.
                freq[old] -= 1
                # If the old port's count reached zero...
                if freq[old] == 0:
                    # ...removes it from the dictionary to keep it small.
                    del freq[old]
            # Stores how many times the current port appears in the window (including this flow).
            counts[i] = freq[p]
        # Adds the counts as the dport_repeat_count column.
        g["dport_repeat_count"] = counts
        # Saves this file's piece.
        out.append(g)
    # Joins all pieces back into one table and renumbers the rows.
    return pd.concat(out, axis=0).reset_index(drop=True)


# Function that loads all three files into one clean table.
def load_clean_dataset() -> pd.DataFrame:
    # List to collect the three cleaned tables.
    frames = []
    # Loops over the file names (the attack type is not needed here).
    for fname, _ in FILES.items():
        # Builds the full path to the CSV file.
        path = f"{DATASET_DIR}/{fname}"
        # Cleans the file and adds the result to the list.
        frames.append(_clean_one(path, fname))
    # Stacks the three tables into one.
    df = pd.concat(frames, axis=0, ignore_index=True)
    # Adds the rolling repeat-count feature.
    df = _add_rolling_features(df)
    # FTP (21) / SSH (22): the only ports Patator brute-force targets. A raw
    # port number is hard for an MLP to isolate after standard scaling, and
    # only ~0.6% of benign flows use these ports.
    # Creates the is_ftp_ssh_port column: 1 if the destination port is 21 or 22, else 0.
    df["is_ftp_ssh_port"] = df["Destination Port"].isin([21, 22]).astype(np.float32)
    # Returns the finished table.
    return df


# Function that splits the data by time (default: first 80% train, rest test).
def chronological_split(df: pd.DataFrame, train_frac: float = 0.8):
    # Docstring: the split is done per file, so all attack types appear in both parts and later traffic never
    # leaks into training. (The simulator code does its own 70/10/20 split.)
    """Per-file chronological split: first train_frac rows of each file -> train,
    remainder -> test. Keeps all attack types represented in both splits while
    never letting later-captured traffic leak into training (see Section 11)."""
    # Two empty lists for the training and test pieces.
    train_parts, test_parts = [], []
    # Loops over each file's rows.
    for _, g in df.groupby("source_file", sort=False):
        # Number of rows in this file.
        n = len(g)
        # Position where the split happens (80% of the rows).
        cut = int(n * train_frac)
        # First part (earlier traffic) goes to training.
        train_parts.append(g.iloc[:cut])
        # Remaining part (later traffic) goes to testing.
        test_parts.append(g.iloc[cut:])
    # Joins the training pieces and renumbers rows.
    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    # Joins the test pieces and renumbers rows.
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)
    # Returns both tables.
    return train_df, test_df


# Runs the code below only when this file is run directly.
if __name__ == "__main__":
    # Loads and cleans the dataset.
    df = load_clean_dataset()
    # Prints the total number of rows after cleaning.
    print("Total rows after cleaning:", len(df))
    # Prints how many rows each class has.
    print(df["Label"].value_counts())
    # Splits into train and test.
    train_df, test_df = chronological_split(df)
    # Prints the size of each part.
    print("Train:", len(train_df), "Test:", len(test_df))
    # Prints class counts in the training part.
    print(train_df["Label"].value_counts())
    # Prints class counts in the test part.
    print(test_df["Label"].value_counts())
