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

from collections import deque

import numpy as np
import pandas as pd

DATASET_DIR = "DATASET"

FILES = {
    "Tuesday-WorkingHours.pcap_ISCX.csv": "BruteForce",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "DDoS",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "PortScan",
}

# Canonical attack classes used by the Detection agent's action space.
LABEL_MAP = {
    "BENIGN": "BENIGN",
    "DDOS": "DDoS",
    "PORTSCAN": "PortScan",
    "FTP-PATATOR": "BruteForce",
    "SSH-PATATOR": "BruteForce",
}

CLASSES = ["BENIGN", "DDoS", "PortScan", "BruteForce"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

ROLLING_WINDOW = 50  # flows looked back when computing dport_repeat_count


def _clean_one(path: str, source_file: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    df["Label"] = df["Label"].astype(str).str.strip().str.upper()
    df["Label"] = df["Label"].map(LABEL_MAP)
    df = df.dropna(subset=["Label"])  # drop anything unexpected

    # inf/-inf -> NaN, then drop rows with NaN in the affected columns
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Flow Bytes/s", "Flow Packets/s"])

    df = df.drop_duplicates()

    df["source_file"] = source_file
    df["attack_type"] = df["Label"]

    # Preserve original row order (chronological within the capture) as the
    # engineered rolling feature and the chronological split both depend on it.
    df = df.reset_index(drop=True)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-file rolling repeat-count proxy, computed in original row order."""
    out = []
    for _, g in df.groupby("source_file", sort=False):
        g = g.copy()
        dport = g["Destination Port"].values
        counts = np.zeros(len(g), dtype=np.float32)
        buf = deque()
        freq = {}
        for i, p in enumerate(dport):
            buf.append(p)
            freq[p] = freq.get(p, 0) + 1
            if len(buf) > ROLLING_WINDOW:
                old = buf.popleft()
                freq[old] -= 1
                if freq[old] == 0:
                    del freq[old]
            counts[i] = freq[p]
        g["dport_repeat_count"] = counts
        out.append(g)
    return pd.concat(out, axis=0).reset_index(drop=True)


def load_clean_dataset() -> pd.DataFrame:
    frames = []
    for fname, _ in FILES.items():
        path = f"{DATASET_DIR}/{fname}"
        frames.append(_clean_one(path, fname))
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = _add_rolling_features(df)
    # FTP (21) / SSH (22): the only ports Patator brute-force targets. A raw
    # port number is hard for an MLP to isolate after standard scaling, and
    # only ~0.6% of benign flows use these ports.
    df["is_ftp_ssh_port"] = df["Destination Port"].isin([21, 22]).astype(np.float32)
    return df


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Per-file chronological split: first train_frac rows of each file -> train,
    remainder -> test. Keeps all attack types represented in both splits while
    never letting later-captured traffic leak into training (see Section 11)."""
    train_parts, test_parts = [], []
    for _, g in df.groupby("source_file", sort=False):
        n = len(g)
        cut = int(n * train_frac)
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)
    return train_df, test_df


if __name__ == "__main__":
    df = load_clean_dataset()
    print("Total rows after cleaning:", len(df))
    print(df["Label"].value_counts())
    train_df, test_df = chronological_split(df)
    print("Train:", len(train_df), "Test:", len(test_df))
    print(train_df["Label"].value_counts())
    print(test_df["Label"].value_counts())
