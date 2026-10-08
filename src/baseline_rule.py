"""
Rule-based static-threshold baseline (Section 12): mimics a traditional
signature/threshold IDS like Snort -- fixed rules, no learning. Thresholds are
simple heuristics tuned against the general shape of CICIDS2017 attacks:
SYN floods / DDoS show very high packet rate with few distinct ports; port
scans show a high flag count with very short/zero-duration flows and a high
`dport_repeat_count` (many flows hitting different-but-nearby ports fast);
brute force (FTP/SSH-Patator) shows many repeated short flows to the same
(low) destination port with low payload size.
"""

import numpy as np

CLASSES = ["BENIGN", "DDoS", "PortScan", "BruteForce"]


def rule_based_detect(row) -> int:
    pkt_rate = row["Flow Packets/s"]
    duration = row["Flow Duration"]
    syn = row["SYN Flag Count"]
    dport = row["Destination Port"]
    repeat = row["dport_repeat_count"]
    fwd_pkts = row["Total Fwd Packets"]

    if pkt_rate > 2000 and fwd_pkts > 10:
        return CLASSES.index("DDoS")
    if syn >= 1 and duration < 1000 and repeat >= 10:
        return CLASSES.index("PortScan")
    if dport in (21, 22) and repeat >= 5 and duration < 2_000_000:
        return CLASSES.index("BruteForce")
    return CLASSES.index("BENIGN")


def rule_based_firewall(det_action: int) -> int:
    # Allow / Rate-limit / Block
    if det_action == 0:
        return 0
    return 2  # block anything flagged as an attack


def rule_based_response(fw_action: int, repeat_count: int, threshold: int = 5) -> int:
    # No action / Alert / Isolate
    if fw_action == 0:
        return 0
    if repeat_count > threshold:
        return 2
    return 1
