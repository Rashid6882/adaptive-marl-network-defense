# Module docstring: a fixed-threshold rule-based defense that imitates a traditional IDS (like Snort). It does
# no learning; the thresholds are simple guesses based on how the attacks look.
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

# Imports numpy (not used in this file).
import numpy as np

# Class names; list position is the class number.
CLASSES = ["BENIGN", "DDoS", "PortScan", "BruteForce"]


# Function that classifies one flow (a row of data) with fixed rules and returns a class number.
def rule_based_detect(row) -> int:
    # Reads the packet rate of the flow.
    pkt_rate = row["Flow Packets/s"]
    # Reads the flow duration.
    duration = row["Flow Duration"]
    # Reads the SYN flag count.
    syn = row["SYN Flag Count"]
    # Reads the destination port.
    dport = row["Destination Port"]
    # Reads the repeat-count feature (flows to the same port in the last 50).
    repeat = row["dport_repeat_count"]
    # Reads the number of forward packets.
    fwd_pkts = row["Total Fwd Packets"]

    # DDoS rule: a very high packet rate (above 2,000 per second) with more than 10 forward packets.
    if pkt_rate > 2000 and fwd_pkts > 10:
        # If the DDoS rule matched, return the DDoS class number.
        return CLASSES.index("DDoS")
    # PortScan rule: at least one SYN flag, a very short flow (under 1,000 microseconds) and a repeat count of
    # 10 or more.
    if syn >= 1 and duration < 1000 and repeat >= 10:
        # If the PortScan rule matched, return the PortScan class number.
        return CLASSES.index("PortScan")
    # BruteForce rule: destination port is FTP or SSH, repeat count at least 5 and a duration under 2,000,000
    # microseconds.
    if dport in (21, 22) and repeat >= 5 and duration < 2_000_000:
        # If the BruteForce rule matched, return the BruteForce class number.
        return CLASSES.index("BruteForce")
    # If no rule matched, call the flow benign.
    return CLASSES.index("BENIGN")


# Firewall rule: takes the detection verdict and returns a Firewall action.
def rule_based_firewall(det_action: int) -> int:
    # Allow / Rate-limit / Block
    # Checks whether the verdict was benign.
    if det_action == 0:
        # Benign flows are allowed (action 0).
        return 0
    # Anything flagged as an attack is blocked (action 2).
    return 2  # block anything flagged as an attack


# Response rule: takes the Firewall action, the repeat count and a threshold (default 5).
def rule_based_response(fw_action: int, repeat_count: int, threshold: int = 5) -> int:
    # No action / Alert / Isolate
    # Checks whether the Firewall allowed the flow.
    if fw_action == 0:
        # If allowed, take no action (0).
        return 0
    # Checks whether the source is a repeat offender (count above the threshold).
    if repeat_count > threshold:
        # Repeat offender: isolate (2).
        return 2
    # Otherwise a first-time offender: send an alert (1).
    return 1
