"""Per-agent feature lists and normalization stats, and reward functions.

Reward design follows Section 8/9 of the project doc: asymmetric, role-specific
costs (false negative >> false positive at Detection; over-blocking cost at
Firewall; unnecessary-isolation cost at Response).
"""

import numpy as np

CLASSES = ["BENIGN", "DDoS", "PortScan", "BruteForce"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

FIREWALL_ACTIONS = ["Allow", "Rate-limit", "Block"]
RESPONSE_ACTIONS = ["No action", "Alert", "Isolate"]

# ~19 features: flow duration, packet counts/rates, packet length stats,
# SYN/ACK/FIN flag counts, destination port, + rolling repeat-count proxy.
DETECTION_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "Min Packet Length",
    "Max Packet Length",
    "SYN Flag Count",
    "ACK Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "Flow IAT Mean",
    "Flow IAT Std",
    "dport_repeat_count",
    "is_ftp_ssh_port",
]

REPEAT_OFFENDER_THRESHOLD = 5  # dport_repeat_count above this => "repeat offender"

# Inverse-frequency weights (BENIGN 73.5%, DDoS 14.9%, PortScan 10.6%,
# BruteForce 1.1% of cleaned rows -- see src/data.py). Without this, PPO's
# cumulative-reward objective lets a policy that never predicts the rarest
# classes still score well overall, since correctly handling the BENIGN/DDoS
# majority dominates total return. Scaling the correct-prediction reward and
# the false-negative penalty by class rarity amplifies the gradient signal
# minority classes get per occurrence, compensating for seeing them less often.
DETECTION_CLASS_WEIGHT = {
    "BENIGN": 1.0,
    "DDoS": 2.5,
    "PortScan": 2.5,
    "BruteForce": 2.5,
}


def detection_reward(pred_idx: int, true_idx: int) -> float:
    w = DETECTION_CLASS_WEIGHT[CLASSES[true_idx]]
    if pred_idx == true_idx:
        return 1.0 * w
    if true_idx != 0 and pred_idx == 0:  # attack misclassified as benign (FN)
        return -1.0 * w
    if true_idx == 0 and pred_idx != 0:  # benign misclassified as attack
        return -0.5
    return -0.5  # one attack type confused for another


def firewall_reward(fw_action: int, true_idx: int) -> float:
    is_attack = true_idx != 0
    if is_attack:
        if fw_action == 2:  # Block
            return 1.0
        if fw_action == 1:  # Rate-limit: under-reacting, smaller penalty
            return 0.3
        return -1.0  # Allow on a real attack: worst case
    else:
        if fw_action == 0:  # Allow
            return 1.0
        if fw_action == 1:  # Rate-limit: mild over-reaction
            return -0.3
        return -1.0  # Block on benign: over-blocking


def response_reward(resp_action: int, true_idx: int, is_repeat_offender: bool) -> float:
    is_attack = true_idx != 0
    if is_attack:
        if is_repeat_offender:
            if resp_action == 2:  # Isolate on confirmed repeat attacker
                return 1.0
            if resp_action == 1:  # Alert: correct low-cost middle action
                return 0.2
            return -0.5  # No action on a repeat attacker
        else:
            if resp_action == 1:  # Alert on a first-seen attack: good
                return 0.5
            if resp_action == 2:  # Isolate: premature, smaller penalty
                return -0.2
            return -0.3  # No action on an attack
    else:
        if resp_action == 0:  # No action on benign: correct
            return 0.2
        if resp_action == 1:  # Alert on benign: mild over-caution
            return -0.2
        return -1.0  # Isolate on benign: worst case
