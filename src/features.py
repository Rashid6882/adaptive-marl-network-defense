# Module docstring: this file holds the feature lists, the class names and the label-based reward tables (the
# "shaping" terms) for the three agents. A missed attack costs more than a false alarm.
"""Per-agent feature lists and normalization stats, and reward functions.

Reward design follows Section 8/9 of the project doc: asymmetric, role-specific
costs (false negative >> false positive at Detection; over-blocking cost at
Firewall; unnecessary-isolation cost at Response).
"""

# Imports numpy (it is not actually used by the functions below).
import numpy as np

# The four traffic classes. Their order fixes the Detection agent's action numbers: 0 = BENIGN, 1 = DDoS, 2 =
# PortScan, 3 = BruteForce.
CLASSES = ["BENIGN", "DDoS", "PortScan", "BruteForce"]
# Dictionary that maps a class name to its number (built by enumerating CLASSES).
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
# Number of classes (4), used for sizes of action spaces and arrays.
N_CLASSES = len(CLASSES)

# Firewall agent's three actions; list position is the action number (0 Allow, 1 Rate-limit, 2 Block).
FIREWALL_ACTIONS = ["Allow", "Rate-limit", "Block"]
# Response agent's three actions (0 No action, 1 Alert, 2 Isolate).
RESPONSE_ACTIONS = ["No action", "Alert", "Isolate"]

# ~19 features: flow duration, packet counts/rates, packet length stats,
# SYN/ACK/FIN flag counts, destination port, + rolling repeat-count proxy.
# Starts the list of column names that Detection observes.
DETECTION_FEATURES = [
    # 19 raw CICIDS2017 flow columns: destination port, duration, packet counts, byte and packet rates, packet
    # length statistics, TCP flag counts and inter-arrival-time statistics.
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
    # Engineered feature: how many of the last 50 flows used the same destination port (stand-in for "repeat
    # offender", because the data has no Source IP).
    "dport_repeat_count",
    # Engineered feature: 1 if the destination port is FTP (21) or SSH (22), the ports brute-force attacks
    # hit.
    "is_ftp_ssh_port",
# Closes the feature list (21 features in total).
]

# A destination-port repeat count above 5 marks a "repeat offender".
REPEAT_OFFENDER_THRESHOLD = 5  # dport_repeat_count above this => "repeat offender"

# Inverse-frequency weights (BENIGN 73.5%, DDoS 14.9%, PortScan 10.6%,
# BruteForce 1.1% of cleaned rows -- see src/data.py). Without this, PPO's
# cumulative-reward objective lets a policy that never predicts the rarest
# classes still score well overall, since correctly handling the BENIGN/DDoS
# majority dominates total return. Scaling the correct-prediction reward and
# the false-negative penalty by class rarity amplifies the gradient signal
# minority classes get per occurrence, compensating for seeing them less often.
# Starts the dictionary of Detection reward weights per true class.
DETECTION_CLASS_WEIGHT = {
    # Benign traffic has weight 1.0.
    "BENIGN": 1.0,
    # DDoS weight 2.5 (equal weights for all attacks, which fixed the failing seed).
    "DDoS": 2.5,
    # PortScan weight 2.5.
    "PortScan": 2.5,
    # BruteForce weight 2.5.
    "BruteForce": 2.5,
# Closes the weight dictionary.
}


# Detection's shaping reward: takes the predicted class number and the true class number and returns a float.
def detection_reward(pred_idx: int, true_idx: int) -> float:
    # Looks up the weight for the true class.
    w = DETECTION_CLASS_WEIGHT[CLASSES[true_idx]]
    # Checks whether the prediction is correct.
    if pred_idx == true_idx:
        # Correct prediction: reward +1 times the class weight.
        return 1.0 * w
    # Checks for a false negative: a real attack that was called benign.
    if true_idx != 0 and pred_idx == 0:  # attack misclassified as benign (FN)
        # Missed attack: penalty -1 times the class weight (the worst Detection mistake).
        return -1.0 * w
    # Checks for a false alarm: benign traffic called an attack.
    if true_idx == 0 and pred_idx != 0:  # benign misclassified as attack
        # False alarm: penalty of -0.5.
        return -0.5
    # Any other case (an attack named as the wrong attack type): penalty of -0.5.
    return -0.5  # one attack type confused for another


# Firewall's shaping reward: takes the Firewall action and the true class number.
def firewall_reward(fw_action: int, true_idx: int) -> float:
    # True if the flow is really an attack (class number not 0).
    is_attack = true_idx != 0
    # Branch for real attacks.
    if is_attack:
        # Checks whether the action was Block.
        if fw_action == 2:  # Block
            # Blocking a real attack: reward +1.
            return 1.0
        # Checks whether the action was Rate-limit.
        if fw_action == 1:  # Rate-limit: under-reacting, smaller penalty
            # Rate-limiting an attack: small reward +0.3 (under-reacting).
            return 0.3
        # Allowing a real attack: penalty -1 (worst case).
        return -1.0  # Allow on a real attack: worst case
    # Else branch: the flow is benign.
    else:
        # Checks whether the action was Allow.
        if fw_action == 0:  # Allow
            # Allowing benign traffic: reward +1.
            return 1.0
        # Checks whether the action was Rate-limit.
        if fw_action == 1:  # Rate-limit: mild over-reaction
            # Rate-limiting benign traffic: mild penalty -0.3.
            return -0.3
        # Blocking benign traffic: penalty -1 (over-blocking).
        return -1.0  # Block on benign: over-blocking


# Response's shaping reward: takes the Response action, the true class number and a repeat-offender flag.
def response_reward(resp_action: int, true_idx: int, is_repeat_offender: bool) -> float:
    # True if the flow is a real attack.
    is_attack = true_idx != 0
    # Branch for real attacks.
    if is_attack:
        # Sub-branch: the attacker has been seen repeatedly.
        if is_repeat_offender:
            # Checks whether the action was Isolate.
            if resp_action == 2:  # Isolate on confirmed repeat attacker
                # Isolating a confirmed repeat attacker: reward +1.
                return 1.0
            # Checks whether the action was Alert.
            if resp_action == 1:  # Alert: correct low-cost middle action
                # Alerting on a repeat attacker: small reward +0.2.
                return 0.2
            # Doing nothing about a repeat attacker: penalty -0.5.
            return -0.5  # No action on a repeat attacker
        # Else: first-seen attacker.
        else:
            # Checks whether the action was Alert.
            if resp_action == 1:  # Alert on a first-seen attack: good
                # Alerting on a first-seen attack: reward +0.5 (the right low-cost step).
                return 0.5
            # Checks whether the action was Isolate.
            if resp_action == 2:  # Isolate: premature, smaller penalty
                # Isolating a first-seen attack: small penalty -0.2 (premature).
                return -0.2
            # Doing nothing about an attack: penalty -0.3.
            return -0.3  # No action on an attack
    # Else: the flow is benign.
    else:
        # Checks whether the action was "No action".
        if resp_action == 0:  # No action on benign: correct
            # No action on benign traffic: reward +0.2 (correct).
            return 0.2
        # Checks whether the action was Alert.
        if resp_action == 1:  # Alert on benign: mild over-caution
            # Alerting on benign traffic: penalty -0.2 (mild over-caution).
            return -0.2
        # Isolating benign traffic: penalty -1 (worst case).
        return -1.0  # Isolate on benign: worst case
