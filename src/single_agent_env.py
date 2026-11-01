# Module docstring: the single-agent baseline picks all three decisions at once as one of 36 joint actions (4
# x 3 x 3). The environment itself lives in sim_env.py.
"""
Action encoding for the single-agent baseline: one PPO agent picks the joint
(detection, firewall, response) action, a flattened product of the three
agents' actions (4 x 3 x 3 = 36). The environment itself is SingleSimEnv in
src/sim_env.py.
"""

# Imports the number of detection classes (4).
from src.features import N_CLASSES

# Number of Firewall actions (3).
N_FW = 3
# Number of Response actions (3).
N_RESP = 3
# Total joint actions: 4 x 3 x 3 = 36.
N_ACTIONS = N_CLASSES * N_FW * N_RESP


# Function that turns one joint action number (0-35) back into the three separate actions.
def decode_action(a: int):
    # Detection action = the number divided by 9 (integer division), because each detection choice covers 9
    # firewall/response pairs.
    det = a // (N_FW * N_RESP)
    # Remainder after removing the detection part (a number from 0 to 8).
    rem = a % (N_FW * N_RESP)
    # Firewall action = remainder divided by 3.
    fw = rem // N_RESP
    # Response action = remainder modulo 3.
    resp = rem % N_RESP
    # Returns the three actions as a tuple (detection, firewall, response).
    return det, fw, resp
