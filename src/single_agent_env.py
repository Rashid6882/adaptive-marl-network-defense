"""
Action encoding for the single-agent baseline: one PPO agent picks the joint
(detection, firewall, response) action, a flattened product of the three
agents' actions (4 x 3 x 3 = 36). The environment itself is SingleSimEnv in
src/sim_env.py.
"""

from src.features import N_CLASSES

N_FW = 3
N_RESP = 3
N_ACTIONS = N_CLASSES * N_FW * N_RESP


def decode_action(a: int):
    det = a // (N_FW * N_RESP)
    rem = a % (N_FW * N_RESP)
    fw = rem // N_RESP
    resp = rem % N_RESP
    return det, fw, resp
