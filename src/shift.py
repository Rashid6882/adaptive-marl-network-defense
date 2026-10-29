"""
Synthetic attack-variant shifts for the adaptation experiment (see runs/PREREGISTRATION.md).

A "slow" variant of one attack class: rates are multiplied by `r` and times divided by `r`, as if the
attacker sent the same flows more slowly. This is a simulated variant, not captured traffic; the rolling
repeat-count feature is left unchanged.
"""

RATE_COLS = ["Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"]
TIME_COLS = ["Flow Duration", "Flow IAT Mean", "Flow IAT Std"]

VARIANTS = {
    "slow_ddos": ("DDoS", 0.05),
    "slow_portscan": ("PortScan", 0.05),
}


def apply_variant(df, name):
    cls, r = VARIANTS[name]
    out = df.copy()
    for col in RATE_COLS + TIME_COLS:
        out[col] = out[col].astype("float64")
    m = (out["Label"] == cls).values
    out.loc[m, RATE_COLS] = out.loc[m, RATE_COLS].values * r
    out.loc[m, TIME_COLS] = out.loc[m, TIME_COLS].values / r
    return out
