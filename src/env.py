"""
Shared feature helpers for the defense simulator (src/sim_env.py): the column
indices of the two raw features the Firewall observes, the feature scaler, and
the scaled-array builder.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.features import CLASSES, DETECTION_FEATURES

DPORT_IDX = DETECTION_FEATURES.index("dport_repeat_count")
PKTS_IDX = DETECTION_FEATURES.index("Flow Packets/s")


def fit_scaler(train_df) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[DETECTION_FEATURES].values)
    return scaler


def _prep_arrays(df, scaler: StandardScaler):
    X = scaler.transform(df[DETECTION_FEATURES].values)
    X = np.clip(X, -8.0, 8.0).astype(np.float32)
    labels = df["Label"].map({c: i for i, c in enumerate(CLASSES)}).values.astype(np.int64)
    return X, labels
