# Module docstring: helper pieces shared by the simulator: the column positions of the two raw features the
# Firewall sees, the feature scaler, and the scaled-array builder.
"""
Shared feature helpers for the defense simulator (src/sim_env.py): the column
indices of the two raw features the Firewall observes, the feature scaler, and
the scaled-array builder.
"""

# Imports numpy for array operations.
import numpy as np
# Imports StandardScaler, which rescales each feature to mean 0 and standard deviation 1.
from sklearn.preprocessing import StandardScaler

# Imports the class list and the Detection feature list from features.py.
from src.features import CLASSES, DETECTION_FEATURES

# Position of "dport_repeat_count" inside the feature list, so the Firewall can read that column from the
# array.
DPORT_IDX = DETECTION_FEATURES.index("dport_repeat_count")
# Position of "Flow Packets/s" inside the feature list (the packet-rate feature the Firewall sees).
PKTS_IDX = DETECTION_FEATURES.index("Flow Packets/s")


# Function that fits the scaler on training data.
def fit_scaler(train_df) -> StandardScaler:
    # Creates an empty StandardScaler.
    scaler = StandardScaler()
    # Learns each feature's mean and standard deviation from the training rows only (no test leakage).
    scaler.fit(train_df[DETECTION_FEATURES].values)
    # Returns the fitted scaler.
    return scaler


# Function that turns a DataFrame into model-ready arrays.
def _prep_arrays(df, scaler: StandardScaler):
    # Applies the scaler to the 21 feature columns.
    X = scaler.transform(df[DETECTION_FEATURES].values)
    # Clips scaled values to the range -8 to 8 (limits extreme outliers) and converts to 32-bit floats.
    X = np.clip(X, -8.0, 8.0).astype(np.float32)
    # Converts the Label column to class numbers (0-3) as 64-bit integers.
    labels = df["Label"].map({c: i for i, c in enumerate(CLASSES)}).values.astype(np.int64)
    # Returns the feature array and the label array.
    return X, labels
