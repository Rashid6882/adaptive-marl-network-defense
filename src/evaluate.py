"""
Evaluate the trained 3-agent MARL system against the rule-based baseline on
the held-out chronological test split (Section 12 metrics), produce plots,
and print a sample episode playback.
"""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from stable_baselines3 import PPO

from src.baseline_rule import rule_based_detect, rule_based_firewall, rule_based_response
from src.data import chronological_split, load_clean_dataset
from src.env import DPORT_IDX, PKTS_IDX, N_CLASSES, _prep_arrays
from src.features import CLASSES, FIREWALL_ACTIONS, RESPONSE_ACTIONS, REPEAT_OFFENDER_THRESHOLD
from src.single_agent_env import decode_action

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


def load_models():
    det = PPO.load(f"{MODELS_DIR}/detection")
    fw = PPO.load(f"{MODELS_DIR}/firewall")
    resp = PPO.load(f"{MODELS_DIR}/response")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.joblib")
    return det, fw, resp, scaler


def run_marl_system(df, det, fw, resp, scaler):
    """Vectorized-ish per-row replay of the full sequential pipeline over the
    whole test split, treated as one long stream (episode-repeat counters
    accumulate across the whole split, matching real deployment)."""
    X, labels = _prep_arrays(df, scaler)
    n = len(df)
    dport_raw = df["Destination Port"].values

    det_preds = np.zeros(n, dtype=np.int64)
    fw_preds = np.zeros(n, dtype=np.int64)
    resp_preds = np.zeros(n, dtype=np.int64)

    dport_counts = {}
    batch = 4096
    for start in range(0, n, batch):
        end = min(start + batch, n)
        obs_batch = X[start:end]
        det_actions, _ = det.predict(obs_batch, deterministic=True)
        det_preds[start:end] = det_actions

        onehot_det = np.zeros((end - start, N_CLASSES), dtype=np.float32)
        onehot_det[np.arange(end - start), det_actions] = 1.0
        fw_obs = np.concatenate(
            [onehot_det, X[start:end, [DPORT_IDX, PKTS_IDX]]], axis=1
        )
        fw_actions, _ = fw.predict(fw_obs, deterministic=True)
        fw_preds[start:end] = fw_actions

        onehot_fw = np.zeros((end - start, 3), dtype=np.float32)
        onehot_fw[np.arange(end - start), fw_actions] = 1.0
        ep_counts_scaled = np.zeros(end - start, dtype=np.float32)
        for i, row_idx in enumerate(range(start, end)):
            dp = dport_raw[row_idx]
            dport_counts[dp] = dport_counts.get(dp, 0) + 1
            ep_counts_scaled[i] = min(dport_counts[dp], 20) / 20.0
        resp_obs = np.concatenate([onehot_fw, ep_counts_scaled[:, None]], axis=1)
        resp_actions, _ = resp.predict(resp_obs, deterministic=True)
        resp_preds[start:end] = resp_actions

    return det_preds, fw_preds, resp_preds


def run_single_agent(df, model, scaler):
    X, _ = _prep_arrays(df, scaler)
    n = len(df)
    det_preds = np.zeros(n, dtype=np.int64)
    fw_preds = np.zeros(n, dtype=np.int64)
    resp_preds = np.zeros(n, dtype=np.int64)
    batch = 4096
    for start in range(0, n, batch):
        end = min(start + batch, n)
        actions, _ = model.predict(X[start:end], deterministic=True)
        for i, a in enumerate(actions):
            d, f, r = decode_action(int(a))
            det_preds[start + i] = d
            fw_preds[start + i] = f
            resp_preds[start + i] = r
    return det_preds, fw_preds, resp_preds


def run_rule_baseline(df):
    n = len(df)
    det_preds = np.zeros(n, dtype=np.int64)
    fw_preds = np.zeros(n, dtype=np.int64)
    resp_preds = np.zeros(n, dtype=np.int64)
    dport_counts = {}
    cols = df[
        [
            "Flow Packets/s",
            "Flow Duration",
            "SYN Flag Count",
            "Destination Port",
            "dport_repeat_count",
            "Total Fwd Packets",
        ]
    ].to_dict("records")
    for i, row in enumerate(cols):
        d = rule_based_detect(row)
        det_preds[i] = d
        f = rule_based_firewall(d)
        fw_preds[i] = f
        dp = row["Destination Port"]
        dport_counts[dp] = dport_counts.get(dp, 0) + 1
        resp_preds[i] = rule_based_response(f, dport_counts[dp], REPEAT_OFFENDER_THRESHOLD)
    return det_preds, fw_preds, resp_preds


def detection_metrics(true_idx, pred_idx, label="system"):
    f1_per_class = f1_score(true_idx, pred_idx, labels=range(N_CLASSES), average=None, zero_division=0)
    acc = (true_idx == pred_idx).mean()
    macro_f1 = f1_score(true_idx, pred_idx, average="macro", zero_division=0)
    print(f"\n--- Detection metrics [{label}] ---")
    print(f"Overall accuracy: {acc:.4f}   Macro F1: {macro_f1:.4f}")
    for c, f1 in zip(CLASSES, f1_per_class):
        print(f"  {c:12s} F1 = {f1:.4f}")
    return {"accuracy": float(acc), "macro_f1": float(macro_f1),
            "per_class_f1": dict(zip(CLASSES, [float(x) for x in f1_per_class]))}


def false_positive_rate(true_idx, fw_preds):
    """Benign flows that were Rate-limited or Blocked."""
    benign_mask = true_idx == 0
    if benign_mask.sum() == 0:
        return 0.0
    over_blocked = (fw_preds[benign_mask] != 0).mean()
    return float(over_blocked)


def plot_reward_curves():
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["detection", "firewall", "response"]:
        path = f"{OUTPUTS_DIR}/{name}_reward_curve.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rewards = json.load(f)
        if not rewards:
            continue
        window = max(1, len(rewards) // 20)
        smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean()
        ax.plot(smoothed, label=name)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode total reward")
    ax.set_title("Training reward curves (smoothed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUTS_DIR}/reward_curves.png", dpi=150)
    print(f"Saved {OUTPUTS_DIR}/reward_curves.png")


def plot_detection_comparison(marl_metrics, rule_metrics):
    classes = CLASSES
    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, [rule_metrics["per_class_f1"][c] for c in classes], width, label="Rule-based baseline")
    ax.bar(x + width / 2, [marl_metrics["per_class_f1"][c] for c in classes], width, label="3-agent MARL")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("F1-score")
    ax.set_title("Detection F1 per attack type: MARL vs. rule-based baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUTS_DIR}/detection_f1_comparison.png", dpi=150)
    print(f"Saved {OUTPUTS_DIR}/detection_f1_comparison.png")


def sample_episode_playback(df, det_preds, fw_preds, resp_preds, n=15, start=None):
    n_total = len(df)
    if start is None:
        start = int(np.random.default_rng(42).integers(0, n_total - n))
    print(f"\n=== Sample episode playback (flows {start}..{start+n-1}) ===")
    true_labels = df["Label"].values
    for i in range(start, start + n):
        true = true_labels[i]
        d = CLASSES[det_preds[i]]
        f = FIREWALL_ACTIONS[fw_preds[i]]
        r = RESPONSE_ACTIONS[resp_preds[i]]
        mark = "OK" if d == true else "MISS"
        print(f"  flow {i:>7} | true={true:10s} | Detection={d:10s} [{mark}] -> Firewall={f:10s} -> Response={r}")


def main():
    print("Loading dataset and trained models...")
    df = load_clean_dataset()
    _, test_df = chronological_split(df)
    det, fw, resp, scaler = load_models()

    print("Running MARL system on test split...")
    marl_det, marl_fw, marl_resp = run_marl_system(test_df, det, fw, resp, scaler)

    print("Running rule-based baseline on test split...")
    rule_det, rule_fw, rule_resp = run_rule_baseline(test_df)

    single_metrics = None
    single_fpr = None
    single_path = f"{MODELS_DIR}/single_agent.zip"
    if os.path.exists(single_path):
        print("Running single-agent RL baseline on test split...")
        single_model = PPO.load(f"{MODELS_DIR}/single_agent")
        single_det, single_fw, single_resp = run_single_agent(test_df, single_model, scaler)

    true_idx = test_df["Label"].map({c: i for i, c in enumerate(CLASSES)}).values

    marl_metrics = detection_metrics(true_idx, marl_det, label="3-agent MARL")
    rule_metrics = detection_metrics(true_idx, rule_det, label="Rule-based baseline")
    if os.path.exists(single_path):
        single_metrics = detection_metrics(true_idx, single_det, label="Single-agent RL")
        single_fpr = false_positive_rate(true_idx, single_fw)

    marl_fpr = false_positive_rate(true_idx, marl_fw)
    rule_fpr = false_positive_rate(true_idx, rule_fw)
    print(f"\nFirewall false positive rate (benign flows rate-limited/blocked):")
    print(f"  MARL: {marl_fpr:.4f}   Rule-based: {rule_fpr:.4f}"
          + (f"   Single-agent: {single_fpr:.4f}" if single_fpr is not None else ""))

    if single_metrics is not None:
        coord_benefit = marl_metrics["macro_f1"] - single_metrics["macro_f1"]
        print(f"\nCoordination benefit (MARL macro F1 - single-agent macro F1): {coord_benefit:+.4f}")

    summary = {
        "marl": {**marl_metrics, "firewall_fpr": marl_fpr},
        "rule_based": {**rule_metrics, "firewall_fpr": rule_fpr},
    }
    if single_metrics is not None:
        summary["single_agent"] = {**single_metrics, "firewall_fpr": single_fpr}
        summary["coordination_benefit_macro_f1"] = marl_metrics["macro_f1"] - single_metrics["macro_f1"]

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(f"{OUTPUTS_DIR}/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_reward_curves()
    plot_detection_comparison(marl_metrics, rule_metrics)
    sample_episode_playback(test_df, marl_det, marl_fw, marl_resp)

    print(f"\nSaved outputs/eval_summary.json")


if __name__ == "__main__":
    main()
