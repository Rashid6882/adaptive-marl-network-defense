"""
Live demo dashboard for the mid-review presentation.

Run with:
    streamlit run src/dashboard.py

Shows:
  - Headline comparison: rule-based baseline vs single-agent RL vs 3-agent MARL
  - Training reward curves per agent
  - A "SOC console" live playback: step/scrub through test-set flows and watch
    Detection -> Firewall -> Response decisions unfold in real time, with a
    ground-truth comparison and running stats.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from src.data import chronological_split, load_clean_dataset
from src.evaluate import (
    MODELS_DIR,
    OUTPUTS_DIR,
    load_models,
    run_marl_system,
    run_rule_baseline,
    run_single_agent,
)
from src.features import CLASSES, FIREWALL_ACTIONS, RESPONSE_ACTIONS
from stable_baselines3 import PPO

st.set_page_config(page_title="Adaptive MARL Network Defense", layout="wide")

CLASS_COLORS = {
    "BENIGN": "#2ecc71",
    "DDoS": "#e74c3c",
    "PortScan": "#e67e22",
    "BruteForce": "#9b59b6",
}


@st.cache_resource(show_spinner="Loading dataset and trained models...")
def load_everything():
    df = load_clean_dataset()
    _, test_df = chronological_split(df)
    det, fw, resp, scaler = load_models()
    single_model = None
    if os.path.exists(f"{MODELS_DIR}/single_agent.zip"):
        single_model = PPO.load(f"{MODELS_DIR}/single_agent")
    return test_df, det, fw, resp, scaler, single_model


@st.cache_resource(show_spinner="Running all systems on the test split (one-time)...")
def run_all_systems(_test_df, _det, _fw, _resp, _scaler, _single_model):
    marl = run_marl_system(_test_df, _det, _fw, _resp, _scaler)
    rule = run_rule_baseline(_test_df)
    single = run_single_agent(_test_df, _single_model, _scaler) if _single_model else None
    return marl, rule, single


@st.cache_data(show_spinner=False)
def load_summary():
    path = f"{OUTPUTS_DIR}/eval_summary.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_reward_curve(name):
    path = f"{OUTPUTS_DIR}/{name}_reward_curve.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


st.title("Adaptive Multi-Agent RL for Intelligent Network Defense")
st.caption("Detection -> Firewall -> Response, trained on real CICIDS2017 traffic (DDoS, Port Scan, Brute Force)")

test_df, det, fw, resp, scaler, single_model = load_everything()
(marl_det, marl_fw, marl_resp), (rule_det, rule_fw, rule_resp), single_preds = run_all_systems(
    test_df, det, fw, resp, scaler, single_model
)
true_idx = test_df["Label"].map({c: i for i, c in enumerate(CLASSES)}).values
summary = load_summary()

tab_about, tab_live, tab_overview, tab_training = st.tabs(
    ["What is this?", "Live Demo", "Baselines & Metrics", "Training Curves"]
)

with tab_about:
    st.subheader("Purpose")
    st.markdown(
        """
Static, rule-based firewalls (fixed thresholds, signature lists) can't adapt when an
attacker tweaks their approach, and they can't tell a real traffic spike from a DDoS.
This system replaces that fixed rulebook with **three cooperating RL agents** that learn
from traffic directly, so the defense adapts as attack patterns change instead of
needing to be manually re-tuned.

It was trained and evaluated on **real captured traffic** (CICIDS2017: DDoS, Port Scan,
and Brute Force attacks mixed with benign traffic), not synthetic data.
"""
    )

    st.subheader("How it works")
    st.markdown(
        """
Every network flow (one connection) is handled by three agents **in sequence** — each one
sees what the agent before it decided, which is the coordination mechanism at the core of
this project:
"""
    )

    pipe_cols = st.columns(3)
    with pipe_cols[0]:
        st.markdown("##### 1. Detection agent")
        st.caption("Looks at the flow's traffic stats (duration, packet rate, flag counts, ...)")
        st.markdown("**Decides:** `Benign` / `DDoS` / `Port Scan` / `Brute Force`")
    with pipe_cols[1]:
        st.markdown("##### 2. Firewall agent")
        st.caption("Sees Detection's verdict + how often this target has been hit recently")
        st.markdown("**Decides:** `Allow` / `Rate-limit` / `Block`")
    with pipe_cols[2]:
        st.markdown("##### 3. Response agent")
        st.caption("Sees the Firewall's action + whether this is a repeat offender")
        st.markdown("**Decides:** `No action` / `Alert` / `Isolate host`")

    st.markdown(
        """
Each agent is trained with its own reward: missing a real attack is penalized much more
heavily than a false alarm, over-blocking benign traffic is penalized, and unnecessarily
isolating a benign host is the worst outcome for the Response agent. This is deliberate —
it mirrors the real cost trade-off a security team faces, instead of optimizing for raw
accuracy alone.
"""
    )

    st.subheader("How to use this dashboard")
    st.markdown(
        """
- **Live Demo** tab — the main thing to look at. It replays real (held-out, never-trained-on)
  network traffic flow by flow and shows what each agent decided, in order. Use the
  "Jump to next DDoS / Port Scan / Brute Force" buttons to skip straight to an attack and
  watch the three agents react, or drag the slider to scrub through traffic yourself.
  Switch the "System to drive the console" radio button to compare against the rule-based
  baseline or a single-agent (no coordination) baseline on the exact same traffic.
- **Baselines & Metrics** tab — the scored comparison (accuracy, F1 per attack type, false
  positive rate) between this system and the two baselines, if you want the numbers behind
  the demo.
- **Training Curves** tab — how each agent's reward improved over training.
"""
    )

    st.info(
        "Known limitation: this CICIDS2017 release doesn't include a Source IP column, "
        "so \"repeated flows from the same attacker\" is approximated using the "
        "destination port instead (documented in the README)."
    )

with tab_overview:
    st.subheader("System comparison on held-out (chronologically later) test traffic")
    if summary:
        rows = []
        for key, label in [("rule_based", "Rule-based baseline"), ("single_agent", "Single-agent RL"), ("marl", "3-agent MARL (proposed)")]:
            if key not in summary:
                continue
            s = summary[key]
            row = {"System": label, "Accuracy": s["accuracy"], "Macro F1": s["macro_f1"], "Firewall FPR": s["firewall_fpr"]}
            row.update({f"{c} F1": s["per_class_f1"][c] for c in CLASSES})
            rows.append(row)
        comp_df = pd.DataFrame(rows).set_index("System")
        st.dataframe(comp_df.style.format("{:.3f}").background_gradient(cmap="RdYlGn", axis=0), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("MARL accuracy", f"{summary['marl']['accuracy']:.1%}",
                   f"{(summary['marl']['accuracy'] - summary['rule_based']['accuracy'])*100:+.1f} pts vs rule-based")
        c2.metric("MARL macro F1", f"{summary['marl']['macro_f1']:.3f}",
                   f"{summary['marl']['macro_f1'] - summary['rule_based']['macro_f1']:+.3f} vs rule-based")
        if "coordination_benefit_macro_f1" in summary:
            c3.metric("Coordination benefit", f"{summary['coordination_benefit_macro_f1']:+.3f}",
                       "MARL macro F1 minus single-agent macro F1")

        st.bar_chart(comp_df[[f"{c} F1" for c in CLASSES]].T)
    else:
        st.warning("Run `python -m src.evaluate` first to generate outputs/eval_summary.json")

    img_path = f"{OUTPUTS_DIR}/detection_f1_comparison.png"
    if os.path.exists(img_path):
        st.image(img_path, caption="Detection F1 per attack type")

with tab_training:
    st.subheader("Per-agent training reward curves")
    curves = {name: load_reward_curve(name) for name in ["detection", "firewall", "response", "single_agent"]}
    curve_df = pd.DataFrame({k: pd.Series(v) for k, v in curves.items() if v})
    if not curve_df.empty:
        smoothed = curve_df.rolling(max(1, len(curve_df) // 30), min_periods=1).mean()
        st.line_chart(smoothed)
        st.caption("Episode total reward, smoothed. X axis = training episode index.")
    else:
        st.warning("No reward curve data found yet -- run training first.")

with tab_live:
    st.subheader("SOC console: live flow-by-flow playback")
    system = st.radio("System to drive the console", ["3-agent MARL", "Rule-based baseline", "Single-agent RL"], horizontal=True)
    if system == "3-agent MARL":
        det_p, fw_p, resp_p = marl_det, marl_fw, marl_resp
    elif system == "Rule-based baseline":
        det_p, fw_p, resp_p = rule_det, rule_fw, rule_resp
    else:
        if single_preds is None:
            st.error("Single-agent model not found. Train it with `python -m src.train_single_agent`.")
            st.stop()
        det_p, fw_p, resp_p = single_preds

    n = len(test_df)
    jump_cols = st.columns(5)
    attack_jumps = {"DDoS": 1, "PortScan": 2, "BruteForce": 3}
    if "cursor" not in st.session_state:
        st.session_state.cursor = 0
    if jump_cols[0].button("Jump to start"):
        st.session_state.cursor = 0
    for i, (label, cls_idx) in enumerate(attack_jumps.items()):
        if jump_cols[i + 1].button(f"Jump to next {label}"):
            candidates = np.where(true_idx[st.session_state.cursor + 1:] == cls_idx)[0]
            if len(candidates) > 0:
                st.session_state.cursor = st.session_state.cursor + 1 + int(candidates[0])
    if jump_cols[4].button("Random flow"):
        st.session_state.cursor = int(np.random.randint(0, n - 10))

    cursor = st.slider("Flow position in test stream", 0, n - 10, key="cursor")

    true_label = CLASSES[true_idx[cursor]]
    det_label = CLASSES[det_p[cursor]]
    fw_label = FIREWALL_ACTIONS[fw_p[cursor]]
    resp_label = RESPONSE_ACTIONS[resp_p[cursor]]
    correct = true_label == det_label

    card = st.container(border=True)
    with card:
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Ground truth**<br><span style='font-size:28px;color:{CLASS_COLORS[true_label]}'>{true_label}</span>", unsafe_allow_html=True)
        verdict_color = "#2ecc71" if correct else "#e74c3c"
        c2.markdown(f"**Detection agent**<br><span style='font-size:28px;color:{verdict_color}'>{det_label}</span>", unsafe_allow_html=True)
        c3.markdown(f"**Firewall agent**<br><span style='font-size:28px'>{fw_label}</span>", unsafe_allow_html=True)
        c4.markdown(f"**Response agent**<br><span style='font-size:28px'>{resp_label}</span>", unsafe_allow_html=True)
        st.caption(("Correct detection" if correct else "Misclassified") + f" -- flow index {cursor} of {n}")

    st.markdown("#### Surrounding flow stream")
    window = 12
    lo, hi = max(0, cursor - window), min(n, cursor + window + 1)
    disp = pd.DataFrame({
        "flow": range(lo, hi),
        "true_label": [CLASSES[t] for t in true_idx[lo:hi]],
        "detection": [CLASSES[d] for d in det_p[lo:hi]],
        "firewall": [FIREWALL_ACTIONS[a] for a in fw_p[lo:hi]],
        "response": [RESPONSE_ACTIONS[a] for a in resp_p[lo:hi]],
    })
    disp["correct"] = disp["true_label"] == disp["detection"]

    def highlight_row(row):
        is_current = row["flow"] == cursor
        base = "background-color: #333355; color: white;" if is_current else ""
        return [base] * len(row)

    st.dataframe(disp.style.apply(highlight_row, axis=1), use_container_width=True, hide_index=True)

    st.markdown("#### Running stats over flows seen so far in this session's window")
    seen = slice(0, cursor + 1)
    acc_so_far = (true_idx[seen] == det_p[seen]).mean()
    blocked = (fw_p[seen] == 2).sum()
    isolated = (resp_p[seen] == 2).sum()
    s1, s2, s3 = st.columns(3)
    s1.metric("Detection accuracy so far", f"{acc_so_far:.1%}")
    s2.metric("Flows blocked", int(blocked))
    s3.metric("Hosts isolated", int(isolated))
