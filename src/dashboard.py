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

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from src.evaluate_sim import (
    make_policy_single,
    make_policy_v1,
    make_policy_v2,
    policy_none,
    policy_rule,
)
from src.sim_env import DefenseSim, load_split_cached
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
    _, test_df = load_split_cached()
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

tab_about, tab_live, tab_v2, tab_overview, tab_training = st.tabs(
    ["What is this?", "Live Demo", "v2: Interactive RL", "Baselines & Metrics", "Training Curves"]
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

@st.cache_resource(show_spinner="Loading v2 simulator and policies...")
def load_v2():
    scaler_v2 = joblib.load(f"{MODELS_DIR}/scaler.joblib")
    sim_v2 = DefenseSim(test_df, scaler_v2, seed=0)
    policies = {"No defense": policy_none, "Rule-based": policy_rule}
    for label, make in (
        ("v2 MARL (interactive RL)", make_policy_v2),
        ("v1 MARL (offline-trained)", make_policy_v1),
        ("v2 single-agent", make_policy_single),
    ):
        pol = make()
        if pol is not None:
            policies[label] = pol
    return sim_v2, policies


def simulate_window(sim_v2, policy, start, steps):
    sim_v2.reset_stats()
    sim_v2.reset(start=int(start), max_steps=steps)
    first = sim_v2.pos
    rows = []
    cum = 0.0
    while not sim_v2.done:
        pos = sim_v2.pos
        d, f, r = policy(sim_v2)
        s0 = (sim_v2.s["damage"], sim_v2.s["disruption"], sim_v2.s["dropped"])
        true = sim_v2.apply(d, f, r)[4]
        dmg = sim_v2.s["damage"] - s0[0]
        cum += dmg
        rows.append(
            {
                "flow": pos - first,
                "true": CLASSES[true],
                "detection": CLASSES[d],
                "firewall": FIREWALL_ACTIONS[f],
                "response": RESPONSE_ACTIONS[r],
                "damage": round(dmg, 2),
                "benign disruption": round(sim_v2.s["disruption"] - s0[1], 2),
                "attacker flows dropped after": sim_v2.s["dropped"] - s0[2],
                "cumulative damage": cum,
            }
        )
    stats = dict(sim_v2.s)
    stats["uncontained"] += sim_v2.open_onsets()
    return pd.DataFrame(rows), stats


with tab_v2:
    st.subheader("v2: agents trained by interacting with a simulator that reacts to their actions")
    st.markdown(
        """
In v1 the recorded traffic is replayed independently of what the agents do. In **v2** the
traffic is still recorded, but the agents' actions change what happens next:

- **Block / Isolate** an attack flow -> that attacker's later flows are dropped until the block expires.
- **Rate-limit** -> the attacker is throttled (less damage per flow).
- **Alert** -> an analyst contains the attacker after a delay (cheap but slow).
- **Letting an attack through** -> damage that grows the longer the attack runs, so speed matters.
- **Blocking/isolating benign traffic** -> service disruption (isolation causes a short outage).

The agents get a reward from these *outcomes* (plus a small label-based shaping term) and
learn with PPO. They never see the labels; only the simulator uses them to compute outcomes.
"""
    )

    st.markdown("#### Closed-loop results on the held-out test stream")
    sim_path = f"{OUTPUTS_DIR}/eval_sim_summary.json"
    if os.path.exists(sim_path):
        with open(sim_path) as f:
            sim_summary = json.load(f)
        names = {
            "none": "No defense",
            "rule": "Rule-based",
            "v1_marl": "v1 MARL (offline-trained)",
            "v2_marl": "v2 MARL (interactive RL)",
            "v2_single": "v2 single-agent",
        }
        rows = []
        for key, label in names.items():
            if key in sim_summary:
                r = sim_summary[key]
                rows.append(
                    {
                        "System": label,
                        "Total loss (lower=better)": r["total_loss"],
                        "Attack damage": r["damage"],
                        "Benign disruption": r["disruption"],
                        "Damage reduction": r["damage_reduction"],
                        "Benign flows disrupted": r["benign_disrupted_rate"],
                        "Mean latency (flows)": r["mean_latency_flows"] if r["mean_latency_flows"] is not None else float("nan"),
                    }
                )
        res_df = pd.DataFrame(rows).set_index("System")
        st.dataframe(
            res_df.style.format(
                {
                    "Total loss (lower=better)": "{:,.0f}",
                    "Attack damage": "{:,.0f}",
                    "Benign disruption": "{:,.0f}",
                    "Damage reduction": "{:.1%}",
                    "Benign flows disrupted": "{:.2%}",
                    "Mean latency (flows)": "{:.1f}",
                },
                na_rep="n/a",
            ),
            use_container_width=True,
        )
        loss_img = f"{OUTPUTS_DIR}/eval_sim_loss.png"
        if os.path.exists(loss_img):
            st.image(loss_img, caption="Total loss on the test stream, split into damage / disruption / cost")
        zoom = res_df.loc[[n for n in res_df.index if "MARL" in n], ["Attack damage", "Benign disruption"]]
        st.markdown("**Zoomed in: v1 vs v2 3-agent systems only:**")
        st.bar_chart(zoom)
        st.caption(
            "Caveats: the single-agent baseline had less training than the 3-agent system; the "
            "BruteForce fix was diagnosed on this same test stream (no separate validation split); "
            "one seed; attackers are identified by attack class in the simulator (no Source IP in the data)."
        )
    else:
        st.warning("Run `python -m src.evaluate_sim` first to generate outputs/eval_sim_summary.json")

    st.markdown("#### Watch an attack get contained")
    sim_v2, policies = load_v2()
    c1, c2, c3, c4 = st.columns(4)
    attack_cls = c1.selectbox("Attack type", ["DDoS", "PortScan", "BruteForce"], index=2)
    burst = c2.slider("Which attack burst in the test stream", 0, 100, 10)
    n_steps = c3.slider("Flows to simulate", 100, 1500, 400, step=50)
    system_label = c4.selectbox("System", list(policies.keys()), index=list(policies.keys()).index("v2 MARL (interactive RL)") if "v2 MARL (interactive RL)" in policies else 0)

    cls_idx = {"DDoS": 0, "PortScan": 1, "BruteForce": 2}[attack_cls]
    idxs = sim_v2._attack_idx[cls_idx]
    anchor = int(idxs[int(burst / 100 * (len(idxs) - 1))])
    start = max(int(sim_v2.file_start[anchor]), anchor - 30)

    chosen_df, chosen_stats = simulate_window(sim_v2, policies[system_label], start, n_steps)
    base_df, base_stats = simulate_window(sim_v2, policy_none, start, n_steps)

    lat = chosen_stats["latencies"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Attack damage", f"{chosen_stats['damage']:.0f}", f"{chosen_stats['damage'] - base_stats['damage']:+.0f} vs no defense", delta_color="inverse")
    m2.metric("Benign traffic disrupted", int(chosen_stats["benign_disrupted"]))
    m3.metric("Time to first containment", f"{lat[0]} flows" if lat else "not contained")
    m4.metric("Attacker flows dropped", int(chosen_stats["dropped"]))

    span = int(max(chosen_df["flow"].max(), base_df["flow"].max())) + 1
    series = {"No defense": base_df.set_index("flow")["cumulative damage"]}
    if system_label != "No defense":
        series = {system_label: chosen_df.set_index("flow")["cumulative damage"], **series}
    curve = pd.concat(series, axis=1).reindex(range(span)).ffill().fillna(0.0)
    st.line_chart(curve, color=["#2ecc71", "#e74c3c"][: len(curve.columns)] if len(curve.columns) == 2 else ["#e74c3c"])
    st.caption("Cumulative attack damage over the simulated stretch of traffic. A flat line means the attacker is contained.")

    notable = st.checkbox("Show only notable flows (attacks, non-trivial actions, damage, disruption)", value=True)
    log = chosen_df.drop(columns=["cumulative damage"])
    if notable:
        mask = (
            (log["true"] != "BENIGN")
            | (log["firewall"] != "Allow")
            | (log["response"] != "No action")
            | (log["damage"] > 0)
            | (log["benign disruption"] > 0)
        )
        log = log[mask]
    st.dataframe(log.head(200), use_container_width=True, hide_index=True)

    with st.expander("v2 training reward curves"):
        sim_curves = {}
        for key, label in (("sim_detection", "detection"), ("sim_firewall", "firewall"), ("sim_response", "response"), ("sim_single_agent", "single-agent")):
            p = f"{OUTPUTS_DIR}/{key}_reward_curve.json"
            if os.path.exists(p):
                with open(p) as f:
                    sim_curves[label] = pd.Series(json.load(f))
        if sim_curves:
            cdf = pd.DataFrame(sim_curves)
            st.line_chart(cdf.rolling(max(1, len(cdf) // 30), min_periods=1).mean())
            st.caption("Episode reward per agent, smoothed (rounds are concatenated; later rounds train against learned partners).")
        else:
            st.info("No v2 reward curves found.")

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
