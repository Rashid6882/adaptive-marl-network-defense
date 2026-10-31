"""
Dashboard for the interactive closed-loop defense system.

Run with:
    streamlit run src/dashboard.py

Shows:
  - What the system is and how the three agents work
  - Closed-loop results vs. no defense, a rule-based baseline and a single-agent baseline
  - A "watch an attack get contained" simulation on held-out traffic
  - Training reward curves
"""

import json
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from src.evaluate_sim import (
    OUTPUTS_DIR,
    V2_DIR,
    make_policy_single,
    make_policy_v2,
    policy_none,
    policy_rule,
)
from src.features import CLASSES, FIREWALL_ACTIONS, RESPONSE_ACTIONS
from src.sim_env import DefenseSim, load_split_cached

st.set_page_config(page_title="Adaptive MARL Network Defense", layout="wide")

st.title("Adaptive Multi-Agent RL for Intelligent Network Defense")
st.caption("Detection -> Firewall -> Response, trained on real CICIDS2017 traffic (DDoS, Port Scan, Brute Force)")


@st.cache_resource(show_spinner="Loading simulator and trained policies...")
def load_sim():
    _, test_df = load_split_cached()
    scaler = joblib.load(f"{V2_DIR}/scaler.joblib")
    sim = DefenseSim(test_df, scaler, seed=0)
    policies = {"No defense": policy_none, "Rule-based": policy_rule}
    for label, make in (("3-agent MARL (interactive RL)", make_policy_v2), ("Single-agent RL", make_policy_single)):
        pol = make()
        if pol is not None:
            policies[label] = pol
    return sim, policies


def simulate_window(sim, policy, start, steps):
    sim.reset_stats()
    sim.reset(start=int(start), max_steps=steps)
    first = sim.pos
    rows = []
    cum = 0.0
    while not sim.done:
        pos = sim.pos
        d, f, r = policy(sim)
        s0 = (sim.s["damage"], sim.s["disruption"], sim.s["dropped"])
        true = sim.apply(d, f, r)[4]
        dmg = sim.s["damage"] - s0[0]
        cum += dmg
        rows.append(
            {
                "flow": pos - first,
                "true": CLASSES[true],
                "detection": CLASSES[d],
                "firewall": FIREWALL_ACTIONS[f],
                "response": RESPONSE_ACTIONS[r],
                "damage": round(dmg, 2),
                "benign disruption": round(sim.s["disruption"] - s0[1], 2),
                "attacker flows dropped after": sim.s["dropped"] - s0[2],
                "cumulative damage": cum,
            }
        )
    stats = dict(sim.s)
    stats["uncontained"] += sim.open_onsets()
    return pd.DataFrame(rows), stats


tab_about, tab_rl = st.tabs(["What is this?", "Interactive RL"])

with tab_about:
    st.subheader("Purpose")
    st.markdown(
        """
Static, rule-based firewalls (fixed thresholds, signature lists) can't adapt when an
attacker tweaks their approach, and they can't tell a real traffic spike from a DDoS.
This system replaces that fixed rulebook with **three cooperating RL agents** that learn
from the outcomes of their own decisions.

It was trained and evaluated on **real captured traffic** (CICIDS2017: DDoS, Port Scan,
and Brute Force attacks mixed with benign traffic), replayed through a simulator.
"""
    )

    st.subheader("How it works")
    st.markdown(
        "Every network flow (one connection) is handled by three agents **in sequence**, "
        "each one seeing what the agent before it decided:"
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
The simulator reacts to those decisions: **blocking** an attack flow contains that attacker (its
later flows are dropped until the block expires), **rate-limiting** throttles it, an **alert**
makes an analyst contain it after a delay, an attack that gets through causes **damage that grows
the longer it runs**, and wrongly blocking or isolating normal traffic causes **disruption**.
The agents are rewarded on these outcomes and learn with PPO. They never see the true labels.
"""
    )

    st.subheader("How to use this dashboard")
    st.markdown(
        """
- **Interactive RL** tab: the closed-loop results table, then a simulation where you pick an attack
  type, a burst of traffic and a system, and watch cumulative damage with and without a defense.
  Switch the system to compare the 3-agent system with the rule-based and single-agent baselines.
"""
    )
    st.info(
        "Known limitation: this CICIDS2017 release doesn't include a Source IP column, so an "
        "attacker is identified by its attack class within an episode, and \"repeated flows\" "
        "are approximated using the destination port."
    )

with tab_rl:
    st.subheader("Agents trained by interacting with a simulator that reacts to their actions")

    st.markdown("#### Closed-loop results on the held-out test stream (mean of 4 seeds)")
    sim_path = "runs/summary_test.json"
    if os.path.exists(sim_path):
        with open(sim_path) as f:
            sim_summary = json.load(f)
        names = {
            "none": "No defense",
            "rule": "Rule-based",
            "rule_tuned": "Rule-based (tuned)",
            "v2_marl": "3-agent MARL (interactive RL)",
            "v2_single": "Single-agent RL",
        }
        rows = []
        for key, label in names.items():
            if key in sim_summary:
                r = {metric: v["mean"] if v else None for metric, v in sim_summary[key].items()}
                lat = r.get("mean_latency_flows")
                rows.append(
                    {
                        "System": label,
                        "Total loss (lower=better)": r["total_loss"],
                        "Attack damage": r["damage"],
                        "Benign disruption": r["disruption"],
                        "Damage reduction": r["damage_reduction"],
                        "Benign flows disrupted": r["benign_disrupted_rate"],
                        "Mean latency (flows)": lat if lat is not None else float("nan"),
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
        st.caption(
            "Both learned systems had the same 1.2M-step budget, 4 seeds each. MARL and the single agent "
            "are statistically tied on this stream (see README for per-seed numbers and the std). "
            "Attackers are identified by attack class in the simulator (no Source IP in the data). "
            "The demo below uses the seed-1 models."
        )
    else:
        st.warning("Run the evaluation first (see README) to generate runs/summary_test.json")

    st.markdown("#### Watch an attack get contained")
    sim, policies = load_sim()
    c1, c2, c3, c4 = st.columns(4)
    attack_cls = c1.selectbox("Attack type", ["DDoS", "PortScan", "BruteForce"], index=2)
    burst = c2.slider("Which attack burst in the test stream", 0, 100, 10)
    n_steps = c3.slider("Flows to simulate", 100, 1500, 400, step=50)
    default_label = "3-agent MARL (interactive RL)"
    system_label = c4.selectbox(
        "System", list(policies.keys()), index=list(policies.keys()).index(default_label) if default_label in policies else 0
    )

    cls_idx = {"DDoS": 0, "PortScan": 1, "BruteForce": 2}[attack_cls]
    idxs = sim._attack_idx[cls_idx]
    anchor = int(idxs[int(burst / 100 * (len(idxs) - 1))])
    start = max(int(sim.file_start[anchor]), anchor - 30)

    chosen_df, chosen_stats = simulate_window(sim, policies[system_label], start, n_steps)
    base_df, base_stats = simulate_window(sim, policy_none, start, n_steps)

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
    st.line_chart(curve, color=["#2ecc71", "#e74c3c"] if len(curve.columns) == 2 else ["#e74c3c"])
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

    with st.expander("Training reward curves"):
        sim_curves = {}
        for key, label in (("sim_detection", "detection"), ("sim_firewall", "firewall"), ("sim_response", "response"), ("sim_single_agent", "single-agent")):
            p = f"{V2_DIR}/{key}_reward_curve.json"
            if os.path.exists(p):
                with open(p) as f:
                    sim_curves[label] = pd.Series(json.load(f))
        if sim_curves:
            cdf = pd.DataFrame(sim_curves)
            st.line_chart(cdf.rolling(max(1, len(cdf) // 30), min_periods=1).mean())
            st.caption("Episode reward per agent, smoothed (rounds are concatenated; later rounds train against learned partners).")
        else:
            st.info("No reward curves found.")
