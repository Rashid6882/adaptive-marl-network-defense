# Module docstring title: the interactive dashboard for the closed-loop defense system.
# How to run it: streamlit run src/dashboard.py.
# What the dashboard shows: an explanation page, the results versus baselines, a "watch an attack get
# contained" simulation and the training reward curves.
# Ends the docstring.
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

# Imports json to read result files.
import json
# Imports os to check that files exist.
import os
# Imports sys to change the import path.
import sys

# Imports joblib to load the saved scaler.
import joblib

# Adds the project root folder to the import path so "src" imports work when streamlit runs this file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports pandas for tables.
import pandas as pd
# Imports streamlit, the web dashboard library.
import streamlit as st

# Imports from the evaluation file: folders, the policy builders for the learned systems and the two baseline
# policies.
from src.evaluate_sim import (
    OUTPUTS_DIR,
    V2_DIR,
    make_policy_single,
    make_policy_v2,
    policy_none,
    policy_rule,
)
# Imports class and action names.
from src.features import CLASSES, FIREWALL_ACTIONS, RESPONSE_ACTIONS
# Imports the simulator and the test-data loader.
from src.sim_env import DefenseSim, load_split_cached

# Sets the browser tab title and wide layout.
st.set_page_config(page_title="Adaptive MARL Network Defense", layout="wide")

# Page title.
st.title("Adaptive Multi-Agent RL for Intelligent Network Defense")
# Subtitle under the title.
st.caption("Detection -> Firewall -> Response, trained on real CICIDS2017 traffic (DDoS, Port Scan, Brute Force)")


# Decorator: load the simulator and policies once and reuse them (cached).
@st.cache_resource(show_spinner="Loading simulator and trained policies...")
# Function that loads the simulator and all policies.
def load_sim():
    # Loads the test table.
    _, test_df = load_split_cached()
    # Loads the scaler saved with the seed-1 models.
    scaler = joblib.load(f"{V2_DIR}/scaler.joblib")
    # Creates the simulator on the test data.
    sim = DefenseSim(test_df, scaler, seed=0)
    # Starts the policies dictionary with the two baselines.
    policies = {"No defense": policy_none, "Rule-based": policy_rule}
    # Loops over the two learned systems and their builder functions.
    for label, make in (("3-agent MARL (interactive RL)", make_policy_v2), ("Single-agent RL", make_policy_single)):
        # Builds the policy from the saved models.
        pol = make()
        # If the models exist...
        if pol is not None:
            # ...add the policy to the dictionary.
            policies[label] = pol
    # Returns the simulator and the policies.
    return sim, policies


# Function that runs one policy over a short stretch of traffic and records every decision.
def simulate_window(sim, policy, start, steps):
    # Clears the simulator statistics.
    sim.reset_stats()
    # Starts an episode at the chosen position for the chosen number of steps.
    sim.reset(start=int(start), max_steps=steps)
    # Remembers the first flow's position.
    first = sim.pos
    # List that will hold one record per flow.
    rows = []
    # Running total of damage.
    cum = 0.0
    # Loops until the episode ends.
    while not sim.done:
        # Current position.
        pos = sim.pos
        # The policy's three decisions.
        d, f, r = policy(sim)
        # Snapshot of damage, disruption and dropped counts before the step.
        s0 = (sim.s["damage"], sim.s["disruption"], sim.s["dropped"])
        # Applies the decisions and gets the true class.
        true = sim.apply(d, f, r)[4]
        # Damage caused by this flow (difference from the snapshot).
        dmg = sim.s["damage"] - s0[0]
        # Adds it to the cumulative damage.
        cum += dmg
        # Starts adding a record.
        rows.append(
            # Opens the record dictionary.
            {
                # Flow number within the window.
                "flow": pos - first,
                # True class name.
                "true": CLASSES[true],
                # Detection's verdict name.
                "detection": CLASSES[d],
                # Firewall's action name.
                "firewall": FIREWALL_ACTIONS[f],
                # Response's action name.
                "response": RESPONSE_ACTIONS[r],
                # Damage from this flow.
                "damage": round(dmg, 2),
                # Benign disruption from this flow.
                "benign disruption": round(sim.s["disruption"] - s0[1], 2),
                # Attacker flows dropped since the snapshot.
                "attacker flows dropped after": sim.s["dropped"] - s0[2],
                # Cumulative damage so far.
                "cumulative damage": cum,
            # Closes the dictionary.
            }
        # Closes the append call.
        )
    # Copies the final statistics.
    stats = dict(sim.s)
    # Adds attacks that were still open at the end to the uncontained count.
    stats["uncontained"] += sim.open_onsets()
    # Returns the table of records and the statistics.
    return pd.DataFrame(rows), stats


# Creates two tabs: "What is this?" and "Interactive RL".
tab_about, tab_rl = st.tabs(["What is this?", "Interactive RL"])

# Starts the content of the first tab.
with tab_about:
    # Section heading "Purpose".
    st.subheader("Purpose")
    # Text explaining the purpose: static rule-based firewalls cannot adapt, so three cooperating RL agents
    # learn from the outcomes; it was trained and evaluated on real CICIDS2017 traffic replayed in a
    # simulator.
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

    # Section heading "How it works".
    st.subheader("How it works")
    # Text saying each flow is handled by three agents in sequence, each seeing the previous agent's decision.
    st.markdown(
        "Every network flow (one connection) is handled by three agents **in sequence**, "
        "each one seeing what the agent before it decided:"
    )
    # Creates three side-by-side columns.
    pipe_cols = st.columns(3)
    # First column: the Detection agent.
    with pipe_cols[0]:
        # Heading "1. Detection agent".
        st.markdown("##### 1. Detection agent")
        # Caption describing its input (flow statistics).
        st.caption("Looks at the flow's traffic stats (duration, packet rate, flag counts, ...)")
        # Lists its possible decisions.
        st.markdown("**Decides:** `Benign` / `DDoS` / `Port Scan` / `Brute Force`")
    # Second column: the Firewall agent.
    with pipe_cols[1]:
        # Heading "2. Firewall agent".
        st.markdown("##### 2. Firewall agent")
        # Caption describing its input (Detection's verdict and recent hits).
        st.caption("Sees Detection's verdict + how often this target has been hit recently")
        # Lists its possible decisions.
        st.markdown("**Decides:** `Allow` / `Rate-limit` / `Block`")
    # Third column: the Response agent.
    with pipe_cols[2]:
        # Heading "3. Response agent".
        st.markdown("##### 3. Response agent")
        # Caption describing its input (Firewall's action and repeat offender).
        st.caption("Sees the Firewall's action + whether this is a repeat offender")
        # Lists its possible decisions.
        st.markdown("**Decides:** `No action` / `Alert` / `Isolate host`")

    # Text explaining how the simulator reacts: blocking contains an attacker, rate-limiting throttles it,
    # alerts act after a delay, attacks cause growing damage and wrongly blocking benign traffic causes
    # disruption; agents learn with PPO and never see labels.
    st.markdown(
        """
The simulator reacts to those decisions: **blocking** an attack flow contains that attacker (its
later flows are dropped until the block expires), **rate-limiting** throttles it, an **alert**
makes an analyst contain it after a delay, an attack that gets through causes **damage that grows
the longer it runs**, and wrongly blocking or isolating normal traffic causes **disruption**.
The agents are rewarded on these outcomes and learn with PPO. They never see the true labels.
"""
    )

    # Section heading "How to use this dashboard".
    st.subheader("How to use this dashboard")
    # Text describing the Interactive RL tab: the results table and the simulation where you pick an attack, a
    # burst and a system.
    st.markdown(
        """
- **Interactive RL** tab: the closed-loop results table, then a simulation where you pick an attack
  type, a burst of traffic and a system, and watch cumulative damage with and without a defense.
  Switch the system to compare the 3-agent system with the rule-based and single-agent baselines.
"""
    )
    # Info box stating the limitation: no Source IP column, so attackers are identified by class and repeats
    # use the destination port.
    st.info(
        "Known limitation: this CICIDS2017 release doesn't include a Source IP column, so an "
        "attacker is identified by its attack class within an episode, and \"repeated flows\" "
        "are approximated using the destination port."
    )

# Starts the content of the second tab.
with tab_rl:
    # Section heading.
    st.subheader("Agents trained by interacting with a simulator that reacts to their actions")

    # Heading for the results table (mean of 4 seeds).
    st.markdown("#### Closed-loop results on the held-out test stream (mean of 4 seeds)")
    # Path of the aggregated test results file.
    sim_path = "runs/summary_test.json"
    # If the file exists...
    if os.path.exists(sim_path):
        # ...open it...
        with open(sim_path) as f:
            # ...and load the numbers.
            sim_summary = json.load(f)
        # Starts the dictionary of display names for the systems.
        names = {
            # "none" is shown as "No defense".
            "none": "No defense",
            # "rule" is shown as "Rule-based".
            "rule": "Rule-based",
            # "rule_tuned" is shown as "Rule-based (tuned)".
            "rule_tuned": "Rule-based (tuned)",
            # "v2_marl" is shown as "3-agent MARL".
            "v2_marl": "3-agent MARL (interactive RL)",
            # "v2_single" is shown as "Single-agent RL".
            "v2_single": "Single-agent RL",
        # Ends the dictionary.
        }
        # Empty list for the table rows.
        rows = []
        # Loops over the systems.
        for key, label in names.items():
            # Only if that system is in the file.
            if key in sim_summary:
                # Takes the mean of each metric (None if missing).
                r = {metric: v["mean"] if v else None for metric, v in sim_summary[key].items()}
                # Gets the latency value.
                lat = r.get("mean_latency_flows")
                # Starts adding a row.
                rows.append(
                    # Opens the row dictionary.
                    {
                        # System name.
                        "System": label,
                        # Total loss.
                        "Total loss (lower=better)": r["total_loss"],
                        # Attack damage.
                        "Attack damage": r["damage"],
                        # Benign disruption.
                        "Benign disruption": r["disruption"],
                        # Damage reduction.
                        "Damage reduction": r["damage_reduction"],
                        # Share of benign flows disrupted.
                        "Benign flows disrupted": r["benign_disrupted_rate"],
                        # Mean latency (or not-a-number if missing).
                        "Mean latency (flows)": lat if lat is not None else float("nan"),
                    # Closes the dictionary.
                    }
                # Closes the append call.
                )
        # Builds the table with the system name as the row label.
        res_df = pd.DataFrame(rows).set_index("System")
        # Starts displaying the table.
        st.dataframe(
            # Starts formatting the numbers.
            res_df.style.format(
                # Opens the format dictionary.
                {
                    # Total loss with thousands separators.
                    "Total loss (lower=better)": "{:,.0f}",
                    # Attack damage with thousands separators.
                    "Attack damage": "{:,.0f}",
                    # Benign disruption with thousands separators.
                    "Benign disruption": "{:,.0f}",
                    # Damage reduction as a percentage.
                    "Damage reduction": "{:.1%}",
                    # Benign flows disrupted as a percentage with two decimals.
                    "Benign flows disrupted": "{:.2%}",
                    # Latency with one decimal.
                    "Mean latency (flows)": "{:.1f}",
                # Closes the format dictionary.
                },
                # Shows "n/a" for missing numbers.
                na_rep="n/a",
            # Closes the format call.
            ),
            # Makes the table the full width of the page.
            use_container_width=True,
        # Closes the dataframe call.
        )
        # Path of the saved loss chart image.
        loss_img = f"{OUTPUTS_DIR}/eval_sim_loss.png"
        # If the image exists...
        if os.path.exists(loss_img):
            # ...display it with a caption.
            st.image(loss_img, caption="Total loss on the test stream, split into damage / disruption / cost")
        # Caption under the table: both learned systems had the same 1.2M-step budget and 4 seeds, MARL and
        # single agent are statistically tied, attackers are identified by class, and the demo uses seed-1
        # models.
        st.caption(
            "Both learned systems had the same 1.2M-step budget, 4 seeds each. MARL and the single agent "
            "are statistically tied on this stream (see README for per-seed numbers and the std). "
            "Attackers are identified by attack class in the simulator (no Source IP in the data). "
            "The demo below uses the seed-1 models."
        )
    # Otherwise (no results file)...
    else:
        # ...show a warning telling you to run the evaluation first.
        st.warning("Run the evaluation first (see README) to generate runs/summary_test.json")

    # Heading "Watch an attack get contained".
    st.markdown("#### Watch an attack get contained")
    # Loads the simulator and policies (cached).
    sim, policies = load_sim()
    # Creates four columns for the controls.
    c1, c2, c3, c4 = st.columns(4)
    # Dropdown to choose the attack type (default BruteForce).
    attack_cls = c1.selectbox("Attack type", ["DDoS", "PortScan", "BruteForce"], index=2)
    # Slider to choose which attack burst in the test stream (0-100).
    burst = c2.slider("Which attack burst in the test stream", 0, 100, 10)
    # Slider for how many flows to simulate (100-1500).
    n_steps = c3.slider("Flows to simulate", 100, 1500, 400, step=50)
    # Name of the default system (3-agent MARL).
    default_label = "3-agent MARL (interactive RL)"
    # Dropdown to choose the system, with the 3-agent system selected by default if available.
    system_label = c4.selectbox(
        "System", list(policies.keys()), index=list(policies.keys()).index(default_label) if default_label in policies else 0
    )

    # Converts the attack name to a class number.
    cls_idx = {"DDoS": 0, "PortScan": 1, "BruteForce": 2}[attack_cls]
    # Gets the positions of all flows of that attack class.
    idxs = sim._attack_idx[cls_idx]
    # Picks the anchor flow according to the burst slider.
    anchor = int(idxs[int(burst / 100 * (len(idxs) - 1))])
    # Starts 30 flows before the anchor (but not before the file start).
    start = max(int(sim.file_start[anchor]), anchor - 30)

    # Runs the chosen system over the window.
    chosen_df, chosen_stats = simulate_window(sim, policies[system_label], start, n_steps)
    # Runs "no defense" over the same window for comparison.
    base_df, base_stats = simulate_window(sim, policy_none, start, n_steps)

    # Containment latencies of the chosen system.
    lat = chosen_stats["latencies"]
    # Creates four columns for the numbers.
    m1, m2, m3, m4 = st.columns(4)
    # Shows attack damage and the change versus no defense (lower is better).
    m1.metric("Attack damage", f"{chosen_stats['damage']:.0f}", f"{chosen_stats['damage'] - base_stats['damage']:+.0f} vs no defense", delta_color="inverse")
    # Shows how many benign flows were disrupted.
    m2.metric("Benign traffic disrupted", int(chosen_stats["benign_disrupted"]))
    # Shows the time to the first containment (or "not contained").
    m3.metric("Time to first containment", f"{lat[0]} flows" if lat else "not contained")
    # Shows how many attacker flows were dropped.
    m4.metric("Attacker flows dropped", int(chosen_stats["dropped"]))

    # Length of the chart's x-axis (the longer of the two runs).
    span = int(max(chosen_df["flow"].max(), base_df["flow"].max())) + 1
    # Starts the chart data with the "no defense" cumulative damage.
    series = {"No defense": base_df.set_index("flow")["cumulative damage"]}
    # If the chosen system is not "No defense"...
    if system_label != "No defense":
        # ...add its curve as well.
        series = {system_label: chosen_df.set_index("flow")["cumulative damage"], **series}
    # Combines the curves into one table with equal lengths (filling gaps forward).
    curve = pd.concat(series, axis=1).reindex(range(span)).ffill().fillna(0.0)
    # Draws the line chart (green for the system, red for no defense).
    st.line_chart(curve, color=["#2ecc71", "#e74c3c"] if len(curve.columns) == 2 else ["#e74c3c"])
    # Caption: a flat line means the attacker is contained.
    st.caption("Cumulative attack damage over the simulated stretch of traffic. A flat line means the attacker is contained.")

    # Checkbox: show only notable flows.
    notable = st.checkbox("Show only notable flows (attacks, non-trivial actions, damage, disruption)", value=True)
    # Drops the cumulative column from the log.
    log = chosen_df.drop(columns=["cumulative damage"])
    # If the checkbox is ticked...
    if notable:
        # Starts building a filter mask.
        mask = (
            # Keep attack flows...
            (log["true"] != "BENIGN")
            # ...flows where the Firewall did something...
            | (log["firewall"] != "Allow")
            # ...flows where Response did something...
            | (log["response"] != "No action")
            # ...flows with damage...
            | (log["damage"] > 0)
            # ...and flows with benign disruption.
            | (log["benign disruption"] > 0)
        # Ends the mask.
        )
        # Applies the filter.
        log = log[mask]
    # Shows the first 200 rows of the decision log.
    st.dataframe(log.head(200), use_container_width=True, hide_index=True)

    # Collapsible section "Training reward curves".
    with st.expander("Training reward curves"):
        # Dictionary for the curves.
        sim_curves = {}
        # Loops over the four reward-curve files.
        for key, label in (("sim_detection", "detection"), ("sim_firewall", "firewall"), ("sim_response", "response"), ("sim_single_agent", "single-agent")):
            # Builds the file path.
            p = f"{V2_DIR}/{key}_reward_curve.json"
            # If it exists...
            if os.path.exists(p):
                # ...open it...
                with open(p) as f:
                    # ...and load it as a series.
                    sim_curves[label] = pd.Series(json.load(f))
        # If any curves were found...
        if sim_curves:
            # ...combine them into one table...
            cdf = pd.DataFrame(sim_curves)
            # ...draw them smoothed with a rolling average.
            st.line_chart(cdf.rolling(max(1, len(cdf) // 30), min_periods=1).mean())
            # Caption explaining the curves.
            st.caption("Episode reward per agent, smoothed (rounds are concatenated; later rounds train against learned partners).")
        # Otherwise...
        else:
            # ...show a message that no curves were found.
            st.info("No reward curves found.")
