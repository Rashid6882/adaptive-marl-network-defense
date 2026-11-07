# Module docstring title: the "Live pipeline" demo tab of the dashboard.
# What it shows: real flows from the held-out test stream travelling one by one through Detection, Firewall and
# Response, with a plain-English result for each flow and running counters.
# Ends the docstring.
"""
Live pipeline demo (a tab of src/dashboard.py).

Real flows from the held-out test stream travel one by one through the three agents:

    incoming flow -> Detection -> Firewall -> Response -> outcome

For every flow the page shows what each agent decided, what really happened (the hidden truth is revealed only
at the end of the pipeline) and running counters, next to the same traffic with no defense.
"""

# Imports time to pause between flows during the animation.
import time

# Imports pandas for the small tables behind the chart.
import pandas as pd
# Imports numpy for fast cumulative sums.
import numpy as np
# Imports streamlit, the web dashboard library.
import streamlit as st

# Imports the "no defense" policy, used as the comparison line.
from src.evaluate_sim import policy_none
# Imports the data loader for the held-out test stream.
from src.sim_env import load_split_cached

# Plain-English names for the destination ports that matter in this data.
PORT_NAMES = {21: "FTP", 22: "SSH", 53: "DNS", 80: "web", 443: "secure web", 445: "file sharing", 3389: "remote desktop"}
# Friendly names for the four traffic classes (list position = class number).
CLASS_LABELS = ["Normal traffic", "DDoS flood", "Port scan", "Brute-force login"]
# Colours: green = safe, amber = cautious, red = harsh, blue = information, grey = idle.
GREEN, AMBER, RED, BLUE, GREY = "#2e9e5b", "#e08e0b", "#d64545", "#3b7ddd", "#8a8f98"
# What each Firewall action looks like: (name, colour, what it does).
FW_LOOK = [
    ("Allow", GREEN, "lets the flow through"),
    ("Rate-limit", AMBER, "slows the attacker to 30% of its damage"),
    ("Block", RED, "cuts the attacker off for 200 flows"),
]
# What each Response action looks like: (name, colour, what it does).
RESP_LOOK = [
    ("No action", GREY, "nothing more to do"),
    ("Alert", AMBER, "an analyst steps in after 40 flows"),
    ("Isolate", RED, "cuts the host off for 400 flows"),
]
# The arrow drawn between two boxes of the pipeline.
ARROW = '<div style="font-size:1.6rem;opacity:0.5;padding:0 2px">&#10140;</div>'


# Decorator: load the raw test columns once and reuse them (cached).
@st.cache_resource(show_spinner="Loading traffic...")
# Function that returns the human-readable columns of the test stream as arrays.
def load_test_raw():
    # Loads the test table (the same rows, in the same order, as the simulator uses).
    _, test_df = load_split_cached()
    # Keeps only the columns shown on the "incoming flow" card.
    return {
        # Packets per second of each flow.
        "pkts": test_df["Flow Packets/s"].to_numpy(),
        # Duration of each flow in microseconds.
        "dur": test_df["Flow Duration"].to_numpy(),
        # Destination port of each flow.
        "dport": test_df["Destination Port"].to_numpy(),
        # Number of SYN flags in each flow.
        "syn": test_df["SYN Flag Count"].to_numpy(),
    }


# Function that runs one policy over a stretch of traffic and records one row per decided flow.
def simulate_trace(sim, policy, start, steps):
    # Clears the simulator statistics.
    sim.reset_stats()
    # Starts an episode at the chosen position for the chosen number of decisions.
    sim.reset(start=int(start), max_steps=steps)
    # List that will hold one record per flow the agents decided on.
    rows = []
    # Loops until the episode ends.
    while not sim.done:
        # Position of the current flow in the test stream.
        pos = sim.pos
        # The policy's three decisions (Detection, Firewall, Response).
        d, f, r = policy(sim)
        # Snapshot of damage, disruption and dropped counts before the step.
        s0 = (sim.s["damage"], sim.s["disruption"], sim.s["dropped"])
        # Applies the decisions and gets the true class of this flow.
        true = sim.apply(d, f, r)[4]
        # Records everything about this flow.
        rows.append(
            # Opens the record.
            dict(
                # Position in the test stream and the true class.
                pos=int(pos), true=int(true),
                # The three decisions.
                det=int(d), fw=int(f), resp=int(r),
                # Damage and benign disruption caused by this flow.
                damage=sim.s["damage"] - s0[0], disruption=sim.s["disruption"] - s0[1],
                # Attacker flows dropped right after this decision (they never reach the agents).
                dropped=int(sim.s["dropped"] - s0[2]),
            # Closes the record.
            )
        # Closes the append call.
        )
    # Returns the list of records.
    return rows


# Function that builds one coloured box of the pipeline.
def _box(title, big, sub, color):
    # A rounded box: small title, big coloured text, small description.
    return (
        f'<div style="flex:1;border:2px solid {color};background:{color}22;border-radius:12px;'
        f'padding:10px 8px;text-align:center;min-height:125px">'
        f'<div style="font-size:0.78rem;opacity:0.75">{title}</div>'
        f'<div style="font-size:1.2rem;font-weight:700;color:{color};margin:4px 0">{big}</div>'
        f'<div style="font-size:0.78rem;opacity:0.85">{sub}</div></div>'
    )


# Function that says in plain words what happened to one flow: returns (headline, detail, colour).
def _outcome(ev):
    # True class, Firewall action and Response action of this flow.
    true, fw, resp = ev["true"], ev["fw"], ev["resp"]
    # Branch: the flow is really an attack.
    if true != 0:
        # Block or Isolate contains the attacker.
        if fw == 2 or resp == 2:
            # Attack stopped.
            return "Attack stopped", "the attacker is contained, so its next flows are dropped", GREEN
        # Rate-limit lets 30% of the damage through.
        if fw == 1:
            # Attack slowed down.
            return "Attack slowed", f"only {ev['damage']:.2f} damage got through (30% of full)", AMBER
        # Allow lets the full damage through; an alert may follow.
        extra = "; an alert was raised" if resp == 1 else ""
        # Attack got through.
        return "Attack got through", f"{ev['damage']:.2f} damage{extra}", RED
    # Branch: a normal user was hurt by a wrong action or by an outage.
    if ev["disruption"] > 0:
        # Real user hurt.
        return "Real user hurt", f"penalty {ev['disruption']:.1f}", RED
    # Branch: a normal user passed untouched.
    return "Normal user passed", "no harm done", GREEN


# Function that builds the whole pipeline picture for one flow as one block of HTML.
def _pipeline_html(ev, raw):
    # Position of the flow in the test stream.
    p = ev["pos"]
    # Destination port and its friendly name.
    port = int(raw["dport"][p])
    # Text on the "incoming flow" card: port, speed, length and SYN flags.
    flow_sub = (
        f'{raw["pkts"][p]:,.0f} packets/s<br>lasts {raw["dur"][p] / 1000:,.1f} ms<br>'
        f'SYN flags: {int(raw["syn"][p])}'
    )
    # Detection's verdict: red if it raised an alarm, green if it said normal.
    det_color = GREEN if ev["det"] == 0 else RED
    # Firewall and Response looks for the chosen actions.
    fw_name, fw_color, fw_sub = FW_LOOK[ev["fw"]]
    # Response look for the chosen action.
    rp_name, rp_color, rp_sub = RESP_LOOK[ev["resp"]]
    # Plain-English result of this flow.
    head, detail, out_color = _outcome(ev)
    # Reveals the hidden truth and whether Detection was right.
    verdict_ok = "Detection was right" if ev["det"] == ev["true"] else "Detection was wrong"
    # Truth line for the outcome box.
    truth = f'Truth: {CLASS_LABELS[ev["true"]]}<br>({verdict_ok})'
    # Joins the five boxes with arrows into one row.
    row = (
        '<div style="display:flex;align-items:center;gap:6px">'
        + _box(f'Incoming flow #{ev["n"]}', f"port {port} ({PORT_NAMES.get(port, 'other')})", flow_sub, BLUE)
        + ARROW
        + _box("1. Detection says", CLASS_LABELS[ev["det"]], "looks at the flow's statistics", det_color)
        + ARROW
        + _box("2. Firewall does", fw_name, fw_sub, fw_color)
        + ARROW
        + _box("3. Response does", rp_name, rp_sub, rp_color)
        + ARROW
        + _box("Result", head, f"{detail}<br><i>{truth}</i>", out_color)
        + "</div>"
    )
    # Banner shown when the block made later attacker flows disappear before reaching any agent.
    shield = ""
    # Only if some flows were dropped after this decision.
    if ev["dropped"] > 0:
        # Shield message.
        shield = (
            f'<div style="margin-top:8px;padding:8px 12px;border-radius:10px;background:{GREEN}22;'
            f'border:1px solid {GREEN}">&#128737; Attacker blocked for the next {400 if ev["resp"] == 2 else 200} flows. '
            f'<b>{ev["dropped"]}</b> attacker connections in that time were turned away at the door, so no agent '
            f'has to check them. Normal traffic still passes.</div>'
        )
    # Returns the row of boxes and the optional shield banner.
    return row + shield


# Function that draws the animated page for one chosen scenario (called from the dashboard tab).
def render(sim, policies):
    # Heading of the tab.
    st.subheader("Watch real traffic travel through the three agents")
    # Short explanation in everyday words.
    st.markdown(
        "Each card is **one real network connection** from the test traffic (traffic the agents never trained on). "
        "It enters on the left and is handled by three agents in turn. The coloured boxes show what each agent "
        "decided. The **Result** box at the end reveals the truth that the agents were never told."
    )
    # Collapsible legend.
    with st.expander("How to read this"):
        # Legend text.
        st.markdown(
            "- **Detection** names the traffic type. **Firewall** decides what to do with it. **Response** decides "
            "whether to raise an alert or cut the host off.\n"
            "- **Green** is safe, **amber** is cautious, **red** is a harsh action or a harmful outcome.\n"
            "- When an attacker is blocked, its later connections are turned away at the door, so no agent has to "
            "check them.\n"
            "- The graph shows total attack damage: red is no defense, green is the chosen defense. A flat line means "
            "the attacker is contained."
        )
    # Loads the raw columns for the flow card.
    raw = load_test_raw()

    # Four controls in a row.
    c1, c2, c3, c4 = st.columns(4)
    # Which attack to watch.
    attack = c1.selectbox("Which attack?", ["DDoS", "PortScan", "BruteForce"], key="live_attack")
    # Where in the test stream to look (0 = early, 100 = late).
    where = c2.slider(
        "Where in the traffic? (0 = early, 100 = late)", 0, 100, 10, key="live_where",
        help="The test traffic is one long recording with many attacks in it. This picks which attack to replay.",
    )
    # Which defense is protecting the network.
    names = list(policies.keys())
    # Default to the 3-agent system when it is available.
    default_label = "3-agent MARL (interactive RL)"
    # Dropdown with the systems.
    system = c3.selectbox("Which defense?", names, index=names.index(default_label) if default_label in names else 0, key="live_system")
    # How many decisions to run.
    steps = c4.slider(
        "How many flows to run", 50, 600, 150, step=50, key="live_steps",
        help="A flow is one network connection. The agents check this many connections, then the replay stops. "
        "Connections from a blocked attacker are turned away and are not counted.",
    )
    # Seconds each flow stays on screen.
    speed = 0.8

    # Class number of the chosen attack (0 DDoS, 1 PortScan, 2 BruteForce in the simulator's attack index).
    cls_idx = {"DDoS": 0, "PortScan": 1, "BruteForce": 2}[attack]
    # Positions of every flow of that attack in the test stream.
    idxs = sim._attack_idx[cls_idx]
    # Picks the anchor attack flow according to the slider.
    anchor = int(idxs[int(where / 100 * (len(idxs) - 1))])
    # Starts 30 flows before the anchor (but not before the file start), so a few normal flows come first.
    start = max(int(sim.file_start[anchor]), anchor - 30)

    # Runs the chosen defense over the stretch of traffic.
    tr = pd.DataFrame(simulate_trace(sim, policies[system], start, steps))
    # First traffic position after everything the defense covered: its last decision plus the flows dropped after it.
    end = int(tr["pos"].iloc[-1]) + 1 + int(tr["dropped"].iloc[-1])
    # Runs "no defense" over the same stretch of traffic for comparison. With no blocks nothing is dropped, so
    # every position is one decision and the run needs end - start decisions to cover the same traffic.
    base = pd.DataFrame(simulate_trace(sim, policy_none, start, end - start))
    # Running totals for the chosen defense.
    tr["n"] = np.arange(1, len(tr) + 1)
    # Cumulative damage so far.
    tr["cum_damage"] = tr["damage"].cumsum()
    # Cumulative attacker flows dropped so far.
    tr["cum_dropped"] = tr["dropped"].cumsum()
    # Cumulative real users hurt so far.
    tr["cum_hurt"] = (tr["disruption"] > 0).cumsum()
    # Cumulative damage of "no defense" at the same stream position as each flow above.
    base_cum = base["damage"].cumsum().to_numpy()
    # Stream positions of the no-defense run.
    base_pos = base["pos"].to_numpy()
    # For every flow, how many no-defense flows came at or before it.
    k = np.searchsorted(base_pos, tr["pos"].to_numpy(), side="right")
    # No-defense damage at that point (0 before the first flow).
    tr["no_defense"] = np.where(k > 0, base_cum[np.maximum(k - 1, 0)], 0.0)

    # Keeps only the interesting flows: attacks, actions other than Allow / No action, damage and disruption.
    keep = (tr["true"] != 0) | (tr["fw"] != 0) | (tr["resp"] != 0) | (tr["damage"] > 0) | (tr["disruption"] > 0)
    # Applies the filter.
    events = tr[keep].reset_index(drop=True)
    # Nothing to show.
    if events.empty:
        # Tells the user and stops this tab.
        st.info("No flows to show here. Try another position.")
        # Leaves the function.
        return

    # Play button that starts the animation.
    play = st.button("▶ Play: send the flows through the pipeline", type="primary", key="live_play")
    # Placeholder where the pipeline picture is redrawn for each flow.
    frame = st.empty()
    # Placeholder where the damage chart is redrawn for each flow.
    chart = st.empty()
    # Caption explaining the chart.
    caption = st.empty()

    # Function that draws the page for the i-th shown flow.
    def draw(i):
        # The flow to draw, with whole-number fields kept as integers (a table row would turn them into floats).
        ev = {k: (int(v) if k in ("pos", "true", "det", "fw", "resp", "dropped", "n", "cum_dropped", "cum_hurt") else float(v)) for k, v in events.iloc[i].items()}
        # Row of the full trace this flow belongs to (flow number - 1).
        j = int(ev["n"]) - 1
        # On the last flow shown, the chart and the totals cover the whole run (the flows after it were routine).
        if i == len(events) - 1:
            # Uses the last row of the full trace.
            j = len(tr) - 1
        # Damage added up so far with no defense and with the chosen defense.
        damage_none = float(tr["no_defense"].iloc[j])
        # Damage with the chosen defense.
        damage_sys = float(tr["cum_damage"].iloc[j])
        # Redraws the pipeline picture and the damage numbers.
        with frame.container():
            # The five boxes with arrows.
            st.markdown(_pipeline_html(ev, raw), unsafe_allow_html=True)
            # Two damage totals side by side (one only when "No defense" itself is chosen).
            n_cols = st.columns(1 if system == "No defense" else 2)
            # Total damage with no defense so far.
            n_cols[0].metric("Total damage with no defense", f"{damage_none:,.1f}")
            # Total damage with the chosen defense so far.
            if system != "No defense":
                # Shows the chosen defense's total.
                n_cols[1].metric(f"Total damage with {system}", f"{damage_sys:,.1f}")
        # Data for the damage chart up to this flow, indexed by position in the stretch of traffic.
        upto = tr.iloc[: j + 1]
        # X axis: flows since the start of the stretch.
        x = upto["pos"] - int(tr["pos"].iloc[0])
        # Chart table: the chosen defense and no defense.
        data = pd.DataFrame({system: upto["cum_damage"].to_numpy(), "No defense": upto["no_defense"].to_numpy()}, index=x.to_numpy())
        # With "No defense" chosen there is only one line.
        if system == "No defense":
            # Drops the duplicate column.
            data = data[["No defense"]]
        # Draws the chart (green for the defense, red for no defense).
        chart.line_chart(
            data, color=[GREEN, RED] if data.shape[1] == 2 else [RED],
            x_label="Connections since the replay started", y_label="Total attack damage so far",
        )
        # Caption under the chart.
        caption.caption("A flat line means the attacker is contained.")

    # If Play was pressed...
    if play:
        # ...draw every flow in turn with a pause in between.
        for i in range(len(events)):
            # Draws this flow.
            draw(i)
            # Waits so the audience can read it.
            time.sleep(speed)
    # Otherwise...
    else:
        # ...asks the user to press Play.
        frame.info("Press Play to send the flows through the pipeline.")
