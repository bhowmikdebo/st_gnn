
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import json
from pathlib import Path

# Import our predictor
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.predictor import WastePredictor
from utils.explainability import WPI_LEVELS

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chennai Waste Management — ST-GNN",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8F9FA; }

    /* WPI card styling */
    .wpi-card {
        color: #212121;   /* dark text */
        padding: 16px 20px;
        border-radius: 12px;
        margin: 6px 0;
        border-left: 6px solid;
        background: white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .wpi-critical  { border-color: #D32F2F; background: #FFEBEE; }
    .wpi-high      { border-color: #F57C00; background: #FFF3E0; }
    .wpi-elevated  { border-color: #FBC02D; background: #FFFDE7; }
    .wpi-normal    { border-color: #388E3C; background: #E8F5E9; }
    .wpi-low       { border-color: #1976D2; background: #E3F2FD; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        /* color: #263238; */
        color: #ECEFF1;
        margin: 12px 0 6px 0;
        padding-bottom: 4px;
        border-bottom: 2px solid #E0E0E0;
    }

    /* Metric tiles */
    .metric-tile {
        background: white;
        border-radius: 10px;
        padding: 14px 18px;
        color: #212121;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }

    /* Explanation box */
    .explanation-box {
        background: #F5F5F5;
        border-radius: 8px;
        padding: 12px 16px;
        background: #F5F5F5;
        color: #212121;
        margin: 4px 0;
        font-size: 0.9rem;
        border-left: 4px solid #7986CB;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model (cached so it only loads once) ────────────────────────────────

@st.cache_resource
def load_predictor():
    """Load the trained model. Cached — only runs once per session."""
    predictor = WastePredictor()
    try:
        predictor.load()
        return predictor, None
    except FileNotFoundError as e:
        return None, str(e)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/3/34/Greater_Chennai_Corporation_logo.svg/200px-Greater_Chennai_Corporation_logo.svg.png",
                     width=100)
    st.sidebar.title("♻️ Waste Management AI")
    st.sidebar.markdown("**Chennai Municipal Corporation**")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 📅 Prediction Date")
    pred_date = st.sidebar.date_input(
        "Predict waste for:",
        value=date.today() + timedelta(days=1),
        min_value=date.today(),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown(
        "This system uses a **Spatio-Temporal Graph Neural Network (ST-GNN)** "
        "to forecast next-day waste generation for all 15 Chennai zones."
    )
    st.sidebar.markdown(
        "**Model features:**\n"
        "- Spatial zone dependencies\n"
        "- Temporal attention (7-day window)\n"
        "- Festival/holiday effects\n"
        "- Explainable predictions\n"
        "- Waste Pressure Index (WPI)"
    )

    return pred_date


# ─── Tab 1: Real-Time Data Entry ──────────────────────────────────────────────

def render_data_entry_tab(predictor):
    st.markdown('<div class="section-header">📥 Enter Today\'s Actual Waste Data</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Enter the actual waste collected today for each zone. "
        "Leave as 0 to use historical averages from the dataset. "
        "These values will be used in the sliding window for tomorrow's prediction."
    )

    real_time_data = {}

    # Show zones in 3 columns
    zones = predictor.zones
    cols  = st.columns(3)

    for i, zone in enumerate(zones):
        col = cols[i % 3]
        baseline = predictor.baselines.get(zone, 1.0)
        val = col.number_input(
            label    = f"{zone}",
            min_value= 0.0,
            max_value= 2000.0,
            value    = 0.0,
            step     = 0.5,
            format   = "%.1f",
            help     = f"Historical average: {baseline:.1f} MT/day",
            key      = f"rt_{zone}",
        )
        if val > 0:
            real_time_data[zone] = [val]   # single latest value

    if real_time_data:
        n_entered = len(real_time_data)
        st.success(f"✅ Real-time data entered for **{n_entered} zone(s)**. "
                   "These will override historical values in the prediction window.")
    else:
        st.info("ℹ️ No real-time data entered. Using historical averages from dataset.")

    return real_time_data


# ─── Tab 2: Festival / Holiday Flag ───────────────────────────────────────────

def render_festival_tab(predictor, pred_date):
    st.markdown('<div class="section-header">🎉 Festival / Holiday / Special Event</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Mark if there is a festival, fair, or public holiday on the prediction date. "
        "The model will adjust waste forecasts accordingly."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        is_festival = st.checkbox("This date has a festival / holiday / special event")

    festival_info = {"is_festival": False}

    if is_festival:
        with col2:
            fest_name = st.text_input(
                "Event name",
                placeholder="e.g. Deepavali, Pongal, City Marathon...",
                key="fest_name"
            )

        col3, col4, col5 = st.columns(3)
        with col3:
            affected_all = st.checkbox("Affects all zones", value=True)

        affected_zones = ["all"]
        if not affected_all:
            with col4:
                affected_zones = st.multiselect(
                    "Select affected zones:",
                    options=predictor.zones,
                    default=predictor.zones[:3],
                )

        with col5:
            spike_pct = st.slider(
                "Expected waste increase (%)",
                min_value=5,
                max_value=60,
                value=25,
                step=5,
                help="How much more waste do you expect compared to a normal day?"
            )

        festival_info = {
            "is_festival":    True,
            "festival_name":  fest_name or "Special Event",
            "affected_zones": affected_zones,
            "spike":          spike_pct / 100.0,
        }

        st.warning(
            f"⚠️ **{festival_info['festival_name']}** on **{pred_date.strftime('%d %b %Y')}**. "
            f"Expected waste increase: **+{spike_pct}%** for "
            f"{'all zones' if 'all' in affected_zones else ', '.join(affected_zones)}."
        )

    return festival_info


# ─── WPI Chart ────────────────────────────────────────────────────────────────

def render_wpi_chart(wpi_results: list):
    """Render a beautiful horizontal bar chart for WPI."""

    zones  = [r["zone"]        for r in wpi_results]
    scores = [r["wpi_score"]   for r in wpi_results]
    colors = [r["color"]       for r in wpi_results]
    levels = [r["level"]       for r in wpi_results]
    preds  = [r["predicted_mt"] for r in wpi_results]

    # Sort by WPI score descending (critical on top)
    sorted_idx = np.argsort(scores)[::-1]
    zones_s  = [zones[i]  for i in sorted_idx]
    scores_s = [scores[i] for i in sorted_idx]
    colors_s = [colors[i] for i in sorted_idx]
    levels_s = [levels[i] for i in sorted_idx]
    preds_s  = [preds[i]  for i in sorted_idx]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x          = scores_s,
        y          = zones_s,
        orientation= "h",
        marker_color = colors_s,
        text       = [f"{s:.2f}  ({l})  |  {p:.1f} MT"
                      for s, l, p in zip(scores_s, levels_s, preds_s)],
        textposition = "outside",
        hovertemplate = (
            "<b>%{y}</b><br>"
            "WPI Score: %{x:.3f}<br>"
            "Predicted: %{customdata:.1f} MT<br>"
            "<extra></extra>"
        ),
        customdata = preds_s,
    ))

    # Reference line at WPI = 1.0 (normal baseline)
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Baseline (1.0)",
                  annotation_position="top right")

    fig.update_layout(
        title       = dict(
            text    = "🗂️ Waste Pressure Index (WPI) — All Zones",
            font    = dict(size=16, color="#263238"),
        ),
        xaxis_title = "WPI Score  (1.0 = normal baseline)",
        yaxis_title = "",
        height      = 550,
        plot_bgcolor= "white",
        paper_bgcolor="white",
        xaxis       = dict(gridcolor="#EEEEEE", range=[0, max(scores_s) * 1.35]),
        yaxis       = dict(gridcolor="#EEEEEE"),
        margin      = dict(l=160, r=200, t=60, b=40),
        font        = dict(family="Inter, sans-serif", size=12),
        showlegend  = False,
    )

    # Add WPI threshold zone bands
    threshold_zones = [
        (0.90, 1.05, "rgba(56,142,60,0.06)",  "Normal"),
        (1.05, 1.15, "rgba(251,192,45,0.08)", "Elevated"),
        (1.15, 1.30, "rgba(245,124,0,0.08)",  "High"),
        (1.30, max(scores_s) * 1.4, "rgba(211,47,47,0.08)", "Critical"),
    ]
    for x0, x1, fill, label in threshold_zones:
        if x1 > 0:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=fill, line_width=0,
                          annotation_text=label, annotation_position="top left",
                          annotation_font_size=10, annotation_font_color="gray")

    return fig


def render_wpi_summary_cards(wpi_results: list):
    """Show summary tiles: count per level."""
    level_counts = {}
    for r in wpi_results:
        level_counts[r["level"]] = level_counts.get(r["level"], 0) + 1

    cols = st.columns(len(WPI_LEVELS))
    for i, lvl in enumerate(WPI_LEVELS):
        cnt = level_counts.get(lvl["label"], 0)
        with cols[i]:
            st.markdown(
                f"""
                <div class="metric-tile" style="border-top: 4px solid {lvl['color']};">
                    <div style="font-size:1.8rem">{lvl['emoji']}</div>
                    <div style="font-size:1.5rem; font-weight:700; color:{lvl['color']}">{cnt}</div>
                    <div style="font-size:0.85rem; color:#666">{lvl['label']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ─── Explainability Panel ─────────────────────────────────────────────────────

def render_explainability(wpi_results: list, explanations: list):
    st.markdown('<div class="section-header">🧠 Explainability — Why These Predictions?</div>',
                unsafe_allow_html=True)

    # Let user pick a zone to inspect
    zone_names = [r["zone"] for r in wpi_results]
    # Pre-select the highest WPI zone
    default_zone = max(wpi_results, key=lambda r: r["wpi_score"])["zone"]

    selected_zone = st.selectbox(
        "Select a zone to see its explanation:",
        options=zone_names,
        index=zone_names.index(default_zone),
        key="explain_zone"
    )

    # Find the matching WPI and explanation
    wpi_info = next(r for r in wpi_results    if r["zone"] == selected_zone)
    exp_info = next(e for e in explanations   if e["zone"] == selected_zone)

    # ── Zone summary card ────────────────────────────────────────────────────
    level_class = {
        "Critical": "wpi-critical",
        "High":     "wpi-high",
        "Elevated": "wpi-elevated",
        "Normal":   "wpi-normal",
        "Low":      "wpi-low",
    }.get(wpi_info["level"], "wpi-normal")

    deviation_sign = "+" if exp_info["deviation_pct"] >= 0 else ""

    st.markdown(
        f"""
        <div class="wpi-card {level_class}">
            <b style="font-size:1.1rem">{wpi_info['emoji']} {selected_zone}</b>
            &nbsp;&nbsp;
            <span style="font-size:1.3rem; font-weight:700">{wpi_info['level']}</span>
            <br>
            <span style="font-size:0.95rem">
                Predicted: <b>{wpi_info['predicted_mt']:.1f} MT</b>
                &nbsp;|&nbsp;
                Baseline: {wpi_info['baseline_mt']:.1f} MT
                &nbsp;|&nbsp;
                WPI: {wpi_info['wpi_score']:.2f}
                ({deviation_sign}{exp_info['deviation_pct']:.1f}% vs baseline)
            </span>
            <br><br>
            <span style="color:#444"><b>Recommended Action:</b> {wpi_info['action']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Reasons ───────────────────────────────────────────────────────────────
    st.markdown("**Factors influencing this prediction:**")
    if exp_info["reasons"]:
        for reason in exp_info["reasons"]:
            st.markdown(
                f'<div class="explanation-box">💡 {reason}</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="explanation-box">ℹ️ Prediction follows normal historical pattern for this zone.</div>',
            unsafe_allow_html=True
        )


# ─── Predictions Table ────────────────────────────────────────────────────────

def render_predictions_table(wpi_results: list):
    """Render a coloured summary table of all zones."""
    rows = []
    for r in sorted(wpi_results, key=lambda x: x["wpi_score"], reverse=True):
        deviation_str = (f"+{r['wpi_percent']:.1f}%" if r["wpi_percent"] >= 0
                         else f"{r['wpi_percent']:.1f}%")
        rows.append({
            "Zone":           r["zone"],
            "Predicted (MT)": r["predicted_mt"],
            "Baseline (MT)":  r["baseline_mt"],
            "vs Baseline":    deviation_str,
            "WPI Score":      r["wpi_score"],
            "Level":          f"{r['emoji']} {r['level']}",
            "Action":         r["action"],
        })

    df = pd.DataFrame(rows)

    # Colour the Level column
    def colour_level(val):
        colours = {
            "🔴 Critical": "background-color: #212121",
            "🟠 High":     "background-color: #212121",
            "🟡 Elevated": "background-color: #212121",
            "🟢 Normal":   "background-color: #212121",
            "🔵 Low":      "background-color: #212121",
        }
        return colours.get(val, "")

    styled = (df.style
              .applymap(colour_level, subset=["Level"])
              .format({"Predicted (MT)": "{:.1f}",
                       "Baseline (MT)":  "{:.1f}",
                       "WPI Score":      "{:.3f}"})
              .set_properties(**{"font-size": "13px"})
             )

    st.dataframe(styled, use_container_width=True, hide_index=True)


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    # Header
    st.title("♻️ Chennai Smart Waste Management")
    st.markdown("**AI-powered next-day waste prediction using Spatio-Temporal GNN**")
    st.markdown("---")

    # Load model
    predictor, error = load_predictor()

    if error:
        st.error(f"⚠️ Model not loaded: {error}")
        st.info(
            "**To use this dashboard:**\n\n"
            "1. Run `python data/preprocess.py` to prepare data\n"
            "2. Run `python train.py` to train the model\n"
            "3. Reload this page"
        )
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    pred_date = render_sidebar()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📥 Data Entry & Prediction",
        "📊 WPI Dashboard",
        "🧠 Explainability"
    ])

    # ── Tab 1: Data Entry + Prediction ────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns([3, 1])

        with col_left:
            real_time_data = render_data_entry_tab(predictor)

        with col_right:
            st.markdown('<div class="section-header">🎉 Festival / Event Flag</div>',
                        unsafe_allow_html=True)
            festival_info = render_festival_tab(predictor, pred_date)

        st.markdown("---")

        # ── Predict button ────────────────────────────────────────────────────
        predict_col, _ = st.columns([1, 3])
        with predict_col:
            predict_btn = st.button(
                f"🔮 Predict Waste for {pred_date.strftime('%d %b %Y')}",
                type="primary",
                use_container_width=True,
            )

        if predict_btn:
            with st.spinner("Running ST-GNN prediction..."):
                try:
                    results = predictor.predict_next_day(
                        real_time_entries = real_time_data if real_time_data else None,
                        festival_info     = festival_info if festival_info.get("is_festival") else None,
                    )
                    # Store in session state so other tabs can use it
                    st.session_state["results"] = results
                    st.session_state["pred_date"] = pred_date
                    st.success("✅ Prediction complete!")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    return

        # Show prediction summary table in Tab 1
        if "results" in st.session_state:
            results = st.session_state["results"]
            st.markdown(f'<div class="section-header">📋 Prediction Results — '
                        f'{st.session_state["pred_date"].strftime("%d %b %Y")}</div>',
                        unsafe_allow_html=True)
            render_predictions_table(results["wpi_results"])

    # ── Tab 2: WPI Dashboard ──────────────────────────────────────────────────
    with tab2:
        if "results" not in st.session_state:
            st.info("ℹ️ Run a prediction first (Tab 1) to see the WPI dashboard.")
        else:
            results = st.session_state["results"]
            wpi     = results["wpi_results"]

            st.markdown(f"### Waste Pressure Index — "
                        f"{st.session_state['pred_date'].strftime('%d %b %Y')}")
            st.markdown(
                "WPI = Predicted MT ÷ Historical Baseline. "
                "Values > 1.0 mean above-average waste generation."
            )

            # Summary tiles
            render_wpi_summary_cards(wpi)
            st.markdown("")

            # Main WPI chart
            fig = render_wpi_chart(wpi)
            st.plotly_chart(fig, use_container_width=True)

            # Legend
            st.markdown("**WPI Level Guide:**")
            legend_cols = st.columns(5)
            for i, lvl in enumerate(WPI_LEVELS):
                legend_cols[i].markdown(
                    f"<span style='color:{lvl['color']};font-weight:700'>"
                    f"{lvl['emoji']} {lvl['label']}</span><br>"
                    f"<small>WPI ≥ {lvl['min_factor']}</small>",
                    unsafe_allow_html=True
                )

    # ── Tab 3: Explainability ─────────────────────────────────────────────────
    with tab3:
        if "results" not in st.session_state:
            st.info("ℹ️ Run a prediction first (Tab 1) to see explanations.")
        else:
            results = st.session_state["results"]
            render_explainability(results["wpi_results"], results["explanations"])


if __name__ == "__main__":
    main()
