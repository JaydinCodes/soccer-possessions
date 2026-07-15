"""Live Streamlit possession dashboard.

Re-reads outputs/possession_log.json every second (via a fragment) so the
numbers update in real time while run.py is processing a video or phone feed.

    streamlit run app.py
"""

import json

import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="Soccer Possession", page_icon="⚽", layout="centered")
st.title("⚽ Soccer Possession — Live")

A_HEX = config.TEAM_HEX[0]
B_HEX = config.TEAM_HEX[1]


def load_data():
    try:
        with open(config.OUTPUT_JSON) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.fragment(run_every="1s")
def dashboard():
    data = load_data()
    if data is None:
        st.info("Waiting for data… start `python run.py --source data/match.mp4`.")
        return

    summary = data["summary"]
    meta = data.get("meta", {})
    pct_a = summary["team_a_pct"]
    pct_b = summary["team_b_pct"]

    if not meta.get("teams_ready", True):
        st.warning("Learning the two team colours from the opening frames…")

    st.subheader("Possession")
    c1, c2, c3 = st.columns(3)
    c1.metric("Team A", f"{pct_a:.0f}%")
    c2.metric("Team B", f"{pct_b:.0f}%")
    total = summary["team_a_frames"] + summary["team_b_frames"] + summary["loose_frames"]
    loose_pct = 100 * summary["loose_frames"] / total if total else 0
    c3.metric("Loose", f"{loose_pct:.0f}%")

    # Stacked possession bar, coloured to match the video overlay.
    bar_html = f"""
    <div style="display:flex; height:30px; border-radius:6px; overflow:hidden;
                font-family:sans-serif; font-weight:600;">
      <div style="width:{pct_a}%; background:{A_HEX}; color:#000;
                  text-align:center; line-height:30px;">{pct_a:.0f}%</div>
      <div style="width:{pct_b}%; background:{B_HEX}; color:#fff;
                  text-align:center; line-height:30px;">{pct_b:.0f}%</div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

    # Possession over time -- rolling share of the ball for Team A.
    timeline = data.get("timeline", [])
    if timeline:
        df = pd.DataFrame(timeline)
        df["a_has"] = df["team"].map({0: 1, 1: 0})   # None -> NaN (loose, ignored)
        df["a_share"] = df["a_has"].rolling(window=20, min_periods=1).mean() * 100
        st.subheader("Possession over time — Team A %")
        st.line_chart(df.set_index("frame")["a_share"])

    st.caption(
        f"Frames processed: {meta.get('frames_processed', '?')} · "
        f"updates every second while the tracker runs."
    )


dashboard()
