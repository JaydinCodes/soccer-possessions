import json
import streamlit as st
import pandas as pd

st.title("Soccer Possession Dashboard")

with open("outputs/possession_log.json") as f:
    data = json.load(f)

summary = data["summary"]
pct_a = summary["team_a_pct"]
pct_b = summary["team_b_pct"]

st.header("Possession")

col1, col2 = st.columns(2)
col1.metric("Team A", f"{pct_a}%")
col2.metric("Team B", f"{pct_b}%")

bar_html = f"""
<div style="display:flex; height:30px; border-radius:6px; overflow:hidden; font-family:sans-serif;">
  <div style="width:{pct_a}%; background:#3b82f6; color:white; text-align:center; line-height:30px;">{pct_a}%</div>
  <div style="width:{pct_b}%; background:#ef4444; color:white; text-align:center; line-height:30px;">{pct_b}%</div>
</div>
"""
st.markdown(bar_html, unsafe_allow_html=True)

timeline = data["timeline"]
df = pd.DataFrame(timeline)

df["a_has"] = df["team"].map({0: 1, 1: 0})
df["a_share"] = df["a_has"].rolling(window=20, min_periods=1).mean() * 100

st.header("Possession over time (Team A %)")
st.line_chart(df.set_index("frame")["a_share"])