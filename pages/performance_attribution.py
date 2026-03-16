"""
pages/performance_attribution.py
---------------------------------
Performance Attribution page — wires the attribution backend with the UI layer.

Data flow
---------
1. Sidebar collects: stock, start_date, end_date, window_size
2. On "Run Attribution": compute_attribution() downloads data from Yahoo Finance
   and FRED, runs static + rolling OLS, and returns an AttributionResults object.
3. Results are persisted in st.session_state["attr_results"] so that switching
   away and back to this page does not trigger a re-download.
"""
import os

import streamlit as st
from dotenv import load_dotenv

from src.dashboard.attribution_dashboard import (
    render_attribution_charts,
    render_attribution_header,
    render_attribution_metrics,
    render_attribution_sidebar,
)
from src.portfolio_construction.performance_attribution import compute_attribution

# Load FRED_API_KEY from .env if present
load_dotenv()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
inputs = render_attribution_sidebar()

# ── Wait for first run ────────────────────────────────────────────────────────
if not inputs["run"] and "attr_results" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **▶ Run Attribution** to begin.")
    st.stop()

# ── Run attribution when button is clicked ────────────────────────────────────
if inputs["run"]:
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        st.error(
            "**FRED API key not found.** "
            "Add `FRED_API_KEY=your_key` to a `.env` file in the project root. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        st.stop()

    with st.spinner(f"Running attribution for **{inputs['stock']}**…"):
        try:
            result = compute_attribution(
                stock=inputs["stock"],
                start_date=str(inputs["start_date"]),
                end_date=str(inputs["end_date"]),
                window_size=inputs["window_size"],
                fred_api_key=fred_key,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            # print(e)
            st.error(f"Attribution failed: {e}")
            st.stop()

    st.session_state["attr_results"] = result

# ── Render results ────────────────────────────────────────────────────────────
res = st.session_state["attr_results"]

render_attribution_header(res.stock, res.window_size)
render_attribution_metrics(res)
render_attribution_charts(res)
