"""
__main__.py
-----------
Multi-page app router.

Handles the single st.set_page_config call (must be first) and CSS injection,
then delegates to the selected page via st.navigation / st.Page.

Pages
-----
  📈  Efficient Frontier     — pages/efficient_frontier.py
  🔍  Performance Attribution — pages/performance_attribution.py

Run
---
  streamlit run __main__.py
"""
import streamlit as st

from src.dashboard.frontier_dashboard import _CSS

st.set_page_config(
    page_title="Portfolio Construction",
    page_icon="📊",
    layout="wide",
)
st.markdown(_CSS, unsafe_allow_html=True)

pages = [
    st.Page("src/pages/efficient_frontier.py", title="Efficient Frontier", icon="📈"),
    st.Page("src/pages/performance_attribution.py", title="Performance Attribution", icon="🔍"),
]

pg = st.navigation(pages)
pg.run()
