"""Streamlit entry point and navigation for the Job Search Agent."""

from __future__ import annotations

import streamlit as st

from src.ui.components import (
    apply_app_styles,
    render_sidebar_brand,
    render_sidebar_navigation,
)
from src.ui.session import ensure_session_defaults

st.set_page_config(
    page_title="Job search agent",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
ensure_session_defaults(st.session_state)
apply_app_styles()

pages = [
    st.Page(
        "views/1_Input.py",
        title="Set up search",
        icon=":material/tune:",
        default=True,
    ),
    st.Page(
        "views/2_Execution.py",
        title="Run progress",
        icon=":material/timeline:",
    ),
    st.Page(
        "views/3_Review.py",
        title="Review drafts",
        icon=":material/rate_review:",
    ),
    st.Page(
        "views/4_Results.py",
        title="Results",
        icon=":material/folder_open:",
    ),
]
page = st.navigation(pages, position="hidden")
render_sidebar_brand()
render_sidebar_navigation(pages)
page.run()
