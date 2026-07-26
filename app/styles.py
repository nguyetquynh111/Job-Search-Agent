"""Application-wide Streamlit styling."""

from __future__ import annotations

import streamlit as st


def apply_app_styles() -> None:
    """Apply the visual system shared by every application page."""

    st.markdown(
        """
        <style>
            :root {
                --ink: #182230;
                --muted: #5d6b7c;
                --soft: #f6f8fb;
                --line: #d8e0ea;
                --panel: #ffffff;
                --brand: #2563eb;
                --brand-dark: #1d4ed8;
                --mint: #0f8f6f;
                --gold: #b7791f;
                --rose: #b42318;
            }

            .stApp {
                background: #f6f8fb;
                color: var(--ink);
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stAppViewContainer"] > .main .block-container {
                max-width: 1200px;
                padding-top: 2.25rem;
                padding-bottom: 5rem;
            }

            [data-testid="stSidebar"] {
                background: #111827;
                border-right: 1px solid #0b1220;
            }

            [data-testid="stSidebar"] * {
                color: #dbe5f2;
            }

            [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
                border-radius: 8px;
                margin-bottom: 0.2rem;
            }

            [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover,
            [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
                background: rgba(255, 255, 255, 0.09);
            }

            [data-testid="stSidebar"] [data-testid="stPageLink"] a {
                border-radius: 8px;
                font-weight: 520;
                margin-bottom: 0.16rem;
                padding: 0.48rem 0.65rem;
                width: 100%;
            }

            [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover,
            [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
                background: rgba(255, 255, 255, 0.09);
            }

            [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(255, 255, 255, 0.055);
                border-color: rgba(255, 255, 255, 0.10);
            }

            [data-testid="stSidebar"] hr {
                border-color: rgba(255, 255, 255, 0.12);
            }

            .app-brand {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin: 0.15rem 0 1.25rem;
            }

            .app-brand__mark {
                width: 2.25rem;
                height: 2.25rem;
                display: grid;
                place-items: center;
                border-radius: 8px;
                color: white !important;
                background: #2563eb;
                box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
                font-weight: 750;
            }

            .app-brand__name {
                color: white !important;
                font-size: 1rem;
                font-weight: 680;
                letter-spacing: 0;
            }

            .app-brand__tagline {
                color: #91a1b8 !important;
                font-size: 0.72rem;
                margin-top: 0.08rem;
            }

            .page-header {
                margin-bottom: 1.25rem;
                max-width: 760px;
            }

            .page-header__eyebrow {
                color: var(--brand);
                font-size: 0.77rem;
                font-weight: 720;
                letter-spacing: 0;
                margin-bottom: 0.55rem;
            }

            .page-header h1 {
                color: var(--ink);
                font-size: clamp(2.15rem, 4vw, 3.15rem);
                letter-spacing: 0;
                line-height: 1.05;
                margin: 0;
            }

            .page-header p {
                color: var(--muted);
                font-size: 1.04rem;
                line-height: 1.65;
                margin: 0.8rem 0 0;
            }

            h2, h3 {
                color: var(--ink);
                letter-spacing: 0;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
                box-shadow: 0 8px 24px rgba(20, 32, 48, 0.05);
            }

            [data-testid="stMetric"] {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.9rem 1rem;
            }

            [data-testid="stMetricValue"] {
                color: var(--ink);
                font-size: 1.45rem;
            }

            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            [data-baseweb="select"] > div {
                background: #fbfcfe;
                border-color: #d6dfeb;
                border-radius: 8px;
            }

            [data-testid="stFileUploaderDropzone"] {
                background: #f9fbfd;
                border-color: #cdd8e7;
                border-radius: 8px;
                padding: 0.8rem;
            }

            [data-testid^="stBaseButton-primary"] {
                background: var(--brand);
                border: 0;
                box-shadow: 0 8px 18px rgba(37, 99, 235, 0.18);
            }

            [data-testid^="stBaseButton-primary"]:hover {
                background: var(--brand-dark);
            }

            [data-testid="stBaseButton-secondary"] {
                border-color: #ced8e6;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: 8px;
                overflow: hidden;
            }

            .workflow-map {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 8px;
                display: grid;
                gap: 0;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                margin: 0 0 1.25rem;
                overflow: hidden;
            }

            .workflow-map__item {
                display: grid;
                gap: 0.65rem;
                grid-template-columns: 2rem 1fr;
                min-height: 7.4rem;
                padding: 0.95rem;
                position: relative;
            }

            .workflow-map__body {
                min-width: 0;
            }

            .workflow-map__item + .workflow-map__item {
                border-left: 1px solid var(--line);
            }

            .workflow-map__marker {
                align-items: center;
                background: #eef2f7;
                border: 1px solid #d9e2ed;
                border-radius: 50%;
                color: #46566b;
                display: flex;
                font-size: 0.8rem;
                font-weight: 760;
                height: 2rem;
                justify-content: center;
                width: 2rem;
            }

            .workflow-map__item--active {
                background: #eff6ff;
            }

            .workflow-map__item--active .workflow-map__marker {
                background: var(--brand);
                border-color: var(--brand);
                color: #ffffff;
            }

            .workflow-map__item--complete .workflow-map__marker {
                background: #e7f7f1;
                border-color: #b7e3d2;
                color: #087255;
            }

            .workflow-map__short {
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 730;
                letter-spacing: 0;
                margin-bottom: 0.18rem;
                text-transform: uppercase;
            }

            .workflow-map__title {
                color: var(--ink);
                font-size: 0.95rem;
                font-weight: 720;
                line-height: 1.25;
            }

            .workflow-map__copy {
                color: var(--muted);
                font-size: 0.8rem;
                line-height: 1.45;
                margin-top: 0.25rem;
            }

            .file-guide {
                display: grid;
                gap: 0.6rem;
                margin-top: 0.4rem;
            }

            .file-guide__item {
                background: #f8fafc;
                border: 1px solid #dce4ee;
                border-radius: 8px;
                padding: 0.72rem 0.8rem;
            }

            .file-guide__name {
                color: var(--ink);
                font-size: 0.9rem;
                font-weight: 720;
            }

            .file-guide__format {
                color: var(--brand);
                font-size: 0.78rem;
                font-weight: 730;
                margin-top: 0.12rem;
            }

            .file-guide__copy {
                color: var(--muted);
                font-size: 0.78rem;
                line-height: 1.45;
                margin-top: 0.28rem;
            }

            .workflow-step {
                display: grid;
                grid-template-columns: 2rem 1fr;
                gap: 0.75rem;
                padding: 0.7rem 0;
            }

            .workflow-step__number {
                align-items: center;
                background: #edf1ff;
                border-radius: 50%;
                color: var(--brand);
                display: flex;
                font-size: 0.78rem;
                font-weight: 750;
                height: 2rem;
                justify-content: center;
            }

            .workflow-step__title {
                color: var(--ink);
                font-size: 0.93rem;
                font-weight: 680;
                margin-top: 0.05rem;
            }

            .workflow-step__copy {
                color: var(--muted);
                font-size: 0.82rem;
                line-height: 1.45;
                margin-top: 0.15rem;
            }

            .sidebar-kicker {
                color: #91a1b8 !important;
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0;
                margin: 1rem 0 0.55rem;
            }

            .sidebar-status {
                align-items: center;
                display: flex;
                gap: 0.5rem;
                font-size: 0.9rem;
                font-weight: 620;
            }

            .sidebar-status__dot {
                background: #55d6ad;
                border-radius: 50%;
                box-shadow: 0 0 0 4px rgba(85, 214, 173, 0.12);
                height: 0.5rem;
                width: 0.5rem;
            }

            [data-testid="stSidebar"] button[kind="secondary"] {
                background: transparent;
                border-color: rgba(255, 255, 255, 0.14);
            }

            [data-testid="stSidebar"] button[kind="secondary"]:hover {
                background: rgba(255, 255, 255, 0.07);
                border-color: rgba(255, 255, 255, 0.24);
            }

            [data-testid="stSidebar"] button:disabled {
                opacity: 0.42;
            }

            .empty-state {
                background: #ffffff;
                border: 1px dashed #cbd6e5;
                border-radius: 8px;
                color: var(--muted);
                padding: 2rem;
                text-align: left;
            }

            .empty-state__title {
                color: var(--ink);
                font-size: 1.05rem;
                font-weight: 740;
                margin-bottom: 0.35rem;
            }

            .empty-state__copy {
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.55;
            }

            .review-gate {
                align-items: center;
                background: #fff8eb;
                border: 1px solid #cbd6ff;
                border-left: 5px solid var(--gold);
                border-radius: 8px;
                display: flex;
                gap: 0.85rem;
                margin: -0.25rem 0 1.5rem;
                padding: 1rem 1.1rem;
            }

            .review-gate__icon {
                align-items: center;
                background: var(--gold);
                border-radius: 50%;
                color: white;
                display: flex;
                flex: 0 0 auto;
                font-size: 0.9rem;
                font-weight: 760;
                height: 2rem;
                justify-content: center;
                width: 2rem;
            }

            .review-gate__title {
                color: var(--ink);
                font-size: 0.98rem;
                font-weight: 740;
            }

            .review-gate__copy {
                color: var(--muted);
                font-size: 0.82rem;
                margin-top: 0.14rem;
            }

            .status-pill,
            .evidence-badge,
            .source-badge {
                border-radius: 999px;
                display: inline-flex;
                font-size: 0.72rem;
                font-weight: 700;
                line-height: 1;
                margin: 0.1rem 0.22rem 0.1rem 0;
                padding: 0.35rem 0.55rem;
            }

            .status-pill--success {
                background: #e6f8f1;
                color: #087255;
            }

            .status-pill--warning {
                background: #fff2d9;
                color: #8a5a00;
            }

            .status-pill--neutral {
                background: #edf1f6;
                color: #526276;
            }

            .status-pill--brand {
                background: #e9edff;
                color: #2947b8;
            }

            .evidence-badge,
            .source-badge {
                background: #edf1ff;
                color: #3156b8;
            }

            .change-card,
            .decision-card,
            .tool-card,
            .fit-card,
            .score-card,
            .swap-card {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 8px;
                margin-bottom: 0.8rem;
                padding: 1rem;
            }

            .change-card__header,
            .tool-card__header,
            .score-card__header {
                align-items: center;
                display: flex;
                gap: 0.65rem;
                justify-content: space-between;
                margin-bottom: 0.7rem;
            }

            .change-card__title,
            .tool-card__title,
            .score-card__title,
            .fit-card__title {
                color: var(--ink);
                font-size: 0.93rem;
                font-weight: 720;
            }

            .diff-grid {
                display: grid;
                gap: 0.7rem;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                margin: 0.65rem 0;
            }

            .diff-block {
                border-radius: 0.7rem;
                min-height: 4.5rem;
                padding: 0.8rem;
            }

            .diff-block--removed {
                background: #fff1f1;
                border: 1px solid #f2caca;
            }

            .diff-block--added {
                background: #eaf8f2;
                border: 1px solid #bfe7d8;
            }

            .diff-block__label,
            .card-label {
                color: var(--muted);
                font-size: 0.68rem;
                font-weight: 760;
                letter-spacing: 0;
                margin-bottom: 0.38rem;
                text-transform: uppercase;
            }

            .diff-block__text,
            .card-copy {
                color: var(--ink);
                font-size: 0.84rem;
                line-height: 1.5;
                overflow-wrap: anywhere;
                white-space: pre-wrap;
            }

            .card-meta {
                color: var(--muted);
                font-size: 0.8rem;
                line-height: 1.5;
                margin-top: 0.55rem;
            }

            .skill-groups {
                display: grid;
                gap: 0.8rem;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                margin: 0.85rem 0 1.1rem;
            }

            .skill-group {
                border-radius: 8px;
                min-height: 8rem;
                padding: 0.85rem;
            }

            .skill-group--aligned {
                background: #eaf8f2;
                border: 1px solid #bfe7d8;
            }

            .skill-group--evidenced {
                background: #eef2ff;
                border: 1px solid #cfd8ff;
            }

            .skill-group--gap {
                background: #fff4e5;
                border: 1px solid #f2d5a8;
            }

            .skill-group__title {
                color: var(--ink);
                font-size: 0.82rem;
                font-weight: 730;
                margin-bottom: 0.55rem;
            }

            .skill-group__item {
                color: var(--ink);
                font-size: 0.8rem;
                line-height: 1.45;
                margin-bottom: 0.45rem;
            }

            .score-components {
                display: grid;
                gap: 0.55rem;
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .score-component {
                background: #f7f9fc;
                border-radius: 8px;
                padding: 0.65rem;
            }

            .score-total {
                color: var(--brand);
                font-size: 1.35rem;
                font-weight: 760;
            }

            .deterministic-note {
                align-items: center;
                color: #42536a;
                display: flex;
                font-size: 0.78rem;
                gap: 0.4rem;
                margin-bottom: 0.65rem;
            }

            .upload-validation {
                background: #f8fafc;
                border: 1px solid #e0e7f0;
                border-radius: 8px;
                font-size: 0.78rem;
                margin: -0.35rem 0 0.9rem;
                padding: 0.4rem 0.55rem;
            }

            .upload-validation--ok {
                background: #edfdf6;
                border-color: #bfe7d8;
                color: #087255;
            }

            .upload-validation--error {
                background: #fff7ed;
                border-color: #fed7aa;
                color: #a33a3a;
            }

            .preview-shell {
                background: #edf1f6;
                border: 1px solid var(--line);
                border-radius: 8px;
                overflow: hidden;
            }

            .technical-note {
                color: var(--muted);
                font-size: 0.78rem;
                line-height: 1.45;
            }

            .reset-zone {
                border-top: 1px solid rgba(255, 255, 255, 0.12);
                margin-top: 0.8rem;
                padding-top: 0.8rem;
            }

            @media (max-width: 780px) {
                [data-testid="stAppViewContainer"] > .main .block-container {
                    padding-top: 2rem;
                }

                .page-header h1 {
                    font-size: 2.15rem;
                }

                .diff-grid,
                .skill-groups,
                .score-components,
                .workflow-map {
                    grid-template-columns: 1fr;
                }

                .workflow-map__item {
                    min-height: auto;
                }

                .workflow-map__item + .workflow-map__item {
                    border-left: 0;
                    border-top: 1px solid var(--line);
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
