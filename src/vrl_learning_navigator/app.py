import base64
from collections import Counter
from html import escape
import importlib.util
import os
from pathlib import Path
import re
from typing import Iterable
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "course_details.csv"
INDEX_PATH = PROJECT_ROOT / "artifacts" / "recommendation_index.joblib"
LOGO_PATH = PROJECT_ROOT / "assets" / "vrl_logo_transparent_sharp.png"

ALL_OPTION = "All"
MAX_FEATURES = 25000
MAX_RECOMMENDATIONS = 12
SORT_SIMILARITY = "Similarity first"
SORT_HIGH_TO_LOW = "Rating high to low"
SORT_LOW_TO_HIGH = "Rating low to high"
DEFAULT_LANGUAGE = "English"
VALIDATOR_MODE_ENV = "VRL_VALIDATOR_MODE"
VALIDATOR_BACKEND_ENV = "VRL_VALIDATOR_BACKEND"
VALIDATOR_PROFILE_ENV = "VRL_AI_PROFILE"
VALIDATOR_MODEL_ENV = "VRL_VALIDATOR_MODEL"
VALIDATOR_FREE = "free"
VALIDATOR_SMART = "smart"
VALIDATOR_TFIDF = "tfidf"
VALIDATOR_MINILM = "minilm"

SMART_FILTER_STOPWORDS = {
    "a",
    "about",
    "and",
    "any",
    "are",
    "best",
    "can",
    "coursera",
    "course",
    "courses",
    "courseware",
    "find",
    "for",
    "from",
    "futurelearn",
    "give",
    "good",
    "i",
    "in",
    "learn",
    "learning",
    "me",
    "microsoft",
    "mit",
    "need",
    "on",
    "online",
    "please",
    "recommend",
    "show",
    "that",
    "the",
    "to",
    "training",
    "want",
    "with",
}

REQUIRED_COLUMNS = [
    "Course Name",
    "University",
    "Difficulty Level",
    "Rating",
    "Course URL",
    "Course Description",
    "Skills",
    "Tags",
]

CATALOG_COLUMNS = [
    *REQUIRED_COLUMNS,
    "Language",
    "Provider",
    "Category",
    "Course Key",
    "Last Verified",
]


st.set_page_config(
    page_title="VRL Learning Navigator",
    page_icon=str(LOGO_PATH),
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --vrl-bg: #05070b;
            --vrl-bg-2: #0a111a;
            --vrl-border: #263344;
            --vrl-border-strong: #35516d;
            --vrl-muted: #a3b2c6;
            --vrl-text: #f2f6fb;
            --vrl-panel: #101821;
            --vrl-panel-2: #0c131c;
            --vrl-soft: #172332;
            --vrl-blue: #2f7de1;
            --vrl-blue-bright: #5ba9ff;
            --vrl-silver: #c7d0db;
            --vrl-platinum: #e7e1d4;
            --vrl-gold: #d7b56d;
        }

        .stApp {
            background:
                linear-gradient(135deg, rgba(47, 125, 225, 0.13) 0%, rgba(47, 125, 225, 0.03) 36%, rgba(5, 7, 11, 0) 64%),
                linear-gradient(180deg, #070b11 0%, var(--vrl-bg-2) 48%, #05070b 100%);
            color: var(--vrl-text);
        }

        .stApp,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp div {
            color: var(--vrl-text);
        }

        .main .block-container {
            max-width: 1440px;
            padding-top: 1.3rem;
            padding-bottom: 2.4rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(12, 19, 28, 0.99), rgba(6, 9, 14, 0.99));
            border-right: 1px solid var(--vrl-border);
        }

        [data-testid="stSidebar"] * {
            color: var(--vrl-text);
        }

        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="popover"] {
            background: var(--vrl-panel-2);
            border-color: var(--vrl-border);
        }

        .vrl-sidebar-brand {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 0.85rem 0.4rem;
            margin: 0 0 1rem;
            border: 1px solid rgba(199, 208, 219, 0.12);
            border-radius: 8px;
            background:
                linear-gradient(180deg, rgba(242, 246, 251, 0.055), rgba(47, 125, 225, 0.045));
        }

        .vrl-sidebar-brand img {
            width: min(188px, 88%);
            height: auto;
            display: block;
        }

        [data-testid="stMetric"] {
            background: var(--vrl-panel);
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--vrl-border);
            border-radius: 8px;
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.26);
            background:
                linear-gradient(180deg, rgba(16, 24, 33, 0.97), rgba(10, 16, 24, 0.97));
        }

        .vrl-header {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(47, 125, 225, 0.18), rgba(16, 24, 33, 0.92) 48%, rgba(8, 12, 18, 0.96));
            padding: 1.05rem 1.25rem 1.05rem 1.45rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.26);
        }

        .vrl-header::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: linear-gradient(180deg, var(--vrl-blue-bright), var(--vrl-gold));
        }

        .vrl-header-content {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.25rem;
        }

        .vrl-header-copy {
            min-width: 0;
        }

        .vrl-eyebrow {
            color: var(--vrl-gold);
            font-size: 0.78rem;
            font-weight: 760;
            margin: 0 0 0.28rem;
        }

        .vrl-title {
            font-size: 1.58rem;
            font-weight: 760;
            line-height: 1.14;
            letter-spacing: 0;
            margin: 0;
            color: var(--vrl-text);
        }

        .vrl-subtitle {
            color: var(--vrl-muted);
            font-size: 0.95rem;
            line-height: 1.42;
            margin: 0.32rem 0 0;
            max-width: 42rem;
        }

        .vrl-header-pills {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.45rem;
            min-width: 15rem;
        }

        .vrl-header-pill {
            border: 1px solid rgba(91, 169, 255, 0.3);
            border-radius: 6px;
            background: rgba(5, 7, 11, 0.34);
            color: #d8e8fb;
            font-size: 0.78rem;
            font-weight: 680;
            padding: 0.3rem 0.52rem;
            white-space: nowrap;
        }

        @media (max-width: 760px) {
            .vrl-header-content {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.85rem;
            }

            .vrl-header-pills {
                justify-content: flex-start;
                min-width: 0;
            }
        }

        .vrl-section-title {
            font-size: 1.05rem;
            font-weight: 720;
            margin: 0.25rem 0 0.6rem;
            color: var(--vrl-text);
        }

        .vrl-section-copy {
            color: var(--vrl-muted);
            font-size: 0.9rem;
            line-height: 1.45;
            margin: -0.2rem 0 0.8rem;
            max-width: 48rem;
        }

        .vrl-mode-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.25rem 0 1rem;
        }

        .vrl-mode-card {
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            background:
                linear-gradient(180deg, rgba(16, 24, 33, 0.94), rgba(10, 16, 24, 0.94));
            padding: 0.82rem 0.9rem;
            min-height: 6.2rem;
        }

        .vrl-mode-kicker {
            color: var(--vrl-gold);
            font-size: 0.75rem;
            font-weight: 760;
            margin-bottom: 0.28rem;
        }

        .vrl-mode-title {
            color: var(--vrl-text);
            font-size: 0.96rem;
            font-weight: 760;
            line-height: 1.25;
            margin-bottom: 0.28rem;
        }

        .vrl-mode-copy {
            color: var(--vrl-muted);
            font-size: 0.83rem;
            line-height: 1.38;
        }

        .vrl-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.6rem;
            margin-bottom: 0.35rem;
        }

        .vrl-rank {
            color: var(--vrl-gold);
            font-weight: 760;
            font-size: 0.9rem;
        }

        .vrl-badge {
            border: 1px solid var(--vrl-border);
            border-radius: 6px;
            color: var(--vrl-platinum);
            background: var(--vrl-soft);
            font-size: 0.78rem;
            font-weight: 650;
            padding: 0.18rem 0.45rem;
            white-space: nowrap;
        }

        .vrl-course-title {
            font-size: 1.02rem;
            font-weight: 730;
            line-height: 1.32;
            letter-spacing: 0;
            margin: 0 0 0.35rem;
            overflow-wrap: anywhere;
            color: var(--vrl-text);
        }

        .vrl-meta {
            color: var(--vrl-muted);
            font-size: 0.86rem;
            line-height: 1.35;
            margin-bottom: 0.62rem;
        }

        .vrl-description {
            color: #cad4df;
            font-size: 0.9rem;
            line-height: 1.45;
            min-height: 3.9rem;
            margin: 0.65rem 0 0.75rem;
        }

        .vrl-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin: 0.25rem 0 0.8rem;
        }

        .vrl-chip {
            border: 1px solid rgba(91, 169, 255, 0.34);
            border-radius: 6px;
            color: #cfe7ff;
            background: rgba(47, 125, 225, 0.15);
            font-size: 0.78rem;
            font-weight: 620;
            padding: 0.18rem 0.45rem;
            max-width: 100%;
        }

        .vrl-empty {
            border: 1px dashed var(--vrl-border);
            border-radius: 8px;
            background: rgba(16, 24, 39, 0.72);
            color: var(--vrl-muted);
            padding: 1.15rem;
        }

        .vrl-empty strong {
            color: var(--vrl-text);
        }

        .vrl-small-note {
            color: var(--vrl-muted);
            font-size: 0.86rem;
            margin-top: -0.15rem;
        }

        .vrl-chat-bubble {
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            background: rgba(16, 24, 33, 0.82);
            color: var(--vrl-text);
            font-size: 0.86rem;
            line-height: 1.4;
            margin: 0.35rem 0;
            padding: 0.62rem 0.72rem;
        }

        .vrl-chat-user {
            border-color: rgba(91, 169, 255, 0.38);
            background: rgba(47, 125, 225, 0.13);
        }

        .vrl-chat-assistant {
            border-color: rgba(215, 181, 109, 0.32);
        }

        .vrl-advisor-head {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            border: 1px solid rgba(91, 169, 255, 0.28);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(47, 125, 225, 0.16), rgba(16, 24, 33, 0.92));
            padding: 0.82rem 0.95rem;
            margin-bottom: 0.72rem;
        }

        .vrl-bot-avatar {
            display: grid;
            place-items: center;
            width: 2.45rem;
            height: 2.45rem;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--vrl-blue), var(--vrl-gold));
            color: #05070b;
            font-weight: 800;
            box-shadow: 0 10px 22px rgba(47, 125, 225, 0.24);
            flex: 0 0 auto;
        }

        .vrl-advisor-name {
            margin: 0;
            font-weight: 760;
            font-size: 1.05rem;
        }

        .vrl-advisor-status {
            color: var(--vrl-muted);
            font-size: 0.84rem;
            margin: 0.15rem 0 0;
        }

        .vrl-similarity-summary {
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) repeat(3, minmax(0, 0.75fr));
            gap: 0.65rem;
            margin: 0.8rem 0 0.2rem;
        }

        .vrl-similarity-stat {
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            background: rgba(16, 24, 33, 0.72);
            padding: 0.68rem 0.72rem;
            min-width: 0;
        }

        .vrl-similarity-label {
            color: var(--vrl-muted);
            font-size: 0.75rem;
            font-weight: 680;
            margin-bottom: 0.22rem;
        }

        .vrl-similarity-value {
            color: var(--vrl-text);
            font-size: 0.92rem;
            font-weight: 740;
            line-height: 1.28;
            overflow-wrap: anywhere;
        }

        .vrl-similarity-value-muted {
            color: var(--vrl-muted);
            font-weight: 650;
        }

        .vrl-roadmap {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.75rem 0 1rem;
        }

        .vrl-roadmap-step {
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            background: rgba(16, 24, 33, 0.82);
            padding: 0.78rem;
            min-height: 8rem;
        }

        .vrl-roadmap-label {
            color: var(--vrl-gold);
            font-size: 0.78rem;
            font-weight: 760;
            margin-bottom: 0.28rem;
        }

        .vrl-roadmap-title {
            font-size: 0.94rem;
            font-weight: 730;
            line-height: 1.28;
            margin-bottom: 0.35rem;
        }

        .vrl-match-reason {
            border-left: 3px solid var(--vrl-gold);
            background: rgba(215, 181, 109, 0.1);
            color: #efe5ce;
            border-radius: 6px;
            font-size: 0.84rem;
            line-height: 1.4;
            margin: 0.2rem 0 0.72rem;
            padding: 0.5rem 0.62rem;
        }

        .vrl-practice-brief {
            border: 1px solid rgba(91, 169, 255, 0.24);
            border-radius: 8px;
            background: rgba(47, 125, 225, 0.08);
            color: #d9e8f8;
            line-height: 1.45;
            margin: 0.4rem 0 0.85rem;
            padding: 0.74rem 0.82rem;
        }

        .vrl-check-list {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.45rem;
            margin: 0.35rem 0 0.8rem;
        }

        .vrl-check-item {
            border: 1px solid var(--vrl-border);
            border-radius: 6px;
            background: rgba(16, 24, 33, 0.72);
            color: #dbe6f2;
            font-size: 0.84rem;
            font-weight: 620;
            line-height: 1.35;
            padding: 0.45rem 0.56rem;
        }

        .vrl-progress-summary {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.65rem 0 0.2rem;
        }

        .vrl-progress-stat {
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            background: rgba(16, 24, 33, 0.72);
            padding: 0.72rem;
        }

        .vrl-progress-label {
            color: var(--vrl-muted);
            font-size: 0.78rem;
            font-weight: 680;
            margin-bottom: 0.22rem;
        }

        .vrl-progress-value {
            color: var(--vrl-text);
            font-size: 0.98rem;
            font-weight: 760;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }

        @media (max-width: 920px) {
            .vrl-mode-grid,
            .vrl-roadmap {
                grid-template-columns: 1fr;
            }

            .vrl-check-list {
                grid-template-columns: 1fr;
            }

            .vrl-similarity-summary,
            .vrl-progress-summary {
                grid-template-columns: 1fr;
            }
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            background: rgba(16, 24, 33, 0.78);
            padding: 0.45rem 0.65rem;
            margin-bottom: 0.45rem;
        }

        div[data-testid="stChatInput"] {
            border-radius: 8px;
        }

        div[data-testid="stChatInput"] textarea {
            background: var(--vrl-panel);
            border-color: var(--vrl-border);
            color: var(--vrl-text);
        }

        h1, h2, h3, h4 {
            letter-spacing: 0;
            color: var(--vrl-text);
        }

        div[data-testid="stTabs"] button p {
            color: var(--vrl-muted);
            font-weight: 650;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] p {
            color: var(--vrl-blue-bright);
        }

        div[data-testid="stExpander"] {
            background: var(--vrl-panel);
            border-color: var(--vrl-border);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--vrl-border);
            border-radius: 8px;
            overflow: hidden;
        }

        div[data-testid="stAlert"] {
            background: rgba(251, 191, 36, 0.12);
            border-color: rgba(251, 191, 36, 0.28);
            color: #fde68a;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] {
            background: var(--vrl-panel);
            border-color: var(--vrl-border);
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="input"] input {
            color: var(--vrl-text);
        }

        div[data-baseweb="popover"] {
            background: var(--vrl-panel);
        }

        div[data-testid="stProgress"] > div > div {
            background: rgba(91, 169, 255, 0.18);
        }

        div.stButton > button,
        div[data-testid="stLinkButton"] > a {
            border-radius: 6px;
            font-weight: 650;
            border-color: var(--vrl-border);
            min-height: 2.9rem;
            white-space: nowrap;
        }

        div.stButton > button[kind="primary"],
        div[data-testid="stLinkButton"] > a[kind="primary"] {
            background: linear-gradient(135deg, #1f5fb8, #2f7de1);
            border: 1px solid rgba(91, 169, 255, 0.5);
            color: #ffffff;
        }

        div.stButton > button[kind="primary"] p,
        div[data-testid="stLinkButton"] > a[kind="primary"] p {
            color: #ffffff;
        }

        button[data-testid="stBaseButton-primary"],
        button[data-testid="stBaseButton-primary"] p,
        a[data-testid="stBaseLinkButton-primary"],
        a[data-testid="stBaseLinkButton-primary"] p {
            color: #ffffff !important;
        }

        div.stButton > button:hover,
        div[data-testid="stLinkButton"] > a:hover {
            border-color: var(--vrl-blue-bright);
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_skills(skills: str) -> tuple[str, ...]:
    return tuple(
        token.strip().lower()
        for token in str(skills).split()
        if token.strip()
    )


def format_skill_label(skill: str) -> str:
    return skill.replace("-", " ").replace("_", " ").title()


def normalize_phrase(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_in_text(phrase: str, text: str) -> bool:
    if not phrase:
        return False
    return f" {phrase} " in f" {text} "


def truncate_text(text: str, max_chars: int = 230) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized

    truncated = normalized[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."


def normalize_course_key(value: object) -> str:
    normalized = " ".join(str(value or "").lower().split())
    return "".join(char if char.isalnum() else "-" for char in normalized).strip("-")


def course_key_from_row(row: pd.Series) -> str:
    provider = normalize_course_key(row.get("Provider", "Coursera")) or "coursera"
    source = row.get("Course URL") or row.get("Course Name")
    return f"{provider}:{normalize_course_key(source)}"


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def logo_data_uri() -> str | None:
    if not LOGO_PATH.exists():
        return None

    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_data(show_spinner="Loading course catalog...")
def load_courses(data_mtime: float) -> pd.DataFrame:
    courses = pd.read_csv(DATA_PATH, index_col=0)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in courses.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"data/course_details.csv is missing required columns: {missing}")

    for column in CATALOG_COLUMNS:
        if column not in courses.columns:
            if column == "Provider":
                courses[column] = "Coursera"
            elif column == "Category":
                courses[column] = "Coursera"
            elif column == "Language":
                courses[column] = DEFAULT_LANGUAGE
            else:
                courses[column] = ""

    courses = courses[CATALOG_COLUMNS].copy()
    courses["Rating"] = pd.to_numeric(courses["Rating"], errors="coerce")
    courses["Rating"] = courses["Rating"].fillna(0.0)
    courses = courses.dropna(subset=["Course Name", "Tags", "Course URL"])

    text_columns = [
        "Course Name",
        "University",
        "Difficulty Level",
        "Course URL",
        "Course Description",
        "Skills",
        "Tags",
        "Language",
        "Provider",
        "Category",
        "Course Key",
        "Last Verified",
    ]
    for column in text_columns:
        courses[column] = courses[column].fillna("").astype(str).str.strip()

    courses["Provider"] = courses["Provider"].replace("", "Coursera")
    courses["Category"] = courses["Category"].replace("", "Coursera")
    courses["Language"] = courses["Language"].replace("", DEFAULT_LANGUAGE)
    courses = courses.reset_index(drop=True)
    empty_keys = courses["Course Key"] == ""
    courses.loc[empty_keys, "Course Key"] = courses[empty_keys].apply(course_key_from_row, axis=1)
    courses["Skill Tokens"] = courses["Skills"].apply(parse_skills)
    courses["Search Text"] = (
        courses["Course Name"]
        + " "
        + courses["University"]
        + " "
        + courses["Provider"]
        + " "
        + courses["Language"]
        + " "
        + courses["Category"]
        + " "
        + courses["Difficulty Level"]
        + " "
        + courses["Course Description"]
        + " "
        + courses["Skills"]
    ).str.lower()

    return courses


@st.cache_resource(show_spinner="Loading recommendation index...")
def load_recommendation_resources(
    course_keys: tuple[str, ...],
    search_texts: tuple[str, ...],
    index_mtime: float,
):
    key_list = list(course_keys)
    if INDEX_PATH.exists():
        index_data = load(INDEX_PATH)
        if index_data.get("course_keys") == key_list:
            index_data["mode"] = "precomputed"
            index_data["key_to_position"] = {key: index for index, key in enumerate(key_list)}
            return index_data

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
    )
    vectors = vectorizer.fit_transform(search_texts)
    return {
        "mode": "runtime",
        "course_keys": key_list,
        "key_to_position": {key: index for index, key in enumerate(key_list)},
        "vectorizer": vectorizer,
        "vectors": vectors,
    }


@st.cache_data(show_spinner=False)
def top_skill_options(skill_rows: tuple[tuple[str, ...], ...], limit: int = 18) -> list[str]:
    counter: Counter[str] = Counter()
    for skills in skill_rows:
        counter.update(
            skill
            for skill in skills
            if skill not in SMART_FILTER_STOPWORDS and (len(skill) > 2 or skill in {"ai", "ml", "ui", "ux"})
        )
    return [skill for skill, _ in counter.most_common(limit)]


def ensure_session_state() -> None:
    st.session_state.setdefault("shortlist", [])
    st.session_state.setdefault("completed_courses", [])
    st.session_state.setdefault("practice_anchor_course_key", "")
    st.session_state.setdefault("recommendations", None)
    st.session_state.setdefault("recommendation_context", None)
    st.session_state.setdefault("advisor_recommendations", None)
    st.session_state.setdefault("advisor_context", None)
    st.session_state.setdefault("advisor_query_text", "")
    st.session_state.setdefault("advisor_user_goal", "")
    st.session_state.setdefault("pending_toast", None)
    st.session_state.setdefault(
        "advisor_messages",
        [
            {
                "role": "assistant",
                "content": "Tell me your goal. I will build a learning bundle from the catalog.",
            }
        ],
    )


def show_pending_toast() -> None:
    message = st.session_state.pop("pending_toast", None)
    if message:
        st.toast(message)


def reset_filters() -> None:
    for key in (
        "catalog_search",
        "course_provider",
        "course_language",
        "difficulty_level",
        "university",
        "rating_sort",
        "selected_skills",
        "advisor_prompt",
        "course_name",
        "recommendations",
        "recommendation_context",
        "advisor_recommendations",
        "advisor_context",
        "advisor_query_text",
        "advisor_user_goal",
    ):
        st.session_state.pop(key, None)


def toggle_shortlist(course_name: str) -> None:
    shortlist = list(st.session_state.get("shortlist", []))
    if course_name in shortlist:
        shortlist.remove(course_name)
        st.session_state["pending_toast"] = "Removed from shortlist"
    else:
        shortlist.append(course_name)
        st.session_state["pending_toast"] = "Saved to shortlist"
    st.session_state["shortlist"] = shortlist


def is_course_completed(course_key: object) -> bool:
    return str(course_key) in set(st.session_state.get("completed_courses", []))


def mark_course_completed(course_key: object, course_name: object) -> bool:
    key = str(course_key)
    completed = list(st.session_state.get("completed_courses", []))
    already_completed = key in completed
    if not already_completed:
        completed.append(key)
        st.session_state["completed_courses"] = completed

    st.session_state["practice_anchor_course_key"] = key
    st.session_state["pending_toast"] = f"Progress updated: {course_name}"
    return not already_completed


def toggle_completed_course(course_key: object, course_name: object) -> None:
    key = str(course_key)
    completed = list(st.session_state.get("completed_courses", []))
    if key in completed:
        completed.remove(key)
        if st.session_state.get("practice_anchor_course_key") == key:
            st.session_state["practice_anchor_course_key"] = completed[-1] if completed else ""
        st.session_state["pending_toast"] = f"Removed from progress: {course_name}"
    else:
        completed.append(key)
        st.session_state["practice_anchor_course_key"] = key
        st.session_state["pending_toast"] = f"Marked completed: {course_name}"

    st.session_state["completed_courses"] = completed


def completed_course_frame(courses: pd.DataFrame) -> pd.DataFrame:
    completed_keys = list(dict.fromkeys(str(key) for key in st.session_state.get("completed_courses", [])))
    if not completed_keys:
        return courses.iloc[0:0].copy()

    completed = courses[courses["Course Key"].astype(str).isin(completed_keys)].copy()
    order = {key: index for index, key in enumerate(completed_keys)}
    completed["_progress_order"] = completed["Course Key"].astype(str).map(order)
    return completed.sort_values("_progress_order").drop(columns=["_progress_order"])


def latest_completed_course(courses: pd.DataFrame) -> pd.Series | None:
    completed = completed_course_frame(courses)
    if completed.empty:
        return None
    return completed.iloc[-1]


def active_practice_course(courses: pd.DataFrame, fallback_courses: pd.DataFrame | None = None) -> pd.Series | None:
    anchor_key = str(st.session_state.get("practice_anchor_course_key", ""))
    if anchor_key:
        anchor_rows = courses[courses["Course Key"].astype(str) == anchor_key]
        if not anchor_rows.empty:
            return anchor_rows.iloc[0]

    latest = latest_completed_course(courses)
    if latest is not None:
        return latest

    if fallback_courses is not None and not fallback_courses.empty:
        return fallback_courses.iloc[0]
    return None


def options_from(series: pd.Series) -> list[str]:
    values = sorted(value for value in series.dropna().unique() if str(value).strip())
    return [ALL_OPTION, *values]


def difficulty_options(courses: pd.DataFrame) -> list[str]:
    preferred = ["Beginner", "Intermediate", "Advanced", "Mixed_Difficulty"]
    available = set(courses["Difficulty Level"].dropna().unique())
    ordered = [level for level in preferred if level in available]
    ordered.extend(sorted(available.difference(ordered)))
    return [ALL_OPTION, *ordered]


def smart_filter_provider(request_text: str, providers: list[str]) -> str:
    aliases = {
        "Coursera": ("coursera",),
        "FutureLearn": ("future learn", "futurelearn"),
        "Kaggle Learn": ("kaggle", "kaggle learn"),
        "MIT OpenCourseWare": ("mit", "ocw", "open courseware", "opencourseware"),
        "Microsoft Learn": ("microsoft", "microsoft learn", "ms learn", "azure"),
    }
    available = set(providers)
    for provider, provider_aliases in aliases.items():
        if provider in available and any(phrase_in_text(alias, request_text) for alias in provider_aliases):
            return provider

    matches = [
        provider
        for provider in providers
        if provider != ALL_OPTION and phrase_in_text(normalize_phrase(provider), request_text)
    ]
    return max(matches, key=len) if matches else ALL_OPTION


def smart_filter_difficulty(request_text: str, difficulties: list[str]) -> str:
    difficulty_aliases = (
        ("Beginner", ("beginner", "basic", "foundation", "foundational", "intro", "introductory", "starter")),
        ("Intermediate", ("intermediate", "mid level", "practical")),
        ("Advanced", ("advanced", "expert", "deep", "senior")),
        ("Mixed_Difficulty", ("mixed difficulty", "mixed")),
    )
    available = set(difficulties)
    for difficulty, aliases in difficulty_aliases:
        if difficulty in available and any(phrase_in_text(alias, request_text) for alias in aliases):
            return difficulty
    return ALL_OPTION


def smart_filter_language(request_text: str, languages: list[str]) -> str:
    language_aliases = {
        "English": ("english",),
        "Spanish": ("spanish", "espanol", "español"),
        "French": ("french", "francais", "français"),
        "German": ("german", "deutsch"),
        "Hindi": ("hindi",),
        "Arabic": ("arabic",),
        "Chinese": ("chinese", "mandarin"),
        "Japanese": ("japanese",),
        "Korean": ("korean",),
        "Portuguese": ("portuguese",),
        "Russian": ("russian",),
        "Italian": ("italian",),
    }
    available = set(languages)
    for language, aliases in language_aliases.items():
        if language in available and any(phrase_in_text(alias, request_text) for alias in aliases):
            return language

    matches = [
        language
        for language in languages
        if language != ALL_OPTION and phrase_in_text(normalize_phrase(language), request_text)
    ]
    return max(matches, key=len) if matches else ALL_OPTION


def smart_filter_university(request_text: str, courses: pd.DataFrame) -> str:
    matches: list[str] = []
    for university in courses["University"].dropna().unique():
        normalized = normalize_phrase(university)
        if len(normalized) >= 4 and phrase_in_text(normalized, request_text):
            matches.append(str(university))
    return max(matches, key=len) if matches else ALL_OPTION


def smart_filter_skills(request_text: str, courses: pd.DataFrame, limit: int = 8) -> list[str]:
    skill_counter: Counter[str] = Counter()
    for skills in courses["Skill Tokens"]:
        skill_counter.update(skills)

    matches: list[str] = []
    for skill, _ in skill_counter.most_common(240):
        normalized = normalize_phrase(format_skill_label(skill))
        if normalized in SMART_FILTER_STOPWORDS:
            continue
        if len(normalized) <= 2 and normalized not in {"ai", "ml", "ui", "ux"}:
            continue
        if phrase_in_text(normalized, request_text):
            matches.append(skill)
        if len(matches) >= limit:
            break
    return matches


def smart_filter_search_terms(
    request: str,
    provider: str,
    language: str,
    difficulty_level: str,
    university: str,
    selected_skills: Iterable[str],
) -> str:
    tokens = re.findall(r"[a-z0-9+#.]+", request.lower())
    blocked = set(SMART_FILTER_STOPWORDS)
    for value in (provider, language, difficulty_level, university, *selected_skills):
        if value != ALL_OPTION:
            blocked.update(normalize_phrase(value).split())

    retained = [
        token
        for token in tokens
        if token not in blocked and len(token) > 1
    ]
    return " ".join(dict.fromkeys(retained))


def interpret_smart_filter(courses: pd.DataFrame, request: str) -> dict[str, object]:
    request_text = normalize_phrase(request)
    providers = options_from(courses["Provider"])
    languages = options_from(courses["Language"])
    difficulties = difficulty_options(courses)
    provider = smart_filter_provider(request_text, providers)
    language = smart_filter_language(request_text, languages)
    difficulty_level = smart_filter_difficulty(request_text, difficulties)

    scoped = apply_filters(
        courses,
        "",
        provider,
        language,
        difficulty_level,
        ALL_OPTION,
        [],
    )
    university = smart_filter_university(request_text, scoped if not scoped.empty else courses)
    selected_skills = smart_filter_skills(request_text, scoped if not scoped.empty else courses)
    search_query = smart_filter_search_terms(
        request,
        provider,
        language,
        difficulty_level,
        university,
        selected_skills,
    )

    return {
        "search_query": search_query,
        "provider": provider,
        "language": language,
        "difficulty_level": difficulty_level,
        "university": university,
        "selected_skills": selected_skills,
    }


def apply_smart_filter(courses: pd.DataFrame, request: str) -> str:
    lowered = normalize_phrase(request)
    if any(phrase_in_text(word, lowered) for word in ("clear", "reset", "start over")):
        for key in (
            "catalog_search",
            "course_provider",
            "course_language",
            "difficulty_level",
            "university",
            "rating_sort",
            "selected_skills",
            "course_name",
            "recommendations",
            "recommendation_context",
        ):
            st.session_state.pop(key, None)
        return "Cleared the active filters."

    result = interpret_smart_filter(courses, request)
    st.session_state["catalog_search"] = result["search_query"]
    st.session_state["course_provider"] = result["provider"]
    st.session_state["course_language"] = result["language"]
    st.session_state["difficulty_level"] = result["difficulty_level"]
    st.session_state["university"] = result["university"]
    st.session_state["selected_skills"] = list(result["selected_skills"])
    st.session_state.pop("recommendations", None)
    st.session_state.pop("recommendation_context", None)

    summary_parts: list[str] = []
    if result["provider"] != ALL_OPTION:
        summary_parts.append(f"provider {result['provider']}")
    if result["language"] != ALL_OPTION:
        summary_parts.append(f"language {result['language']}")
    if result["difficulty_level"] != ALL_OPTION:
        summary_parts.append(f"difficulty {result['difficulty_level']}")
    if result["university"] != ALL_OPTION:
        summary_parts.append(f"organization {result['university']}")
    if result["selected_skills"]:
        labels = ", ".join(format_skill_label(skill) for skill in result["selected_skills"])
        summary_parts.append(f"skills {labels}")
    if result["search_query"]:
        summary_parts.append(f"search {result['search_query']}")

    if not summary_parts:
        st.session_state["catalog_search"] = request.strip()
        return "I used your full request as the catalog search."
    return "Applied " + "; ".join(summary_parts) + "."


def expand_goal_text(goal: str) -> str:
    normalized = normalize_phrase(goal)
    expansions: list[str] = []
    if any(phrase_in_text(term, normalized) for term in ("data analyst", "analytics", "business analyst")):
        expansions.append("data analysis analytics sql python statistics visualization dashboard business intelligence")
    if any(phrase_in_text(term, normalized) for term in ("software engineer", "developer", "programmer")):
        expansions.append("software engineering programming algorithms data structures python java git github")
    if any(phrase_in_text(term, normalized) for term in ("cloud", "azure", "devops", "kubernetes")):
        expansions.append("cloud azure devops kubernetes infrastructure deployment containers security")
    if any(phrase_in_text(term, normalized) for term in ("web developer", "frontend", "front end", "backend", "back end", "full stack", "fullstack")):
        expansions.append("web development frontend backend full stack javascript html css react api application development")
    if any(phrase_in_text(term, normalized) for term in ("cyber", "security", "secure")):
        expansions.append("cybersecurity security network risk identity protection threat")
    if any(phrase_in_text(term, normalized) for term in ("ai", "ml", "machine learning", "deep learning", "generative", "gen ai", "genai")):
        expansions.append("artificial intelligence generative ai genai large language models llm prompt engineering azure openai machine learning deep learning neural networks python")
    if any(phrase_in_text(term, normalized) for term in ("finance", "financial", "banking")):
        expansions.append("finance financial risk accounting investment fintech")
    if any(phrase_in_text(term, normalized) for term in ("healthcare", "health", "medical")):
        expansions.append("healthcare health medical clinical public health")
    if any(phrase_in_text(term, normalized) for term in ("project manager", "product manager", "management", "leadership")):
        expansions.append("project management product management leadership agile scrum strategy communication")
    if any(phrase_in_text(term, normalized) for term in ("database", "sql", "data engineer", "data engineering")):
        expansions.append("database sql data engineering pipelines data warehouse postgresql mysql big data")
    if any(phrase_in_text(term, normalized) for term in ("career switch", "switch career", "job ready", "job-ready")):
        expansions.append("beginner professional certificate career skills hands on project portfolio")
    if any(phrase_in_text(term, normalized) for term in ("beginner", "new", "start", "foundation")):
        expansions.append("beginner introductory foundations fundamentals")
    if any(phrase_in_text(term, normalized) for term in ("advanced", "expert", "senior")):
        expansions.append("advanced expert architecture optimization")
    return " ".join([goal, *expansions]).strip()


def advisor_query_text(messages: list[dict[str, str]], request: str) -> str:
    previous_user_context = [
        str(message.get("content", ""))
        for message in messages[-6:]
        if message.get("role") == "user"
    ]
    return expand_goal_text(" ".join([*previous_user_context, request]))


def meaningful_tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9+#.]+", normalize_phrase(value))
        if token not in SMART_FILTER_STOPWORDS and len(token) > 2
    }


def request_mentions_completion(request: str) -> bool:
    normalized = normalize_phrase(request)
    return any(
        phrase_in_text(term, normalized)
        for term in (
            "completed",
            "complete",
            "finished",
            "finish",
            "done with",
            "i did",
            "cleared",
            "passed",
        )
    )


def match_course_from_text(request: str, courses: pd.DataFrame) -> pd.Series | None:
    normalized_request = normalize_phrase(request)
    request_tokens = meaningful_tokens(request)
    best_score = 0.0
    best_token_count = 0
    best_row: pd.Series | None = None

    for _, course in courses.iterrows():
        course_name = str(course.get("Course Name", ""))
        normalized_name = normalize_phrase(course_name)
        if not normalized_name:
            continue

        name_tokens = meaningful_tokens(course_name)
        if not name_tokens:
            continue

        score = 0.0
        if phrase_in_text(normalized_name, normalized_request):
            score = 1.0
        elif phrase_in_text(normalized_request, normalized_name) and len(request_tokens) >= 2:
            score = 0.92
        else:
            overlap = request_tokens.intersection(name_tokens)
            precision = len(overlap) / max(len(name_tokens), 1)
            recall = len(overlap) / max(len(request_tokens), 1)
            score = (precision * 0.55) + (recall * 0.45)
            if len(overlap) >= 2:
                score += 0.12
            if len(name_tokens) < 2:
                score *= 0.55

        if score > best_score or (score == best_score and len(name_tokens) > best_token_count):
            best_score = score
            best_token_count = len(name_tokens)
            best_row = course

    return best_row if best_score >= 0.56 else None


def progress_note_from_request(request: str, courses: pd.DataFrame) -> str:
    if not request_mentions_completion(request):
        return ""

    matched_course = match_course_from_text(request, courses)
    if matched_course is None:
        return "I noticed you mentioned a completed course, but I could not confidently match it in the catalog. Use Progress tracker to select it exactly."

    course_name = str(matched_course.get("Course Name", ""))
    difficulty = str(matched_course.get("Difficulty Level", "Mixed"))
    mark_course_completed(matched_course.get("Course Key", ""), course_name)
    return f"Progress updated: I marked {course_name} as completed and set practice to {difficulty} level."


def enrich_query_with_progress(query_text: str, courses: pd.DataFrame) -> str:
    anchor = active_practice_course(courses)
    if anchor is None:
        return query_text

    progress_context = " ".join(
        str(anchor.get(column, ""))
        for column in ("Course Name", "Difficulty Level", "Skills", "Category", "Provider")
    )
    return f"{query_text} completed course context {progress_context}".strip()


def goal_terms(goal_text: str) -> list[str]:
    normalized = normalize_phrase(goal_text)
    terms = [
        token
        for token in re.findall(r"[a-z0-9+#.]+", normalized)
        if token not in SMART_FILTER_STOPWORDS and len(token) > 2
    ]
    phrase_terms = [
        "generative ai",
        "large language",
        "prompt engineering",
        "machine learning",
        "deep learning",
        "data analyst",
        "data analytics",
        "cybersecurity",
        "cloud",
        "devops",
        "azure",
        "sql",
        "python",
    ]
    for phrase in phrase_terms:
        if phrase_in_text(phrase, normalized):
            terms.insert(0, phrase)
    unique_terms = list(dict.fromkeys(terms))
    return [
        term
        for term in unique_terms
        if not any(term != other and phrase_in_text(term, other) for other in unique_terms)
    ]


def roadmap_stage(difficulty: object) -> str:
    normalized = str(difficulty or "").lower()
    if "beginner" in normalized:
        return "Foundation"
    if "advanced" in normalized:
        return "Specialize"
    return "Build"


def explain_course_match(course: pd.Series, query_text: str) -> str:
    course_text = normalize_phrase(
        " ".join(
            str(course.get(column, ""))
            for column in ("Course Name", "Course Description", "Skills", "Category", "Provider")
        )
    )
    matched_terms = [
        term
        for term in goal_terms(query_text)
        if phrase_in_text(normalize_phrase(term), course_text)
    ][:3]
    matched_skills = list(dict.fromkeys(
        format_skill_label(skill)
        for skill in course.get("Skill Tokens", ())
        if normalize_phrase(format_skill_label(skill)) in {normalize_phrase(term) for term in matched_terms}
    ))[:2]

    reason_bits: list[str] = []
    if matched_terms:
        reason_bits.append("matches " + ", ".join(term.title() for term in matched_terms))
    if matched_skills:
        reason_bits.append("skill signals " + ", ".join(matched_skills))
    reason_bits.append(f"{course.get('Difficulty Level', 'Mixed')} level")
    reason_bits.append(str(course.get("Provider", "Catalog")))
    return "; ".join(reason_bits) + "."


def add_advisor_context(recommendations: pd.DataFrame, query_text: str) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    enriched = recommendations.copy()
    enriched["Roadmap Stage"] = enriched["Difficulty Level"].apply(roadmap_stage)
    enriched["Match Reason"] = enriched.apply(
        lambda row: explain_course_match(row, query_text),
        axis=1,
    )
    return enriched


def recommend_for_goal(
    goal_text: str,
    candidate_courses: pd.DataFrame,
    recommendation_resources: dict,
    limit: int = MAX_RECOMMENDATIONS,
) -> pd.DataFrame:
    if candidate_courses.empty or not goal_text.strip():
        return pd.DataFrame()

    key_to_position = recommendation_resources["key_to_position"]
    candidate_pairs = [
        (key, key_to_position[key])
        for key in candidate_courses["Course Key"].astype(str)
        if key in key_to_position
    ]
    if not candidate_pairs:
        return pd.DataFrame()

    candidate_keys = [key for key, _ in candidate_pairs]
    candidate_positions = [position for _, position in candidate_pairs]
    query_vector = recommendation_resources["vectorizer"].transform([goal_text])
    similarities = cosine_similarity(
        query_vector,
        recommendation_resources["vectors"][candidate_positions],
    ).ravel()

    normalized_goal = normalize_phrase(goal_text)
    genai_intent = any(
        phrase_in_text(term, normalized_goal)
        for term in ("gen ai", "genai", "generative ai", "llm", "large language", "prompt engineering", "openai")
    )
    genai_terms = (
        "generative ai",
        "generative artificial intelligence",
        "large language",
        "llm",
        "prompt",
        "openai",
        "copilot",
        "foundation model",
        "transformer",
    )
    ai_gate_terms = (
        "artificial intelligence",
        "generative",
        "machine learning",
        "deep learning",
        "neural",
        "language model",
        "prompt",
        "openai",
        "copilot",
        "transformer",
    )
    candidate_meta = candidate_courses.set_index("Course Key")[
        ["Search Text", "Difficulty Level", "Provider"]
    ].to_dict("index")
    text_by_key = {
        key: str(meta.get("Search Text", ""))
        for key, meta in candidate_meta.items()
    }
    similarity_by_key = {
        key: min(float(score), 1.0)
        for key, score in zip(candidate_keys, similarities)
        if float(score) > 0
    }

    difficulty_intent = smart_filter_difficulty(
        normalized_goal,
        difficulty_options(candidate_courses),
    )
    provider_intent = smart_filter_provider(
        normalized_goal,
        options_from(candidate_courses["Provider"]),
    )
    if difficulty_intent != ALL_OPTION or provider_intent != ALL_OPTION:
        adjusted_scores: dict[str, float] = {}
        for key, score in similarity_by_key.items():
            meta = candidate_meta.get(key, {})
            if difficulty_intent != ALL_OPTION:
                difficulty = str(meta.get("Difficulty Level", ""))
                if difficulty == difficulty_intent:
                    score += 0.12
                elif difficulty == "Mixed_Difficulty":
                    score += 0.03
                else:
                    score *= 0.82
            if provider_intent != ALL_OPTION:
                if str(meta.get("Provider", "")) == provider_intent:
                    score += 0.09
                else:
                    score *= 0.9
            adjusted_scores[key] = min(score, 1.0)
        similarity_by_key = adjusted_scores

    if genai_intent:
        boosted_scores: dict[str, float] = {}
        for key, score in similarity_by_key.items():
            course_text = str(text_by_key.get(key, "")).lower()
            if not any(term in course_text for term in ai_gate_terms):
                continue
            if any(term in course_text for term in genai_terms):
                score += 0.18
            else:
                score += 0.035
            boosted_scores[key] = min(score, 1.0)
        similarity_by_key = boosted_scores

    if not similarity_by_key:
        return pd.DataFrame()

    recommendations = candidate_courses[
        candidate_courses["Course Key"].isin(similarity_by_key)
    ].copy()
    recommendations["Similarity"] = recommendations["Course Key"].map(similarity_by_key)
    return recommendations.sort_values(
        by=["Similarity", "Rating"],
        ascending=[False, False],
    ).head(limit)


def advisor_reply(recommendations: pd.DataFrame, scoped_count: int) -> str:
    if recommendations.empty:
        return "I could not find a strong match in the current catalog scope. Try adding a role, skill, level, or remove some manual filters."

    top = recommendations.iloc[0]
    providers = ", ".join(recommendations["Provider"].dropna().astype(str).unique()[:3])
    return (
        f"I found {len(recommendations)} recommendations from {scoped_count:,} in-scope courses. "
        f"Top match: {top['Course Name']} from {top['University']}. "
        f"Provider mix: {providers}."
    )


def apply_filters(
    courses: pd.DataFrame,
    search_query: str,
    provider: str,
    language: str,
    difficulty_level: str,
    university: str,
    selected_skills: Iterable[str],
) -> pd.DataFrame:
    filtered = courses
    query = search_query.strip().lower()
    skills = tuple(selected_skills or ())

    if query:
        filtered = filtered[filtered["Search Text"].str.contains(query, regex=False)]

    if provider != ALL_OPTION:
        filtered = filtered[filtered["Provider"] == provider]

    if language != ALL_OPTION:
        filtered = filtered[filtered["Language"] == language]

    if difficulty_level != ALL_OPTION:
        filtered = filtered[filtered["Difficulty Level"] == difficulty_level]

    if university != ALL_OPTION:
        filtered = filtered[filtered["University"] == university]

    if skills:
        selected = set(skills)
        filtered = filtered[
            filtered["Skill Tokens"].apply(lambda row_skills: bool(selected.intersection(row_skills)))
        ]

    return filtered


def recommend_courses(
    course_key: str,
    candidate_courses: pd.DataFrame,
    recommendation_resources: dict,
    limit: int = MAX_RECOMMENDATIONS,
) -> pd.DataFrame:
    key_to_position = recommendation_resources["key_to_position"]
    selected_index = key_to_position.get(course_key)
    if selected_index is None:
        return pd.DataFrame()

    candidates = candidate_courses[candidate_courses["Course Key"] != course_key].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidate_keys = set(candidates["Course Key"])
    candidate_positions = [
        key_to_position[key]
        for key in candidate_keys
        if key in key_to_position and key != course_key
    ]

    if not candidate_positions:
        return pd.DataFrame()

    vectors = recommendation_resources["vectors"]
    ranked_rows: list[tuple[str, float]] = []
    if recommendation_resources.get("mode") == "precomputed":
        neighbor_indices = recommendation_resources.get("neighbor_indices")
        neighbor_similarities = recommendation_resources.get("neighbor_similarities")
        course_keys = recommendation_resources["course_keys"]
        for neighbor_index, similarity in zip(
            neighbor_indices[selected_index],
            neighbor_similarities[selected_index],
        ):
            neighbor_key = course_keys[int(neighbor_index)]
            if neighbor_key == course_key or neighbor_key not in candidate_keys:
                continue
            ranked_rows.append((neighbor_key, float(similarity)))
            if len(ranked_rows) >= limit:
                break

    if len(ranked_rows) < limit:
        similarities = cosine_similarity(
            vectors[selected_index],
            vectors[candidate_positions],
        ).ravel()
        ranked_rows = [
            (recommendation_resources["course_keys"][position], float(score))
            for position, score in zip(candidate_positions, similarities)
        ]

    similarity_by_key = dict(ranked_rows)
    candidates = candidates[candidates["Course Key"].isin(similarity_by_key)].copy()
    candidates["Similarity"] = candidates["Course Key"].map(similarity_by_key)

    return candidates.sort_values(
        by=["Similarity", "Rating"],
        ascending=[False, False],
    ).head(limit)


def sort_recommendations(recommendations: pd.DataFrame, sort_order: str) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    if sort_order == SORT_HIGH_TO_LOW:
        return recommendations.sort_values(
            by=["Rating", "Similarity"],
            ascending=[False, False],
        )

    if sort_order == SORT_LOW_TO_HIGH:
        return recommendations.sort_values(
            by=["Rating", "Similarity"],
            ascending=[True, False],
        )

    return recommendations.sort_values(
        by=["Similarity", "Rating"],
        ascending=[False, False],
    )


def filter_key(
    search_query: str,
    provider: str,
    language: str,
    difficulty_level: str,
    university: str,
    selected_skills: Iterable[str],
) -> tuple[str, str, str, str, str, tuple[str, ...]]:
    return (
        search_query.strip().lower(),
        provider,
        language,
        difficulty_level,
        university,
        tuple(sorted(selected_skills or ())),
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="vrl-header">
            <div class="vrl-header-content">
                <div class="vrl-header-copy">
                    <p class="vrl-eyebrow">Learning Intelligence</p>
                    <p class="vrl-title">VRL Learning Navigator</p>
                    <p class="vrl-subtitle">Goal-based course paths, practice prompts, and knowledge checks.</p>
                </div>
                <div class="vrl-header-pills" aria-label="Recommendation context">
                    <span class="vrl-header-pill">Catalog</span>
                    <span class="vrl-header-pill">Progress</span>
                    <span class="vrl-header-pill">Roadmap</span>
                    <span class="vrl-header-pill">Practice</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(courses: pd.DataFrame, filtered_courses: pd.DataFrame) -> None:
    total_courses = len(courses)
    filtered_count = len(filtered_courses)
    universities = filtered_courses["University"].nunique() if filtered_count else 0
    avg_rating = filtered_courses["Rating"].mean() if filtered_count else 0

    cols = st.columns(4)
    cols[0].metric("Total courses", f"{total_courses:,}", border=True)
    cols[1].metric("Filtered catalog", f"{filtered_count:,}", border=True)
    cols[2].metric("Universities", f"{universities:,}", border=True)
    cols[3].metric("Average rating", f"{avg_rating:.2f}", border=True)


def render_section_heading(title: str, copy: str | None = None) -> None:
    body = f'<div class="vrl-section-title">{escape(title)}</div>'
    if copy:
        body += f'<div class="vrl-section-copy">{escape(copy)}</div>'
    st.markdown(body, unsafe_allow_html=True)


def render_recommend_flow() -> None:
    st.markdown(
        """
        <div class="vrl-mode-grid">
            <div class="vrl-mode-card">
                <div class="vrl-mode-kicker">1. Start with intent</div>
                <div class="vrl-mode-title">AI Learning Navigator</div>
                <div class="vrl-mode-copy">Best when the learner has a goal, role, domain, or roadmap question.</div>
            </div>
            <div class="vrl-mode-card">
                <div class="vrl-mode-kicker">2. Match from a course</div>
                <div class="vrl-mode-title">Course similarity</div>
                <div class="vrl-mode-copy">Best when the learner already completed, liked, or selected one course.</div>
            </div>
            <div class="vrl-mode-card">
                <div class="vrl-mode-kicker">3. Keep practice aligned</div>
                <div class="vrl-mode-title">Progress tracker</div>
                <div class="vrl-mode-copy">Best for calibrating practice tasks and checks to the learner's current level.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skill_chips(skills: Iterable[str], limit: int = 5) -> None:
    visible = list(skills or ())[:limit]
    if not visible:
        return

    chip_html = "".join(
        f'<span class="vrl-chip">{escape(format_skill_label(skill))}</span>'
        for skill in visible
    )
    extra = max(0, len(list(skills or ())) - limit)
    if extra:
        chip_html += f'<span class="vrl-chip">+{extra}</span>'

    st.markdown(f'<div class="vrl-chip-row">{chip_html}</div>', unsafe_allow_html=True)


def render_course_card(
    course: pd.Series,
    *,
    rank: int | None = None,
    show_similarity: bool = False,
    show_reason: bool = False,
    key_prefix: str = "course",
) -> None:
    course_name = str(course["Course Name"])
    course_key = normalize_course_key(course.get("Course Key", course.name))
    raw_course_key = str(course.get("Course Key", course_key))
    saved = course_name in st.session_state.get("shortlist", [])
    completed = is_course_completed(raw_course_key)
    similarity = course.get("Similarity", None)
    similarity_value = None
    if similarity is not None and pd.notna(similarity):
        similarity_value = min(max(float(similarity), 0.0), 1.0)

    rank_text = f"#{rank}" if rank is not None else "Course"
    difficulty = escape(str(course["Difficulty Level"]))
    language = escape(str(course.get("Language", DEFAULT_LANGUAGE)))

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="vrl-card-top">
                <span class="vrl-rank">{escape(rank_text)}</span>
                <span class="vrl-badge">{difficulty}</span>
            </div>
            <div class="vrl-course-title">{escape(course_name)}</div>
            <div class="vrl-meta">
                {escape(str(course["University"]))} | {language} | {float(course["Rating"]):.1f}/5
            </div>
            """,
            unsafe_allow_html=True,
        )

        if show_similarity and similarity_value is not None:
            st.progress(similarity_value, text=f"{similarity_value:.0%} match")

        if show_reason and str(course.get("Match Reason", "")).strip():
            st.markdown(
                f'<div class="vrl-match-reason">{escape(str(course["Match Reason"]))}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="vrl-description">{escape(truncate_text(course["Course Description"]))}</div>',
            unsafe_allow_html=True,
        )
        render_skill_chips(course.get("Skill Tokens", ()))

        st.link_button(
            "Open course",
            str(course["Course URL"]),
            type="primary",
            width="stretch",
        )
        save_col, done_col = st.columns(2)
        with save_col:
            st.button(
                "Saved" if saved else "Save",
                key=f"{key_prefix}_save_{course_key}",
                width="stretch",
                on_click=toggle_shortlist,
                args=(course_name,),
            )
        with done_col:
            st.button(
                "Undo" if completed else "Complete",
                key=f"{key_prefix}_done_{course_key}",
                width="stretch",
                on_click=toggle_completed_course,
                args=(raw_course_key, course_name),
            )


def render_course_grid(
    courses: pd.DataFrame,
    *,
    show_similarity: bool = False,
    show_reason: bool = False,
    key_prefix: str = "grid",
) -> None:
    if courses.empty:
        st.markdown(
            '<div class="vrl-empty">No courses match the current selection.</div>',
            unsafe_allow_html=True,
        )
        return

    for start in range(0, len(courses), 3):
        columns = st.columns(3)
        chunk = courses.iloc[start : start + 3]
        for offset, (index, course) in enumerate(chunk.iterrows(), start=1):
            with columns[offset - 1]:
                render_course_card(
                    course,
                    rank=start + offset,
                    show_similarity=show_similarity,
                    show_reason=show_reason,
                    key_prefix=f"{key_prefix}_{index}",
                )


def practice_task(
    title: str,
    task: str,
    goal: str,
    deliverables: tuple[str, ...],
    checks: tuple[tuple[str, tuple[str, ...]], ...],
    minimum_words: int = 45,
) -> dict[str, object]:
    return {
        "title": title,
        "task": task,
        "goal": goal,
        "deliverables": deliverables,
        "checks": checks,
        "minimum_words": minimum_words,
    }


LEARNING_BUNDLES = {
    "genai": {
        "label": "Generative AI",
        "practice": [
            practice_task(
                "Prompt comparison lab",
                "Write two prompts for the same task, compare output quality, and note which instruction improved accuracy.",
                "Shows whether you can design instructions, compare outputs, and improve a GenAI result intentionally.",
                ("Task and audience", "Two prompt versions", "Output comparison", "Improvement decision"),
                (
                    ("Task context", ("task", "audience", "goal", "user")),
                    ("Two prompt versions", ("prompt", "version", "instruction")),
                    ("Comparison evidence", ("compare", "difference", "quality", "accuracy")),
                    ("Improvement decision", ("improve", "better", "constraint", "format")),
                ),
            ),
            practice_task(
                "Document summarizer",
                "Pick a public article, design a summary prompt, then add a checklist for hallucination checks.",
                "Validates that you can ask for grounded summaries and define checks before trusting the answer.",
                ("Source context", "Summary prompt", "Grounding checks", "Risk notes"),
                (
                    ("Source context", ("article", "source", "document", "link")),
                    ("Summary instruction", ("summary", "summarize", "bullet", "brief")),
                    ("Hallucination check", ("hallucination", "evidence", "verify", "source")),
                    ("Risk handling", ("risk", "uncertain", "citation", "fact")),
                ),
            ),
            practice_task(
                "Mini assistant brief",
                "Define a narrow assistant persona, required inputs, refusal boundaries, and a success metric.",
                "Checks whether you can translate a GenAI idea into a usable assistant specification.",
                ("Assistant scope", "Required inputs", "Boundaries", "Success metric"),
                (
                    ("Assistant scope", ("assistant", "scope", "persona", "role")),
                    ("Inputs", ("input", "fields", "required", "context")),
                    ("Boundaries", ("boundary", "refuse", "limit", "policy")),
                    ("Success metric", ("metric", "success", "evaluate", "quality")),
                ),
            ),
        ],
        "mcqs": [
            ("Which signal most strongly indicates an LLM hallucination risk?", ["A source-backed answer", "A confident answer without evidence", "A short answer", "A rewritten answer"], 1),
            ("Prompt engineering usually improves outputs by making what clearer?", ["The model weights", "The user interface color", "Task, context, constraints, and format", "The internet speed"], 2),
            ("Why are evaluation examples useful for GenAI systems?", ["They replace all testing", "They show whether outputs meet expected behavior", "They remove the need for prompts", "They make models private"], 1),
        ],
        "resource_query": "generative AI prompt engineering large language models",
    },
    "data": {
        "label": "Data Science",
        "practice": [
            practice_task(
                "EDA notebook",
                "Choose a CSV, profile missing values, create three charts, and write five findings.",
                "Validates that you can inspect a dataset before modeling and turn raw columns into useful observations.",
                ("Dataset context", "Missing-value profile", "Three chart ideas", "Five findings"),
                (
                    ("Dataset context", ("dataset", "csv", "rows", "columns", "source")),
                    ("Missing values", ("missing", "null", "nan", "blank")),
                    ("Charts", ("chart", "plot", "visual", "histogram", "bar")),
                    ("Findings", ("finding", "insight", "trend", "pattern")),
                ),
            ),
            practice_task(
                "SQL insight drill",
                "Write queries for filtering, grouping, joins, and ranking on one dataset.",
                "Checks whether you can move from a question to the SQL patterns used in real analysis work.",
                ("Business question", "Filter query", "Group/join query", "Ranked insight"),
                (
                    ("Business question", ("question", "business", "goal", "metric")),
                    ("Filtering", ("where", "filter", "condition")),
                    ("Grouping or joins", ("group", "join", "aggregate", "sum", "count")),
                    ("Ranking", ("rank", "order", "top", "limit")),
                ),
            ),
            practice_task(
                "Dashboard brief",
                "Turn one business question into metrics, visuals, and a recommendation.",
                "Shows whether you can design a dashboard around decisions instead of only placing charts on a page.",
                ("Decision question", "Metrics", "Visual layout", "Recommendation"),
                (
                    ("Decision question", ("decision", "question", "stakeholder", "business")),
                    ("Metrics", ("metric", "kpi", "measure", "rate")),
                    ("Visual layout", ("dashboard", "visual", "chart", "layout")),
                    ("Recommendation", ("recommend", "action", "next", "decision")),
                ),
            ),
        ],
        "mcqs": [
            ("What is the main goal of exploratory data analysis?", ["Deploy a model", "Understand patterns and data quality", "Encrypt the dataset", "Write production APIs"], 1),
            ("Which metric is best for a heavily imbalanced classification dataset?", ["Accuracy only", "Precision/recall or F1", "File size", "Column count"], 1),
            ("Why split data into train and test sets?", ["To make charts prettier", "To estimate performance on unseen data", "To delete outliers", "To speed up typing"], 1),
        ],
        "resource_query": "data science python sql dashboard analytics",
    },
    "cybersecurity": {
        "label": "Cybersecurity",
        "practice": [
            practice_task(
                "Threat model",
                "Pick a simple app and list assets, entry points, threats, and mitigations.",
                "Validates that you can reason about what needs protection and where risk enters the system.",
                ("System scope", "Assets", "Entry points", "Mitigations"),
                (
                    ("System scope", ("app", "system", "scope", "user")),
                    ("Assets", ("asset", "data", "credential", "service")),
                    ("Entry points", ("entry", "endpoint", "login", "input")),
                    ("Mitigations", ("mitigation", "control", "protect", "reduce")),
                ),
            ),
            practice_task(
                "Security checklist",
                "Create checks for password policy, logging, access control, and patching.",
                "Checks whether you can convert security fundamentals into operational review items.",
                ("Password control", "Access control", "Logging", "Patch review"),
                (
                    ("Password control", ("password", "mfa", "authentication")),
                    ("Access control", ("access", "permission", "role", "privilege")),
                    ("Logging", ("log", "audit", "monitor", "alert")),
                    ("Patch review", ("patch", "update", "vulnerability", "version")),
                ),
            ),
            practice_task(
                "Incident response drill",
                "Write a short response plan for a suspicious login event.",
                "Shows whether you understand the first practical steps when an alert becomes an incident.",
                ("Trigger", "Containment", "Investigation", "Communication"),
                (
                    ("Trigger", ("alert", "login", "suspicious", "event")),
                    ("Containment", ("contain", "disable", "revoke", "block")),
                    ("Investigation", ("investigate", "log", "timeline", "evidence")),
                    ("Communication", ("notify", "escalate", "owner", "report")),
                ),
            ),
        ],
        "mcqs": [
            ("What does least privilege mean?", ["Give users admin access", "Give only required access", "Disable logs", "Use one shared account"], 1),
            ("Which control helps detect suspicious activity?", ["Audit logging", "Bigger fonts", "Unused accounts", "Plain text passwords"], 0),
            ("Why is patching important?", ["It improves monitor brightness", "It fixes known vulnerabilities", "It removes backups", "It avoids authentication"], 1),
        ],
        "resource_query": "cybersecurity fundamentals threat modeling incident response",
    },
    "cloud": {
        "label": "Cloud DevOps",
        "practice": [
            practice_task(
                "Deployment map",
                "Draw source control, build, test, deploy, monitoring, and rollback steps for one app.",
                "Validates that you understand the flow from code change to a monitored release.",
                ("Source flow", "Build/test step", "Deployment step", "Rollback/monitoring"),
                (
                    ("Source flow", ("git", "source", "commit", "branch")),
                    ("Build and test", ("build", "test", "ci", "artifact")),
                    ("Deployment", ("deploy", "release", "environment", "service")),
                    ("Recovery", ("rollback", "monitor", "alert", "health")),
                ),
            ),
            practice_task(
                "Pipeline checklist",
                "Define checks for tests, secrets, approvals, artifacts, and release notes.",
                "Checks whether you can design release guardrails before production deployment.",
                ("Test gate", "Secrets handling", "Approval", "Release evidence"),
                (
                    ("Test gate", ("test", "quality", "gate", "check")),
                    ("Secrets handling", ("secret", "credential", "token", "vault")),
                    ("Approval", ("approval", "review", "owner")),
                    ("Release evidence", ("artifact", "release", "notes", "version")),
                ),
            ),
            practice_task(
                "Cloud cost review",
                "Estimate compute, storage, and network drivers for a small service.",
                "Shows whether you can identify the cloud choices that affect cost before scaling.",
                ("Service context", "Compute driver", "Storage driver", "Cost action"),
                (
                    ("Service context", ("service", "users", "traffic", "workload")),
                    ("Compute", ("compute", "cpu", "instance", "container")),
                    ("Storage or network", ("storage", "database", "network", "egress")),
                    ("Cost action", ("cost", "optimize", "budget", "reduce")),
                ),
            ),
        ],
        "mcqs": [
            ("What is a CI/CD pipeline for?", ["Manual copy-paste releases", "Automating build, test, and deployment", "Writing invoices", "Replacing source control"], 1),
            ("Why keep secrets out of code?", ["They slow tests", "They can leak credentials", "They reduce comments", "They break CSS"], 1),
            ("What does rollback support?", ["Recovering from bad releases", "Increasing logo size", "Deleting monitoring", "Skipping tests"], 0),
        ],
        "resource_query": "cloud devops azure ci cd kubernetes",
    },
    "general": {
        "label": "Learning Path",
        "practice": [
            practice_task(
                "Concept map",
                "Write the top ten terms in this topic and connect each term to one practical use.",
                "Validates that you can explain the subject vocabulary in a useful context.",
                ("Topic terms", "Connections", "Practical use", "Learning gap"),
                (
                    ("Topic terms", ("term", "concept", "topic", "definition")),
                    ("Connections", ("connect", "relationship", "depends", "relates")),
                    ("Practical use", ("use", "example", "apply", "scenario")),
                    ("Learning gap", ("gap", "unclear", "next", "improve")),
                ),
            ),
            practice_task(
                "Mini project",
                "Build or outline a small project that applies the first two courses.",
                "Checks whether you can turn course content into something concrete.",
                ("Project goal", "Inputs", "Build steps", "Result"),
                (
                    ("Project goal", ("project", "goal", "problem", "user")),
                    ("Inputs", ("input", "data", "resource", "tool")),
                    ("Build steps", ("step", "build", "implement", "create")),
                    ("Result", ("result", "output", "demo", "deliverable")),
                ),
            ),
            practice_task(
                "Portfolio note",
                "Summarize what you learned, what you built, and what you would improve next.",
                "Shows whether you can capture learning as evidence for interviews or future review.",
                ("Learning summary", "Built artifact", "Evidence", "Next improvement"),
                (
                    ("Learning summary", ("learned", "skill", "concept", "understand")),
                    ("Built artifact", ("built", "created", "project", "artifact")),
                    ("Evidence", ("evidence", "result", "screenshot", "link")),
                    ("Next improvement", ("improve", "next", "better", "iterate")),
                ),
            ),
        ],
        "mcqs": [
            ("What makes a learning goal useful?", ["It is vague", "It has a skill, context, and outcome", "It avoids practice", "It has no deadline"], 1),
            ("Why do small projects help learning?", ["They replace all theory", "They force applied recall", "They hide mistakes", "They remove feedback"], 1),
            ("What should you do after completing a course?", ["Never revisit it", "Apply, review, and document the skill", "Delete notes", "Avoid examples"], 1),
        ],
        "resource_query": "online learning practical project skills",
    },
}


def infer_bundle_domain(query_text: str, recommendations: pd.DataFrame) -> str:
    combined = normalize_phrase(query_text)
    if not recommendations.empty:
        combined = normalize_phrase(
            combined
            + " "
            + " ".join(recommendations.head(5)["Search Text"].fillna("").astype(str))
        )

    domain_terms = (
        ("genai", ("gen ai", "genai", "generative", "large language", "llm", "prompt", "openai", "copilot")),
        ("data", ("data science", "data analyst", "analytics", "dashboard", "sql", "statistics", "machine learning")),
        ("cybersecurity", ("cyber", "security", "threat", "incident", "vulnerability", "privacy")),
        ("cloud", ("cloud", "devops", "azure", "kubernetes", "pipeline", "deployment")),
    )
    for domain, terms in domain_terms:
        if any(phrase_in_text(term, combined) for term in terms):
            return domain
    return "general"


def bundle_resource_links(query_text: str, bundle: dict[str, object]) -> list[tuple[str, str]]:
    base_query = str(bundle.get("resource_query") or query_text or "online learning")
    encoded = quote_plus(base_query)
    return [
        ("Medium", f"https://medium.com/search?q={encoded}"),
        ("YouTube", f"https://www.youtube.com/results?search_query={encoded}"),
        ("Google Scholar", f"https://scholar.google.com/scholar?q={encoded}"),
        ("GitHub", f"https://github.com/search?q={encoded}&type=repositories"),
        ("Kaggle", f"https://www.kaggle.com/search?q={encoded}"),
        ("Microsoft Learn", f"https://learn.microsoft.com/en-us/search/?terms={encoded}"),
    ]


def practice_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def practice_unique_word_count(text: str) -> int:
    return len(
        {
            token
            for token in re.findall(r"[A-Za-z0-9+#.]+", normalize_phrase(text))
            if token not in SMART_FILTER_STOPWORDS and len(token) > 2
        }
    )


def validator_mode() -> str:
    mode = normalize_phrase(os.getenv(VALIDATOR_MODE_ENV, ""))
    if mode in {"smart", "ai", "large", "advanced"}:
        return VALIDATOR_SMART
    if mode in {"free", "light", "basic", "streamlit"}:
        return VALIDATOR_FREE

    legacy_backend = normalize_phrase(os.getenv(VALIDATOR_BACKEND_ENV, ""))
    legacy_profile = normalize_phrase(os.getenv(VALIDATOR_PROFILE_ENV, ""))
    if legacy_backend in {"minilm", "sentence transformers", "sentence transformer", "embedding"}:
        return VALIDATOR_SMART
    if legacy_backend in {"tfidf", "light", "semantic light", "streamlit"}:
        return VALIDATOR_FREE
    if legacy_profile in {"scaled", "server", "production"}:
        return VALIDATOR_SMART

    return VALIDATOR_FREE


def optional_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def choose_validation_backend() -> dict[str, object]:
    mode = validator_mode()
    model_name = os.getenv(VALIDATOR_MODEL_ENV, "sentence-transformers/all-MiniLM-L6-v2")
    smart_available = optional_module_available("sentence_transformers")

    if mode == VALIDATOR_SMART and smart_available:
        backend = VALIDATOR_MINILM
    else:
        backend = VALIDATOR_TFIDF

    if backend == VALIDATOR_MINILM:
        return {
            "backend": VALIDATOR_MINILM,
            "label": "Smart",
            "mode": VALIDATOR_SMART,
            "model": model_name,
            "threshold": 0.42,
            "note": "Uses a local open-source model when the deployment has enough resources.",
        }

    note = "No API key, no model download, safe for Streamlit Cloud."
    if mode == VALIDATOR_SMART and not smart_available:
        note = "Smart mode was requested, but the model package is not installed; using Free mode safely."

    return {
        "backend": VALIDATOR_TFIDF,
        "label": "Free",
        "mode": VALIDATOR_FREE,
        "model": "built-in semantic scorer",
        "threshold": 0.08,
        "note": note,
    }


@st.cache_resource(show_spinner=False)
def load_minilm_validator(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def tfidf_semantic_similarity(left_text: str, right_text: str) -> float:
    left = normalize_phrase(left_text)
    right = normalize_phrase(right_text)
    if not left or not right:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        vectors = vectorizer.fit_transform([left, right])
        return float(cosine_similarity(vectors[0], vectors[1]).ravel()[0])
    except ValueError:
        return 0.0


def minilm_semantic_similarity(left_text: str, right_text: str, model_name: str) -> float:
    model = load_minilm_validator(model_name)
    vectors = model.encode(
        [left_text, right_text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return float(cosine_similarity([vectors[0]], [vectors[1]]).ravel()[0])


def semantic_similarity(left_text: str, right_text: str, backend_info: dict[str, object]) -> float:
    if backend_info.get("backend") == VALIDATOR_MINILM:
        try:
            return minilm_semantic_similarity(left_text, right_text, str(backend_info.get("model", "")))
        except Exception:
            backend_info["backend"] = VALIDATOR_TFIDF
            backend_info["label"] = "Free"
            backend_info["mode"] = VALIDATOR_FREE
            backend_info["model"] = "built-in semantic scorer"
            backend_info["threshold"] = 0.08
            backend_info["note"] = "Smart mode was unavailable, so validation fell back to Free mode."
    return tfidf_semantic_similarity(left_text, right_text)


def criterion_reference_text(
    label: str,
    keywords: tuple[str, ...],
    task_context: str,
) -> str:
    return " ".join([label, *keywords, task_context])


def validate_practice_attempt(
    attempt_text: str,
    checks: tuple[tuple[str, tuple[str, ...]], ...],
    minimum_words: int = 45,
    field_texts: tuple[tuple[str, str], ...] = (),
    task_context: str = "",
) -> dict[str, object]:
    normalized_attempt = normalize_phrase(attempt_text)
    passed: list[str] = []
    missing: list[str] = []
    semantic_scores: list[dict[str, object]] = []
    backend_info = choose_validation_backend()
    semantic_threshold = float(backend_info["threshold"])
    word_count = practice_word_count(attempt_text)
    unique_words = practice_unique_word_count(attempt_text)
    minimum_unique_words = max(10, min(18, minimum_words // 3))

    if word_count >= minimum_words:
        passed.append("Enough detail")
    else:
        missing.append(f"Add at least {minimum_words} words across the structured fields")

    if unique_words >= minimum_unique_words:
        passed.append("Varied evidence")
    else:
        missing.append("Use more specific, varied details instead of repeated or random text")

    for field_label, field_text in field_texts:
        if practice_word_count(field_text) >= 5 and practice_unique_word_count(field_text) >= 3:
            passed.append(f"{field_label} completed")
        else:
            missing.append(f"Add meaningful detail in {field_label}")

    for label, keywords in checks:
        reference_text = criterion_reference_text(label, keywords, task_context)
        score = semantic_similarity(attempt_text, reference_text, backend_info)
        exact_signal = any(phrase_in_text(keyword, normalized_attempt) for keyword in keywords)
        criterion_passed = score >= semantic_threshold or (
            exact_signal and word_count >= minimum_words and unique_words >= minimum_unique_words
        )
        semantic_scores.append(
            {
                "Requirement": label,
                "Match score": round(score, 3),
                "Passed": "Yes" if criterion_passed else "No",
            }
        )
        if criterion_passed:
            passed.append(label)
        else:
            missing.append(label)

    total = len(passed) + len(missing)
    score = int(round((len(passed) / total) * 100)) if total else 0
    return {
        "score": score,
        "passed": passed,
        "missing": missing,
        "valid": not missing,
        "backend": backend_info,
        "semantic_scores": semantic_scores,
    }


def render_check_items(items: Iterable[str]) -> None:
    if not items:
        return
    item_markup = "".join(
        f'<div class="vrl-check-item">{escape(str(item))}</div>'
        for item in items
    )
    st.markdown(f'<div class="vrl-check-list">{item_markup}</div>', unsafe_allow_html=True)


def difficulty_practice_profile(difficulty: object) -> dict[str, object]:
    normalized = normalize_phrase(difficulty)
    if "advanced" in normalized:
        return {
            "label": "Advanced",
            "scope": "design tradeoffs, edge cases, and a small extension",
            "minimum_words": 65,
            "deliverable_prefix": "Advanced proof",
        }
    if "intermediate" in normalized:
        return {
            "label": "Intermediate",
            "scope": "a practical mini-build with reasoning",
            "minimum_words": 55,
            "deliverable_prefix": "Applied proof",
        }
    return {
        "label": "Beginner",
        "scope": "guided fundamentals and a small one-sitting exercise",
        "minimum_words": 35,
        "deliverable_prefix": "Foundation proof",
    }


def skill_labels_for_course(course: pd.Series | None, limit: int = 4) -> list[str]:
    if course is None:
        return []
    return [
        format_skill_label(skill)
        for skill in course.get("Skill Tokens", ())[:limit]
        if str(skill).strip()
    ]


def build_course_practice_items(
    course: pd.Series | None,
    fallback_practice: list[dict[str, object]],
) -> list[dict[str, object]]:
    if course is None:
        return fallback_practice

    course_name = str(course.get("Course Name", "Selected course"))
    difficulty = str(course.get("Difficulty Level", "Beginner"))
    profile = difficulty_practice_profile(difficulty)
    skills = skill_labels_for_course(course)
    skill_text = ", ".join(skills) if skills else "the main course skills"
    minimum_words = int(profile["minimum_words"])
    scope = str(profile["scope"])
    proof_label = str(profile["deliverable_prefix"])

    return [
        practice_task(
            f"{profile['label']} skill drill",
            f"Practice {skill_text} from {course_name} through {scope}.",
            f"This is anchored to the completed course {course_name}, so the difficulty stays at {difficulty} instead of jumping to a harder track.",
            (f"{proof_label}: course concept used", "Input or example", "Step-by-step attempt", "Result"),
            (
                ("Course concept", tuple(skill.lower() for skill in skills) or ("concept", "skill", "topic")),
                ("Input or example", ("input", "example", "sample", "scenario")),
                ("Attempt steps", ("step", "approach", "method", "solve")),
                ("Result", ("result", "output", "answer", "finding")),
            ),
            minimum_words=minimum_words,
        ),
        practice_task(
            f"{profile['label']} mini build",
            f"Create a tiny artifact using {skill_text}: a script, notebook cell, query, diagram, prompt, or checklist based on the course topic.",
            f"Checks whether you can turn {course_name} into a concrete output at the same learning level.",
            ("Goal", "Tool or format", "Build steps", "Working output"),
            (
                ("Goal", ("goal", "problem", "task", "objective")),
                ("Tool or format", ("script", "notebook", "query", "diagram", "prompt", "checklist", "tool")),
                ("Build steps", ("build", "create", "step", "implement")),
                ("Working output", ("output", "works", "result", "demo")),
            ),
            minimum_words=minimum_words,
        ),
        practice_task(
            f"{profile['label']} review check",
            f"Explain one concept from {course_name}, show one mistake a learner might make, and write how to fix it.",
            f"Validates retention from the completed course before recommending harder practice.",
            ("Concept explanation", "Common mistake", "Fix", "Confidence note"),
            (
                ("Concept explanation", ("concept", "means", "explain", "definition")),
                ("Common mistake", ("mistake", "error", "wrong", "bug")),
                ("Fix", ("fix", "correct", "solution", "improve")),
                ("Confidence note", ("confident", "unclear", "practice", "review")),
            ),
            minimum_words=minimum_words,
        ),
    ]


def render_advisor_roadmap(
    recommendations: pd.DataFrame,
    catalog_courses: pd.DataFrame,
    query_text: str = "",
) -> None:
    if recommendations.empty or "Roadmap Stage" not in recommendations.columns:
        return

    domain = infer_bundle_domain(query_text, recommendations)
    bundle = LEARNING_BUNDLES.get(domain, LEARNING_BUNDLES["general"])
    stage_order = ["Foundation", "Build", "Specialize"]
    stage_copy = {
        "Foundation": "Start here",
        "Build": "Practice next",
        "Specialize": "Go deeper",
    }
    selected_indexes: set[object] = set()
    roadmap_rows: list[pd.Series] = []
    for stage in stage_order:
        stage_rows = recommendations[
            (recommendations["Roadmap Stage"] == stage)
            & (~recommendations.index.isin(selected_indexes))
        ]
        if stage_rows.empty:
            continue
        row = stage_rows.iloc[0]
        roadmap_rows.append(row)
        selected_indexes.add(row.name)

    if len(roadmap_rows) < 3:
        for _, row in recommendations.iterrows():
            if row.name in selected_indexes:
                continue
            roadmap_rows.append(row)
            selected_indexes.add(row.name)
            if len(roadmap_rows) >= 3:
                break

    if not roadmap_rows:
        return

    roadmap_rows = roadmap_rows[:3]
    st.markdown(
        f'<div class="vrl-section-title">{escape(str(bundle["label"]))} learning bundle</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="vrl-small-note">Courses, practice, self-check, and external resources in one compact path.</div>',
        unsafe_allow_html=True,
    )

    start_tab, practice_tab, deeper_tab, resources_tab = st.tabs(
        ["Start here", "Practice next", "Go deeper", "Resources"]
    )

    with start_tab:
        step_options = list(range(len(roadmap_rows)))
        roadmap_key = "advisor_roadmap_step"
        if st.session_state.get(roadmap_key) not in step_options:
            st.session_state[roadmap_key] = 0

        selected_step = st.pills(
            "Roadmap steps",
            step_options,
            format_func=lambda index: f"{index + 1}. {stage_copy.get(str(roadmap_rows[index].get('Roadmap Stage', 'Build')), 'Recommended')}",
            key=roadmap_key,
            label_visibility="collapsed",
            width="stretch",
        )
        selected_step = 0 if selected_step is None else int(selected_step)

        columns = st.columns(len(roadmap_rows))
        for index, row in enumerate(roadmap_rows):
            stage = str(row.get("Roadmap Stage", "Build"))
            label = stage_copy.get(stage, "Recommended")
            course_key = normalize_course_key(row.get("Course Key", row.name))
            similarity = row.get("Similarity", 0.0)
            similarity_value = min(max(float(similarity), 0.0), 1.0) if pd.notna(similarity) else 0.0
            with columns[index]:
                with st.container(border=True):
                    st.markdown(f"**{index + 1}. {label}**")
                    st.markdown(f"##### {str(row.get('Course Name', ''))}")
                    st.caption(f"{row.get('Provider', '')} | {row.get('Difficulty Level', '')}")
                    st.progress(similarity_value, text=f"{similarity_value:.0%} match")
                    st.link_button(
                        "Open",
                        str(row.get("Course URL", "")),
                        width="stretch",
                        key=f"roadmap_open_{course_key}_{index}",
                    )

        selected_row = roadmap_rows[selected_step]
        selected_course_key = normalize_course_key(selected_row.get("Course Key", selected_row.name))
        selected_course_completed = is_course_completed(selected_row.get("Course Key", ""))
        st.markdown("##### Current step")
        with st.container(border=True):
            left_col, right_col = st.columns([0.68, 0.32], vertical_alignment="center")
            with left_col:
                st.markdown(f"### {selected_step + 1}. {str(selected_row.get('Course Name', ''))}")
                st.caption(
                    f"{selected_row.get('Provider', '')} | "
                    f"{selected_row.get('University', '')} | "
                    f"{selected_row.get('Difficulty Level', '')}"
                )
                if str(selected_row.get("Match Reason", "")).strip():
                    st.markdown(
                        f'<div class="vrl-match-reason">{escape(str(selected_row["Match Reason"]))}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(truncate_text(str(selected_row.get("Course Description", "")), max_chars=320))
            with right_col:
                st.link_button(
                    "Open current step",
                    str(selected_row.get("Course URL", "")),
                    type="primary",
                    width="stretch",
                )
                st.button(
                    "Save step",
                    key=f"roadmap_save_{selected_course_key}",
                    width="stretch",
                    on_click=toggle_shortlist,
                    args=(str(selected_row.get("Course Name", "")),),
                )
                st.button(
                    "Undo completed" if selected_course_completed else "Complete",
                    key=f"roadmap_complete_{selected_course_key}",
                    width="stretch",
                    on_click=toggle_completed_course,
                    args=(
                        str(selected_row.get("Course Key", "")),
                        str(selected_row.get("Course Name", "")),
                    ),
                )

    with practice_tab:
        roadmap_frame = pd.DataFrame(roadmap_rows)
        practice_anchor = active_practice_course(catalog_courses, roadmap_frame)
        anchor_key = normalize_course_key(
            practice_anchor.get("Course Key", "bundle") if practice_anchor is not None else "bundle"
        )
        practice_items = build_course_practice_items(practice_anchor, list(bundle["practice"]))
        task_options = list(range(len(practice_items)))
        task_key = f"advisor_practice_task_{domain}_{anchor_key}"
        if st.session_state.get(task_key) not in task_options:
            st.session_state[task_key] = 0
        selected_task = st.pills(
            "Practice tasks",
            task_options,
            format_func=lambda index: str(practice_items[index].get("title", "Practice task")),
            key=task_key,
            label_visibility="collapsed",
            width="stretch",
        )
        selected_task = 0 if selected_task is None else int(selected_task)
        task = practice_items[selected_task]
        task_title = str(task.get("title", "Practice task"))
        task_body = str(task.get("task", "Complete a practical checkpoint for this topic."))
        task_goal = str(task.get("goal", "Practice the current topic with a concrete output."))
        deliverables = tuple(str(item) for item in task.get("deliverables", ()))
        checks = tuple(task.get("checks", ()))
        minimum_words = int(task.get("minimum_words", 45))
        task_context = " ".join([task_title, task_body, task_goal, *deliverables])
        validator_info = choose_validation_backend()
        field_prefix = f"advisor_practice_{domain}_{anchor_key}_{selected_task}"
        result_key = f"{field_prefix}_validation"

        with st.container(border=True):
            if practice_anchor is not None:
                completed_count = len(st.session_state.get("completed_courses", []))
                skill_preview = ", ".join(skill_labels_for_course(practice_anchor, limit=3)) or "Course skills"
                st.markdown(
                    f"""
                    <div class="vrl-progress-summary">
                        <div class="vrl-progress-stat">
                            <div class="vrl-progress-label">Practice anchor</div>
                            <div class="vrl-progress-value">{escape(str(practice_anchor.get("Course Name", "")))}</div>
                        </div>
                        <div class="vrl-progress-stat">
                            <div class="vrl-progress-label">Current level</div>
                            <div class="vrl-progress-value">{escape(str(practice_anchor.get("Difficulty Level", "Mixed")))}</div>
                        </div>
                        <div class="vrl-progress-stat">
                            <div class="vrl-progress-label">Progress</div>
                            <div class="vrl-progress-value">{completed_count} completed | {escape(skill_preview)}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(f"### {task_title}")
            st.markdown(
                f'<div class="vrl-practice-brief"><strong>What this section validates:</strong> '
                f'{escape(task_goal)} It checks for concrete evidence, so a random note will show missing criteria.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Task:** {task_body}")
            st.caption(f"Minimum evidence target: {minimum_words} words. Complete a course to make this practice follow your progress.")
            st.caption(
                f"Validator mode: {validator_info['label']} - {validator_info['note']}"
            )
            st.markdown("##### Expected deliverables")
            render_check_items(deliverables)
            st.markdown("##### Validation rubric")
            render_check_items(label for label, _ in checks)

            plan = st.text_area(
                "1. Plan or setup",
                placeholder="Mention the dataset, app, prompt, source, or scenario you used and what you are trying to prove.",
                key=f"{field_prefix}_plan",
                height=92,
            )
            evidence = st.text_area(
                "2. Evidence created",
                placeholder="List the queries, charts, checks, prompts, controls, or outputs you actually produced.",
                key=f"{field_prefix}_evidence",
                height=108,
            )
            reflection = st.text_area(
                "3. Result and next improvement",
                placeholder="Write the key finding, decision, mistake, or next improvement from this practice.",
                key=f"{field_prefix}_reflection",
                height=92,
            )
            attempt_text = "\n".join([plan, evidence, reflection])
            attempt_signature = normalize_phrase(attempt_text)
            if st.button("Validate practice", type="primary", width="stretch", key=f"{field_prefix}_validate"):
                validation_result = validate_practice_attempt(
                    attempt_text,
                    checks,
                    minimum_words=minimum_words,
                    field_texts=(
                        ("Plan", plan),
                        ("Evidence", evidence),
                        ("Result", reflection),
                    ),
                    task_context=task_context,
                )
                validation_result["attempt_signature"] = attempt_signature
                st.session_state[result_key] = validation_result

            result = st.session_state.get(result_key)
            if result and result.get("attempt_signature") != attempt_signature:
                st.info("Your fields changed after the last validation. Run validation again for the latest attempt.")
            elif result:
                score = int(result.get("score", 0))
                st.progress(score / 100, text=f"{score}% validation score")
                if result.get("valid"):
                    st.success("Validated: this looks like a meaningful practice attempt.")
                elif score >= 60:
                    st.warning("Partially valid: add the missing evidence below before treating it as complete.")
                else:
                    st.error("Not validated yet: the answer is too thin or misses the required proof points.")

                covered_col, missing_col = st.columns(2)
                with covered_col:
                    st.markdown("##### Covered")
                    render_check_items(result.get("passed", []))
                with missing_col:
                    st.markdown("##### Missing")
                    render_check_items(result.get("missing", []))
                with st.expander("Why this result?", expanded=False):
                    backend = result.get("backend", {})
                    st.caption(
                        f"Mode: {backend.get('label', 'Validator')} | Engine: {backend.get('model', 'local scorer')}"
                    )
                    st.dataframe(
                        pd.DataFrame(result.get("semantic_scores", [])),
                        hide_index=True,
                        width="stretch",
                    )
            else:
                st.info("Fill the three fields and run validation. Default mode is Free, which works on Streamlit Cloud without credentials.")

    with deeper_tab:
        for index, (question, options, answer_index) in enumerate(bundle["mcqs"], start=1):
            with st.container(border=True):
                st.markdown(f"**Q{index}. {question}**")
                selected_answer = st.radio(
                    "Choose one",
                    options,
                    index=None,
                    key=f"advisor_mcq_{domain}_{index}",
                    label_visibility="collapsed",
                )
                if selected_answer:
                    if options.index(selected_answer) == answer_index:
                        st.success("Correct")
                    else:
                        st.warning(f"Review this: {options[answer_index]}")

    with resources_tab:
        st.markdown("##### External resources")
        resource_cols = st.columns(3)
        for index, (label, url) in enumerate(bundle_resource_links(query_text, bundle)):
            with resource_cols[index % 3]:
                st.link_button(label, url, width="stretch")
        st.markdown("##### Bundle courses")
        for index, row in enumerate(roadmap_rows, start=1):
            st.link_button(
                f"{index}. {str(row.get('Course Name', ''))}",
                str(row.get("Course URL", "")),
                width="stretch",
            )


def render_smart_filter(courses: pd.DataFrame) -> None:
    st.sidebar.markdown("### Smart filter")
    for message in st.session_state.get("smart_filter_messages", [])[-3:]:
        role = "user" if message.get("role") == "user" else "assistant"
        st.sidebar.markdown(
            f'<div class="vrl-chat-bubble vrl-chat-{role}">{escape(str(message.get("content", "")))}</div>',
            unsafe_allow_html=True,
        )

    request = st.sidebar.text_input(
        "Ask smart filter",
        placeholder="Beginner Azure security from Microsoft",
        key="smart_filter_prompt",
    )
    submitted = st.sidebar.button("Apply smart filter", width="stretch")

    if submitted and request.strip():
        reply = apply_smart_filter(courses, request.strip())
        st.session_state["smart_filter_messages"] = [
            *st.session_state.get("smart_filter_messages", [])[-4:],
            {"role": "user", "content": request.strip()},
            {"role": "assistant", "content": reply},
        ]
        st.rerun()


def submit_advisor_request(
    request: str,
    filtered_courses: pd.DataFrame,
    catalog_courses: pd.DataFrame,
    recommendation_resources: dict,
    active_filter_key: tuple[str, str, str, str, str, tuple[str, ...]],
    rating_sort: str,
) -> None:
    progress_note = progress_note_from_request(request, catalog_courses)
    query_text = advisor_query_text(
        st.session_state.get("advisor_messages", []),
        request.strip(),
    )
    query_text = enrich_query_with_progress(query_text, catalog_courses)
    with st.status("Searching the course catalog", expanded=False) as status:
        status.write("Reading your goal")
        recommendations = recommend_for_goal(
            query_text,
            filtered_courses,
            recommendation_resources,
            limit=MAX_RECOMMENDATIONS,
        )
        status.write("Ranking best-fit courses")
        recommendations = sort_recommendations(recommendations, rating_sort)
        recommendations = add_advisor_context(recommendations, query_text)
        reply = advisor_reply(recommendations, len(filtered_courses))
        if progress_note:
            reply = f"{progress_note} {reply}"
        status.update(label="Recommendations ready", state="complete", expanded=False)

    st.session_state["advisor_recommendations"] = recommendations
    st.session_state["advisor_context"] = active_filter_key
    st.session_state["advisor_query_text"] = query_text
    st.session_state["advisor_user_goal"] = request.strip()
    st.session_state["advisor_messages"] = [
        *st.session_state.get("advisor_messages", [])[-6:],
        {"role": "user", "content": request.strip()},
        {"role": "assistant", "content": reply},
    ]
    st.session_state["pending_toast"] = "Recommendations ready"
    st.rerun()


def render_advisor_chat(
    filtered_courses: pd.DataFrame,
    catalog_courses: pd.DataFrame,
    recommendation_resources: dict,
    active_filter_key: tuple[str, str, str, str, str, tuple[str, ...]],
    rating_sort: str,
) -> None:
    st.markdown(
        """
        <div class="vrl-advisor-head">
            <div class="vrl-bot-avatar">AI</div>
            <div>
                <p class="vrl-advisor-name">AI Learning Navigator</p>
                <p class="vrl-advisor-status">Describe a goal and get a focused learning bundle</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    quick_prompts = [
        ("Gen AI", "I am looking for Gen AI courses"),
        ("Cybersecurity", "Beginner cybersecurity courses"),
        ("Data analyst", "Python data analyst SQL dashboard courses"),
        ("Cloud DevOps", "Cloud DevOps Azure courses"),
    ]
    quick_cols = st.columns(4)
    quick_request = ""
    for index, (label, prompt) in enumerate(quick_prompts):
        with quick_cols[index]:
            if st.button(label, key=f"advisor_quick_{index}", width="stretch"):
                quick_request = prompt

    clear_advisor = st.button("Clear conversation", width="content")

    if clear_advisor:
        st.session_state["advisor_messages"] = [
            {
                "role": "assistant",
                "content": "Tell me your goal. I will build a learning bundle from the catalog.",
            }
        ]
        st.session_state["advisor_recommendations"] = None
        st.session_state["advisor_context"] = None
        st.session_state["advisor_query_text"] = ""
        st.session_state["advisor_user_goal"] = ""
        st.rerun()

    if quick_request:
        submit_advisor_request(
            quick_request,
            filtered_courses,
            catalog_courses,
            recommendation_resources,
            active_filter_key,
            rating_sort,
        )

    for message in st.session_state.get("advisor_messages", [])[-6:]:
        role = "user" if message.get("role") == "user" else "assistant"
        avatar = "user" if role == "user" else "assistant"
        with st.chat_message(role, avatar=avatar):
            st.write(str(message.get("content", "")))

    advisor_recommendations = st.session_state.get("advisor_recommendations")
    advisor_context = st.session_state.get("advisor_context")
    if advisor_recommendations is not None:
        if advisor_context != active_filter_key:
            st.info("Manual filters changed after the last advisor answer. Ask again to refresh these recommendations.")
        else:
            render_advisor_roadmap(
                advisor_recommendations,
                catalog_courses,
                st.session_state.get("advisor_query_text", ""),
            )
            with st.expander("All matched courses", expanded=False):
                render_course_grid(
                    advisor_recommendations,
                    show_similarity=True,
                    show_reason=True,
                    key_prefix="advisor",
                )

    chat_request = st.chat_input(
        "Tell me what you want to learn",
        key="advisor_chat_input",
    )
    if chat_request and str(chat_request).strip():
        submit_advisor_request(
            str(chat_request).strip(),
            filtered_courses,
            catalog_courses,
            recommendation_resources,
            active_filter_key,
            rating_sort,
        )


def render_sidebar(courses: pd.DataFrame) -> tuple[str, str, str, str, str, str, list[str]]:
    logo_uri = logo_data_uri()
    if logo_uri:
        st.sidebar.markdown(
            f"""
            <div class="vrl-sidebar-brand">
                <img src="{logo_uri}" alt="VRL logo" />
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("### Filters")
    search_query = st.sidebar.text_input(
        "Search catalog",
        placeholder="Python, leadership, finance",
        key="catalog_search",
    )

    providers = options_from(courses["Provider"])
    if st.session_state.get("course_provider") not in providers:
        st.session_state["course_provider"] = ALL_OPTION
    provider = st.sidebar.selectbox(
        "Course provider",
        providers,
        key="course_provider",
    )

    languages = options_from(courses["Language"])
    if st.session_state.get("course_language") not in languages:
        st.session_state["course_language"] = ALL_OPTION
    language = st.sidebar.selectbox(
        "Spoken language",
        languages,
        key="course_language",
    )

    difficulties = difficulty_options(courses)
    if st.session_state.get("difficulty_level") not in difficulties:
        st.session_state["difficulty_level"] = ALL_OPTION
    difficulty_level = st.sidebar.segmented_control(
        "Difficulty",
        difficulties,
        key="difficulty_level",
        width="stretch",
    )

    university_base = apply_filters(
        courses,
        search_query,
        provider,
        language,
        difficulty_level,
        ALL_OPTION,
        st.session_state.get("selected_skills", []),
    )
    universities = options_from(university_base["University"])
    if st.session_state.get("university") not in universities:
        st.session_state["university"] = ALL_OPTION
    university = st.sidebar.selectbox(
        "University / company",
        universities,
        key="university",
    )

    sort_options = [SORT_SIMILARITY, SORT_HIGH_TO_LOW, SORT_LOW_TO_HIGH]
    if st.session_state.get("rating_sort") not in sort_options:
        st.session_state["rating_sort"] = SORT_SIMILARITY
    rating_sort = st.sidebar.selectbox(
        "Recommendation sort",
        sort_options,
        key="rating_sort",
    )

    base_skill_options = top_skill_options(tuple(courses["Skill Tokens"]))
    stored_skills = list(st.session_state.get("selected_skills", []))
    skill_options = [
        *base_skill_options,
        *[skill for skill in stored_skills if skill not in base_skill_options],
    ]
    selected_skills = [
        skill
        for skill in stored_skills
        if skill in skill_options
    ]
    if st.session_state.get("selected_skills") != selected_skills:
        st.session_state["selected_skills"] = selected_skills

    selected_skills = st.sidebar.pills(
        "Skills",
        skill_options,
        selection_mode="multi",
        format_func=format_skill_label,
        key="selected_skills",
        width="stretch",
    )
    selected_skills = list(selected_skills or [])

    st.sidebar.button(
        "Reset filters",
        width="stretch",
        on_click=reset_filters,
    )

    return search_query, provider, language, difficulty_level, university, rating_sort, selected_skills


def render_progress_tracker(courses: pd.DataFrame) -> None:
    render_section_heading(
        "Progress tracker",
        "Completed courses keep practice tasks, checks, and roadmap anchors aligned to the learner's current level.",
    )
    with st.container(border=True):
        course_label_map = courses.set_index("Course Key")["Course Name"].to_dict()
        course_options = courses.sort_values("Course Name")["Course Key"].astype(str).to_list()
        if not course_options:
            st.warning("No courses available to track.")
            return

        anchor = active_practice_course(courses)
        default_key = str(anchor.get("Course Key", "")) if anchor is not None else course_options[0]
        if default_key not in course_options:
            default_key = course_options[0]

        tracker_col, action_col = st.columns([0.72, 0.28], vertical_alignment="bottom")
        with tracker_col:
            selected_key = st.selectbox(
                "Completed course",
                course_options,
                index=course_options.index(default_key),
                format_func=lambda key: course_label_map.get(key, key),
                key="progress_course_picker",
            )
        selected_row = courses[courses["Course Key"].astype(str) == selected_key].iloc[0]
        with action_col:
            st.button(
                "Mark completed",
                type="primary",
                width="stretch",
                key="progress_mark_completed",
                on_click=mark_course_completed,
                args=(selected_key, str(selected_row.get("Course Name", ""))),
            )

        completed = completed_course_frame(courses)
        active_course = active_practice_course(courses)
        active_name = str(active_course.get("Course Name", "No course selected")) if active_course is not None else "No course selected"
        active_level = str(active_course.get("Difficulty Level", "Mixed")) if active_course is not None else "Not set"
        active_skills = ", ".join(skill_labels_for_course(active_course, limit=3)) or "No skill anchor yet"
        st.markdown(
            f"""
            <div class="vrl-progress-summary">
                <div class="vrl-progress-stat">
                    <div class="vrl-progress-label">Completed courses</div>
                    <div class="vrl-progress-value">{len(completed)}</div>
                </div>
                <div class="vrl-progress-stat">
                    <div class="vrl-progress-label">Practice anchor</div>
                    <div class="vrl-progress-value">{escape(active_name)}</div>
                </div>
                <div class="vrl-progress-stat">
                    <div class="vrl-progress-label">Practice level</div>
                    <div class="vrl-progress-value">{escape(active_level)} | {escape(active_skills)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not completed.empty:
            with st.expander("Completed courses", expanded=False):
                display = completed[["Course Name", "Provider", "Language", "Difficulty Level", "Rating"]].copy()
                st.dataframe(display, hide_index=True, width="stretch")


def render_recommend_tab(
    courses: pd.DataFrame,
    filtered_courses: pd.DataFrame,
    recommendation_resources: dict,
    active_filter_key: tuple[str, str, str, str, str, tuple[str, ...]],
    rating_sort: str,
) -> None:
    render_recommend_flow()
    render_section_heading(
        "AI Learning Navigator",
        "Start here when the learner describes a goal, role, domain, or roadmap question in plain language.",
    )
    render_advisor_chat(
        filtered_courses,
        courses,
        recommendation_resources,
        active_filter_key,
        rating_sort,
    )

    st.divider()
    render_section_heading(
        "Course similarity",
        "Use this when there is already a completed, liked, or target course and the next step should stay close to it.",
    )

    course_label_map = filtered_courses.set_index("Course Key")["Course Name"].to_dict()
    course_options = (
        filtered_courses.sort_values("Course Name")["Course Key"].astype(str).to_list()
    )
    if not course_options:
        st.warning("No courses match the selected filters.")
        return

    if st.session_state.get("course_name") not in course_options:
        st.session_state["course_name"] = course_options[0]

    selected_course = st.selectbox(
        "Completed, liked, or target course",
        course_options,
        format_func=lambda key: course_label_map.get(key, key),
        key="course_name",
    )

    selected_row = filtered_courses[
        filtered_courses["Course Key"].astype(str) == str(selected_course)
    ].iloc[0]
    selected_skills = ", ".join(skill_labels_for_course(selected_row, limit=3)) or "No clear skill tags"
    selected_provider = str(selected_row.get("Provider", selected_row.get("University", "Catalog")))
    selected_language = str(selected_row.get("Language", DEFAULT_LANGUAGE))
    selected_level = str(selected_row.get("Difficulty Level", "Mixed"))
    selected_rating = float(selected_row.get("Rating", 0) or 0)

    action_col, note_col = st.columns([0.32, 0.68], vertical_alignment="center")
    with action_col:
        refresh_matches = st.button(
            "Refresh matches",
            type="primary",
            width="stretch",
        )
    with note_col:
        st.markdown(
            f'<div class="vrl-small-note">{len(filtered_courses):,} courses in scope after filters</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="vrl-similarity-summary">
            <div class="vrl-similarity-stat">
                <div class="vrl-similarity-label">Selected course</div>
                <div class="vrl-similarity-value">{escape(str(selected_row.get("Course Name", "")))}</div>
            </div>
            <div class="vrl-similarity-stat">
                <div class="vrl-similarity-label">Source</div>
                <div class="vrl-similarity-value">{escape(selected_provider)} <span class="vrl-similarity-value-muted">|</span> {escape(selected_language)}</div>
            </div>
            <div class="vrl-similarity-stat">
                <div class="vrl-similarity-label">Level</div>
                <div class="vrl-similarity-value">{escape(selected_level)}</div>
            </div>
            <div class="vrl-similarity-stat">
                <div class="vrl-similarity-label">Signals</div>
                <div class="vrl-similarity-value">{selected_rating:.1f}/5 <span class="vrl-similarity-value-muted">|</span> {escape(selected_skills)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_context = (selected_course, active_filter_key)
    context_changed = st.session_state.get("recommendation_context") != active_context
    if context_changed:
        st.session_state["recommendations"] = None

    if refresh_matches or st.session_state.get("recommendations") is None:
        with st.spinner("Matching nearby courses"):
            recommendations = recommend_courses(
                selected_course,
                filtered_courses,
                recommendation_resources,
            )
        st.session_state["recommendations"] = recommendations
        st.session_state["recommendation_context"] = active_context
        if refresh_matches:
            st.toast("Similarity matches refreshed")

    recommendations = st.session_state.get("recommendations")
    if recommendations is None:
        st.markdown(
            '<div class="vrl-empty"><strong>Ready for a match.</strong><br>Select a course to calculate nearby options from the catalog.</div>',
            unsafe_allow_html=True,
        )
        return

    if recommendations.empty:
        st.warning("No similar course is available under the selected filters.")
        return

    sorted_recommendations = sort_recommendations(recommendations, rating_sort)
    render_course_grid(
        sorted_recommendations,
        show_similarity=True,
        key_prefix="recommend",
    )

    st.divider()
    render_progress_tracker(courses)


def render_explore_tab(filtered_courses: pd.DataFrame) -> None:
    st.markdown('<div class="vrl-section-title">Catalog</div>', unsafe_allow_html=True)

    preview = filtered_courses.sort_values(
        by=["Rating", "Course Name"],
        ascending=[False, True],
    ).head(12)
    render_course_grid(preview, key_prefix="explore")

    table = filtered_courses[
        ["Course Name", "Provider", "Language", "University", "Difficulty Level", "Rating", "Course URL"]
    ].sort_values(by=["Rating", "Course Name"], ascending=[False, True])
    st.dataframe(
        table.head(200),
        hide_index=True,
        width="stretch",
    )


def render_shortlist_tab(courses: pd.DataFrame) -> None:
    st.markdown('<div class="vrl-section-title">Shortlist</div>', unsafe_allow_html=True)

    shortlist = st.session_state.get("shortlist", [])
    if not shortlist:
        st.markdown(
            '<div class="vrl-empty">Saved courses will appear here for this session.</div>',
            unsafe_allow_html=True,
        )
        return

    shortlisted_courses = courses[courses["Course Name"].isin(shortlist)]
    render_course_grid(shortlisted_courses, key_prefix="shortlist")


def top_skills_frame(courses: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for skills in courses["Skill Tokens"]:
        counter.update(skills)

    return pd.DataFrame(
        [
            {"Skill": format_skill_label(skill), "Courses": count}
            for skill, count in counter.most_common(limit)
        ]
    )


def render_insights_tab(courses: pd.DataFrame, filtered_courses: pd.DataFrame) -> None:
    st.markdown('<div class="vrl-section-title">Insights</div>', unsafe_allow_html=True)

    if filtered_courses.empty:
        st.warning("No data available for the selected filters.")
        return

    difficulty_frame = (
        filtered_courses["Difficulty Level"]
        .value_counts()
        .rename_axis("Difficulty")
        .reset_index(name="Courses")
    )
    provider_frame = (
        filtered_courses["University"]
        .value_counts()
        .head(10)
        .rename_axis("Provider")
        .reset_index(name="Courses")
    )
    skills_frame = top_skills_frame(filtered_courses)

    chart_col, provider_col = st.columns(2)
    with chart_col:
        st.markdown("##### Difficulty distribution")
        st.bar_chart(difficulty_frame, x="Difficulty", y="Courses", width="stretch")
    with provider_col:
        st.markdown("##### Top providers")
        st.bar_chart(provider_frame, x="Provider", y="Courses", width="stretch")

    skill_col, rating_col = st.columns(2)
    with skill_col:
        st.markdown("##### Top skills")
        st.bar_chart(skills_frame, x="Skill", y="Courses", width="stretch")
    with rating_col:
        st.markdown("##### Rating profile")
        rating_summary = pd.DataFrame(
            {
                "Metric": ["Filtered average", "Catalog average"],
                "Rating": [
                    filtered_courses["Rating"].mean(),
                    courses["Rating"].mean(),
                ],
            }
        )
        st.bar_chart(rating_summary, x="Metric", y="Rating", width="stretch")


def main() -> None:
    inject_styles()
    ensure_session_state()
    show_pending_toast()

    courses = load_courses(file_mtime(DATA_PATH))
    recommendation_resources = load_recommendation_resources(
        tuple(courses["Course Key"]),
        tuple(courses["Search Text"]),
        file_mtime(INDEX_PATH),
    )

    (
        search_query,
        provider,
        language,
        difficulty_level,
        university,
        rating_sort,
        selected_skills,
    ) = render_sidebar(courses)

    filtered_courses = apply_filters(
        courses,
        search_query,
        provider,
        language,
        difficulty_level,
        university,
        selected_skills,
    )
    active_filter_key = filter_key(
        search_query,
        provider,
        language,
        difficulty_level,
        university,
        selected_skills,
    )

    render_header()
    render_metrics(courses, filtered_courses)

    if filtered_courses.empty:
        st.warning("No courses match the selected filters. Try a different combination.")
        st.stop()

    recommend_tab, explore_tab, shortlist_tab, insights_tab = st.tabs(
        ["Navigator", "Explore", "Shortlist", "Insights"]
    )

    with recommend_tab:
        render_recommend_tab(
            courses,
            filtered_courses,
            recommendation_resources,
            active_filter_key,
            rating_sort,
        )

    with explore_tab:
        render_explore_tab(filtered_courses)

    with shortlist_tab:
        render_shortlist_tab(courses)

    with insights_tab:
        render_insights_tab(courses, filtered_courses)


if __name__ == "__main__":
    main()
