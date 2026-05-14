"""Refresh the course catalog and build offline recommendation artifacts.

The Streamlit app stays read-only and fast. This script is the controlled,
offline ingestion path for public course catalogs and course pages.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "course_details.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
INDEX_PATH = ARTIFACT_DIR / "recommendation_index.joblib"
QUARANTINE_PATH = REPORT_DIR / "link_quarantine.csv"
REFRESH_REPORT_PATH = REPORT_DIR / "catalog_refresh_report.json"
DEFAULT_LANGUAGE = "English"

APP_COLUMNS = [
    "Course Name",
    "University",
    "Difficulty Level",
    "Rating",
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

USER_AGENT = (
    "Mozilla/5.0 (compatible; VRL-Course-Recommender/1.0; "
    "+https://github.com/vikramsh2002/VRL-E-Learning-Course-Recommender)"
)

COURSE_PATH_RE = re.compile(
    r"^/(learn|projects|specializations|professional-certificates)/[^?#/]+/?$"
)
FUTURELEARN_COURSE_RE = re.compile(r"^/courses/[a-z0-9][a-z0-9-]+/?$")
MICROSOFT_LEARN_RE = re.compile(
    r"^/en-us/(training/[^?#]+|credentials/applied-skills/[^?#/]+)/?$"
)
MIT_OCW_COURSE_RE = re.compile(r"^/courses/[a-z0-9][a-z0-9.-]+/?$")
KAGGLE_LEARN_RE = re.compile(r"^/learn/[a-z0-9][a-z0-9-]+/?$")

TECHNICAL_KEYWORDS = (
    "ai",
    "algorithm",
    "analytics",
    "android",
    "api",
    "application",
    "artificial-intelligence",
    "automation",
    "azure",
    "cloud",
    "code",
    "coding",
    "computation",
    "computational",
    "computer",
    "cyber",
    "data",
    "database",
    "deep-learning",
    "developer",
    "devops",
    "digital",
    "electrical",
    "electronics",
    "engineering",
    "generative",
    "github",
    "infrastructure",
    "intelligence",
    "kubernetes",
    "linux",
    "machine-learning",
    "mathematics",
    "network",
    "optimization",
    "programming",
    "python",
    "robot",
    "security",
    "software",
    "sql",
    "statistics",
    "systems",
    "tensorflow",
    "web",
)

KAGGLE_LEARN_COURSES = [
    ("Intro to Programming", "intro-to-programming", "Programming"),
    ("Python", "python", "Programming"),
    ("Pandas", "pandas", "Data Science"),
    ("Intro to Machine Learning", "intro-to-machine-learning", "Machine Learning"),
    ("Intermediate Machine Learning", "intermediate-machine-learning", "Machine Learning"),
    ("Data Visualization", "data-visualization", "Data Science"),
    ("Feature Engineering", "feature-engineering", "Machine Learning"),
    ("Intro to SQL", "intro-to-sql", "Databases"),
    ("Advanced SQL", "advanced-sql", "Databases"),
    ("Intro to Deep Learning", "intro-to-deep-learning", "Artificial Intelligence"),
    ("Computer Vision", "computer-vision", "Artificial Intelligence"),
    ("Time Series", "time-series", "Data Science"),
    ("Data Cleaning", "data-cleaning", "Data Science"),
    ("Intro to AI Ethics", "intro-to-ai-ethics", "Artificial Intelligence"),
    ("Geospatial Analysis", "geospatial-analysis", "Data Science"),
    ("Machine Learning Explainability", "machine-learning-explainability", "Machine Learning"),
    ("Intro to Game AI and Reinforcement Learning", "intro-to-game-ai-and-reinforcement-learning", "Artificial Intelligence"),
]


@dataclass(frozen=True)
class LinkItem:
    url: str
    text: str


class AnchorParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._href_stack: list[str | None] = []
        self._text_stack: list[list[str]] = []
        self.links: list[LinkItem] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return

        attr_map = dict(attrs)
        href = attr_map.get("href")
        self._href_stack.append(href)
        self._text_stack.append([])

    def handle_data(self, data: str) -> None:
        if self._text_stack:
            self._text_stack[-1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._href_stack:
            return

        href = self._href_stack.pop()
        text_parts = self._text_stack.pop()
        text = normalize_text(" ".join(text_parts))
        if href and text:
            self.links.append(LinkItem(urllib.parse.urljoin(self.base_url, href), text))


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


SCRIPT_LANGUAGE_RANGES = (
    ("Japanese", ((0x3040, 0x309F), (0x30A0, 0x30FF))),
    ("Korean", ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))),
    ("Arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF))),
    ("Hindi", ((0x0900, 0x097F),)),
    ("Hebrew", ((0x0590, 0x05FF),)),
    ("Thai", ((0x0E00, 0x0E7F),)),
    ("Greek", ((0x0370, 0x03FF),)),
    ("Russian", ((0x0400, 0x04FF),)),
)

LATIN_LANGUAGE_TERMS = {
    "Spanish": {
        "análisis",
        "atención",
        "cálculo",
        "claves",
        "comercio",
        "conceptos",
        "conformidad",
        "datos",
        "diseño",
        "domina",
        "eléctricos",
        "enfermo",
        "escala",
        "fundamentos",
        "gestión",
        "introducción",
        "inmigración",
        "movilidad",
        "niños",
        "prehospitalaria",
        "precisión",
        "protección",
        "público",
        "semicrítico",
        "reciclaje",
        "residuos",
        "salud",
        "técnica",
        "sostenible",
        "transformación",
        "vehículos",
    },
    "French": {
        "archéologie",
        "associés",
        "avènement",
        "contenu",
        "création",
        "créer",
        "données",
        "étudiants",
        "informatique",
        "mise",
        "octets",
        "païens",
        "publicités",
        "réseau",
        "réseaux",
        "visuelle",
    },
    "Portuguese": {
        "centrada",
        "cliente",
        "complexidade",
        "educação",
        "esportivas",
        "federações",
        "gerente",
        "gestão",
        "jornada",
        "liderança",
        "organização",
        "prática",
        "simulações",
    },
    "Hungarian": {
        "eszközök",
        "hatékony",
        "megbirkózni",
        "mentális",
        "melyek",
        "segítenek",
        "tanulás",
        "tantárgyakkal",
    },
    "German": {
        "einführung",
        "grundlagen",
    },
    "Italian": {
        "introduzione",
        "programmazione",
    },
}


def character_count_in_ranges(text: str, ranges: Iterable[tuple[int, int]]) -> int:
    return sum(
        1
        for char in text
        if any(start <= ord(char) <= end for start, end in ranges)
    )


def detect_course_language(title: object, fallback: str = DEFAULT_LANGUAGE) -> str:
    text = normalize_text(title)
    if not text:
        return fallback or DEFAULT_LANGUAGE

    for language, ranges in SCRIPT_LANGUAGE_RANGES:
        if character_count_in_ranges(text, ranges):
            return language

    cjk_count = character_count_in_ranges(text, ((0x4E00, 0x9FFF), (0x3400, 0x4DBF)))
    if cjk_count:
        return "Chinese"

    lowered = text.casefold()
    if " à " in f" {lowered} ":
        return "French"

    tokens = set(re.findall(r"[a-zÀ-ÖØ-öø-ÿ]+", lowered))
    scores = {
        language: len(tokens.intersection(terms))
        for language, terms in LATIN_LANGUAGE_TERMS.items()
    }
    language, score = max(scores.items(), key=lambda item: item[1])
    if score >= 2 or (score == 1 and any(char in lowered for char in "áéíóúñçãõàèìòùâêîôûäëïöüőű")):
        return language

    return fallback or DEFAULT_LANGUAGE


def canonical_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(str(url).strip())
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = re.sub(r"/+$", "", parsed.path)
    return urllib.parse.urlunsplit((scheme, netloc, path, "", ""))


def slug_key(provider: str, url: str, title: str) -> str:
    canonical = canonical_url(url)
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:14]
    provider_slug = re.sub(r"[^a-z0-9]+", "-", provider.lower()).strip("-")
    title_slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:42]
    title_slug = title_slug or "course"
    return f"{provider_slug}:{title_slug}:{digest}"


def split_title_org(text: str) -> tuple[str, str]:
    match = re.match(r"(?P<title>.+?)\s+\((?P<org>[^()]{2,120})\)$", text)
    if match:
        return normalize_text(match.group("title")), normalize_text(match.group("org"))
    return normalize_text(text), ""


def parse_rating(text: str, default: float = 0.0) -> float:
    match = re.search(r"\b([0-5]\.\d)\b", text)
    if not match:
        return default
    return float(match.group(1))


def infer_difficulty(text: str, default: str = "Mixed_Difficulty") -> str:
    lowered = text.lower()
    if "beginner" in lowered or "introduct" in lowered:
        return "Beginner"
    if "advanced" in lowered or "expert" in lowered:
        return "Advanced"
    if "intermediate" in lowered:
        return "Intermediate"
    return default


def strip_html(text: object) -> str:
    return normalize_text(re.sub(r"<[^>]+>", " ", str(text or "")))


def technical_score(text: str) -> int:
    lowered = text.lower().replace(" ", "-")
    return sum(1 for keyword in TECHNICAL_KEYWORDS if keyword in lowered)


def title_from_slug(slug: str) -> str:
    title = re.sub(r"-(spring|fall|summer|january-iap)-\d{4}$", "", slug)
    title = re.sub(r"-(spring|fall|summer|january-iap)$", "", title)
    title = re.sub(r"-\d{4}$", "", title)
    return normalize_text(title.replace("-", " ").title())


def mit_title_from_url(url: str) -> str:
    slug = urllib.parse.urlsplit(url).path.strip("/").split("/", 1)[1]
    parts = slug.split("-")
    while parts and (any(char.isdigit() for char in parts[0]) or parts[0] in {"res", "mas", "es", "hst", "sts", "ids", "ec", "cms"}):
        parts.pop(0)
    return title_from_slug("-".join(parts) or slug)


def mit_category_from_slug(slug: str) -> str:
    if "artificial-intelligence" in slug or "machine-learning" in slug or "deep-learning" in slug:
        return "Artificial Intelligence"
    if "data" in slug or "statistics" in slug or "analytics" in slug:
        return "Data Science"
    if "computer" in slug or "software" in slug or "programming" in slug or "algorithm" in slug:
        return "Computer Science"
    if "electrical" in slug or "engineering" in slug:
        return "Engineering"
    return "OpenCourseWare"


def difficulty_from_levels(levels: Iterable[str]) -> str:
    joined = " ".join(levels).lower()
    if "beginner" in joined:
        return "Beginner"
    if "advanced" in joined:
        return "Advanced"
    if "intermediate" in joined:
        return "Intermediate"
    return "Mixed_Difficulty"


def make_record(
    *,
    provider: str,
    title: str,
    university: str,
    difficulty: str,
    rating: float,
    url: str,
    description: str,
    skills: str,
    category: str,
    language: str = DEFAULT_LANGUAGE,
    verified: str = "",
) -> dict[str, object]:
    title = normalize_text(title)
    university = normalize_text(university) or provider
    description = normalize_text(description) or title
    skills = normalize_text(skills) or category or provider
    category = normalize_text(category) or provider
    language = normalize_text(language)
    if not language or language == DEFAULT_LANGUAGE:
        language = detect_course_language(title, language or DEFAULT_LANGUAGE)
    url = canonical_url(url)
    tags = normalize_text(
        f"{title} {description} taught by {university}. "
        f"The provider is {provider}. The level of course is {difficulty}. "
        f"The spoken language is {language}. You will learn {skills} {category}."
    ).lower()
    return {
        "Course Name": title,
        "University": university,
        "Difficulty Level": difficulty,
        "Rating": float(rating or 0.0),
        "Course URL": url,
        "Course Description": description,
        "Skills": skills,
        "Tags": tags,
        "Language": language,
        "Provider": provider,
        "Category": category,
        "Course Key": slug_key(provider, url, title),
        "Last Verified": verified,
    }


def fetch_text(url: str, *, timeout: int = 20, retries: int = 2) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(content_type, errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Unable to fetch {url}: {last_error}")


def parse_links(url: str, html_text: str) -> list[LinkItem]:
    parser = AnchorParser(url)
    parser.feed(html_text)
    return parser.links


def existing_records() -> list[dict[str, object]]:
    df = pd.read_csv(DATA_PATH, index_col=0)
    for column in APP_COLUMNS:
        if column not in df.columns:
            if column == "Provider":
                df[column] = "Coursera"
            elif column == "Category":
                df[column] = "Coursera"
            elif column == "Language":
                df[column] = DEFAULT_LANGUAGE
            elif column == "Course Key":
                df[column] = [
                    slug_key("Coursera", row["Course URL"], row["Course Name"])
                    for _, row in df.iterrows()
                ]
            elif column == "Last Verified":
                df[column] = ""
            else:
                df[column] = ""
    df = df[APP_COLUMNS].copy()
    df["Provider"] = df["Provider"].fillna("").replace("", "Coursera")
    df["Category"] = df["Category"].fillna("").replace("", "Coursera")
    df["Language"] = df["Language"].fillna("").replace("", DEFAULT_LANGUAGE)
    df["Language"] = [
        detect_course_language(row["Course Name"], row["Language"])
        if row["Language"] == DEFAULT_LANGUAGE
        else row["Language"]
        for _, row in df.iterrows()
    ]
    df["Course URL"] = df["Course URL"].map(canonical_url)
    df["Course Key"] = [
        row["Course Key"] or slug_key(row["Provider"], row["Course URL"], row["Course Name"])
        for _, row in df.iterrows()
    ]
    return df.to_dict("records")


def discover_coursera(max_pages: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        url = "https://www.coursera.org/directory/courses"
        if page > 1:
            url = f"{url}?page={page}"
        try:
            links = parse_links(url, fetch_text(url))
        except RuntimeError as exc:
            print(exc)
            continue

        for link in links:
            parsed = urllib.parse.urlsplit(link.url)
            if parsed.netloc and parsed.netloc != "www.coursera.org":
                continue
            if not COURSE_PATH_RE.match(parsed.path):
                continue
            canonical = canonical_url(link.url)
            if canonical in seen:
                continue
            seen.add(canonical)
            title, org = split_title_org(link.text)
            path_kind = parsed.path.strip("/").split("/", 1)[0].replace("-", " ").title()
            records.append(
                make_record(
                    provider="Coursera",
                    title=title,
                    university=org or "Coursera",
                    difficulty=infer_difficulty(title),
                    rating=0.0,
                    url=canonical,
                    description=f"{title} from {org or 'Coursera'} on Coursera.",
                    skills=title,
                    category=path_kind,
                )
            )
        time.sleep(0.2)
    return records


def futurelearn_title_parts(text: str) -> tuple[str, str, float, str]:
    cleaned = re.sub(r"\bFind out more\b", " ", text)
    cleaned = re.sub(r"\bShort Course\b", " ", cleaned)
    cleaned = normalize_text(cleaned)
    rating = parse_rating(cleaned, 0.0)
    cleaned = re.sub(r"\b[0-5]\.\d\s*\(\s*[\d,]+\s*reviews?\s*\).*", "", cleaned)
    cleaned = re.sub(r"\b\d+\s+weeks?.*", "", cleaned)
    cleaned = normalize_text(cleaned)
    if not cleaned:
        return "", "FutureLearn", rating, "FutureLearn"
    words = cleaned.split()
    if len(words) > 5:
        university = " ".join(words[: min(4, len(words) // 3)])
        title = " ".join(words[min(4, len(words) // 3) :])
    else:
        university = "FutureLearn"
        title = cleaned
    return title or cleaned, university or "FutureLearn", rating, "FutureLearn"


def enrich_futurelearn_record(record: dict[str, object]) -> dict[str, object]:
    try:
        page = fetch_text(str(record["Course URL"]), timeout=16, retries=1)
    except RuntimeError:
        return record

    text = normalize_text(re.sub(r"<[^>]+>", " ", page))
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", page, flags=re.I | re.S)
    if h1_match:
        record["Course Name"] = normalize_text(re.sub(r"<[^>]+>", " ", h1_match.group(1)))
    else:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", page, flags=re.I | re.S)
        if title_match:
            page_title = normalize_text(title_match.group(1))
            page_title = re.sub(
                r"\s+-\s+(Free\s+)?Online Course\s+-\s+FutureLearn\s*$",
                "",
                page_title,
                flags=re.I,
            )
            page_title = re.sub(r"\s+\|\s+FutureLearn\s*$", "", page_title, flags=re.I)
            if page_title:
                record["Course Name"] = page_title

    rating = parse_rating(text, float(record["Rating"]))
    record["Rating"] = rating
    record["Difficulty Level"] = infer_difficulty(text)

    for meta_tag in re.findall(r"<meta\s+[^>]*>", page, flags=re.I | re.S):
        if "description" not in meta_tag.lower():
            continue
        content_match = re.search(r"content=[\"'](.*?)[\"']", meta_tag, flags=re.I | re.S)
        if content_match:
            record["Course Description"] = normalize_text(content_match.group(1))
            break

    category = "FutureLearn"
    crumbs = re.findall(r"/courses/categories/[^\"']+[^>]*>(.*?)</a>", page, flags=re.I | re.S)
    if crumbs:
        category = normalize_text(re.sub(r"<[^>]+>", " ", crumbs[-1]))
    record["Category"] = category
    record["Skills"] = normalize_text(f"{record['Course Name']} {category}")
    refreshed = make_record(
        provider="FutureLearn",
        title=str(record["Course Name"]),
        university=str(record["University"]),
        difficulty=str(record["Difficulty Level"]),
        rating=float(record["Rating"]),
        url=str(record["Course URL"]),
        description=str(record["Course Description"]),
        skills=str(record["Skills"]),
        category=str(record["Category"]),
        verified=str(record["Last Verified"]),
    )
    return refreshed


def enrich_existing_futurelearn_records(
    records: list[dict[str, object]],
    detail_limit: int,
) -> list[dict[str, object]]:
    if detail_limit <= 0:
        return records

    refreshed: list[dict[str, object]] = []
    enriched_count = 0
    for record in records:
        if record.get("Provider") == "FutureLearn" and enriched_count < detail_limit:
            refreshed.append(enrich_futurelearn_record(record))
            enriched_count += 1
            time.sleep(0.2)
        else:
            refreshed.append(record)
    return refreshed


def refresh_existing_provider_metadata(records: list[dict[str, object]]) -> list[dict[str, object]]:
    refreshed: list[dict[str, object]] = []
    for record in records:
        updated = {**record}
        if updated.get("Provider") == "MIT OpenCourseWare":
            url = canonical_url(str(updated.get("Course URL", "")))
            slug = urllib.parse.urlsplit(url).path.strip("/").split("/", 1)[-1]
            title = mit_title_from_url(url)
            updated["Course Name"] = title
            updated["Course Description"] = f"MIT OpenCourseWare materials for {title}."
            updated["Skills"] = title
            updated["Category"] = mit_category_from_slug(slug)
            updated["Course Key"] = slug_key("MIT OpenCourseWare", url, title)
        refreshed.append(updated)
    return refreshed


def discover_futurelearn(max_pages: int, detail_limit: int) -> list[dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for page in range(1, max_pages + 1):
        url = "https://www.futurelearn.com/courses"
        if page > 1:
            url = f"{url}?page={page}"
        try:
            links = parse_links(url, fetch_text(url))
        except RuntimeError as exc:
            print(exc)
            continue
        for link in links:
            parsed = urllib.parse.urlsplit(link.url)
            if parsed.netloc and parsed.netloc != "www.futurelearn.com":
                continue
            if not FUTURELEARN_COURSE_RE.match(parsed.path):
                continue
            canonical = canonical_url(link.url)
            if link.text.lower() in {"find out more", "join course"}:
                continue
            title, org, rating, category = futurelearn_title_parts(link.text)
            if not title or len(title) < 4:
                continue
            previous = records.get(canonical)
            if previous and len(str(previous["Course Description"])) >= len(link.text):
                continue
            records[canonical] = make_record(
                provider="FutureLearn",
                title=title,
                university=org,
                difficulty=infer_difficulty(link.text),
                rating=rating,
                url=canonical,
                description=link.text,
                skills=title,
                category=category,
            )
        time.sleep(0.2)

    result = list(records.values())
    if detail_limit > 0:
        enriched: list[dict[str, object]] = []
        for record in result[:detail_limit]:
            enriched.append(enrich_futurelearn_record(record))
            time.sleep(0.2)
        result = enriched + result[detail_limit:]
    return result


def discover_microsoft_learn(limit: int) -> list[dict[str, object]]:
    if limit <= 0:
        return []

    catalog_url = "https://learn.microsoft.com/api/catalog/?locale=en-us"
    try:
        data = json.loads(fetch_text(catalog_url, timeout=40, retries=2))
    except RuntimeError as exc:
        print(exc)
        return []

    candidates: list[tuple[float, dict[str, object]]] = []
    for section, category_label in (
        ("learningPaths", "Learning Path"),
        ("modules", "Module"),
        ("courses", "Instructor-Led Course"),
        ("appliedSkills", "Applied Skill"),
    ):
        for item in data.get(section, []):
            title = normalize_text(item.get("title", ""))
            url = canonical_url(urllib.parse.urljoin("https://learn.microsoft.com", str(item.get("url", ""))))
            if not title or not url:
                continue

            subjects = item.get("subjects") or []
            products = item.get("products") or []
            roles = item.get("roles") or []
            summary = strip_html(item.get("summary", ""))
            skill_text = " ".join([*subjects, *products, *roles]).replace("-", " ")
            score = technical_score(f"{title} {summary} {skill_text}")
            if score <= 0:
                continue

            popularity = float(item.get("popularity") or 0.0)
            duration_minutes = item.get("duration_in_minutes")
            duration_hours = item.get("duration_in_hours")
            if duration_minutes:
                summary = normalize_text(f"{summary} Duration: {duration_minutes} minutes.")
            elif duration_hours:
                summary = normalize_text(f"{summary} Duration: {duration_hours} hours.")

            record = make_record(
                provider="Microsoft Learn",
                title=title,
                university="Microsoft",
                difficulty=difficulty_from_levels(item.get("levels") or ()),
                rating=0.0,
                url=url,
                description=summary or f"{title} on Microsoft Learn.",
                skills=skill_text or title,
                category=", ".join(subjects) or category_label,
                verified="",
            )
            candidates.append((score + popularity, record))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [record for _, record in candidates[:limit]]


def discover_mit_ocw(limit: int) -> list[dict[str, object]]:
    if limit <= 0:
        return []

    try:
        sitemap = fetch_text("https://ocw.mit.edu/sitemap.xml", timeout=40, retries=2)
    except RuntimeError as exc:
        print(exc)
        return []

    urls = [
        canonical_url(match.group(1).replace("/sitemap.xml", ""))
        for match in re.finditer(r"<loc>(https://ocw\.mit\.edu/courses/[^<]+/sitemap\.xml)</loc>", sitemap)
    ]
    candidates: list[tuple[int, dict[str, object]]] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        slug = urllib.parse.urlsplit(url).path.strip("/").split("/", 1)[1]
        score = technical_score(slug)
        if score <= 0:
            continue
        title = mit_title_from_url(url)
        category = mit_category_from_slug(slug)
        record = make_record(
            provider="MIT OpenCourseWare",
            title=title,
            university="MIT OpenCourseWare",
            difficulty=infer_difficulty(slug),
            rating=0.0,
            url=url,
            description=f"MIT OpenCourseWare materials for {title}.",
            skills=title,
            category=category,
            verified="",
        )
        candidates.append((score, record))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [record for _, record in candidates[:limit]]


def discover_kaggle_learn(enabled: bool) -> list[dict[str, object]]:
    if not enabled:
        return []

    records = []
    for title, slug, category in KAGGLE_LEARN_COURSES:
        url = f"https://www.kaggle.com/learn/{slug}"
        records.append(
            make_record(
                provider="Kaggle Learn",
                title=title,
                university="Kaggle",
                difficulty=infer_difficulty(title, default="Beginner"),
                rating=0.0,
                url=url,
                description=f"Hands-on Kaggle Learn micro-course for {title}.",
                skills=f"{title} {category} Kaggle notebooks exercises",
                category=category,
                verified="",
            )
        )
    return records


def validate_url(url: str) -> tuple[bool, str, int | None]:
    head_status: int | None = None
    try:
        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
        )
        with urllib.request.urlopen(request, timeout=12) as response:
            head_status = int(response.status)
    except urllib.error.HTTPError as exc:
        head_status = int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        head_status = None

    if head_status and head_status >= 400:
        return False, url, head_status

    try:
        request = urllib.request.Request(
            url,
            method="GET",
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
        )
        with urllib.request.urlopen(request, timeout=16) as response:
            final_url = canonical_url(response.geturl())
            status = int(response.status)
            body = response.read(350000).decode(
                response.headers.get_content_charset() or "utf-8",
                errors="replace",
            )
    except urllib.error.HTTPError as exc:
        return False, url, int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False, url, None

    if not page_content_is_available(url, final_url, status, body):
        return False, final_url, status

    return True, final_url, status


def validate_url_fast(url: str) -> tuple[bool, str, int | None]:
    try:
        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            final_url = canonical_url(response.geturl())
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        return False, url, int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        try:
            request = urllib.request.Request(
                url,
                method="GET",
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
            )
            with urllib.request.urlopen(request, timeout=12) as response:
                final_url = canonical_url(response.geturl())
                status = int(response.status)
        except urllib.error.HTTPError as exc:
            return False, url, int(exc.code)
        except (urllib.error.URLError, TimeoutError, OSError):
            return False, url, None

    parsed_original = urllib.parse.urlsplit(url)
    parsed_final = urllib.parse.urlsplit(final_url)
    if status >= 400:
        return False, url, status
    if "coursera.org" in parsed_original.netloc:
        if parsed_final.netloc != "www.coursera.org" or not COURSE_PATH_RE.match(parsed_final.path):
            return False, final_url, status
    if "futurelearn.com" in parsed_original.netloc:
        if parsed_final.netloc != "www.futurelearn.com" or not FUTURELEARN_COURSE_RE.match(parsed_final.path):
            return False, final_url, status
    if "learn.microsoft.com" in parsed_original.netloc:
        if parsed_final.netloc != "learn.microsoft.com" or not MICROSOFT_LEARN_RE.match(parsed_final.path):
            return False, final_url, status
    if "ocw.mit.edu" in parsed_original.netloc:
        if parsed_final.netloc != "ocw.mit.edu" or not MIT_OCW_COURSE_RE.match(parsed_final.path):
            return False, final_url, status
    if "kaggle.com" in parsed_original.netloc:
        if parsed_final.netloc not in {"www.kaggle.com", "kaggle.com"} or not KAGGLE_LEARN_RE.match(parsed_final.path):
            return False, final_url, status
    return True, final_url, status


def page_content_is_available(
    original_url: str,
    final_url: str,
    status: int,
    body: str,
) -> bool:
    if status != 200:
        return False

    parsed_original = urllib.parse.urlsplit(original_url)
    parsed_final = urllib.parse.urlsplit(final_url)
    lowered_body = body.lower()
    lowered_title_match = re.search(r"<title[^>]*>(.*?)</title>", lowered_body, flags=re.I | re.S)
    lowered_title = normalize_text(lowered_title_match.group(1)) if lowered_title_match else ""

    if "coursera.org" in parsed_original.netloc:
        if parsed_final.netloc != "www.coursera.org":
            return False
        if not COURSE_PATH_RE.match(parsed_final.path):
            return False
        unavailable_terms = (
            "page not found",
            "not found | coursera",
            "we can't find the page",
            "we can’t find the page",
            "course is no longer available",
            "course no longer available",
            "this course is not available",
            "this course is currently unavailable",
        )
        if any(term in lowered_title or term in lowered_body for term in unavailable_terms):
            return False

    if "futurelearn.com" in parsed_original.netloc:
        if parsed_final.netloc != "www.futurelearn.com":
            return False
        if not FUTURELEARN_COURSE_RE.match(parsed_final.path):
            return False
        unavailable_terms = (
            "page not found",
            "course not found",
            "no longer available",
            "this course is not currently available",
        )
        if any(term in lowered_title or term in lowered_body for term in unavailable_terms):
            return False

    if "learn.microsoft.com" in parsed_original.netloc:
        if parsed_final.netloc != "learn.microsoft.com":
            return False
        if not MICROSOFT_LEARN_RE.match(parsed_final.path):
            return False
        unavailable_terms = (
            "404 - content not found",
            "content not found",
            "page not found",
            "this content is no longer available",
        )
        if any(term in lowered_title or term in lowered_body for term in unavailable_terms):
            return False

    if "ocw.mit.edu" in parsed_original.netloc:
        if parsed_final.netloc != "ocw.mit.edu":
            return False
        if not MIT_OCW_COURSE_RE.match(parsed_final.path):
            return False
        unavailable_terms = (
            "page not found",
            "the page you are looking for",
            "course not found",
        )
        if any(term in lowered_title or term in lowered_body for term in unavailable_terms):
            return False

    if "kaggle.com" in parsed_original.netloc:
        if parsed_final.netloc not in {"www.kaggle.com", "kaggle.com"}:
            return False
        if not KAGGLE_LEARN_RE.match(parsed_final.path):
            return False
        unavailable_terms = (
            "404 - page not found",
            "page not found",
            "we can't find the page",
        )
        if any(term in lowered_title or term in lowered_body for term in unavailable_terms):
            return False

    return True


def source_backed_link_is_publishable(
    record: dict[str, object],
    checked_url: str,
    status: int | None,
) -> bool:
    """Accept trusted catalog links when providers throttle scripted checks."""
    provider = str(record.get("Provider", ""))
    parsed = urllib.parse.urlsplit(checked_url)

    if provider == "Microsoft Learn":
        return (
            status == 429
            and parsed.netloc == "learn.microsoft.com"
            and MICROSOFT_LEARN_RE.match(parsed.path) is not None
        )

    if provider == "Kaggle Learn":
        known_paths = {f"/learn/{slug}" for _, slug, _ in KAGGLE_LEARN_COURSES}
        return (
            status in {403, 404, 429}
            and parsed.netloc in {"www.kaggle.com", "kaggle.com"}
            and parsed.path in known_paths
        )

    return False


def validate_new_records(records: list[dict[str, object]], workers: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    published: list[dict[str, object]] = []
    quarantined: list[dict[str, object]] = []
    today = date.today().isoformat()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(validate_url, str(record["Course URL"])): record for record in records}
        for future in concurrent.futures.as_completed(future_map):
            record = future_map[future]
            ok, final_url, status = future.result()
            if not ok and source_backed_link_is_publishable(record, canonical_url(final_url), status):
                ok = True
                final_url = canonical_url(final_url)
            if ok:
                record["Course URL"] = final_url
                record["Last Verified"] = today
                record["Course Key"] = slug_key(str(record["Provider"]), final_url, str(record["Course Name"]))
                published.append(record)
            else:
                quarantined.append({**record, "Status": status or "unreachable"})
    return published, quarantined


def validate_existing_records(
    records: list[dict[str, object]],
    workers: int,
    *,
    deep_ui_check: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    published: list[dict[str, object]] = []
    quarantined: list[dict[str, object]] = []
    today = date.today().isoformat()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        validator = validate_url if deep_ui_check else validate_url_fast
        future_map = {executor.submit(validator, str(record["Course URL"])): record for record in records}
        for future in concurrent.futures.as_completed(future_map):
            record = future_map[future]
            ok, final_url, status = future.result()
            if ok:
                record["Course URL"] = final_url
                record["Last Verified"] = today
                record["Course Key"] = slug_key(str(record["Provider"]), final_url, str(record["Course Name"]))
                published.append(record)
            else:
                quarantined.append({**record, "Status": status or "unreachable"})
    return published, quarantined


def dedupe(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    by_url: dict[str, dict[str, object]] = {}
    by_title_org: set[tuple[str, str, str]] = set()
    for record in records:
        url = canonical_url(str(record["Course URL"]))
        if not url:
            continue
        title = normalize_text(record["Course Name"]).lower()
        org = normalize_text(record["University"]).lower()
        provider = normalize_text(record["Provider"]).lower()
        title_key = (provider, title, org)
        if url in by_url or title_key in by_title_org:
            continue
        record["Course URL"] = url
        by_url[url] = record
        by_title_org.add(title_key)
    return list(by_url.values())


def text_for_index(row: pd.Series) -> str:
    return " ".join(
        normalize_text(row.get(column, ""))
        for column in (
            "Course Name",
            "Course Description",
            "Skills",
            "Tags",
            "Language",
            "Category",
            "University",
            "Provider",
            "Difficulty Level",
        )
    )


def build_index(df: pd.DataFrame, top_k: int) -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    texts = df.apply(text_for_index, axis=1).tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=25000,
        min_df=1,
    )
    vectors = vectorizer.fit_transform(texts)
    n_neighbors = min(top_k + 1, len(df))
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    model.fit(vectors)
    distances, indices = model.kneighbors(vectors, return_distance=True)
    similarities = 1.0 - distances
    joblib.dump(
        {
            "schema_version": 1,
            "created_at": date.today().isoformat(),
            "course_keys": df["Course Key"].astype(str).tolist(),
            "vectorizer": vectorizer,
            "vectors": vectors,
            "neighbor_indices": indices,
            "neighbor_similarities": similarities,
        },
        INDEX_PATH,
        compress=3,
    )


def write_reports(quarantined: list[dict[str, object]], report: dict[str, object]) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    if quarantined:
        with QUARANTINE_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*APP_COLUMNS, "Status"])
            writer.writeheader()
            writer.writerows(quarantined)
    else:
        QUARANTINE_PATH.write_text("", encoding="utf-8")
    REFRESH_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coursera-pages", type=int, default=25)
    parser.add_argument("--futurelearn-pages", type=int, default=25)
    parser.add_argument("--futurelearn-detail-limit", type=int, default=80)
    parser.add_argument("--microsoft-learn-limit", type=int, default=700)
    parser.add_argument("--mit-ocw-limit", type=int, default=250)
    parser.set_defaults(kaggle_learn=True)
    parser.add_argument(
        "--kaggle-learn",
        dest="kaggle_learn",
        action="store_true",
        help="Include the curated public Kaggle Learn technical micro-course catalog.",
    )
    parser.add_argument(
        "--skip-kaggle-learn",
        dest="kaggle_learn",
        action="store_false",
        help="Skip the curated Kaggle Learn technical micro-course catalog.",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate existing baseline links too using fast HTTP/redirect checks.",
    )
    parser.add_argument(
        "--deep-existing-ui-check",
        action="store_true",
        help="When validating existing links, fetch page bodies and reject provider UI not-found/unavailable pages.",
    )
    parser.add_argument(
        "--max-existing-failure-rate",
        type=float,
        default=0.15,
        help="Safety guard for full audits. If more existing links fail than this fraction, retain them and report instead.",
    )
    args = parser.parse_args()

    baseline = existing_records()
    baseline = enrich_existing_futurelearn_records(
        baseline,
        args.futurelearn_detail_limit,
    )
    baseline = refresh_existing_provider_metadata(baseline)
    baseline_count = len(baseline)
    print(f"Loaded {baseline_count} baseline records")

    coursera = discover_coursera(args.coursera_pages)
    futurelearn = discover_futurelearn(args.futurelearn_pages, args.futurelearn_detail_limit)
    microsoft_learn = discover_microsoft_learn(args.microsoft_learn_limit)
    mit_ocw = discover_mit_ocw(args.mit_ocw_limit)
    kaggle_learn = discover_kaggle_learn(args.kaggle_learn)
    print(
        "Discovered "
        f"{len(coursera)} Coursera, "
        f"{len(futurelearn)} FutureLearn, "
        f"{len(microsoft_learn)} Microsoft Learn, "
        f"{len(mit_ocw)} MIT OpenCourseWare, and "
        f"{len(kaggle_learn)} Kaggle Learn candidates"
    )

    existing_urls = {canonical_url(str(record["Course URL"])) for record in baseline}
    candidates = [
        record
        for record in dedupe([*coursera, *futurelearn, *microsoft_learn, *mit_ocw, *kaggle_learn])
        if canonical_url(str(record["Course URL"])) not in existing_urls
    ]
    print(f"Validating {len(candidates)} new candidate links")
    valid_new, quarantined = validate_new_records(candidates, args.workers)

    retained_unverified_existing = 0
    if args.validate_existing:
        valid_existing, existing_quarantine = validate_existing_records(
            baseline,
            args.workers,
            deep_ui_check=args.deep_existing_ui_check,
        )
        existing_failure_rate = len(existing_quarantine) / max(1, baseline_count)
        if existing_failure_rate <= args.max_existing_failure_rate:
            baseline = valid_existing
            quarantined.extend(existing_quarantine)
        else:
            retained_unverified_existing = len(existing_quarantine)

    final_records = dedupe([*baseline, *valid_new])
    df = pd.DataFrame(final_records, columns=APP_COLUMNS)
    df = df[df["Course URL"].fillna("").astype(str).str.strip() != ""].copy()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0.0)
    df["Language"] = [
        detect_course_language(row["Course Name"], row["Language"])
        for _, row in df.iterrows()
    ]
    df = df.sort_values(by=["Provider", "Course Name"], kind="stable").reset_index(drop=True)
    df.to_csv(DATA_PATH)
    build_index(df, args.top_k)

    report = {
        "baseline_records": baseline_count,
        "published_records": len(df),
        "new_records_added": max(0, len(df) - baseline_count),
        "quarantined_new_records": len(quarantined),
        "retained_unverified_existing_records": retained_unverified_existing,
        "providers": df["Provider"].value_counts().to_dict(),
        "languages": df["Language"].value_counts().to_dict(),
        "index_path": str(INDEX_PATH.relative_to(ROOT)),
        "data_path": str(DATA_PATH.relative_to(ROOT)),
    }
    write_reports(quarantined, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
