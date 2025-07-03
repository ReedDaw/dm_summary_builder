"""
dm_summary_builder.py
─────────────────────
Batch‑generates ~100‑word, neutral‑tone summaries and best‑effort profile‑image
URLs for Change.org Decision‑Maker pages.

INPUT  : CSV with at least column `name` (and optional `summary_web`,
          `image_url` columns)
OUTPUT : Same CSV with new/updated `summary_web` and `image_url` columns

USAGE EXAMPLE
-------------
$ python dm_summary_builder.py \
    --infile  dms_needing_summaries.csv \
    --outfile dm_summaries_completed.csv \
    --model   gpt-4o-mini \
    --max     100               # optional row limit per run

ENV VARS NEEDED
---------------
OPENAI_API_KEY   – ChatGPT / GPT‑4O key
CIVIC_API_KEY    – (optional) Google Civic Information API key (for extra portraits)

DEPENDENCIES
------------
openai>=1.14.0
wikipedia==1.4.0
pandas, requests, tqdm, tenacity, bs4, lxml (for faster BeautifulSoup)

Install:
$ pip install openai wikipedia pandas requests tqdm tenacity bs4 lxml
"""

from __future__ import annotations

import argparse
import os
import re
import textwrap
import urllib.parse
from typing import Optional, List, Tuple

import warnings

# Silence noisy third‑party warnings (LibreSSL + BeautifulSoup)
try:
    import urllib3
    warnings.filterwarnings(
        "ignore", category=urllib3.exceptions.NotOpenSSLWarning
    )
except Exception:
    pass

try:
    import bs4.builder  # noqa: F401 – only for warning filter
    warnings.filterwarnings(
        "ignore", category=bs4.builder.GuessedAtParserWarning
    )
except Exception:
    pass

import pandas as pd
import openai
import wikipedia
import requests
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_MODEL_DEFAULT = "gpt-4o-mini"
WIKI_API = "https://en.wikipedia.org/w/api.php"
COMMONS_FILE_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/{}?width=400"
SESSION = requests.Session()

# ──────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────────────────────────────────────
PROMPT_TMPL = textwrap.dedent(
    """
    You are an impartial civic‑information editor writing a single paragraph
    (~100 words, max 120) about the person named below for a Change.org
    Decision‑Maker profile.

    • Start with current (or most recent) elected or appointed office.
    • Include district/state and party if discoverable.
    • Mention landmark bills, leadership posts, or signature issues.
    • If no longer in office, note that fact neutrally.
    • Do NOT urge action or express opinion.
    • Do NOT invent facts; if sourcing is thin, write a shorter bio.

    Person: {name}
    Wikipedia extract (may be empty): {wiki}
    ---
    ONE neutral third‑person paragraph only:
    """
)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def call_openai(system_prompt: str, user_prompt: str, model: str) -> str:
    """Wrapper with exponential back‑off for transient errors."""
    rsp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=256,
    )
    return rsp.choices[0].message.content.strip()

# ──────────────────────────────────────────────────────────────────────────────
# Wikipedia extract
# ──────────────────────────────────────────────────────────────────────────────

def get_wikipedia_extract(person: str, sentences: int = 3) -> str:
    """Return first *sentences* sentences of Wikipedia summary or ''."""
    try:
        page = wikipedia.page(person, auto_suggest=False, redirect=True, preload=False)
        return " ".join(wikipedia.summary(page.title, sentences=sentences).split())
    except Exception:
        try:
            hit = wikipedia.search(person, results=1)
            if hit:
                return " ".join(wikipedia.summary(hit[0], sentences=sentences).split())
        except Exception:
            pass
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Image‑lookup helpers
# ──────────────────────────────────────────────────────────────────────────────

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(4))
def _wiki_thumbnail(title: str) -> str:
    """Return Wikipedia lead thumbnail URL (empty if none)."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageimages",
        "pithumbsize": 400,
        "format": "json",
    }
    r = SESSION.get(WIKI_API, params=params, timeout=8).json()
    pages = r.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    return page.get("thumbnail", {}).get("source", "")

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(4))
def _wikidata_image(title: str) -> str:
    """Return Commons image URL via Wikidata P18 property (or '')."""
    params = {"action": "query", "titles": title, "prop": "pageprops", "format": "json"}
    item = SESSION.get(WIKI_API, params=params, timeout=8).json()
    page = next(iter(item.get("query", {}).get("pages", {}).values()), {})
    qid = page.get("pageprops", {}).get("wikibase_item")
    if not qid:
        return ""
    wd = SESSION.get(
        "https://www.wikidata.org/w/api.php",
        params={"action": "wbgetclaims", "entity": qid, "property": "P18", "format": "json"},
        timeout=8,
    ).json()
    claims = wd.get("claims", {}).get("P18")
    if not claims:
        return ""
    filename = claims[0]["mainsnak"]["datavalue"]["value"]
    return COMMONS_FILE_URL.format(urllib.parse.quote(filename))

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(4))
def _google_civic_photo(name: str, api_key: str) -> str:
    """Fetch portrait from Google Civic Information API (may return '')."""
    if not api_key:
        return ""
    url = (
        "https://civicinfo.googleapis.com/civicinfo/v2/representatives"
        f"?key={api_key}&includeOffices=true&address={urllib.parse.quote(name)}"
    )
    data = SESSION.get(url, timeout=8).json()
    officials = data.get("officials", [])
    if officials:
        return officials[0].get("photoUrl", "")
    return ""

def get_profile_image(name: str, civic_api_key: Optional[str] = None) -> str:
    """Return best‑effort portrait URL or ''."""
    for grabber in (
        lambda: _wiki_thumbnail(name),
        lambda: _wikidata_image(name),
        lambda: _google_civic_photo(name, civic_api_key),
    ):
        try:
            url = grabber()
            if url:
                return url
        except Exception:
            continue
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_row(idx: int, name: str, civic_key: Optional[str], model: str) -> Tuple[str, str]:
    """Return (bio, image_url) for one DM."""
    wiki = get_wikipedia_extract(name)
    prompt = PROMPT_TMPL.format(name=name, wiki=wiki or "None available")
    bio = call_openai("You are a factual civic‑information editor.", prompt, model)
    bio = re.sub(r"\s+", " ", bio).strip()
    image_url = get_profile_image(name, civic_api_key=civic_key)
    return bio, image_url


def main(infile: str, outfile: str, model: str, limit: Optional[int]):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("❌ OPENAI_API_KEY is missing. Export it or set openai.api_key directly.")

    civic_key = os.getenv("CIVIC_API_KEY")  # optional, may be None

    df = pd.read_csv(infile)

    # Ensure required columns exist
    for col in ("summary_web", "image_url"):
        if col not in df.columns:
            df[col] = ""

    todo_idx = df[(df["summary_web"].isna()) | (df["summary_web"].eq(""))].index
    if limit:
        todo_idx = todo_idx[:limit]

    for idx in tqdm(todo_idx, desc="Generating bios", unit="DM"):
        name = df.at[idx, "name"]
        try:
            bio, img = process_row(idx, name, civic_key, model)
            df.at[idx, "summary_web"] = bio
            if img:
                df.at[idx, "image_url"] = img
        except Exception as e:
            df.at[idx, "summary_web"] = f"ERROR: {e}"
            df.at[idx, "image_url"] = ""

    df.to_csv(outfile, index=False)
    print(f"\n✅ Finished {len(todo_idx)} summaries. → {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DM bios + portraits")
    parser.add_argument("--infile", required=True, help="Input CSV with at least a 'name' column")
    parser.add_argument("--outfile", required=True, help="Output CSV path")
    parser.add_argument("--model", default=OPENAI_MODEL_DEFAULT, help="OpenAI model, default gpt-4o-mini")
    parser.add_argument("--max", type=int, default=None, help="Optional max rows to process")
    args = parser.parse_args()
    main(args.infile, args.outfile, args.model, args.max)
