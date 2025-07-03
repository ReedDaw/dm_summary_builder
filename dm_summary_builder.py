"""
dm_summary_builder.py
─────────────────────
Batch-generates ~100-word, neutral-tone summaries for Change.org
Decision-Maker pages, matching the style now live on /decision-makers/*.

INPUT  : CSV with at least 2 columns – `name`  and an optional `summary`
OUTPUT : Same CSV plus a new `summary_web` column
REQUIRES:
  • openai>=1.14.0      (ChatGPT or GPT-4o)
  • wikipedia==1.4.0    (MediaWiki REST search)
  • python-env-config   (or set OPENAI_API_KEY in shell)
  • pandas, tqdm, tenacity
---------------------------------------------------------------------
Run example:
$ python dm_summary_builder.py \
    --infile  dms_needing_summaries.csv \
    --outfile dm_summaries_completed.csv \
    --model   gpt-4o-mini \
    --max     100          # limit rows per call (omit for all)
"""
import argparse, os, re, textwrap, time, json
import pandas as pd
import openai
import wikipedia
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt

OPENAI_MODEL = "gpt-4o-mini"
TOKENS_PER_REQ = 2048         # safety buffer

PROMPT_TMPL = textwrap.dedent(
    """\
    You are an impartial civic-information editor writing a single paragraph
    (~100 words, max 120) about the person named below for a Change.org
    Decision-Maker profile.

    • Start with current (or most recent) elected/appointed office.
    • Include district/state and party if discoverable.
    • Mention landmark bills, leadership posts, or signature issues.
    • If no longer in office, note that fact neutrally.
    • Do NOT urge action or express opinion.
    • Do NOT invent facts; if sourcing is thin, write a shorter bio.

    Person: {name}
    Wikipedia extract (may be empty): {wiki}
    ---
    ONE neutral third-person paragraph only:
    """
)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def call_openai(system_prompt, user_prompt, model=OPENAI_MODEL):
    """Best-practice wrapper with exponential back-off."""
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

def get_wikipedia_extract(person):
    """Returns first 3 sentences of the matching Wiki page, else ''. """
    try:
        page = wikipedia.page(person, auto_suggest=False, redirect=True, preload=False)
        return " ".join(wikipedia.summary(page.title, sentences=3).split())
    except Exception:
        # fallback: first search hit
        try:
            hit = wikipedia.search(person, results=1)
            if hit:
                return " ".join(wikipedia.summary(hit[0], sentences=3).split())
        except Exception:
            pass
    return ""

def main(infile, outfile, model, limit):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise SystemExit("❌  Set OPENAI_API_KEY env variable.")

    df = pd.read_csv(infile)
    df["summary_web"] = df.get("summary_web")  # keeps existing ones
    todo = df[df["summary_web"].isna() | df["summary_web"].eq("")]

    if limit:
        todo = todo.head(limit)

    summaries = []
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Generating bios"):
        name = row["name"]
        wiki = get_wikipedia_extract(name)
        prompt = PROMPT_TMPL.format(name=name, wiki=wiki or "None available")
        try:
            bio = call_openai("You are a factual civic information editor.", prompt, model)
            bio = re.sub(r"\s+", " ", bio)  # tidy whitespace
            summaries.append((_, bio))
        except Exception as e:
            summaries.append((_, f"ERROR: {e}"))

    # write back
    for idx, bio in summaries:
        df.at[idx, "summary_web"] = bio

    df.to_csv(outfile, index=False)
    print(f"\n✅  Finished {len(summaries)} new summaries → {outfile}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--model", default=OPENAI_MODEL)
    ap.add_argument("--max", type=int, default=None,
                    help="Limit rows processed this run")
    args = ap.parse_args()
    main(args.infile, args.outfile, args.model, args.max)
