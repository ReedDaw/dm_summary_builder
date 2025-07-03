How it works

Wikipedia first-pass – Cheap, rate-limit-free scrape grabs a few sentences.

LLM polishing – Feeds the extract (or “None”) plus a tight prompt to GPT-4o (or 3.5-Turbo if cost-sensitive) to craft a sleek, Column-O-style bio.

Error handling & retries – tenacity ensures transient 502/RateLimit errors get retried automatically.

Idempotent – Rows that already have a summary_web value are skipped, letting you rerun endlessly as new DM rows are appended.

One-time setup: 
pip install openai wikipedia pandas tqdm tenacity python-env-config
echo 'export OPENAI_API_KEY="sk-paste-your-key-here"' >> ~/.zshrc
source ~/.zshrc     # or restart Terminal


Running the first big back-fill: 
python3 dm_summary_builder.py \
    --infile  dms_needing_summaries.csv \
    --outfile dm_summaries_completed.csv
