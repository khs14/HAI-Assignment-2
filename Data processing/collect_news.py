"""
Collect Google News RSS headlines for a sustainability relevance classifier.

Builds two datasets:
  positive_dataset.csv  (label = 1)  environment / sustainability headlines
  negative_dataset.csv  (label = 0)  hard + adjacent + easy negatives
and a combined_dataset.csv (title, label, ...) ready for training.

Template used:
  https://news.google.com/rss/search?q=<query>+after:YYYY-MM-DD+before:YYYY-MM-DD&hl=en-US&gl=US&ceid=US:en

Loop order: year (outer) -> window inside the year -> query -> region.
Google News returns at most ~100 items per query, so each year is sliced into
smaller windows (month by default) to get more headlines per keyword.
Use --window year for one request per query per year.

Usage examples:
  python collect_news.py --dry-run                 # count requests, no network
  python collect_news.py --test                    # tiny live test
  python collect_news.py                           # full run, both datasets
  python collect_news.py --dataset positive        # one dataset only
  python collect_news.py --window year --years 2024 2025 2026
  python collect_news.py --small --dry-run         # small balanced dataset: show quota + requests
  python collect_news.py --small                   # ~3000 rows per label; quota = 3000 / keywords,
                                                   # move to the next keyword once its quota is met
  python collect_news.py --small --target-rows 2000
  python collect_news.py --small --finalize-only   # re-pick rows from already collected data
  python collect_news.py --validation --dry-run    # held-out validation set (300 rows)
  python collect_news.py --validation              # collect, then build participant + ground-truth files
For a clean time split, collect training with --end-date 2026-06-30 and let
--validation use the later period (default: the last 100 days).
The run is resumable: finished (query, region, window) jobs are skipped on restart.
"""
import argparse
import csv
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE_URL = "https://news.google.com/rss/search"
REGIONS = {
    "US": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    "IN": {"hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
}
# Appended to Tier 2 hard negatives only, to avoid accidentally collecting
# real environment stories. A term is skipped if the query itself contains it
# (e.g. "political climate" keeps its own word). Verify the '-' operator on a
# few samples, or turn off with --no-exclusions.
EXCLUDE_TERMS = ["climate", "emissions", "environmental"]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; news-dataset-builder/1.0)"}

# --------------------------------------------------------------------------- #
# Keywords.  Region: "BOTH" runs in US and IN, otherwise only that region.
# --------------------------------------------------------------------------- #
POSITIVE_BOTH = {
    "climate": [
        "climate change", "global warming", "greenhouse gas emissions",
        "carbon emissions", "carbon footprint", "net zero target",
        "decarbonization", "climate adaptation", "climate policy",
        "climate summit",
    ],
    "energy_transition": [
        "renewable energy", "solar power", "wind energy", "green hydrogen",
        "clean energy transition", "energy efficiency", "battery storage",
        "coal phase-out", "offshore wind",
    ],
    "climate_finance": [
        "green bonds", "sustainable finance", "carbon credits", "carbon market",
        "carbon tax", "climate risk disclosure",
    ],
    "circular_waste": [
        "circular economy", "recycling", "plastic pollution",
        "single-use plastic ban", "zero waste", "e-waste", "waste management",
    ],
    "nature": [
        "biodiversity loss", "deforestation", "wildlife conservation",
        "ocean pollution", "ecosystem restoration", "endangered species",
        "coral reef",
    ],
    "water_land": [
        "water scarcity", "groundwater depletion", "drought climate",
        "sustainable agriculture", "regenerative farming", "food waste",
    ],
    "pollution": [
        "air pollution", "air quality index", "water pollution",
        "industrial pollution", "soil contamination",
    ],
    "corporate_env": [
        "sustainability report", "greenwashing", "science-based targets",
        "scope 3 emissions", "supply chain emissions", "net zero pledge",
    ],
    "transport": [
        "electric vehicle emissions", "EV adoption emissions",
        "sustainable aviation fuel", "green shipping",
        "public transport electrification",
    ],
    "extreme_weather": [
        "climate-driven flooding", "heatwave climate change", "wildfire climate",
        "sea level rise", "glacier melt",
    ],
}
POSITIVE_US = [
    "EPA emissions rule", "Clean Air Act", "Inflation Reduction Act clean energy",
    "clean energy tax credit", "wildfire smoke air quality",
    "offshore wind project", "Paris Agreement US", "national parks conservation",
    "PFAS contamination",
]
POSITIVE_IN = [
    "Delhi air pollution", "stubble burning", "Namami Gange",
    "PM Surya Ghar rooftop solar", "National Green Hydrogen Mission",
    "CPCB pollution", "National Green Tribunal", "Project Tiger conservation",
    "Western Ghats", "heatwave India climate", "India net zero 2070",
    "carbon credit trading scheme India",
]

# Tier 2: hard negatives (same words, different meaning)
HARD_NEG = [
    # (query, region)
    ("golf green putt", "US"), ("Green Bay Packers", "US"),
    ("green card visa", "US"), ('"green light" project', "US"),
    ("Miami Heat", "US"), ("Seattle Storm", "US"), ("Chicago Fire", "US"),
    ("Carolina Hurricanes", "US"), ("Minnesota Wild", "US"),
    ("Nottingham Forest", "US"), ("Frank Ocean", "US"), ("Ocean's Eleven", "US"),
    ("Delhi Capitals", "IN"), ("Chennai Super Kings", "IN"), ("Tiger Shroff", "IN"),
    ("carbon fiber bike", "BOTH"), ('"carbon copy"', "BOTH"),
    ('"political climate"', "BOTH"), ('"economic climate"', "BOTH"),
    ('"business climate"', "BOTH"), ('"investment climate"', "BOTH"),
    ("Climate Pledge Arena", "BOTH"),
    ('"scoring drought"', "BOTH"), ('"title drought"', "BOTH"),
    ('"sustained injury"', "BOTH"), ('"sustain momentum"', "BOTH"),
    ('"sustainable growth" earnings', "BOTH"), ('"sustainable dividend"', "BOTH"),
    ('"sustainable pace"', "BOTH"),
    ('"clean sheet"', "BOTH"), ('"clean sweep"', "BOTH"),
    ('"store footprint"', "BOTH"), ('"digital footprint"', "BOTH"),
    ("contract renewal", "BOTH"), ("subscription renews", "BOTH"),
    ("solar eclipse", "BOTH"), ("solar flare", "BOTH"),
    ("energy drink", "BOTH"), ('"positive energy"', "BOTH"),
    ('"waste of time"', "BOTH"), ('"waste of money"', "BOTH"),
    ('"recycled plot"', "BOTH"), ('"recycled storyline"', "BOTH"),
    ('"offset the cost"', "BOTH"), ("offset printing", "BOTH"),
    ('"assault and battery"', "BOTH"), ("phone battery life", "BOTH"),
    ("plastic surgery", "BOTH"),
    ("startup ecosystem", "BOTH"), ("tech ecosystem", "BOTH"),
    ("partner ecosystem", "BOTH"),
    ("Air India", "BOTH"), ("air strike", "BOTH"), ("air force", "BOTH"),
    ("decision tree", "BOTH"), ("family tree", "BOTH"), ("fossil watch", "BOTH"),
    ("starting grid", "BOTH"), ("bitcoin mining", "BOTH"),
    ("Detroit Tigers", "BOTH"), ("Tiger Woods", "BOTH"),
]

# Tier 3: adjacent negatives (nearby topics, not environmental)
ADJACENT_NEG = [
    ("workplace diversity", "BOTH"), ("DEI policy", "BOTH"),
    ("labor union strike", "BOTH"), ("human rights report", "BOTH"),
    ("gender pay gap", "BOTH"), ("worker safety", "BOTH"),
    ("board of directors", "BOTH"), ("executive compensation", "BOTH"),
    ("shareholder vote", "BOTH"), ("corporate governance", "BOTH"),
    ("audit", "BOTH"),
    ("corporate philanthropy", "BOTH"), ("charity donation", "BOTH"),
    ("employee volunteering", "BOTH"),
    ("oil prices", "BOTH"), ("OPEC", "BOTH"), ("natural gas futures", "BOTH"),
    ("gasoline prices", "BOTH"), ("utility rate hike", "BOTH"),
    ("petrol diesel price hike", "IN"), ("crude oil Brent", "IN"),
    ("Tesla earnings", "US"), ("EV stock",
                               "US"), ("automaker quarterly sales", "US"),
    ("Tata Motors sales", "IN"), ("Ola Electric shares", "IN"),
    ("wheat futures", "US"), ("crop prices", "US"), ("cattle prices", "US"),
    ("mandi prices", "IN"), ("MSP wheat", "IN"),
    ("NASA launch", "US"), ("astronomy", "US"), ("particle physics", "US"),
    ("ISRO launch", "IN"),
    ("Earth Day sale", "BOTH"), ("eco-friendly product sale", "BOTH"),
]

# Tier 1: easy negatives (general news)
EASY_NEG = [
    ("NFL scores", "US"), ("NBA playoffs", "US"), ("tennis grand slam", "BOTH"),
    ("box office", "US"), ("celebrity wedding", "US"), ("album release", "US"),
    ("election polls", "US"), ("Supreme Court ruling", "US"),
    ("smartphone launch", "US"), ("AI chatbot",
                                  "US"), ("cybersecurity breach", "US"),
    ("quarterly earnings", "US"), ("IPO",
                                   "US"), ("Fed interest rate decision", "US"),
    ("clinical trial", "US"), ("vaccine", "US"),
    ("arrested", "US"), ("lawsuit settlement", "US"),
    ("recipe", "US"), ("fashion week", "US"), ("airline fares", "US"),
    ("IPL match", "IN"), ("cricket match", "IN"), ("Premier League transfer", "IN"),
    ("Bollywood", "IN"), ("TV series finale", "IN"),
    ("parliament vote", "IN"), ("cabinet reshuffle", "IN"),
    ("startup funding", "IN"), ("software update", "IN"),
    ("Sensex", "IN"), ("Nifty", "IN"), ("RBI repo rate", "IN"),
    ("hospital", "IN"), ("diabetes study", "IN"),
    ("court sentenced", "IN"), ("police investigation", "IN"),
    ("restaurant review", "IN"), ("board exam results", "IN"),
]


def build_query_rows(dataset: str, use_exclusions: bool):
    """Return list of dicts: query, base_query, label, tier, theme, region."""
    rows = []

    def add(q, label, tier, theme, region, exclude=False):
        full = q
        if exclude:
            low = q.lower()
            for t in EXCLUDE_TERMS:
                if t not in low:
                    full += f" -{t}"
        rows.append(dict(query=full, base_query=q, label=label, tier=tier,
                         theme=theme, region=region))

    if dataset == "positive":
        def pq(q):  # exact-phrase quoting for positives
            return q if '"' in q or " " not in q else f'"{q}"'
        for theme, qs in POSITIVE_BOTH.items():
            for q in qs:
                add(pq(q), 1, "positive", theme, "BOTH")
        for q in POSITIVE_US:
            add(pq(q), 1, "positive", "us_specific", "US")
        for q in POSITIVE_IN:
            add(pq(q), 1, "positive", "in_specific", "IN")
    else:
        for q, r in HARD_NEG:
            add(q, 0, "hard", "hard_negative", r, exclude=use_exclusions)
        for q, r in ADJACENT_NEG:
            add(q, 0, "adjacent", "adjacent_negative", r)
        for q, r in EASY_NEG:
            add(q, 0, "easy", "easy_negative", r)
    return rows


# --------------------------------------------------------------------------- #
# Date windows
# --------------------------------------------------------------------------- #
def windows(year: int, mode: str, end_date: date):
    """Yield (start, end) windows inside a year; `before:` is the next start."""
    step = {"year": 12, "quarter": 3, "month": 1}[mode]
    cur = date(year, 1, 1)
    stop = min(date(year + 1, 1, 1), end_date + timedelta(days=1))
    while cur < stop:
        m = cur.month - 1 + step
        nxt = date(cur.year + m // 12, m % 12 + 1, 1)
        yield cur, min(nxt, stop)
        cur = nxt


# --------------------------------------------------------------------------- #
# Fetch + parse
# --------------------------------------------------------------------------- #
def build_url(query: str, start: date, end: date, region: str) -> str:
    q = f"{query} after:{start.isoformat()} before:{end.isoformat()}"
    p = REGIONS[region]
    return f"{BASE_URL}?q={quote_plus(q)}&hl={p['hl']}&gl={p['gl']}&ceid={p['ceid']}"


def fetch(session, url, retries=4):
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.content
            if r.status_code in (429, 503):
                wait = 5 * (2 ** attempt) + random.random()
                print(
                    f"    HTTP {r.status_code}, backing off {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"    HTTP {r.status_code} for {url}", file=sys.stderr)
            return None
        except requests.RequestException as e:
            print(f"    request error: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


def parse_feed(content: bytes):
    items = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return items
    for it in root.iter("item"):
        raw_title = (it.findtext("title") or "").strip()
        src_el = it.find("source")
        source = (src_el.text or "").strip() if src_el is not None else ""
        title = raw_title
        if source and title.endswith(f" - {source}"):
            title = title[: -(len(source) + 3)]
        else:  # fall back: Google appends " - Publisher"
            m = re.match(r"^(.*)\s-\s([^-]+)$", title)
            if m:
                title, source = m.group(1), source or m.group(2).strip()
        pub = it.findtext("pubDate") or ""
        try:
            pub_iso = parsedate_to_datetime(pub).isoformat()
        except (TypeError, ValueError):
            pub_iso = ""
        items.append(dict(title=title.strip(), source=source,
                          link=(it.findtext("link") or "").strip(),
                          published=pub_iso))
    return items


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
RAW_FIELDS = ["title", "source", "link", "published", "label", "tier", "theme",
              "query", "base_query", "region", "year", "window_start", "window_end"]


def collect(dataset, args, end_date):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / f"raw_{dataset}.csv"
    done_path = outdir / f"done_{dataset}.txt"

    qrows = build_query_rows(dataset, not args.no_exclusions)
    if args.test:
        qrows = qrows[:3]
    years = args.years[-1:] if args.test else args.years
    mode = "year" if args.test else args.window

    done = set(done_path.read_text().splitlines()
               ) if done_path.exists() else set()

    if args.validation:  # one held-out window, one request per keyword/region
        spans = [(args.val_start.year, args.val_start,
                  end_date + timedelta(days=1))]
    elif mode == "all":  # one request per keyword/region spanning every year
        spans = [(years[0], date(years[0], 1, 1),
                  end_date + timedelta(days=1))]
    else:                # outer loop: year by year, windows inside each year
        spans = [(year, s, e)
                 for year in years for s, e in windows(year, mode, end_date)]
    jobs = []
    for year, start, end in spans:
        for qi, qr in enumerate(qrows):
            if qr["region"] != "BOTH":
                regions = [qr["region"]]
            elif args.one_region:  # alternate US / IN so each region gets half
                regions = [list(REGIONS)[qi % 2]]
            else:
                regions = list(REGIONS)
            for region in regions:
                key = f"{qr['query']}|{region}|{start}|{end}"
                jobs.append((year, start, end, qr, region, key))

    todo = [j for j in jobs if j[5] not in done]
    print(f"[{dataset}] {len(jobs)} requests total, {len(todo)} remaining "
          f"(~{len(todo) * args.delay / 60:.0f} min at {args.delay}s delay)")
    if args.dry_run:
        for year, start, end, qr, region, _ in todo[:3]:
            print("   e.g.", build_url(qr["query"], start, end, region))
        return

    new_file = not raw_path.exists()
    session = requests.Session()
    with open(raw_path, "a", newline="", encoding="utf-8") as fh, \
            open(done_path, "a", encoding="utf-8") as dh:
        writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
        if new_file:
            writer.writeheader()
        for i, (year, start, end, qr, region, key) in enumerate(todo, 1):
            content = fetch(session, build_url(
                qr["query"], start, end, region))
            if content is None:
                continue  # not marked done -> retried on next run
            items = parse_feed(content)
            for it in items:
                writer.writerow({**it, "label": qr["label"], "tier": qr["tier"],
                                 "theme": qr["theme"], "query": qr["base_query"],
                                 "base_query": qr["base_query"], "region": region,
                                 "year": year, "window_start": start,
                                 "window_end": end})
            dh.write(key + "\n")
            fh.flush()
            dh.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  [{dataset}] {i}/{len(todo)}  last: {qr['base_query']!r} "
                      f"{region} {start} -> {len(items)} items")
            time.sleep(args.delay + random.random() * 0.5)


# --------------------------------------------------------------------------- #
# Build final datasets
# --------------------------------------------------------------------------- #
def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def finalize(outdir):
    outdir = Path(outdir)
    frames = {}
    for ds in ("positive", "negative"):
        p = outdir / f"raw_{ds}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df = df[df["title"].fillna("").str.len() > 0].copy()
            df["norm"] = df["title"].map(norm)
            frames[ds] = df
    if not frames:
        print("No raw data to finalize.")
        return

    # Drop headlines that appear in both datasets (conflicting labels)
    if len(frames) == 2:
        conflict = set(frames["positive"]["norm"]) & set(
            frames["negative"]["norm"])
        print(f"Removing {len(conflict)} headlines found in both datasets")
        for ds in frames:
            frames[ds] = frames[ds][~frames[ds]["norm"].isin(conflict)]

    for ds, df in frames.items():
        before = len(df)
        df = df.drop_duplicates("norm").drop(columns=["norm"])
        df.to_csv(outdir / f"{ds}_dataset.csv", index=False)
        print(f"{ds}_dataset.csv: {len(df)} rows (from {before} raw)")
        frames[ds] = df
    if len(frames) == 2:
        comb = pd.concat(frames.values(), ignore_index=True)
        comb = comb.sample(frac=1, random_state=42).reset_index(drop=True)
        comb.to_csv(outdir / "combined_dataset.csv", index=False)
        print(f"combined_dataset.csv: {len(comb)} rows")
        print(comb.groupby(["label", "tier"]).size().to_string())


# --------------------------------------------------------------------------- #
# Small keyword-balanced dataset
# --------------------------------------------------------------------------- #
# Negative headlines that clearly are about the environment are dropped, since
# they would be label noise (e.g. an "oil prices" story about emissions rules).
CONTAMINATION_RE = re.compile(
    r"climate change|global warming|greenhouse gas|emissions|renewable energy|"
    r"decarboni[sz]|deforestation|biodiversity|air pollution|plastic pollution|"
    r"carbon (?:credit|tax|market|neutral|capture|footprint)|"
    r"net[- ]zero (?:target|emission|pledge|goal|by)|clean energy|green hydrogen",
    re.I,
)


def attempts_for(qi, qr, args, end_date, years):
    """Ordered (region, start, end) requests to try for one keyword.
    First request covers the whole period in one region (alternating US/IN
    across keywords); the rest are fallbacks used only if it falls short."""
    if qr["region"] == "BOTH":
        names = list(REGIONS)
        regs = [names[qi % 2], names[(qi + 1) % 2]]
    else:
        regs = [qr["region"]]
    if args.validation:
        full, extra = (args.val_start, end_date + timedelta(days=1)), []
    else:
        full = (date(years[0], 1, 1), end_date + timedelta(days=1))
        extra = [w for y in years for w in windows(y, "year", end_date)]
    att = [(r, *full) for r in regs]
    att += [(r, s, e) for r in regs for s, e in extra]
    return att


def collect_quota(dataset, args, end_date, target):
    """Quota = target / number of keywords. Collect that many unique headlines
    for one keyword, then move on to the next keyword."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / f"raw_{dataset}.csv"
    done_path = outdir / f"done_{dataset}.txt"
    rng = random.Random(args.seed)

    qrows = build_query_rows(dataset, not args.no_exclusions)
    if args.test:
        qrows, target = qrows[:3], 30
    quota = -(-int(target * args.oversample) // len(qrows))  # ceiling division
    years = args.years

    have_titles, have_count = set(), {}
    if raw_path.exists():
        old = pd.read_csv(raw_path, usecols=["title", "base_query"])
        have_titles = set(old["title"].dropna().map(norm))
        have_count = old.groupby("base_query").size().to_dict()
    done = set(done_path.read_text().splitlines()
               ) if done_path.exists() else set()

    pending = [(qi, qr) for qi, qr in enumerate(qrows)
               if have_count.get(qr["base_query"], 0) < quota]
    print(f"[{dataset}] {len(qrows)} keywords, target {target} x {args.oversample} "
          f"-> {quota} rows per keyword. {len(pending)} keywords still to collect.")
    print(f"  minimum {len(pending)} requests; more only if a keyword returns "
          f"fewer than {quota} new headlines.")
    if args.dry_run:
        for qi, qr in pending[:3]:
            region, s, e = attempts_for(qi, qr, args, end_date, years)[0]
            print("   e.g.", build_url(qr["query"], s, e, region))
        return

    new_file = not raw_path.exists()
    session = requests.Session()
    fails = requests_made = 0
    with open(raw_path, "a", newline="", encoding="utf-8") as fh, \
            open(done_path, "a", encoding="utf-8") as dh:
        writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
        if new_file:
            writer.writeheader()
        for n_done, (qi, qr) in enumerate(pending, 1):
            kw = qr["base_query"]
            have = have_count.get(kw, 0)
            for region, start, end in attempts_for(qi, qr, args, end_date, years):
                if have >= quota:
                    break  # quota reached: move on to the next keyword
                key = f"{qr['query']}|{region}|{start}|{end}"
                if key in done:
                    continue
                content = fetch(session, build_url(
                    qr["query"], start, end, region))
                requests_made += 1
                if content is None:
                    fails += 1
                    if fails >= 5:
                        print("Stopping: 5 failed requests in a row (network blocked or "
                              "rate limited). Rerun later; progress is saved.")
                        return
                    continue
                fails = 0
                fresh = {}
                for it in parse_feed(content):
                    n = norm(it["title"])
                    if n and n not in have_titles and n not in fresh:
                        fresh[n] = it
                keys = list(fresh)
                for n in rng.sample(keys, min(quota - have, len(keys))):
                    it = fresh[n]
                    year = int(it["published"][:4]
                               ) if it["published"] else start.year
                    writer.writerow({**it, "label": qr["label"], "tier": qr["tier"],
                                     "theme": qr["theme"], "query": kw, "base_query": kw,
                                     "region": region, "year": year,
                                     "window_start": start, "window_end": end})
                    have_titles.add(n)
                    have += 1
                have_count[kw] = have
                dh.write(key + "\n")
                fh.flush()
                dh.flush()
                time.sleep(args.delay + random.random() * 0.5)
            if n_done % 10 == 0 or n_done == len(pending):
                print(f"  [{dataset}] {n_done}/{len(pending)} keywords, "
                      f"{requests_made} requests so far; last: {kw!r} {have}/{quota}")


def waterfill(avail: dict, total: int, rng) -> dict:
    """Split `total` as equally as possible across keys, capped by `avail`.
    Shortfalls from keys with too few items are redistributed to the others."""
    alloc = {k: 0 for k in avail}
    active = [k for k in avail if avail[k] > 0]
    remaining = total
    while remaining > 0 and active:
        share = max(remaining // len(active), 1)
        rng.shuffle(active)
        progressed = False
        for k in list(active):
            give = min(share, avail[k] - alloc[k], remaining)
            if give > 0:
                alloc[k] += give
                remaining -= give
                progressed = True
            if alloc[k] >= avail[k]:
                active.remove(k)
            if remaining == 0:
                break
        if not progressed:
            break
    return alloc


def build_balanced(outdir, target, seed, filter_contamination=True):
    rng = random.Random(seed)
    outdir = Path(outdir)
    frames = {}
    for ds in ("positive", "negative"):
        p = outdir / f"raw_{ds}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df = df[df["title"].fillna("").str.len() > 0].copy()
            df["norm"] = df["title"].map(norm)
            # real publication year, so rows can be spread across years even
            # when one request covers several years
            pub_year = pd.to_datetime(
                df["published"], errors="coerce", utc=True).dt.year
            df["year"] = pub_year.fillna(df["year"]).astype(int)
            # shuffle first so that duplicate headlines are not always
            # credited to the same (first) keyword
            frames[ds] = df.sample(frac=1, random_state=seed)
    if not frames:
        print("No raw data found in", outdir)
        return

    if len(frames) == 2:
        conflict = set(frames["positive"]["norm"]) & set(
            frames["negative"]["norm"])
        print(f"Removing {len(conflict)} headlines found in both datasets")
        for ds in frames:
            frames[ds] = frames[ds][~frames[ds]["norm"].isin(conflict)]

    selected, report = {}, []
    for ds, df in frames.items():
        df = df.drop_duplicates("norm")
        if ds == "negative" and filter_contamination:
            bad = df["title"].str.contains(CONTAMINATION_RE)
            print(
                f"negative: dropped {int(bad.sum())} headlines that look environmental")
            df = df[~bad]

        groups = {k: g for k, g in df.groupby("base_query")}
        alloc = waterfill({k: len(g) for k, g in groups.items()}, target, rng)
        picks = []
        for k, g in groups.items():
            n = alloc[k]
            if n:
                cells = {c: cg for c, cg in g.groupby(["region", "year"])}
                calloc = waterfill({c: len(cg)
                                   for c, cg in cells.items()}, n, rng)
                for c, cg in cells.items():
                    if calloc[c]:
                        picks.append(cg.sample(n=calloc[c], random_state=seed))
            report.append(dict(dataset=ds, keyword=k, tier=g["tier"].iloc[0],
                               available=len(g), selected=n))
        out = pd.concat(picks).drop(columns=["norm"])
        out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
        out.to_csv(outdir / f"small_{ds}.csv", index=False)
        selected[ds] = out

        rep = pd.DataFrame([r for r in report if r["dataset"] == ds])
        print(f"\nsmall_{ds}.csv: {len(out)} rows (target {target}) across "
              f"{(rep['selected'] > 0).sum()}/{len(rep)} keywords")
        print(f"  rows per keyword: min {rep['selected'].min()}, "
              f"median {int(rep['selected'].median())}, max {rep['selected'].max()}")
        short = rep[rep["selected"] < rep["selected"].max()].sort_values(
            "selected").head(8)
        if len(short):
            print("  lowest keywords:", ", ".join(
                f"{r.keyword} ({r.selected})" for r in short.itertuples()))
        print("  by tier:", out["tier"].value_counts().to_dict())
        print("  by region:", out["region"].value_counts().to_dict())

    pd.DataFrame(report).to_csv(outdir / "small_report.csv", index=False)
    if len(selected) == 2:
        comb = pd.concat(selected.values(), ignore_index=True)
        comb = comb.sample(frac=1, random_state=seed).reset_index(drop=True)
        comb.to_csv(outdir / "small_combined.csv", index=False)
        print(f"\nsmall_combined.csv: {len(comb)} rows "
              f"({int(comb['label'].sum())} positive, {int((comb['label'] == 0).sum())} negative)")
        print("Per-keyword counts are in small_report.csv")


# --------------------------------------------------------------------------- #
# Held-out validation set (for human participants)
# --------------------------------------------------------------------------- #
def _training_titles(dirs):
    """Normalized titles (and 50-char prefixes) already used in training files."""
    seen, prefixes = set(), set()
    names = ("small_positive.csv", "small_negative.csv",
             "positive_dataset.csv", "negative_dataset.csv")
    for d in dirs:
        for name in names:
            p = Path(d) / name
            if p.exists():
                for t in pd.read_csv(p, usecols=["title"])["title"].dropna():
                    n = norm(t)
                    seen.add(n)
                    prefixes.add(n[:50])
    return seen, prefixes


def build_validation(outdir, exclude_dirs, n_rows, pos_frac, seed, filter_contamination=True):
    rng = random.Random(seed)
    outdir = Path(outdir)
    train_titles, train_prefix = _training_titles(exclude_dirs)
    print(f"Excluding {len(train_titles)} headlines already used for training "
          f"(from {', '.join(map(str, exclude_dirs))})")

    frames = {}
    for ds in ("positive", "negative"):
        p = outdir / f"raw_{ds}.csv"
        if not p.exists():
            print(f"Missing {p}; collect both datasets first.")
            return
        df = pd.read_csv(p)
        df = df[df["title"].fillna("").str.len() > 0].copy()
        df["norm"] = df["title"].map(norm)
        df["prefix"] = df["norm"].str[:50]
        before = len(df)
        df = df[~df["norm"].isin(train_titles) & ~
                df["prefix"].isin(train_prefix)]
        print(
            f"{ds}: {before - len(df)} of {before} collected rows overlap training, removed")
        frames[ds] = df.sample(frac=1, random_state=seed)

    conflict = set(frames["positive"]["norm"]) & set(
        frames["negative"]["norm"])
    print(f"Removing {len(conflict)} headlines found in both datasets")
    for ds in frames:
        frames[ds] = frames[ds][~frames[ds]["norm"].isin(conflict)]

    n_pos = round(n_rows * pos_frac)
    targets = {"positive": n_pos, "negative": n_rows - n_pos}
    picked = []
    for ds, df in frames.items():
        df = df.drop_duplicates("norm").drop_duplicates(
            "prefix")  # also drops near-duplicates
        if ds == "negative" and filter_contamination:
            bad = df["title"].str.contains(CONTAMINATION_RE)
            print(
                f"negative: dropped {int(bad.sum())} headlines that look environmental")
            df = df[~bad]
        groups = {k: g for k, g in df.groupby("base_query")}
        alloc = waterfill({k: len(g)
                          for k, g in groups.items()}, targets[ds], rng)
        parts = []
        for k, g in groups.items():
            if alloc[k]:
                cells = {c: cg for c, cg in g.groupby("region")}
                calloc = waterfill({c: len(cg)
                                   for c, cg in cells.items()}, alloc[k], rng)
                for c, cg in cells.items():
                    if calloc[c]:
                        parts.append(cg.sample(n=calloc[c], random_state=seed))
        out = pd.concat(parts)
        print(f"{ds}: picked {len(out)} of {targets[ds]} from "
              f"{out['base_query'].nunique()}/{len(groups)} keywords")
        picked.append(out)

    val = pd.concat(picked).sample(
        frac=1, random_state=seed).reset_index(drop=True)
    # ids assigned after shuffle
    val.insert(0, "id", [f"V{i + 1:03d}" for i in range(len(val))])

    # utf-8-sig so Excel opens the participant file with correct characters
    val[["id", "title"]].to_csv(outdir / "validation_participant.csv", index=False,
                                encoding="utf-8-sig")
    gt = val[["id", "title", "label", "tier", "base_query", "region", "source",
              "published", "link"]].rename(columns={"base_query": "keyword"})
    gt.to_csv(outdir / "validation_ground_truth.csv",
              index=False, encoding="utf-8-sig")

    print(
        f"\nvalidation_participant.csv: {len(val)} rows (id, title only; give this to participants)")
    print(f"validation_ground_truth.csv: keep private. label counts: "
          f"{val['label'].value_counts().to_dict()}")
    print("  by tier:", val["tier"].value_counts().to_dict())
    print("  by region:", val["region"].value_counts().to_dict())
    print("Labels come from the search keyword, so read through validation_ground_truth.csv "
          "and correct any wrong label before using it.")


def main():
    today = date.today()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dataset", choices=["positive", "negative", "both"], default="both")
    ap.add_argument("--years", type=int, nargs="+",
                    default=[today.year - 2, today.year - 1, today.year])
    ap.add_argument("--window", choices=["all", "year", "quarter", "month"], default=None,
                    help="date slicing: 'all' = one request spanning all --years; "
                         "default is 'all' with --small, otherwise 'month'")
    ap.add_argument("--one-region", action="store_true",
                    help="query each shared keyword in only one region (alternating US/IN) "
                         "instead of both, which cuts those requests in half")
    ap.add_argument("--end-date", type=date.fromisoformat, default=today,
                    help="cap for the latest year (default: today)")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between requests")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--no-exclusions", action="store_true",
                    help="don't append -climate -emissions -environmental to hard negatives")
    ap.add_argument("--dry-run", action="store_true",
                    help="count requests only")
    ap.add_argument("--test", action="store_true",
                    help="3 queries, latest year, 1 window")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--small", action="store_true",
                    help="small keyword-balanced dataset: yearly windows, then pick "
                         "--target-rows per label with an equal share per keyword")
    ap.add_argument("--target-rows", type=int, default=3000,
                    help="rows per label in --small mode (default 3000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oversample", type=float, default=1.1,
                    help="--small/--validation: collect this much extra per keyword to "
                         "cover rows lost to dedup and filtering (default 1.1)")
    ap.add_argument("--no-filter", action="store_true",
                    help="--small: keep negatives that look environmental")
    ap.add_argument("--validation", action="store_true",
                    help="build a held-out validation set from a later time window")
    ap.add_argument("--val-rows", type=int, default=300)
    ap.add_argument("--val-pos-frac", type=float, default=0.5,
                    help="share of positives in the validation set (default 0.5)")
    ap.add_argument("--val-start", type=date.fromisoformat,
                    default=today - timedelta(days=100),
                    help="first day of the validation window (default: 100 days ago)")
    ap.add_argument("--exclude-dir", nargs="+", default=["data_small", "data"],
                    help="folders holding training files whose headlines are excluded")
    args = ap.parse_args()

    if args.window is None:
        args.window = "all" if args.small else "month"

    if args.validation:
        if args.outdir == "data":
            args.outdir = "data_validation"
        if not args.finalize_only:
            n_pos = round(args.val_rows * args.val_pos_frac)
            targets = {"positive": n_pos, "negative": args.val_rows - n_pos}
            for ds in (["positive", "negative"] if args.dataset == "both" else [args.dataset]):
                collect_quota(ds, args, args.end_date, targets[ds])
        if not args.dry_run:
            build_validation(args.outdir, args.exclude_dir, args.val_rows,
                             args.val_pos_frac, args.seed, not args.no_filter)
        return

    if args.small:
        if args.outdir == "data":
            args.outdir = "data_small"
        if not args.finalize_only:
            for ds in (["positive", "negative"] if args.dataset == "both" else [args.dataset]):
                collect_quota(ds, args, args.end_date, args.target_rows)
        if not args.dry_run:
            build_balanced(args.outdir, args.target_rows,
                           args.seed, not args.no_filter)
        return

    if not args.finalize_only:
        for ds in (["positive", "negative"] if args.dataset == "both" else [args.dataset]):
            collect(ds, args, args.end_date)
    if not args.dry_run:
        finalize(args.outdir)


if __name__ == "__main__":
    main()
