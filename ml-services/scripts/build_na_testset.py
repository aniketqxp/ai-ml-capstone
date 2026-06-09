"""
Build a North-American-accent test set from apptek_callcenter_dialogues.

Downloads the parquet index, filters to the requested accents (all domains),
pairs channel1 (agent) + channel2 (customer) into calls, downloads the per-channel
WAVs, and writes a manifest.json the batch runner / evaluator consume.

Resumable: skips WAVs already on disk.

Usage:
  python build_na_testset.py --accents en-CA en-US_General
  python build_na_testset.py --accents en-CA en-US_General --limit-per-accent 3   # pilot
"""

import os
import re
import json
import time
import argparse
import urllib.request
from collections import defaultdict

import pyarrow.parquet as pq

REPO       = "apptek-com/apptek_callcenter_dialogues"
PARQUET_URL = (
    "https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues"
    "/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)
OUT_DIR    = r"d:\Desktop\ai-ml-capstone\data\na_testset"
INDEX_PARQUET = os.path.join(OUT_DIR, "_index.parquet")
MANIFEST   = os.path.join(OUT_DIR, "manifest.json")
UA         = {"User-Agent": "Mozilla/5.0"}


def fetch(url, dest, retries=3):
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
            return True
        except Exception as e:
            print(f"      attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)
    return False


def path_to_url(hf_path):
    # hf://datasets/<repo>@<rev>/test/<accent>/audio/<file>.wav
    m = re.search(r"@([0-9a-f]+)/(.*)$", hf_path)
    rev, path_in_repo = m.group(1), m.group(2)
    return f"https://huggingface.co/datasets/{REPO}/resolve/{rev}/{path_in_repo}"


def call_id_of(hf_path):
    fname = hf_path.split("/")[-1]                 # en_CA_Agriculture_1586885_channel1.wav
    return re.sub(r"_channel[12]\.wav$", "", fname)  # en_CA_Agriculture_1586885


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accents", nargs="+", default=["en-CA", "en-US_General"])
    ap.add_argument("--limit-per-accent", type=int, default=None,
                    help="cap calls per accent (for a pilot run)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Index
    if not os.path.exists(INDEX_PARQUET):
        print("Downloading parquet index...")
        if not fetch(PARQUET_URL, INDEX_PARQUET):
            raise SystemExit("Failed to download parquet index")
    df = pq.read_table(INDEX_PARQUET).to_pandas()
    df["path"] = df["audio"].apply(lambda a: a["path"])

    # 2. Filter + pair channels into calls
    manifest = []
    for accent in args.accents:
        sub = df[df["accent"] == accent]
        calls = defaultdict(dict)
        for _, row in sub.iterrows():
            cid = call_id_of(row["path"])
            ch  = "agent" if row["path"].endswith("channel1.wav") else "customer"
            calls[cid][ch] = row

        complete = {cid: v for cid, v in calls.items() if "agent" in v and "customer" in v}
        cids = sorted(complete)
        if args.limit_per_accent:
            cids = cids[: args.limit_per_accent]
        print(f"\n{accent}: {len(complete)} complete calls"
              f"{f' (using first {len(cids)})' if args.limit_per_accent else ''}")

        adir = os.path.join(OUT_DIR, accent)
        os.makedirs(adir, exist_ok=True)

        for i, cid in enumerate(cids, 1):
            a_row, c_row = complete[cid]["agent"], complete[cid]["customer"]
            a_wav = os.path.join(adir, f"{cid}_agent.wav")
            c_wav = os.path.join(adir, f"{cid}_customer.wav")

            for wav, row in [(a_wav, a_row), (c_wav, c_row)]:
                if os.path.exists(wav) and os.path.getsize(wav) > 0:
                    continue
                print(f"  [{i}/{len(cids)}] {os.path.basename(wav)}")
                if not fetch(path_to_url(row["path"]), wav):
                    print(f"      SKIP (download failed)")

            if os.path.exists(a_wav) and os.path.exists(c_wav):
                manifest.append({
                    "call_id": cid,
                    "accent":  accent,
                    "domain":  a_row["domain"],
                    "agent_wav":    os.path.relpath(a_wav, OUT_DIR),
                    "customer_wav": os.path.relpath(c_wav, OUT_DIR),
                    "agent_gender":    a_row["gender"],
                    "customer_gender": c_row["gender"],
                    "agent_transcript":    a_row["text"],
                    "customer_transcript": c_row["text"],
                })

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    by_accent = defaultdict(int)
    for m in manifest:
        by_accent[m["accent"]] += 1
    print(f"\nManifest written: {MANIFEST}")
    print(f"Total calls: {len(manifest)}  " + "  ".join(f"{k}={v}" for k, v in by_accent.items()))


if __name__ == "__main__":
    main()
