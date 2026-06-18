"""
Export AppTek selected-domain metadata while preserving original source identifiers.

This creates a mapping between:
- our exported call_id, for example APPTEK_BANKING_0048
- AppTek/source audio id, for example en_CA_Banking_1586889

No audio is downloaded or copied here.
"""

from pathlib import Path
import pandas as pd
from datasets import load_dataset, Audio

DATASET_NAME = "apptek-com/apptek_callcenter_dialogues"

ML_SERVICES_ROOT = Path(__file__).resolve().parents[2]

EXISTING_METADATA_PATH = (
    ML_SERVICES_ROOT
    / "data"
    / "processed"
    / "apptek_selected_domains"
    / "apptek_selected_domain_metadata.csv"
)

OUTPUT_PATH = (
    ML_SERVICES_ROOT
    / "data"
    / "processed"
    / "apptek_selected_domains"
    / "apptek_selected_domain_metadata_with_source_ids.csv"
)

DOMAIN_MAPPING = {
    "banking": "banking",
    "health": "healthcare",
    "telecom": "telecommunications",
}


def get_source_id_from_audio(audio_obj):
    if not isinstance(audio_obj, dict):
        return None, None

    audio_path = audio_obj.get("path")
    if not audio_path:
        return None, None

    audio_path = str(audio_path)
    source_id = Path(audio_path).stem

    return source_id, audio_path


def main():
    existing_df = pd.read_csv(EXISTING_METADATA_PATH)

    print("Loading AppTek dataset...")
    ds = load_dataset(DATASET_NAME, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))

    selected_rows = []

    counters = {
        "banking": 0,
        "healthcare": 0,
        "telecommunications": 0,
    }

    for row in ds:
        raw_domain = row.get("domain")

        if raw_domain not in DOMAIN_MAPPING:
            continue

        selected_domain = DOMAIN_MAPPING[raw_domain]
        counters[selected_domain] += 1

        call_id = f"APPTEK_{selected_domain.upper()}_{counters[selected_domain]:04d}"
        source_apptek_id, source_audio_path = get_source_id_from_audio(row.get("audio"))

        selected_rows.append(
            {
                "call_id": call_id,
                "source_apptek_id": source_apptek_id,
                "selected_domain": selected_domain,
                "raw_domain": raw_domain,
                "gender": row.get("gender"),
                "accent": row.get("accent"),
                "source_audio_path": source_audio_path,
            }
        )

    source_df = pd.DataFrame(selected_rows)

    merged = existing_df.merge(
        source_df,
        on=["call_id", "selected_domain", "raw_domain", "gender", "accent"],
        how="left",
    )

    merged.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Rows:", len(merged))
    print()
    print("Missing source IDs:", merged["source_apptek_id"].isna().sum())
    print()
    print("Sample:")
    print(
        merged[
            [
                "call_id",
                "source_apptek_id",
                "selected_domain",
                "raw_domain",
                "gender",
                "accent",
                "duration_seconds",
                "audio_path",
            ]
        ].head(20).to_string(index=False)
    )

    print()
    print("Rows containing 1586889:")
    mask = merged.astype(str).apply(
        lambda col: col.str.contains("1586889", case=False, na=False)
    ).any(axis=1)
    print(merged[mask].to_string(index=False))


if __name__ == "__main__":
    main()
