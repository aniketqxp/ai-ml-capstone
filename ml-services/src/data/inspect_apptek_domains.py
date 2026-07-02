"""
Inspect available domains in the AppTek Call Center Dialogues dataset
without saving the full audio dataset locally.

Run from ml-services:

    python -m src.data.inspect_apptek_domains
"""

from collections import Counter

from datasets import Audio, load_dataset


DATASET_NAME = "apptek-com/apptek_callcenter_dialogues"


def main() -> None:
    print("\nLoading AppTek dataset metadata...")
    print("This may access the dataset, but it will not save the full dataset locally.")
    print("-" * 80)

    ds = load_dataset(DATASET_NAME, split="test")

    # Disable audio decoding so we inspect metadata without loading audio arrays.
    ds = ds.cast_column("audio", Audio(decode=False))

    domain_counts = Counter()
    accent_counts = Counter()
    gender_counts = Counter()

    for row in ds:
        domain_counts[row.get("domain", "unknown")] += 1
        accent_counts[row.get("accent", "unknown")] += 1
        gender_counts[row.get("gender", "unknown")] += 1

    print("\nAvailable AppTek domains:")
    print("-" * 80)
    for domain, count in sorted(domain_counts.items()):
        print(f"{domain}: {count}")

    print("\nAccents:")
    print("-" * 80)
    for accent, count in sorted(accent_counts.items()):
        print(f"{accent}: {count}")

    print("\nGenders:")
    print("-" * 80)
    for gender, count in sorted(gender_counts.items()):
        print(f"{gender}: {count}")

    target_keywords = ["bank", "health", "medical", "tele", "telecom", "communication"]

    print("\nTarget-domain search:")
    print("-" * 80)
    for keyword in target_keywords:
        matches = {
            domain: count
            for domain, count in domain_counts.items()
            if keyword.lower() in domain.lower()
        }
        print(f"{keyword}: {matches if matches else 'no matches'}")


if __name__ == "__main__":
    main()