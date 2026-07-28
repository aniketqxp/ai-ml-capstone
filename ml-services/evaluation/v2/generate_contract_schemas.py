"""Write language-neutral JSON Schemas for pipeline consumers."""
import json
from pathlib import Path

from v2.schemas import CallDecision, SignalBundle

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "contracts"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {
        "signal_bundle.schema.json": SignalBundle,
        "call_decision.schema.json": CallDecision,
    }
    for filename, model in models.items():
        path = OUTPUT_DIR / filename
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
