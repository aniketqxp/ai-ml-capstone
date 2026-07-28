"""Write language-neutral JSON Schemas for pipeline consumers."""
import json
from pathlib import Path

from v2.domain_profiles import (
    DomainProfile,
    ProfileSelection,
    ResolvedDomainPlan,
)
from v2.evaluation_data import (
    CallAnnotation,
    CounterfactualPair,
    EvaluationDataset,
)
from v2.findings import (
    FindingDerivation,
    RequirementAssessmentBatch,
)
from v2.schemas import CallDecision, SignalBundle

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "contracts"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {
        "signal_bundle.schema.json": SignalBundle,
        "call_decision.schema.json": CallDecision,
        "domain_profile.schema.json": DomainProfile,
        "profile_selection.schema.json": ProfileSelection,
        "resolved_domain_plan.schema.json": ResolvedDomainPlan,
        "requirement_assessment_batch.schema.json": (
            RequirementAssessmentBatch
        ),
        "finding_derivation.schema.json": FindingDerivation,
        "call_annotation.schema.json": CallAnnotation,
        "counterfactual_pair.schema.json": CounterfactualPair,
        "evaluation_dataset.schema.json": EvaluationDataset,
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
