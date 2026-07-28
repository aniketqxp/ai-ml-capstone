"""Select the evaluator implementation used by the hosted pipeline."""
import importlib
import os

DEFAULT_EVALUATOR_VERSION = "v1"
KNOWN_EVALUATOR_VERSIONS = ("v1", "v2")
_IMPLEMENTATIONS = {
    "v1": ("graph", "evaluate_call"),
}


class EvaluatorConfigurationError(ValueError):
    """Raised when EVALUATOR_VERSION is not a recognized version."""


class EvaluatorUnavailableError(RuntimeError):
    """Raised when a recognized evaluator has not been implemented yet."""


def resolve_evaluator_version(explicit=None):
    """Return an available evaluator version or raise a descriptive error."""
    version = (
        explicit
        if explicit is not None
        else os.environ.get("EVALUATOR_VERSION", DEFAULT_EVALUATOR_VERSION)
    )
    version = str(version).strip().lower()

    if version not in KNOWN_EVALUATOR_VERSIONS:
        choices = ", ".join(KNOWN_EVALUATOR_VERSIONS)
        raise EvaluatorConfigurationError(
            f"Unknown EVALUATOR_VERSION={version!r}; expected one of: {choices}"
        )
    if version not in _IMPLEMENTATIONS:
        raise EvaluatorUnavailableError(
            f"Evaluator {version!r} is reserved but not implemented"
        )
    return version


def evaluate_call(call_id, results_dir="results", evaluator_version=None):
    """Run the selected evaluator and attach implementation provenance."""
    version = resolve_evaluator_version(evaluator_version)
    module_name, function_name = _IMPLEMENTATIONS[version]
    implementation = importlib.import_module(module_name)
    run = getattr(implementation, function_name)

    evaluation = run(call_id, results_dir=results_dir)
    if not isinstance(evaluation, dict):
        raise TypeError(
            f"Evaluator {version!r} returned {type(evaluation).__name__}, expected dict"
        )

    evaluation["_evaluator"] = {
        "version": version,
        "implementation": module_name,
        "rubric_version": evaluation.get("rubric_version"),
    }
    return evaluation
