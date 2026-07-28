"""Cross-artifact validation for SignalBundle and CallDecision."""
from .schemas import CallDecision, SignalBundle


def validate_decision_references(
    decision: CallDecision,
    bundle: SignalBundle,
) -> CallDecision:
    """Ensure every decision citation resolves inside its signal bundle."""
    if decision.call_id != bundle.call_id:
        raise ValueError("decision and signal bundle call_id values do not match")
    if decision.signal_bundle_schema_version != bundle.schema_version:
        raise ValueError("decision references an incompatible signal bundle schema")

    segment_ids = {segment.segment_id for segment in bundle.segments}
    signal_ids = {signal.signal_id for signal in bundle.signals}
    episode_ids = {episode.episode_id for episode in bundle.episodes}

    findings = decision.triggered_findings + decision.positive_findings
    for finding in findings:
        unknown_rule_signals = (
            set(finding.detection_rule.signal_ids) - signal_ids
        )
        if unknown_rule_signals:
            raise ValueError(
                f"finding {finding.finding_id!r} rule references unknown signals: "
                f"{sorted(unknown_rule_signals)}"
            )

        for evidence in finding.evidence + finding.counter_evidence:
            unknown_segments = set(evidence.segment_ids) - segment_ids
            unknown_signals = set(evidence.signal_ids) - signal_ids
            unknown_episodes = set(evidence.episode_ids) - episode_ids
            if unknown_segments or unknown_signals or unknown_episodes:
                raise ValueError(
                    f"evidence {evidence.evidence_id!r} has unresolved references"
                )

    for uncertainty in decision.uncertainties:
        unknown_signals = set(uncertainty.affected_signal_ids) - signal_ids
        if unknown_signals:
            raise ValueError(
                f"uncertainty {uncertainty.code!r} references unknown signals"
            )
    return decision
