// Human-readable provenance for a hybrid quality score
export function hybridTip(h) {
  if (!h || h.method !== 'weighted_mean') return 'Scored from transcript text only.';
  return `Hybrid score: text model ${h.text_score} × ${h.text_weight} + audio model ${h.acoustic_score} × ${h.acoustic_weight}. ` +
    `Acoustic signal from the ${h.channel.toLowerCase()} channel (audio predictions cover ${Math.round(h.coverage * 100)}% of its sentences).`;
}

export function escTip(h) {
  if (!h || h.acoustic_risk == null) return 'Risk assessed from transcript text only.';
  const base = `Text tier: ${h.text_risk}. Acoustic tier: ${h.acoustic_risk} ` +
    `(customer's vocal escalation — late-call mean ${h.late_mean_escalation}, peak ${h.peak_escalation}).`;
  if (h.method === 'llm_arbitration')
    return `${base} The two channels disagreed, so an arbitration model weighed both against the transcript: ${h.arbitration_rationale}`;
  if (h.method === 'agreement') return `${base} Both channels agree.`;
  return `${base} Fused as the more severe of the two tiers.`;
}

// valence (-1..+1) → heat color: negative=rose, positive=emerald, neutral=faint slate
export function valColor(v) {
  if (v == null) return 'transparent';
  if (v > 0.05) return `rgba(52,211,153,${Math.min(0.85, 0.2 + 0.65 * v)})`;
  if (v < -0.05) return `rgba(244,63,94,${Math.min(0.85, 0.2 + 0.65 * -v)})`;
  return 'rgba(100,116,139,0.3)';
}
