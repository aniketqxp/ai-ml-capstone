import { COMPLIANCE_ITEMS, QUALITY_DIMS, RISK_STYLE } from '../lib/constants';
import { hybridTip, escTip } from '../lib/evidenceTips';
import { InfoTip, InfoDot } from './InfoTip';
import { ScoreDots } from './ScoreDots';

// Live compliance scorecard — checks "fire" at the moment their evidence plays
export function CompliancePanel({ evaluation, time, seek, fmt }) {
  if (!evaluation) return null;
  const c = evaluation.compliance, q = evaluation.quality, e = evaluation.escalation;

  const cleared = COMPLIANCE_ITEMS.filter(([k]) => {
    const it = c[k];
    return it?.passed === true && (it.evidence?.sec == null || time >= it.evidence.sec);
  }).length;
  const applicable = COMPLIANCE_ITEMS.filter(([k]) => c[k]?.passed !== null).length;

  return (
    <section className="lg:col-span-2 bg-white rounded-xl border border-slate-200 shadow-xl flex flex-col min-h-0 dark:bg-slate-950 dark:border-slate-800">
      <div className="px-5 py-3 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Compliance &amp; QA</h2>
        <span className="text-xs font-mono text-slate-500">{cleared}/{applicable} cleared</span>
      </div>
      <div className="overflow-y-auto p-4 space-y-4" style={{ maxHeight: '46vh' }}>
        {/* compliance checklist */}
        <div className="space-y-1">
          {COMPLIANCE_ITEMS.map(([key, name]) => {
            const it = c[key];
            const sec = it?.evidence?.sec;
            const passed = it?.passed;
            const fired = passed === true && (sec == null || time >= sec);
            let icon, ring, txt;
            if (passed === null)       { icon = '–'; ring = 'bg-slate-200 text-slate-500 dark:bg-slate-800 dark:text-slate-600'; txt = 'text-slate-500 dark:text-slate-600'; }
            else if (passed === false) { icon = '✕'; ring = 'bg-red-100 text-red-600 border border-red-300 dark:bg-red-950 dark:text-red-400 dark:border-red-800'; txt = 'text-red-700 dark:text-red-300'; }
            else if (fired)            { icon = '✓'; ring = 'bg-emerald-100 text-emerald-700 border border-emerald-400 dark:bg-emerald-900 dark:text-emerald-300 dark:border-emerald-600'; txt = 'text-emerald-800 dark:text-emerald-200'; }
            else                       { icon = '○'; ring = 'bg-slate-200 text-slate-500 dark:bg-slate-800 dark:text-slate-600'; txt = 'text-slate-600 dark:text-slate-500'; }
            const clickable = sec != null;
            return (
              <button key={key} disabled={!clickable} onClick={() => clickable && seek(sec + 0.02)}
                className={`w-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg text-left transition-all duration-300
                  ${clickable ? 'hover:bg-slate-100 dark:hover:bg-slate-900 cursor-pointer' : 'cursor-default'}
                  ${fired ? 'bg-emerald-100/50 dark:bg-emerald-950/30' : ''}`}>
                <span className={`w-5 h-5 shrink-0 rounded-full flex items-center justify-center text-[11px] font-bold ${ring}`}>{icon}</span>
                <span className={`flex-grow text-xs font-medium ${txt}`}>{name}</span>
                {passed === false && <span className="text-[10px] text-red-600 dark:text-red-500">missed</span>}
                {passed === null && <span className="text-[10px] text-slate-400 dark:text-slate-600">n/a</span>}
                {sec != null && <span className="text-[10px] font-mono text-slate-400 dark:text-slate-600">{fmt(sec)}</span>}
              </button>
            );
          })}
          {c.identity_method && (
            <div className="text-[10px] text-slate-500 pl-9 pt-0.5">verified via {c.identity_method}</div>
          )}
        </div>

        {/* expected workflow (v0.4.0) — checklist derived from the call's
            subject alone, then audited against the actual call */}
        {evaluation.workflow && (
          <div className="border-t border-slate-200 dark:border-slate-800 pt-3 space-y-1">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5 flex items-center gap-1.5">
              Expected workflow
              <InfoTip tip="This checklist was generated from the call's domain and subject alone — the generator never saw the transcript — then each box was audited against the actual call.">
                <InfoDot />
              </InfoTip>
            </div>
            <div className="text-[10px] text-slate-500 italic pb-1">"{evaluation.workflow.subject}"</div>
            {evaluation.workflow.expected_steps.map((s, i) => {
              const sec = s.evidence?.sec;
              const fired = s.met === true && (sec == null || time >= sec);
              let icon, ring, txt;
              if (s.met == null)      { icon = '–'; ring = 'bg-slate-200 text-slate-500 dark:bg-slate-800 dark:text-slate-600'; txt = 'text-slate-500 dark:text-slate-600'; }
              else if (s.met === false) { icon = '✕'; ring = 'bg-red-100 text-red-600 border border-red-300 dark:bg-red-950 dark:text-red-400 dark:border-red-800'; txt = 'text-red-700 dark:text-red-300'; }
              else if (fired)          { icon = '✓'; ring = 'bg-emerald-100 text-emerald-700 border border-emerald-400 dark:bg-emerald-900 dark:text-emerald-300 dark:border-emerald-600'; txt = 'text-emerald-800 dark:text-emerald-200'; }
              else                     { icon = '○'; ring = 'bg-slate-200 text-slate-500 dark:bg-slate-800 dark:text-slate-600'; txt = 'text-slate-600 dark:text-slate-500'; }
              const clickable = sec != null;
              return (
                <button key={i} disabled={!clickable} onClick={() => clickable && seek(sec + 0.02)}
                  className={`w-full flex items-center gap-2.5 px-2 py-1 rounded-lg text-left transition-all duration-300
                    ${clickable ? 'hover:bg-slate-100 dark:hover:bg-slate-900 cursor-pointer' : 'cursor-default'}
                    ${fired ? 'bg-emerald-100/50 dark:bg-emerald-950/30' : ''}`}>
                  <span className={`w-5 h-5 shrink-0 rounded-full flex items-center justify-center text-[11px] font-bold ${ring}`}>{icon}</span>
                  <span className={`flex-grow text-[11px] ${txt}`}>{s.step}</span>
                  {sec != null && <span className="text-[10px] font-mono text-slate-400 dark:text-slate-600">{fmt(sec)}</span>}
                </button>
              );
            })}
          </div>
        )}

        {/* quality — hybrid text+acoustic scores; hover ⓘ for the formula */}
        <div className="border-t border-slate-200 dark:border-slate-800 pt-3 space-y-2">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1.5 flex items-center gap-1.5">
            Quality
            <InfoTip tip="Each dimension is scored by the text model against the QA rubric; where the audio sentiment model produced predictions for this call, its signal is fused in. Hover a dimension for its exact formula.">
              <InfoDot />
            </InfoTip>
          </div>
          {QUALITY_DIMS.map(([key, name]) => {
            const d = q[key];
            if (!d) return null;
            return (
              <div key={key} className="flex items-center gap-2">
                <span className="flex-grow text-xs text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                  {name}
                  <InfoTip tip={hybridTip(d.hybrid)}><InfoDot /></InfoTip>
                </span>
                <ScoreDots score={d.score} />
                <span className="text-xs font-mono text-slate-500 dark:text-slate-400 w-4 text-right">{d.score}</span>
              </div>
            );
          })}
        </div>

        {/* escalation — fused risk (text ∨ acoustic) */}
        <div className="border-t border-slate-200 dark:border-slate-800 pt-3 flex items-center gap-2 flex-wrap">
          <InfoTip tip={escTip(e.hybrid)}>
            <span className={`text-xs font-semibold px-2 py-1 rounded border cursor-help ${RISK_STYLE[e.risk_level]?.cls || ''}`}>
              {RISK_STYLE[e.risk_level]?.label || e.risk_level}
            </span>
          </InfoTip>
          <span className="text-xs text-slate-500 dark:text-slate-400">emotion: <span className="text-slate-700 dark:text-slate-300">{e.customer_emotion_text}</span></span>
        </div>

        {/* summary */}
        {evaluation.overall_summary && (
          <div className="border-t border-slate-200 dark:border-slate-800 pt-3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">Summary</div>
            <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">{evaluation.overall_summary}</p>
          </div>
        )}
        <div className="text-[9px] text-slate-400 dark:text-slate-600 text-right pt-1">rubric {evaluation.rubric_version}{evaluation.served_by ? ` · ${evaluation.served_by}` : ''}</div>
      </div>
    </section>
  );
}
