import { useMemo } from 'react';
import { COMPLIANCE_ITEMS, QUALITY_DIMS, LANE_STYLE } from '../lib/constants';
import { valColor } from '../lib/evidenceTips';

export function AgentLanes({ evaluation, duration, time, seek }) {
  const lanes = useMemo(() => {
    if (!evaluation) return [];
    const c = evaluation.compliance, q = evaluation.quality, e = evaluation.escalation;

    const comp = [];
    COMPLIANCE_ITEMS.forEach(([key, name]) => {
      const it = c[key];
      if (it?.evidence?.sec != null)
        comp.push({
          sec: it.evidence.sec,
          fail: it.passed === false,
          tip: `${name} — "${it.evidence.quote}"`,
        });
    });

    const qual = [];
    QUALITY_DIMS.forEach(([key, name]) => {
      const d = q[key];
      if (!d) return;
      (d.evidence || []).forEach((ev) => {
        if (ev.sec != null)
          qual.push({
            sec: ev.sec,
            fail: false,
            tip: `${name} ${d.score}/5 — "${ev.quote}"`,
          });
      });
    });

    const wf = (evaluation.workflow?.expected_steps || [])
      .filter((s) => s.evidence?.sec != null)
      .map((s) => ({
        sec: s.evidence.sec,
        fail: s.met === false,
        tip: `${s.met === false ? 'missed step' : 'step done'}: ${s.step} — "${s.evidence.quote}"`,
      }));

    const esc = (e.evidence || [])
      .filter((ev) => ev.sec != null)
      .map((ev) => ({
        sec: ev.sec,
        fail: e.risk_level !== 'none',
        tip: `${e.risk_level === 'none' ? 'screened: calm' : `risk: ${e.risk_level}`} — "${ev.quote}"`,
      }));

    return [
      { key: 'compliance', markers: comp },
      { key: 'quality', markers: qual },
      { key: 'workflow', markers: wf },
      { key: 'escalation', markers: esc, spark: true },
    ];
  }, [evaluation]);

  const traj = evaluation?._acoustic?.trajectory || [];
  const escPts = traj.filter((t) => t.esc != null);
  // svg coords: x = position 0-100, y = 20 (score 0) up to 0 (score 1)
  const sparkPath = escPts.map((t) => `${(t.p * 100).toFixed(1)},${(20 - t.esc * 20).toFixed(1)}`).join(' ');
  const progress = duration ? Math.min(1, time / duration) : 0;

  if (!evaluation || !duration) return null;

  return (
    <div className="mt-3 space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
          Agent findings
        </span>
        <span className="text-[9px] text-slate-400 dark:text-slate-600">
          markers = judgments anchored to this moment · click to jump
        </span>
      </div>
      {lanes.map(({ key, markers, spark }) => {
        const s = LANE_STYLE[key];
        const tall = spark && escPts.length > 0;
        return (
          <div key={key} className="flex items-center gap-2">
            <span className="w-20 shrink-0 text-right text-[9px] font-medium text-slate-500">
              {s.label}
            </span>
            <div className={`relative flex-grow ${tall ? 'h-7' : 'h-4'} rounded bg-slate-100 border border-slate-200 dark:bg-slate-900 dark:border-slate-800 overflow-hidden`}>
              {tall && (
                <svg viewBox="0 0 100 20" preserveAspectRatio="none" className="absolute inset-0 w-full h-full">
                  {/* review threshold (late_mean ≥ 0.30) */}
                  <line x1="0" y1="14" x2="100" y2="14" stroke="rgba(251,191,36,0.25)" strokeWidth="0.4" strokeDasharray="2,1.5" />
                  {/* full trajectory, dim */}
                  <polyline points={sparkPath} fill="none" stroke="rgba(251,191,36,0.3)" strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
                  {/* played portion, bright */}
                  <g clipPath="url(#escProgress)">
                    <polyline points={sparkPath} fill="none" stroke="rgba(251,191,36,0.95)" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
                  </g>
                  <defs>
                    <clipPath id="escProgress">
                      <rect x="0" y="0" width={progress * 100} height="20" />
                    </clipPath>
                  </defs>
                </svg>
              )}
              {markers.length === 0 && !tall && (
                <span className="absolute inset-0 flex items-center pl-2 text-[8px] text-slate-400 dark:text-slate-700">
                  no findings anchored
                </span>
              )}
              {markers.map((m, i) => {
                const fired = time >= m.sec;
                const cls = m.fail ? s.fail : fired ? s.on : s.off;
                return (
                  <button
                    key={i}
                    title={m.tip}
                    onClick={() => seek(m.sec + 0.02)}
                    className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2 h-2 rotate-45 border transition-all duration-300 ${cls} hover:scale-150 z-10`}
                    style={{ left: `${(m.sec / duration) * 100}%` }}
                  />
                );
              })}
            </div>
            <span className="w-6 shrink-0 text-[9px] font-mono text-slate-400 dark:text-slate-600">
              {markers.length}
            </span>
          </div>
        );
      })}
      {traj.length > 0 && (
        <div className="flex items-center gap-2">
          <span className="w-20 shrink-0 text-right text-[9px] font-medium text-slate-500" title="Audio model's per-sentence sentiment, bucketed across the call. Green = positive, red = negative, grey = neutral.">
            Audio tone
          </span>
          <div className="flex-grow space-y-px">
            {[['agent_val', 'agent'], ['cust_val', 'customer']].map(([field, who]) => (
              <div key={field} className="flex h-[7px] rounded-sm overflow-hidden bg-slate-100 border border-slate-200 dark:bg-slate-900 dark:border-slate-800">
                {traj.map((t, i) => (
                  <div
                    key={i}
                    className="flex-1"
                    title={t[field] != null ? `${who} tone ${t[field] > 0.05 ? 'positive' : t[field] < -0.05 ? 'negative' : 'neutral'} (${Math.round(t.p * 100)}% through call)` : ''}
                    style={{ backgroundColor: valColor(t[field]), opacity: t.p <= progress ? 1 : 0.35 }}
                  />
                ))}
              </div>
            ))}
          </div>
          <span className="w-6 shrink-0 text-[8px] leading-tight text-slate-400 dark:text-slate-600 text-left">ag<br/>cu</span>
        </div>
      )}
    </div>
  );
}
