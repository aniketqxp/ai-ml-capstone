import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import callData from './call_data.json';
import sentenceData from './sentence_segments.json';
import sentimentStub from './en_CA_Banking_1586889_simple_sentiment.json';

const SENTIMENT_STYLE = {
  Positive: {
    bg:   'bg-emerald-950/40',
    ring: 'ring-emerald-700/50',
    dot:  'bg-emerald-400',
    pill: 'bg-emerald-900/70 text-emerald-300 border-emerald-700',
  },
  Negative: {
    bg:   'bg-red-950/40',
    ring: 'ring-red-700/50',
    dot:  'bg-red-400',
    pill: 'bg-red-900/70 text-red-300 border-red-700',
  },
  Neutral: {
    bg:   '',
    ring: '',
    dot:  'bg-slate-500',
    pill: 'bg-slate-800 text-slate-400 border-slate-700',
  },
};

const CHAPTER_COLORS = [
  '#6366f1', '#0ea5e9', '#10b981', '#f59e0b',
  '#ec4899', '#8b5cf6', '#14b8a6', '#ef4444',
  '#22c55e', '#eab308', '#06b6d4', '#a855f7',
];

const fmt = (s) => {
  if (!isFinite(s)) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
};

// Shows how much of the audio has buffered (grey bar behind the playhead)
function BufferedRanges({ audio, duration }) {
  const [ranges, setRanges] = useState([]);
  useEffect(() => {
    const a = audio.current;
    if (!a) return;
    const update = () => {
      const r = [];
      for (let i = 0; i < a.buffered.length; i++)
        r.push([a.buffered.start(i), a.buffered.end(i)]);
      setRanges(r);
    };
    a.addEventListener('progress', update);
    a.addEventListener('timeupdate', update);
    return () => { a.removeEventListener('progress', update); a.removeEventListener('timeupdate', update); };
  }, [audio]);
  if (!duration) return null;
  return <>
    {ranges.map(([s, e], i) => (
      <div key={i} className="absolute inset-y-0 bg-slate-600 rounded-full pointer-events-none"
        style={{ left: `${(s / duration) * 100}%`, width: `${((e - s) / duration) * 100}%` }} />
    ))}
  </>;
}

const COMPLIANCE_ITEMS = [
  ['name_announced', 'Name announced'],
  ['company_announced', 'Company announced'],
  ['recording_disclosure', 'Recording disclosure'],
  ['identity_verified', 'Identity verified'],
  ['resolution_provided', 'Resolution provided'],
  ['transfer_next_steps', 'Transfer next-steps'],
];
const QUALITY_DIMS = [
  ['efficiency', 'Efficiency'],
  ['problem_resolution', 'Problem resolution'],
  ['clarity', 'Clarity'],
  ['professionalism', 'Professionalism'],
  ['empathy', 'Empathy'],
  ['customer_satisfaction', 'Customer satisfaction'],
];
const RISK_STYLE = {
  none:     { label: 'No escalation', cls: 'bg-emerald-950 text-emerald-300 border-emerald-800' },
  review:   { label: 'Review',        cls: 'bg-amber-950 text-amber-300 border-amber-800' },
  escalate: { label: 'Escalate',      cls: 'bg-red-950 text-red-300 border-red-800' },
};

// Small hover tooltip — used to explain HOW a score was computed without
// cluttering the scorecard with modality tags.
function InfoTip({ tip, children }) {
  return (
    <span className="relative group/tip inline-flex items-center">
      {children}
      <span className="pointer-events-none absolute z-30 hidden group-hover/tip:block bottom-full left-1/2 -translate-x-1/2 mb-1.5 w-60 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-2 text-[10px] leading-relaxed text-slate-300 shadow-xl normal-case tracking-normal font-normal">
        {tip}
      </span>
    </span>
  );
}

function InfoDot() {
  return (
    <span className="text-slate-500 hover:text-slate-300 border border-slate-700 rounded-full w-3.5 h-3.5 inline-flex items-center justify-center text-[8px] cursor-help select-none">i</span>
  );
}

// Human-readable provenance for a hybrid quality score
function hybridTip(h) {
  if (!h || h.method !== 'weighted_mean') return 'Scored from transcript text only.';
  return `Hybrid score: text model ${h.text_score} × ${h.text_weight} + audio model ${h.acoustic_score} × ${h.acoustic_weight}. ` +
    `Acoustic signal from the ${h.channel.toLowerCase()} channel (audio predictions cover ${Math.round(h.coverage * 100)}% of its sentences).`;
}

function escTip(h) {
  if (!h || h.acoustic_risk == null) return 'Risk assessed from transcript text only.';
  const base = `Text tier: ${h.text_risk}. Acoustic tier: ${h.acoustic_risk} ` +
    `(customer's vocal escalation — late-call mean ${h.late_mean_escalation}, peak ${h.peak_escalation}).`;
  if (h.method === 'llm_arbitration')
    return `${base} The two channels disagreed, so an arbitration model weighed both against the transcript: ${h.arbitration_rationale}`;
  if (h.method === 'agreement') return `${base} Both channels agree.`;
  return `${base} Fused as the more severe of the two tiers.`;
}

function ScoreDots({ score }) {
  const color = score >= 4 ? 'bg-emerald-500' : score === 3 ? 'bg-amber-500' : 'bg-red-500';
  return (
    <span className="flex gap-1">
      {[1, 2, 3, 4, 5].map((n) => (
        <span key={n} className={`w-1.5 h-1.5 rounded-full ${n <= score ? color : 'bg-slate-700'}`} />
      ))}
    </span>
  );
}

// Live compliance scorecard — checks "fire" at the moment their evidence plays
function CompliancePanel({ evaluation, time, seek, fmt }) {
  if (!evaluation) return null;
  const c = evaluation.compliance, q = evaluation.quality, e = evaluation.escalation;

  const cleared = COMPLIANCE_ITEMS.filter(([k]) => {
    const it = c[k];
    return it?.passed === true && (it.evidence?.sec == null || time >= it.evidence.sec);
  }).length;
  const applicable = COMPLIANCE_ITEMS.filter(([k]) => c[k]?.passed !== null).length;

  return (
    <section className="lg:col-span-2 bg-slate-950 rounded-xl border border-slate-800 shadow-xl flex flex-col min-h-0">
      <div className="px-5 py-3 border-b border-slate-800 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-300">Compliance &amp; QA</h2>
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
            if (passed === null)       { icon = '–'; ring = 'bg-slate-800 text-slate-600'; txt = 'text-slate-600'; }
            else if (passed === false) { icon = '✕'; ring = 'bg-red-950 text-red-400 border border-red-800'; txt = 'text-red-300'; }
            else if (fired)            { icon = '✓'; ring = 'bg-emerald-900 text-emerald-300 border border-emerald-600'; txt = 'text-emerald-200'; }
            else                       { icon = '○'; ring = 'bg-slate-800 text-slate-600'; txt = 'text-slate-500'; }
            const clickable = sec != null;
            return (
              <button key={key} disabled={!clickable} onClick={() => clickable && seek(sec + 0.02)}
                className={`w-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg text-left transition-all duration-300
                  ${clickable ? 'hover:bg-slate-900 cursor-pointer' : 'cursor-default'}
                  ${fired ? 'bg-emerald-950/30' : ''}`}>
                <span className={`w-5 h-5 shrink-0 rounded-full flex items-center justify-center text-[11px] font-bold ${ring}`}>{icon}</span>
                <span className={`flex-grow text-xs font-medium ${txt}`}>{name}</span>
                {passed === false && <span className="text-[10px] text-red-500">missed</span>}
                {passed === null && <span className="text-[10px] text-slate-600">n/a</span>}
                {sec != null && <span className="text-[10px] font-mono text-slate-600">{fmt(sec)}</span>}
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
          <div className="border-t border-slate-800 pt-3 space-y-1">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5 flex items-center gap-1.5">
              Expected workflow
              <InfoTip tip="This checklist was generated from the call's domain and subject alone — the generator never saw the transcript — then each box was audited against the actual call.">
                <InfoDot />
              </InfoTip>
            </div>
            <div className="text-[10px] text-slate-500 italic pb-1">“{evaluation.workflow.subject}”</div>
            {evaluation.workflow.expected_steps.map((s, i) => {
              const sec = s.evidence?.sec;
              const fired = s.met === true && (sec == null || time >= sec);
              let icon, ring, txt;
              if (s.met === false)      { icon = '✕'; ring = 'bg-red-950 text-red-400 border border-red-800'; txt = 'text-red-300'; }
              else if (s.met == null)   { icon = '–'; ring = 'bg-slate-800 text-slate-600'; txt = 'text-slate-600'; }
              else if (fired)           { icon = '✓'; ring = 'bg-emerald-900 text-emerald-300 border border-emerald-600'; txt = 'text-emerald-200'; }
              else                      { icon = '○'; ring = 'bg-slate-800 text-slate-600'; txt = 'text-slate-500'; }
              const clickable = sec != null;
              return (
                <button key={i} disabled={!clickable} onClick={() => clickable && seek(sec + 0.02)}
                  className={`w-full flex items-center gap-2.5 px-2 py-1 rounded-lg text-left transition-all duration-300
                    ${clickable ? 'hover:bg-slate-900 cursor-pointer' : 'cursor-default'}
                    ${fired ? 'bg-emerald-950/30' : ''}`}>
                  <span className={`w-5 h-5 shrink-0 rounded-full flex items-center justify-center text-[11px] font-bold ${ring}`}>{icon}</span>
                  <span className={`flex-grow text-[11px] ${txt}`}>{s.step}</span>
                  {sec != null && <span className="text-[10px] font-mono text-slate-600">{fmt(sec)}</span>}
                </button>
              );
            })}
          </div>
        )}

        {/* quality — hybrid text+acoustic scores; hover ⓘ for the formula */}
        <div className="border-t border-slate-800 pt-3 space-y-2">
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
                <span className="flex-grow text-xs text-slate-300 flex items-center gap-1.5">
                  {name}
                  <InfoTip tip={hybridTip(d.hybrid)}><InfoDot /></InfoTip>
                </span>
                <ScoreDots score={d.score} />
                <span className="text-xs font-mono text-slate-400 w-4 text-right">{d.score}</span>
              </div>
            );
          })}
        </div>

        {/* escalation — fused risk (text ∨ acoustic) */}
        <div className="border-t border-slate-800 pt-3 flex items-center gap-2 flex-wrap">
          <InfoTip tip={escTip(e.hybrid)}>
            <span className={`text-xs font-semibold px-2 py-1 rounded border cursor-help ${RISK_STYLE[e.risk_level]?.cls || ''}`}>
              {RISK_STYLE[e.risk_level]?.label || e.risk_level}
            </span>
          </InfoTip>
          <span className="text-xs text-slate-400">emotion: <span className="text-slate-300">{e.customer_emotion_text}</span></span>
        </div>

        {/* summary */}
        {evaluation.overall_summary && (
          <div className="border-t border-slate-800 pt-3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">Summary</div>
            <p className="text-xs text-slate-400 leading-relaxed">{evaluation.overall_summary}</p>
          </div>
        )}
        <div className="text-[9px] text-slate-600 text-right pt-1">rubric {evaluation.rubric_version}{evaluation.served_by ? ` · ${evaluation.served_by}` : ''}</div>
      </div>
    </section>
  );
}

// Where each agent grounded its judgments on THIS call. One lane per agent;
// every marker is a piece of evidence the agent cited, anchored to its real
// moment in the audio. Markers light up as playback passes them.
const LANE_STYLE = {
  compliance: {
    label: 'Compliance',
    on:  'bg-indigo-400 border-indigo-300 shadow-[0_0_6px_rgba(129,140,248,0.8)]',
    off: 'bg-indigo-900 border-indigo-700',
    fail: 'bg-red-500 border-red-400',
  },
  quality: {
    label: 'Quality',
    on:  'bg-emerald-400 border-emerald-300 shadow-[0_0_6px_rgba(52,211,153,0.8)]',
    off: 'bg-emerald-900 border-emerald-700',
    fail: 'bg-red-500 border-red-400',
  },
  workflow: {
    label: 'Workflow',
    on:  'bg-sky-400 border-sky-300 shadow-[0_0_6px_rgba(56,189,248,0.8)]',
    off: 'bg-sky-900 border-sky-700',
    fail: 'bg-red-500 border-red-400',
  },
  escalation: {
    label: 'Escalation',
    on:  'bg-amber-400 border-amber-300 shadow-[0_0_6px_rgba(251,191,36,0.8)]',
    off: 'bg-amber-900 border-amber-700',
    fail: 'bg-red-500 border-red-400',
  },
};

// valence (-1..+1) → heat color: negative=rose, positive=emerald, neutral=faint slate
function valColor(v) {
  if (v == null) return 'transparent';
  if (v > 0.05) return `rgba(52,211,153,${Math.min(0.85, 0.2 + 0.65 * v)})`;
  if (v < -0.05) return `rgba(244,63,94,${Math.min(0.85, 0.2 + 0.65 * -v)})`;
  return 'rgba(100,116,139,0.3)';
}

function AgentLanes({ evaluation, duration, time, seek }) {
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
        <span className="text-[9px] text-slate-600">
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
            <div className={`relative flex-grow ${tall ? 'h-7' : 'h-4'} rounded bg-slate-900 border border-slate-800 overflow-hidden`}>
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
                <span className="absolute inset-0 flex items-center pl-2 text-[8px] text-slate-700">
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
            <span className="w-6 shrink-0 text-[9px] font-mono text-slate-600">
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
              <div key={field} className="flex h-[7px] rounded-sm overflow-hidden bg-slate-900 border border-slate-800">
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
          <span className="w-6 shrink-0 text-[8px] leading-tight text-slate-600 text-left">ag<br/>cu</span>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const audioRef = useRef(null);
  const scrollRef = useRef(null);
  const activeTurnRef = useRef(null);
  const scrubberRef = useRef(null);

  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(callData.duration || 0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [dragTime, setDragTime] = useState(0);

  const { turns, chapters, evaluation } = callData;
  const sentences = sentenceData.sentences;

  // seq_id → sentiment entry
  const sentimentMap = useMemo(() => {
    const m = new Map();
    sentimentStub.segments.forEach(s => m.set(s.seq_id, s));
    return m;
  }, []);

  // for each turn, collect the sentences that fall inside it (same speaker, overlapping time)
  const turnSentences = useMemo(() =>
    turns.map(turn =>
      sentences.filter(s =>
        s.speaker === turn.speaker &&
        s.start >= turn.start - 0.1 &&
        s.end   <= turn.end   + 0.1
      )
    ),
  [turns, sentences]);

  // ── audio wiring ────────────────────────────────────────────────────────────
  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    const onTime = () => setCurrentTime(a.currentTime);
    const onMeta = () => setDuration(a.duration || callData.duration);
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    a.addEventListener('timeupdate', onTime);
    a.addEventListener('loadedmetadata', onMeta);
    a.addEventListener('play', onPlay);
    a.addEventListener('pause', onPause);
    return () => {
      a.removeEventListener('timeupdate', onTime);
      a.removeEventListener('loadedmetadata', onMeta);
      a.removeEventListener('play', onPlay);
      a.removeEventListener('pause', onPause);
    };
  }, []);

  const seek = useCallback((t) => {
    const a = audioRef.current;
    if (!a) return;
    const clamped = Math.max(0, Math.min(t, duration));
    a.currentTime = clamped;
    setCurrentTime(clamped);
  }, [duration]);

  const togglePlay = () => {
    const a = audioRef.current;
    if (!a) return;
    a.paused ? a.play() : a.pause();
  };

  const changeRate = () => {
    const next = rate === 1 ? 1.5 : rate === 1.5 ? 2 : 1;
    setRate(next);
    if (audioRef.current) audioRef.current.playbackRate = next;
  };

  // scrubber drag logic
  const timeFromEvent = useCallback((e, el) => {
    const r = (el || scrubberRef.current).getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    return Math.max(0, Math.min(((clientX - r.left) / r.width) * duration, duration));
  }, [duration]);

  const onScrubStart = useCallback((e) => {
    e.preventDefault();
    const t = timeFromEvent(e);
    setIsDragging(true);
    setDragTime(t);
  }, [timeFromEvent]);

  useEffect(() => {
    if (!isDragging) return;
    const onMove = (e) => setDragTime(timeFromEvent(e));
    const onUp = (e) => {
      const t = timeFromEvent(e);
      setIsDragging(false);
      seek(t);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('touchmove', onMove, { passive: false });
    window.addEventListener('touchend', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('touchend', onUp);
    };
  }, [isDragging, timeFromEvent, seek]);

  // ── derived display time (follows drag during scrub, audio otherwise) ────────
  const displayTime = isDragging ? dragTime : currentTime;
  const progress = duration ? (displayTime / duration) * 100 : 0;

  // ── derived active chapter / turn ────────────────────────────────────────────
  const activeChapter = useMemo(() => {
    for (let i = chapters.length - 1; i >= 0; i--)
      if (displayTime >= chapters[i].start) return chapters[i];
    return chapters[0];
  }, [displayTime, chapters]);

  const activeTurnIdx = useMemo(() => {
    let idx = -1;
    for (let i = 0; i < turns.length; i++) {
      if (displayTime >= turns[i].start) idx = i; else break;
    }
    return idx;
  }, [displayTime, turns]);

  // auto-scroll the active turn into view
  useEffect(() => {
    if (activeTurnRef.current && scrollRef.current) {
      activeTurnRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activeTurnIdx]);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
      {/* Header */}
      <header className="px-6 py-4 border-b border-slate-800 bg-slate-950 flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-indigo-400">Call Compliance Player</h1>
          <p className="text-xs text-slate-500 mt-0.5">{callData.model}</p>
        </div>
        <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-300 text-xs font-mono border border-slate-700">
          {callData.call} · {fmt(duration)} · {chapters.length} chapters
        </span>
      </header>

      <main className="flex-grow max-w-6xl w-full mx-auto p-6 flex flex-col gap-5">
        {/* ── Player + scrubber ─────────────────────────────────────────── */}
        <section className="bg-slate-950 rounded-xl border border-slate-800 p-5 shadow-xl">
          <div className="flex items-center gap-4">
            <button onClick={togglePlay}
              className="w-12 h-12 shrink-0 rounded-full bg-indigo-600 hover:bg-indigo-500 transition flex items-center justify-center text-white shadow-lg shadow-indigo-900/40">
              {isPlaying
                ? <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>
                : <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style={{ marginLeft: 2 }}><path d="M8 5v14l11-7z"/></svg>}
            </button>

            <div className="flex-grow">
              <div className="flex justify-between text-xs font-mono text-slate-400 mb-1.5">
                <span>{fmt(displayTime)}</span>
                <span>{fmt(duration)}</span>
              </div>
              {/* scrubber — click + drag + touch */}
              <div ref={scrubberRef}
                className="relative h-3 bg-slate-800 rounded-full cursor-pointer select-none"
                onMouseDown={onScrubStart}
                onTouchStart={onScrubStart}>
                {/* buffered range */}
                <BufferedRanges audio={audioRef} duration={duration} currentTime={currentTime} />
                {/* played */}
                <div className="absolute inset-y-0 left-0 bg-indigo-500 rounded-full pointer-events-none"
                  style={{ width: `${progress}%` }} />
                {/* thumb */}
                <div className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 rounded-full shadow transition-transform pointer-events-none
                  ${isDragging ? 'w-5 h-5 bg-white scale-110' : 'w-4 h-4 bg-white'}`}
                  style={{ left: `${progress}%` }} />
              </div>
            </div>

            <button onClick={changeRate}
              className="shrink-0 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-300 border border-slate-700">
              {rate}×
            </button>
          </div>

          {/* ── Chapter strip — absolute time positioning so playhead aligns exactly ── */}
          <div className="mt-5">
            <div className="relative h-9 rounded-lg overflow-hidden border border-slate-800 bg-slate-900">
              {chapters.map((c, i) => {
                const left = (c.start / duration) * 100;
                const w    = ((c.end - c.start) / duration) * 100;
                const active = c.index === activeChapter.index;
                const color = CHAPTER_COLORS[i % CHAPTER_COLORS.length];
                return (
                  <button key={c.index} onClick={() => seek(c.start + 0.05)}
                    title={`${c.label} (${fmt(c.start)}–${fmt(c.end)})`}
                    className="absolute top-0 h-full transition-colors duration-150 border-r border-slate-950/40 group"
                    style={{
                      left: `${left}%`,
                      width: `${w}%`,
                      backgroundColor: active ? color : `${color}33`,
                    }}>
                    <span className={`absolute inset-0 flex items-center justify-center px-1 text-[10px] font-semibold truncate
                      ${active ? 'text-white' : 'text-slate-400 group-hover:text-slate-200'}`}>
                      {w > 5 ? c.index : ''}
                    </span>
                  </button>
                );
              })}
              {/* playhead */}
              <div className="absolute top-0 bottom-0 w-0.5 bg-white pointer-events-none shadow-[0_0_8px_rgba(255,255,255,0.8)]"
                style={{ left: `${progress}%` }} />
            </div>

            {/* active chapter label + summary */}
            <div className="mt-3 flex items-start gap-3">
              <span className="mt-0.5 px-2 py-0.5 rounded text-xs font-bold text-white shrink-0"
                style={{ backgroundColor: CHAPTER_COLORS[(activeChapter.index - 1) % CHAPTER_COLORS.length] }}>
                {activeChapter.index}/{chapters.length}
              </span>
              <div>
                <div className="text-sm font-semibold text-slate-100">{activeChapter.label}</div>
                <div className="text-xs text-slate-400 mt-0.5">{activeChapter.summary}</div>
              </div>
            </div>

            {/* agent findings — where each agent grounded its judgments */}
            <AgentLanes evaluation={evaluation} duration={duration} time={displayTime} seek={seek} />
          </div>
        </section>

        {/* ── Transcript + live compliance scorecard ──────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 flex-grow min-h-0">
        <section className="lg:col-span-3 bg-slate-950 rounded-xl border border-slate-800 shadow-xl flex flex-col min-h-0">
          <div className="px-5 py-3 border-b border-slate-800 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-300">Conversation</h2>
            <div className="flex gap-4 text-xs items-center">
              <span className="flex items-center gap-1.5 text-indigo-400"><span className="w-2 h-2 rounded-full bg-indigo-500" />Agent</span>
              <span className="flex items-center gap-1.5 text-emerald-400"><span className="w-2 h-2 rounded-full bg-emerald-500" />Customer</span>
              <span className="w-px h-3 bg-slate-700" />
              <span className="flex items-center gap-1.5 text-slate-400 text-[10px]">
                <span className="w-3 h-3 rounded-sm bg-emerald-500/30 border border-emerald-700/50" />positive
              </span>
              <span className="flex items-center gap-1.5 text-slate-400 text-[10px]">
                <span className="w-3 h-3 rounded-sm bg-red-500/30 border border-red-700/50" />negative
              </span>
            </div>
          </div>

          <div ref={scrollRef} className="overflow-y-auto p-5 space-y-3" style={{ maxHeight: '46vh' }}>
            {turns.map((t, i) => {
              const isAgent = t.speaker === 'AGENT';
              const active  = i === activeTurnIdx;
              const sents   = turnSentences[i] || [];
              return (
                <div key={i} ref={active ? activeTurnRef : null}
                  className={`flex ${isAgent ? 'justify-end' : 'justify-start'}`}>
                  <button onClick={() => seek(t.start + 0.02)}
                    className={`max-w-[75%] text-left rounded-2xl px-4 py-2.5 transition-all duration-200 border
                      ${isAgent
                        ? 'bg-indigo-950/60 border-indigo-900 rounded-br-sm'
                        : 'bg-slate-800/60 border-slate-700 rounded-bl-sm'}
                      ${active
                        ? 'ring-2 ring-offset-2 ring-offset-slate-950 scale-[1.01] shadow-lg ' +
                          (isAgent ? 'ring-indigo-400' : 'ring-emerald-400')
                        : 'opacity-70 hover:opacity-100'}`}>
                    <div className={`flex items-center gap-2 mb-1 text-[10px] font-mono uppercase tracking-wide
                      ${isAgent ? 'text-indigo-400' : 'text-emerald-400'}`}>
                      <span>{isAgent ? 'Agent' : 'Customer'}</span>
                      <span className="text-slate-600">{fmt(t.start)}</span>
                    </div>
                    {/* sentence-level emotion color — no labels, pure background */}
                    <div className="text-sm text-slate-200 leading-relaxed">
                      {sents.length > 0
                        ? sents.map((s) => {
                            const sent = sentimentMap.get(s.seq_id);
                            const bg = sent?.sentiment === 'Positive' ? 'bg-emerald-500/20'
                                     : sent?.sentiment === 'Negative' ? 'bg-red-500/25'
                                     : '';
                            return (
                              <span key={s.seq_id} className={`${bg} rounded px-0.5`}>
                                {s.text}{' '}
                              </span>
                            );
                          })
                        : t.text}
                    </div>
                  </button>
                </div>
              );
            })}
          </div>
        </section>

        <CompliancePanel evaluation={evaluation} time={displayTime} seek={seek} fmt={fmt} />
        </div>
      </main>

      <audio ref={audioRef} src="/call_1.mp3" preload="auto" />
    </div>
  );
}
