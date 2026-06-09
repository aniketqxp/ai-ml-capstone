import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import callData from './call_data.json';

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
];
const RISK_STYLE = {
  none:     { label: 'No escalation', cls: 'bg-emerald-950 text-emerald-300 border-emerald-800' },
  review:   { label: 'Review',        cls: 'bg-amber-950 text-amber-300 border-amber-800' },
  escalate: { label: 'Escalate',      cls: 'bg-red-950 text-red-300 border-red-800' },
};

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

        {/* quality */}
        <div className="border-t border-slate-800 pt-3 space-y-2">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1.5">Quality · text-derived</div>
          {QUALITY_DIMS.map(([key, name]) => {
            const d = q[key];
            return (
              <div key={key} className="flex items-center gap-2">
                <span className="flex-grow text-xs text-slate-300">
                  {name}
                  {d.requires_audio && <span className="ml-1.5 text-[9px] text-amber-500/90 border border-amber-800/60 rounded px-1">audio</span>}
                </span>
                <ScoreDots score={d.score} />
                <span className="text-xs font-mono text-slate-400 w-4 text-right">{d.score}</span>
              </div>
            );
          })}
        </div>

        {/* escalation */}
        <div className="border-t border-slate-800 pt-3 flex items-center gap-2 flex-wrap">
          <span className={`text-xs font-semibold px-2 py-1 rounded border ${RISK_STYLE[e.risk_level]?.cls || ''}`}>
            {RISK_STYLE[e.risk_level]?.label || e.risk_level}
          </span>
          <span className="text-xs text-slate-400">emotion: <span className="text-slate-300">{e.customer_emotion_text}</span></span>
          {e.requires_audio && <span className="text-[9px] text-amber-500/90 border border-amber-800/60 rounded px-1 py-0.5">audio confirms</span>}
        </div>

        {/* summary */}
        {evaluation.overall_summary && (
          <div className="border-t border-slate-800 pt-3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">Summary</div>
            <p className="text-xs text-slate-400 leading-relaxed">{evaluation.overall_summary}</p>
          </div>
        )}
        <div className="text-[9px] text-slate-600 text-right pt-1">evaluated by {evaluation.served_by}</div>
      </div>
    </section>
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
          </div>
        </section>

        {/* ── Transcript + live compliance scorecard ──────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 flex-grow min-h-0">
        <section className="lg:col-span-3 bg-slate-950 rounded-xl border border-slate-800 shadow-xl flex flex-col min-h-0">
          <div className="px-5 py-3 border-b border-slate-800 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-300">Conversation</h2>
            <div className="flex gap-4 text-xs">
              <span className="flex items-center gap-1.5 text-indigo-400"><span className="w-2 h-2 rounded-full bg-indigo-500" />Agent</span>
              <span className="flex items-center gap-1.5 text-emerald-400"><span className="w-2 h-2 rounded-full bg-emerald-500" />Customer</span>
            </div>
          </div>

          <div ref={scrollRef} className="overflow-y-auto p-5 space-y-3" style={{ maxHeight: '46vh' }}>
            {turns.map((t, i) => {
              const isAgent = t.speaker === 'AGENT';
              const active = i === activeTurnIdx;
              return (
                <div key={i} ref={active ? activeTurnRef : null}
                  className={`flex ${isAgent ? 'justify-end' : 'justify-start'}`}>
                  <button onClick={() => seek(t.start + 0.02)}
                    className={`max-w-[75%] text-left rounded-2xl px-4 py-2.5 transition-all duration-200 border
                      ${isAgent
                        ? 'bg-indigo-950/60 border-indigo-900 rounded-br-sm'
                        : 'bg-slate-800/60 border-slate-700 rounded-bl-sm'}
                      ${active ? 'ring-2 ring-offset-2 ring-offset-slate-950 scale-[1.01] shadow-lg ' +
                        (isAgent ? 'ring-indigo-400' : 'ring-emerald-400') : 'opacity-70 hover:opacity-100'}`}>
                    <div className={`flex items-center gap-2 mb-1 text-[10px] font-mono uppercase tracking-wide
                      ${isAgent ? 'text-indigo-400' : 'text-emerald-400'}`}>
                      <span>{isAgent ? 'Agent' : 'Customer'}</span>
                      <span className="text-slate-600">{fmt(t.start)}</span>
                    </div>
                    <div className="text-sm text-slate-200 leading-snug">{t.text}</div>
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
