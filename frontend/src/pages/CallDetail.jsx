import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { fmt } from '../lib/format';
import { CHAPTER_COLORS } from '../lib/constants';
import { BufferedRanges } from '../components/BufferedRanges';
import { CompliancePanel } from '../components/CompliancePanel';
import { AgentLanes } from '../components/AgentLanes';
import { ThemeToggle } from '../components/ThemeToggle';

async function fetchJsonOptional(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

// Route wrapper — the `key` forces a full remount (and therefore a clean
// reset of every piece of player state) whenever the user navigates from
// one call's detail page directly to another's.
export default function CallDetail() {
  const { callId } = useParams();
  return <CallDetailInner key={callId} callId={callId} />;
}

function CallDetailInner({ callId }) {
  const audioRef = useRef(null);
  const scrollRef = useRef(null);
  const activeTurnRef = useRef(null);
  const scrubberRef = useRef(null);

  const [callData, setCallData] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [sentences, setSentences] = useState([]);
  const [sentimentMap, setSentimentMap] = useState(new Map());

  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [dragTime, setDragTime] = useState(0);

  // ── load this call's data ───────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    setCallData(null);
    setLoadError(null);
    fetch(`/calls/${callId}.json`)
      .then((res) => {
        if (!res.ok) throw new Error(`call "${callId}" not found`);
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        setCallData(data);
        setDuration(data.duration || 0);
      })
      .catch((err) => !cancelled && setLoadError(err.message));

    // per-sentence sentiment overlay — only generated for a handful of calls
    // so far; missing files degrade to plain (untinted) transcript turns.
    fetchJsonOptional(`/sentence_segments/${callId}.json`).then((data) => {
      if (cancelled) return;
      setSentences(data?.sentences || []);
    });
    fetchJsonOptional(`/sentiment/${callId}.json`).then((data) => {
      if (cancelled) return;
      const m = new Map();
      (data?.segments || []).forEach((s) => m.set(s.seq_id, s));
      setSentimentMap(m);
    });

    return () => { cancelled = true; };
  }, [callId]);

  const turns = callData?.turns || [];
  const chapters = callData?.chapters || [];
  const evaluation = callData?.evaluation;

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
    const onMeta = () => setDuration(a.duration || callData?.duration || 0);
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
  }, [callData]);

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

  if (loadError) {
    return (
      <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100 flex items-center justify-center flex-col gap-3">
        <p className="text-red-600 dark:text-red-400 text-sm">{loadError}</p>
        <Link to="/" className="text-indigo-600 dark:text-indigo-400 text-sm hover:underline">&larr; Back to dashboard</Link>
      </div>
    );
  }

  if (!callData) {
    return (
      <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100 flex items-center justify-center">
        <p className="text-slate-500 text-sm">Loading call…</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100 flex flex-col">
      {/* Header */}
      <header className="px-6 py-4 border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950 flex justify-between items-center">
        <div>
          <Link to="/" className="text-xs text-indigo-600 dark:text-indigo-400 hover:underline">&larr; All calls</Link>
          <h1 className="text-xl font-bold tracking-tight text-indigo-600 dark:text-indigo-400">Call Compliance Player</h1>
          <p className="text-xs text-slate-500 mt-0.5">{callData.model}</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="px-3 py-1 rounded-full bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300 text-xs font-mono border border-slate-300 dark:border-slate-700">
            {callData.call} · {fmt(duration)} · {chapters.length} chapters
          </span>
          <ThemeToggle />
        </div>
      </header>

      <main className="flex-grow max-w-6xl w-full mx-auto p-6 flex flex-col gap-5">
        {/* ── Player + scrubber ─────────────────────────────────────────── */}
        <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-xl dark:bg-slate-950 dark:border-slate-800">
          <div className="flex items-center gap-4">
            <button onClick={togglePlay}
              className="w-12 h-12 shrink-0 rounded-full bg-indigo-600 hover:bg-indigo-500 transition flex items-center justify-center text-white shadow-lg shadow-indigo-900/40">
              {isPlaying
                ? <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>
                : <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style={{ marginLeft: 2 }}><path d="M8 5v14l11-7z"/></svg>}
            </button>

            <div className="flex-grow">
              <div className="flex justify-between text-xs font-mono text-slate-500 dark:text-slate-400 mb-1.5">
                <span>{fmt(displayTime)}</span>
                <span>{fmt(duration)}</span>
              </div>
              {/* scrubber — click + drag + touch */}
              <div ref={scrubberRef}
                className="relative h-3 bg-slate-200 dark:bg-slate-800 rounded-full cursor-pointer select-none"
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
              className="shrink-0 px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-xs font-mono text-slate-700 border border-slate-300 dark:bg-slate-800 dark:hover:bg-slate-700 dark:text-slate-300 dark:border-slate-700">
              {rate}×
            </button>
          </div>

          {/* ── Chapter strip — absolute time positioning so playhead aligns exactly ── */}
          {chapters.length > 0 && (
          <div className="mt-5">
            <div className="relative h-9 rounded-lg overflow-hidden border border-slate-200 bg-slate-100 dark:border-slate-800 dark:bg-slate-900">
              {chapters.map((c, i) => {
                const left = (c.start / duration) * 100;
                const w    = ((c.end - c.start) / duration) * 100;
                const active = activeChapter && c.index === activeChapter.index;
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
                      ${active ? 'text-white' : 'text-slate-500 group-hover:text-slate-700 dark:text-slate-400 dark:group-hover:text-slate-200'}`}>
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
            {activeChapter && (
              <div className="mt-3 flex items-start gap-3">
                <span className="mt-0.5 px-2 py-0.5 rounded text-xs font-bold text-white shrink-0"
                  style={{ backgroundColor: CHAPTER_COLORS[(activeChapter.index - 1) % CHAPTER_COLORS.length] }}>
                  {activeChapter.index}/{chapters.length}
                </span>
                <div>
                  <div className="text-sm font-semibold text-slate-900 dark:text-slate-100">{activeChapter.label}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{activeChapter.summary}</div>
                </div>
              </div>
            )}

            {/* agent findings — where each agent grounded its judgments */}
            <AgentLanes evaluation={evaluation} duration={duration} time={displayTime} seek={seek} />
          </div>
          )}
        </section>

        {/* ── Transcript + live compliance scorecard ──────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 flex-grow min-h-0">
        <section className="lg:col-span-3 bg-white rounded-xl border border-slate-200 shadow-xl flex flex-col min-h-0 dark:bg-slate-950 dark:border-slate-800">
          <div className="px-5 py-3 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Conversation</h2>
            <div className="flex gap-4 text-xs items-center">
              <span className="flex items-center gap-1.5 text-indigo-600 dark:text-indigo-400"><span className="w-2 h-2 rounded-full bg-indigo-500" />Agent</span>
              <span className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400"><span className="w-2 h-2 rounded-full bg-emerald-500" />Customer</span>
              <span className="w-px h-3 bg-slate-300 dark:bg-slate-700" />
              <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400 text-[10px]">
                <span className="w-3 h-3 rounded-sm bg-emerald-500/30 border border-emerald-700/50" />positive
              </span>
              <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400 text-[10px]">
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
                        ? 'bg-indigo-50 border-indigo-200 rounded-br-sm dark:bg-indigo-950/60 dark:border-indigo-900'
                        : 'bg-slate-100 border-slate-200 rounded-bl-sm dark:bg-slate-800/60 dark:border-slate-700'}
                      ${active
                        ? 'ring-2 ring-offset-2 ring-offset-white dark:ring-offset-slate-950 scale-[1.01] shadow-lg ' +
                          (isAgent ? 'ring-indigo-400' : 'ring-emerald-400')
                        : 'opacity-70 hover:opacity-100'}`}>
                    <div className={`flex items-center gap-2 mb-1 text-[10px] font-mono uppercase tracking-wide
                      ${isAgent ? 'text-indigo-600 dark:text-indigo-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                      <span>{isAgent ? 'Agent' : 'Customer'}</span>
                      <span className="text-slate-400 dark:text-slate-600">{fmt(t.start)}</span>
                    </div>
                    {/* sentence-level emotion color — no labels, pure background */}
                    <div className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed">
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

      <audio ref={audioRef} src={`/audio/${callData.call}.mp3`} preload="auto" />
    </div>
  );
}
