import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import callData from './call_data.json';
import sentenceData from './sentence_segments.json';
const API_BASE_URL = 'http://127.0.0.1:8000';
const DEFAULT_SENTIMENT_CALL_ID = 'en_CA_Banking_1586889';

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

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return 'N/A';
  }
  return Number(value).toFixed(digits);
};

const formatLabel = (value) => {
  if (value === null || value === undefined || value === '') {
    return 'N/A';
  }
  return String(value);
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
            if (!d) return null;
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
        <div className="text-[9px] text-slate-600 text-right pt-1">rubric {evaluation.rubric_version}{evaluation.served_by ? ` · ${evaluation.served_by}` : ''}</div>
      </div>
    </section>
  );
}

function AudioFeatureTimeline({ series, duration, currentTime, seek, formatNumber }) {
  const [selectedMetric, setSelectedMetric] = useState('escalation_score');

  if (!series || series.length === 0 || !duration) return null;

  const metrics = [
    {
      key: 'escalation_score',
      label: 'Escalation',
      color: '#f97316',
      format: (v) => formatNumber(v, 3),
    },
    {
      key: 'volume_db_mean',
      label: 'Volume',
      color: '#38bdf8',
      format: (v) => `${formatNumber(v)} dB`,
    },
    {
      key: 'pitch_mean_hz',
      label: 'Pitch',
      color: '#a78bfa',
      format: (v) => `${formatNumber(v)} Hz`,
    },
    {
      key: 'speech_rate_words_per_minute',
      label: 'Speech Rate',
      color: '#22c55e',
      format: (v) => `${formatNumber(v)} WPM`,
    },
  ];

  const selectedMetricConfig = metrics.find((m) => m.key === selectedMetric) || metrics[0];

  const getTime = (point) => {
    if (point.start_time !== null && point.start_time !== undefined) return Number(point.start_time);
    if (point.start !== null && point.start !== undefined) return Number(point.start);
    return 0;
  };

  const getEndTime = (point) => {
    if (point.end_time !== null && point.end_time !== undefined) return Number(point.end_time);
    if (point.end !== null && point.end !== undefined) return Number(point.end);
    return getTime(point) + 0.5;
  };

  const getSentimentColor = (sentiment) => {
    if (sentiment === 'Positive') return '#22c55e';
    if (sentiment === 'Negative') return '#ef4444';
    if (sentiment === 'Neutral') return '#64748b';
    if (sentiment === 'Mixed') return '#eab308';
    return '#334155';
  };

  const currentPoint =
    series.find((point) => {
      const start = getTime(point);
      const end = getEndTime(point);
      return currentTime >= start && currentTime <= end;
    }) || series.reduce((closest, point) => {
      const t = getTime(point);
      const closestTime = getTime(closest);
      return Math.abs(t - currentTime) < Math.abs(closestTime - currentTime) ? point : closest;
    }, series[0]);

  const playheadX = Math.max(0, Math.min((currentTime / duration) * 100, 100));

  const buildPath = (metricKey) => {
    const values = series
      .map((point) => Number(point[metricKey]))
      .filter((value) => Number.isFinite(value));

    if (values.length === 0) return '';

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    return series
      .map((point) => {
        const x = Math.max(0, Math.min((getTime(point) / duration) * 100, 100));
        const rawValue = Number(point[metricKey]);
        const safeValue = Number.isFinite(rawValue) ? rawValue : min;
        const y = 34 - ((safeValue - min) / range) * 28;
        return `${x},${y}`;
      })
      .join(' ');
  };

  return (
    <div className="mt-4 rounded-xl border border-slate-800 bg-slate-900/70 p-4">
      <div className="flex flex-col gap-1 mb-3">
        <h3 className="text-sm font-semibold text-slate-300">
          Sentiment & Audio Timeline
        </h3>
        <p className="text-xs text-slate-500">
          Choose one metric to view over time. Sentiment is shown underneath so the audio pattern can be compared with positive, negative, or neutral moments.
        </p>
      </div>

      <div className="mb-4 flex flex-wrap gap-3">
        {metrics.map((metric) => (
          <label
            key={metric.key}
            className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-xs cursor-pointer transition
              ${selectedMetric === metric.key
                ? 'border-indigo-500 bg-indigo-950/40 text-white'
                : 'border-slate-800 bg-slate-950/60 text-slate-400 hover:text-slate-200'}`}
          >
            <input
              type="checkbox"
              checked={selectedMetric === metric.key}
              onChange={() => setSelectedMetric(metric.key)}
              className="accent-indigo-500"
            />
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: metric.color }}
            />
            {metric.label}
          </label>
        ))}
      </div>

      <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="rounded-lg bg-slate-950/60 p-3 border border-slate-800">
          <p className="text-xs text-slate-500">Current metric</p>
          <div className="mt-1 flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: selectedMetricConfig.color }}
            />
            <p className="font-semibold text-slate-200">
              {selectedMetricConfig.label}: {selectedMetricConfig.format(currentPoint?.[selectedMetricConfig.key])}
            </p>
          </div>
        </div>

        <div className="rounded-lg bg-slate-950/60 p-3 border border-slate-800">
          <p className="text-xs text-slate-500">Current sentiment</p>
          <div className="mt-1 flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: getSentimentColor(currentPoint?.sentiment) }}
            />
            <p className="font-semibold text-slate-200">
              {formatLabel(currentPoint?.sentiment)}
            </p>
          </div>
        </div>

        <div className="rounded-lg bg-slate-950/60 p-3 border border-slate-800">
          <p className="text-xs text-slate-500">Current emotion</p>
          <p className="mt-1 font-semibold text-slate-200">
            {formatLabel(currentPoint?.dominant_emotion)}
          </p>
        </div>
      </div>

      <div
        className="relative rounded-lg border border-slate-800 bg-slate-950/80 p-3 cursor-pointer"
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const nextTime = (x / rect.width) * duration;
          seek(nextTime);
        }}
      >
        <svg viewBox="0 0 100 48" preserveAspectRatio="none" className="h-56 w-full">
          {[4, 12, 20, 28, 36].map((y) => (
            <line
              key={y}
              x1="0"
              x2="100"
              y1={y}
              y2={y}
              stroke="#1e293b"
              strokeWidth="0.25"
            />
          ))}

          <polyline
            points={buildPath(selectedMetricConfig.key)}
            fill="none"
            stroke={selectedMetricConfig.color}
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />

          {series.map((point) => {
            const startX = Math.max(0, Math.min((getTime(point) / duration) * 100, 100));
            const endX = Math.max(startX + 0.15, Math.min((getEndTime(point) / duration) * 100, 100));
            return (
              <rect
                key={`${point.segment_index}-${point.seq_id}`}
                x={startX}
                y="42"
                width={Math.max(endX - startX, 0.15)}
                height="4"
                fill={getSentimentColor(point.sentiment)}
                opacity="0.85"
              />
            );
          })}

          <line
            x1={playheadX}
            x2={playheadX}
            y1="0"
            y2="48"
            stroke="#ffffff"
            strokeWidth="0.6"
            vectorEffect="non-scaling-stroke"
          />
        </svg>

        <div className="mt-3 flex flex-wrap gap-4 text-xs">
          <span className="flex items-center gap-1.5 text-slate-400">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: selectedMetricConfig.color }}
            />
            {selectedMetricConfig.label}
          </span>

          <span className="flex items-center gap-1.5 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            Positive sentiment
          </span>

          <span className="flex items-center gap-1.5 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-red-500" />
            Negative sentiment
          </span>

          <span className="flex items-center gap-1.5 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-slate-500" />
            Neutral sentiment
          </span>
        </div>

        <p className="mt-2 text-[10px] text-slate-600">
          Click anywhere on the chart to jump to that moment in the call.
        </p>
      </div>
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
  const [sentimentPayload, setSentimentPayload] = useState(null);
  const [sentimentLoading, setSentimentLoading] = useState(true);
  const [sentimentError, setSentimentError] = useState(null);

  const { turns, chapters, evaluation } = callData;
  const sentences = sentenceData.sentences;
  const callSummary = sentimentPayload?.call_summary || {};
  const audioSummary = sentimentPayload?.audio_feature_summary || {};
  const speakerSummary = sentimentPayload?.speaker_audio_feature_summary || {};
  const customerSummary = speakerSummary?.customer || {};
  const agentSummary = speakerSummary?.agent || {};
  const customerTrend = sentimentPayload?.customer_escalation_trend || {};
  const featureSeries = sentimentPayload?.dashboard_audio_feature_series || [];
  const managerReview = sentimentPayload?.manager_review_recommendation || {};
  const topRiskySegments = sentimentPayload?.top_risky_segments || [];
  const calibratedSummary = sentimentPayload?.calibrated_sentiment_summary || {};

  const unreliableSpeechCount = sentimentPayload?.segments?.filter(
    (s) => s?.audio_features?.audio_quality_flags?.unrealistic_speech_rate
  ).length || 0;

  const maxEscalation = featureSeries.length > 0
    ? Math.max(...featureSeries.map((s) => Number(s.escalation_score || 0)))
    : null;

  const maxVolume = featureSeries.length > 0
    ? Math.max(...featureSeries.map((s) => Number(s.volume_db_mean ?? -999)))
    : null;

  useEffect(() => {
    async function loadSentiment() {
      try {
        setSentimentLoading(true);
        setSentimentError(null);

        const response = await fetch(`${API_BASE_URL}/calls/${DEFAULT_SENTIMENT_CALL_ID}/sentiment`);

        if (!response.ok) {
          throw new Error(`Backend returned ${response.status}`);
        }

        const data = await response.json();
        setSentimentPayload(data);
      } catch (error) {
        console.error('Failed to load sentiment from backend:', error);
        setSentimentError(error.message);
      } finally {
        setSentimentLoading(false);
      }
    }

    loadSentiment();
  }, []);

  // seq_id → sentiment entry
  const sentimentMap = useMemo(() => {
    const m = new Map();

    if (!sentimentPayload?.segments) {
      return m;
    }

    sentimentPayload.segments.forEach((s) => {
      m.set(s.seq_id, s);
    });

    return m;
  }, [sentimentPayload]);

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
          <p className="text-[10px] text-slate-600 mt-1">
      Sentiment source:{' '}
      {sentimentLoading
        ? 'loading from backend...'
        : sentimentError
          ? `backend error: ${sentimentError}`
          : `backend API · ${sentimentPayload?.model_version}`}
    </p>
        </div>
        <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-300 text-xs font-mono border border-slate-700">
          {callData.call} · {fmt(duration)} · {chapters.length} chapters
        </span>
      </header>

      <main className="flex-grow max-w-6xl w-full mx-auto p-6 flex flex-col gap-5">
              {/* ── Sentiment + audio feature summary ─────────────────────────── */}
        {sentimentPayload && (
          <section className="bg-slate-950 rounded-xl border border-slate-800 p-5 shadow-xl">
            <div className="flex flex-col gap-1 mb-4">
              <h2 className="text-sm font-semibold text-slate-300">
                Sentiment, Escalation & Audio Insights
              </h2>
              <p className="text-xs text-slate-500">
                Wav2Vec2 sentiment output combined with explainable audio features.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-5">
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <p className="text-xs uppercase text-slate-500">Risk Level</p>
                <p className="mt-1 text-2xl font-semibold text-white">
                  {formatLabel(callSummary.risk_level)}
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
  <p className="text-xs uppercase text-slate-500">Overall Sentiment</p>
  <p className="mt-1 text-2xl font-semibold text-white">
    {formatLabel(callSummary.dominant_sentiment)}
  </p>
</div>
<div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
  <p className="text-xs uppercase text-slate-500">Customer Sentiment</p>
<p className="mt-1 text-2xl font-semibold text-white">
  {formatLabel(
    calibratedSummary.customer_sentiment_calibrated ||
    customerSummary.dominant_sentiment
  )}
</p>
</div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <p className="text-xs uppercase text-slate-500">Avg Volume</p>
                <p className="mt-1 text-2xl font-semibold text-white">
                  {formatNumber(audioSummary.average_volume_db)} dB
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <p className="text-xs uppercase text-slate-500">Avg Speech Rate</p>
                <p className="mt-1 text-2xl font-semibold text-white">
                  {formatNumber(audioSummary.average_speech_rate_wpm)} WPM
                </p>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <p className="text-xs uppercase text-slate-500">Customer Escalation Trend</p>
                <p className="mt-1 text-xl font-semibold text-white">
                  {formatLabel(customerTrend.trend)}
                </p>

                <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
                  <div>
                    <p className="text-slate-500">First half</p>
                    <p className="font-semibold text-slate-200">
                      {formatNumber(customerTrend.first_half_average_escalation, 3)}
                    </p>
                  </div>
                  <div>
                    <p className="text-slate-500">Second half</p>
                    <p className="font-semibold text-slate-200">
                      {formatNumber(customerTrend.second_half_average_escalation, 3)}
                    </p>
                  </div>
                  <div>
                    <p className="text-slate-500">Delta</p>
                    <p className="font-semibold text-slate-200">
                      {formatNumber(customerTrend.trend_delta, 3)}
                    </p>
                  </div>
                </div>

                <p className="mt-3 text-xs text-slate-400">
                  {formatLabel(customerTrend.trend_explanation)}
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <p className="text-xs uppercase text-slate-500">Feature Series Check</p>
                <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
                  <div>
                    <p className="text-slate-500">Segments</p>
                    <p className="font-semibold text-slate-200">{featureSeries.length}</p>
                  </div>
                  <div>
                    <p className="text-slate-500">Max escalation</p>
                    <p className="font-semibold text-slate-200">{formatNumber(maxEscalation, 3)}</p>
                  </div>
                  <div>
                    <p className="text-slate-500">Speech-rate warnings</p>
                    <p className="font-semibold text-slate-200">{unreliableSpeechCount}</p>
                  </div>
                </div>
                <p className="mt-3 text-xs text-slate-400">
                  Max volume: {formatNumber(maxVolume)} dB · Audio features loaded: {String(sentimentPayload.has_audio_features)}
                </p>
              </div>
            </div>
              {sentimentPayload?.manager_review_recommendation && (
              <div className={`mt-4 rounded-xl border p-4 ${
                managerReview.review_required
                  ? 'border-amber-800 bg-amber-950/20'
                  : 'border-emerald-800 bg-emerald-950/20'
              }`}>
                <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
                  <div>
                    <p className="text-xs uppercase text-slate-500">
                      Manager Review Recommendation
                    </p>

                    <p className="mt-1 text-xl font-semibold text-white">
                      {managerReview.review_required ? 'Review Recommended' : 'No Review Required'}
                    </p>

                    <p className="mt-1 text-xs text-slate-400">
                      Review level: {formatLabel(managerReview.review_level)}
                    </p>
                  </div>

                  <div className="rounded-lg bg-slate-950/60 px-3 py-2 border border-slate-800">
                    <p className="text-xs text-slate-500">Max escalation</p>
                    <p className="text-lg font-semibold text-slate-200">
                      {formatNumber(managerReview.max_escalation_score, 3)}
                    </p>
                  </div>
                </div>

                {managerReview.reasons?.length > 0 && (
                  <div className="mt-3">
                    <p className="text-xs uppercase text-slate-500 mb-2">Reasons</p>

                    <ul className="space-y-1">
                      {managerReview.reasons.map((reason, index) => (
                        <li key={index} className="text-xs text-slate-300">
                          • {reason}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {managerReview.notes?.length > 0 && (
  <div className="mt-3 border-t border-slate-800 pt-3">
    <p className="text-xs uppercase text-slate-500 mb-2">Model Notes</p>

    <ul className="space-y-1">
      {managerReview.notes.map((note, index) => (
        <li key={index} className="text-xs text-slate-400">
          • {note}
        </li>
      ))}
    </ul>
  </div>
)}
              </div>
            )}

            {topRiskySegments.length > 0 && (

              <div className="mt-4 rounded-xl border border-slate-800 bg-slate-900/70 p-4">

                <div className="flex flex-col gap-1 mb-3">

                  <h3 className="text-sm font-semibold text-slate-300">

                    Flagged Sentiment Moments

                  </h3>

                  <p className="text-xs text-slate-500">

                    These are model-flagged moments with stronger context signals. They are supporting evidence, not automatic escalations.

                  </p>

                </div>

                <div className="space-y-3">

                  {topRiskySegments.map((segment) => (

                    <button

                      key={`${segment.segment_index}-${segment.seq_id}`}

                      onClick={() => seek(Number(segment.start_time || 0) + 0.02)}

                      className="w-full rounded-lg border border-slate-800 bg-slate-950/60 p-3 text-left hover:border-amber-700 hover:bg-amber-950/10 transition"

                    >

                      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">

                        <div>

                          <div className="flex flex-wrap items-center gap-2">

                            <span className="text-xs font-mono text-amber-300">

                              {fmt(Number(segment.start_time || 0))}

                            </span>

                            <span className="text-xs text-slate-500">

                              {formatLabel(segment.speaker)}

                            </span>

                            <span className={`rounded-full border px-2 py-0.5 text-[10px] ${

                              segment.sentiment === 'Negative'

                                ? 'border-red-800 bg-red-950/40 text-red-300'

                                : segment.sentiment === 'Positive'

                                  ? 'border-emerald-800 bg-emerald-950/40 text-emerald-300'

                                  : 'border-slate-700 bg-slate-800 text-slate-300'

                            }`}>

                              {formatLabel(segment.sentiment)}

                            </span>

                            <span className="rounded-full border border-slate-700 bg-slate-800 px-2 py-0.5 text-[10px] text-slate-300">

                              {formatLabel(segment.dominant_emotion)}

                            </span>

                          </div>

                          <p className="mt-2 text-sm text-slate-200 leading-relaxed">

                            {formatLabel(segment.text)}

                          </p>

                        </div>

                        <div className="flex gap-2 md:flex-col md:text-right">

                          <div>

                            <p className="text-[10px] uppercase text-slate-500">

                              Escalation

                            </p>

                            <p className="text-xs font-semibold text-slate-200">

                              {formatNumber(segment.escalation_score, 3)}

                            </p>

                          </div>

                          <div>

                            <p className="text-[10px] uppercase text-slate-500">

                              Flag score

                            </p>

                            <p className="text-xs font-semibold text-slate-200">

                              {formatNumber(segment.review_score, 3)}

                            </p>

                          </div>

                        </div>

                      </div>

                      {segment.reasons?.length > 0 && (

                        <div className="mt-2 flex flex-wrap gap-1.5">

                          {segment.reasons.slice(0, 3).map((reason, index) => (

                            <span

                              key={index}

                              className="rounded-full border border-amber-900/60 bg-amber-950/20 px-2 py-0.5 text-[10px] text-amber-200"

                            >

                              {reason}

                            </span>

                          ))}

                        </div>

                      )}

                      <p className="mt-2 text-[10px] text-slate-600">

                        Click to jump to this moment in the call.

                      </p>

                    </button>

                  ))}

                </div>

              </div>

            )}

            

            {customerSummary?.total_segments !== undefined && agentSummary?.total_segments !== undefined && (
              <div className="mt-4 rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <h3 className="text-sm font-semibold text-slate-300">Customer vs Agent Summary</h3>

                <div className="mt-3 overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="text-slate-500">
                      <tr>
                        <th className="py-2">Metric</th>
                        <th className="py-2">Customer</th>
                        <th className="py-2">Agent</th>
                      </tr>
                    </thead>
                    <tbody className="text-slate-300">
                      <tr className="border-t border-slate-800">
                        <td className="py-2 text-slate-500">Dominant sentiment</td>
                        <td>{formatLabel(customerSummary.dominant_sentiment)}</td>
                        <td>{formatLabel(agentSummary.dominant_sentiment)}</td>
                      </tr>
                      <tr className="border-t border-slate-800">
                        <td className="py-2 text-slate-500">Dominant emotion</td>
                        <td>{formatLabel(customerSummary.dominant_emotion)}</td>
                        <td>{formatLabel(agentSummary.dominant_emotion)}</td>
                      </tr>
                      <tr className="border-t border-slate-800">
                        <td className="py-2 text-slate-500">Avg escalation</td>
                        <td>{formatNumber(customerSummary.average_escalation_score, 3)}</td>
                        <td>{formatNumber(agentSummary.average_escalation_score, 3)}</td>
                      </tr>
                      <tr className="border-t border-slate-800">
                        <td className="py-2 text-slate-500">Avg speech rate</td>
                        <td>{formatNumber(customerSummary.average_speech_rate_wpm)} WPM</td>
                        <td>{formatNumber(agentSummary.average_speech_rate_wpm)} WPM</td>
                      </tr>
                      <tr className="border-t border-slate-800">
                        <td className="py-2 text-slate-500">High pause segments</td>
                        <td>{formatLabel(customerSummary.high_pause_segments)}</td>
                        <td>{formatLabel(agentSummary.high_pause_segments)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
                        <AudioFeatureTimeline
              series={featureSeries}
              duration={duration}
              currentTime={displayTime}
              seek={seek}
              formatNumber={formatNumber}
            />
          </section>
        )}

        {/* ── Player + scrubber ─────────────────────────────────────────── */}
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
                            const audioFlags = sent?.audio_features?.audio_quality_flags || {};
                            const explanations = sent?.escalation_explanation || [];

                            const bg = sent?.sentiment === 'Positive' ? 'bg-emerald-500/20'
                                     : sent?.sentiment === 'Negative' ? 'bg-red-500/25'
                                     : '';

                            return (
                              <span key={s.seq_id} className="inline">
                                <span className={`${bg} rounded px-0.5`}>
                                  {s.text}{' '}
                                </span>

                                {(audioFlags.unrealistic_speech_rate || audioFlags.low_energy_segment) && (
                                  <span className="ml-1 inline-flex gap-1 align-middle">
                                    {audioFlags.unrealistic_speech_rate && (
                                      <span className="rounded-full bg-amber-500/20 px-1.5 py-0.5 text-[9px] text-amber-200 border border-amber-700/50">
                                        speech rate unreliable
                                      </span>
                                    )}

                                    {audioFlags.low_energy_segment && (
                                      <span className="rounded-full bg-blue-500/20 px-1.5 py-0.5 text-[9px] text-blue-200 border border-blue-700/50">
                                        low energy
                                      </span>
                                    )}
                                  </span>
                                )}

                                {explanations.length > 0 &&
  sent?.sentiment === 'Negative' &&
  topRiskySegments.some((item) => item.seq_id === s.seq_id) && (
    <span className="block mt-1 text-[10px] text-slate-400">
      Flagged model signal: {explanations.slice(0, 2).join(' · ')}
    </span>
)}
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
