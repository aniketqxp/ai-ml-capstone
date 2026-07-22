import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { supabase } from './supabaseClient';
import mockCallData from './call_data.json';
import mockSentenceData from './sentence_segments.json';
import mockSentimentData from './en_CA_Banking_1586889_simple_sentiment.json';

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
    return () => { 
      a.removeEventListener('progress', update); 
      a.removeEventListener('timeupdate', update); 
    };
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

function CompliancePanel({ evaluation, time, seek, fmt }) {
  if (!evaluation) {
    return (
      <section className="lg:col-span-2 bg-slate-950 rounded-xl border border-slate-800 p-6 flex flex-col justify-center items-center text-slate-500">
        <svg className="w-8 h-8 mb-2 animate-pulse text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
        </svg>
        <span className="text-xs">No evaluation scorecard available yet for this call.</span>
      </section>
    );
  }

  const c = evaluation.compliance || {};
  const q = evaluation.quality || {};
  const e = evaluation.escalation || { risk_level: 'none', customer_emotion_text: 'unknown' };

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
            if (passed === null || passed === undefined) { icon = '–'; ring = 'bg-slate-800 text-slate-600'; txt = 'text-slate-600'; }
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
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1.5">Quality · AI evaluated</div>
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
        <div className="text-[9px] text-slate-600 text-right pt-1">rubric {evaluation.rubric_version || '0.2.0'}</div>
      </div>
    </section>
  );
}

export default function App() {
  const audioRef = useRef(null);
  const scrollRef = useRef(null);
  const activeTurnRef = useRef(null);
  const scrubberRef = useRef(null);

  // Supabase states
  const [calls, setCalls] = useState([]);
  const [selectedCallId, setSelectedCallId] = useState(null);
  const [selectedCall, setSelectedCall] = useState(null);
  const [transcripts, setTranscripts] = useState([]);
  const [evaluation, setEvaluation] = useState(null);
  const [sentimentSegments, setSentimentSegments] = useState([]);
  const [jobs, setJobs] = useState({});
  const [loading, setLoading] = useState(true);
  const [dbMode, setDbMode] = useState('loading'); // 'live' | 'mock' | 'loading'
  const [populating, setPopulating] = useState(false);

  // Playback states
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [dragTime, setDragTime] = useState(0);

  // Fetch all calls and current job statuses
  const fetchCallsAndJobs = async () => {
    try {
      // 1. Fetch calls
      const { data: fetchedCalls, error: callErr } = await supabase
        .from('calls')
        .select('*');

      if (callErr) throw callErr;

      // 2. Fetch jobs
      const { data: fetchedJobs, error: jobErr } = await supabase
        .from('jobs')
        .select('*');

      if (jobErr) throw jobErr;

      // 3. Map jobs by call_id
      const jobMap = {};
      if (fetchedJobs) {
        fetchedJobs.forEach(job => {
          jobMap[job.call_id] = job;
        });
      }
      setJobs(jobMap);

      if (fetchedCalls && fetchedCalls.length > 0) {
        setCalls(fetchedCalls);
        setDbMode('live');
        // Select first call by default if none selected
        if (!selectedCallId) {
          setSelectedCallId(fetchedCalls[0].call_id);
        }
      } else {
        // Fallback to mock data if DB is empty
        setDbMode('mock');
        loadMockData();
      }
    } catch (err) {
      console.warn("Failed to connect or fetch from Supabase. Falling back to Mock Data. Error:", err.message);
      setDbMode('mock');
      loadMockData();
    } finally {
      setLoading(false);
    }
  };

  const loadMockData = () => {
    // Treat the static mock call as our only call
    const mockCall = {
      call_id: 'mock-call-1',
      filename: mockCallData.call,
      created_at: new Date().toISOString(),
      call_date: new Date().toISOString(),
      duration_seconds: Math.floor(mockCallData.duration || 636),
      audio_path: '/call_1.mp3'
    };
    setCalls([mockCall]);
    setSelectedCallId('mock-call-1');
    setSelectedCall(mockCall);
    setDuration(mockCallData.duration);
    setEvaluation(mockCallData.evaluation);

    // Map mock turns to matching schema
    const mockTurns = mockCallData.turns.map((t, idx) => ({
      transcript_id: `mock-turn-${idx}`,
      call_id: 'mock-call-1',
      turn_id: idx,
      speaker: t.speaker,
      text: t.text,
      start_time: t.start,
      end_time: t.end
    }));
    setTranscripts(mockTurns);

    // Map mock sentiment segments
    setSentimentSegments(mockSentimentData.segments || []);
  };

  useEffect(() => {
    fetchCallsAndJobs();

    // Subscribe to evaluations updates (real-time scorecard updates when processing finishes)
    const evaluationsSub = supabase
      .channel('realtime-evaluations')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'evaluations' }, (payload) => {
        console.log('Realtime evaluations update:', payload);
        if (selectedCallId && payload.new && payload.new.call_id === selectedCallId) {
          setEvaluation(payload.new.evaluation_data || payload.new);
        }
        // Refresh call list to pull in updated scores
        fetchCallsAndJobs();
      })
      .subscribe();

    // Subscribe to jobs updates (real-time processing status bar)
    const jobsSub = supabase
      .channel('realtime-jobs')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'jobs' }, (payload) => {
        console.log('Realtime jobs update:', payload);
        fetchCallsAndJobs();
      })
      .subscribe();

    return () => {
      supabase.removeChannel(evaluationsSub);
      supabase.removeChannel(jobsSub);
    };
  }, [selectedCallId]);

  // Fetch call details when selection changes
  useEffect(() => {
    if (!selectedCallId) return;

    if (selectedCallId === 'mock-call-1') {
      loadMockData();
      return;
    }

    const loadCallDetails = async () => {
      setLoading(true);
      try {
        // Find current call details
        const callObj = calls.find(c => c.call_id === selectedCallId);
        setSelectedCall(callObj);
        if (callObj) {
          setDuration(callObj.duration_seconds || 0);
        }

        // 1. Fetch transcripts
        const { data: transData, error: transErr } = await supabase
          .from('transcripts')
          .select('*')
          .eq('call_id', selectedCallId)
          .order('turn_id', { ascending: true });

        if (transErr) throw transErr;
        setTranscripts(transData || []);

        // 2. Fetch evaluation
        const { data: evalData, error: evalErr } = await supabase
          .from('evaluations')
          .select('*')
          .eq('call_id', selectedCallId)
          .maybeSingle();

        if (evalErr) throw evalErr;
        
        // Check if evaluation data contains structured object
        if (evalData) {
          // If the DB evaluation contains JSON column `evaluation_data`, use it. Otherwise use the fields directly.
          setEvaluation(evalData.evaluation_data || evalData);
        } else {
          setEvaluation(null);
        }

        // 3. Fetch sentiment segments
        const { data: sentData, error: sentErr } = await supabase
          .from('sentiment_segments')
          .select('*')
          .eq('call_id', selectedCallId);

        if (sentErr) throw sentErr;
        setSentimentSegments(sentData || []);

      } catch (err) {
        console.error("Error loading call details from Supabase:", err.message);
      } finally {
        setLoading(false);
      }
    };

    loadCallDetails();
  }, [selectedCallId, calls]);

  // Map sentiment segment to turn by sequence / timing comparison
  const sentimentMap = useMemo(() => {
    const m = new Map();
    sentimentSegments.forEach(s => {
      // Use turn_id, seq_id, or transcript_id
      const id = s.seq_id || s.transcript_id;
      if (id) m.set(id, s);
    });
    return m;
  }, [sentimentSegments]);

  // Audio configuration
  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    const onTime = () => setCurrentTime(a.currentTime);
    const onMeta = () => {
      if (a.duration) setDuration(a.duration);
    };
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
  }, [selectedCallId]);

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

  const displayTime = isDragging ? dragTime : currentTime;
  const progress = duration ? (displayTime / duration) * 100 : 0;

  // Render chapters (load mock chapters as outline fallback or generate based on transcripts)
  const chapters = useMemo(() => {
    if (selectedCallId === 'mock-call-1') return mockCallData.chapters;
    
    // Generate simple dynamic chapters from transcript if DB doesn't have a chapters table
    if (transcripts.length === 0) return [];
    
    // Create 3 basic segments based on elapsed time to map compliance outline
    const len = duration || 600;
    return [
      { index: 1, label: "Opening & Identity verification", summary: "Greeting and call opening validation", start: 0, end: len * 0.25 },
      { index: 2, label: "Transaction / Inquiry processing", summary: "Processing customer request and validation steps", start: len * 0.25, end: len * 0.75 },
      { index: 3, label: "Closing recap & Upsell", summary: "Summary of changes, next steps and sign off", start: len * 0.75, end: len }
    ];
  }, [selectedCallId, transcripts, duration]);

  const activeChapter = useMemo(() => {
    if (chapters.length === 0) return null;
    for (let i = chapters.length - 1; i >= 0; i--)
      if (displayTime >= chapters[i].start) return chapters[i];
    return chapters[0];
  }, [displayTime, chapters]);

  const activeTurnIdx = useMemo(() => {
    let idx = -1;
    for (let i = 0; i < transcripts.length; i++) {
      if (displayTime >= transcripts[i].start_time) idx = i; else break;
    }
    return idx;
  }, [displayTime, transcripts]);

  useEffect(() => {
    if (activeTurnRef.current && scrollRef.current) {
      activeTurnRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activeTurnIdx]);

  // Demo utility to seed database with sample mock data
  const populateDatabase = async () => {
    if (populating) return;
    setPopulating(true);
    try {
      const agentId = 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11';
      const callId = 'c0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11';
      const jobId = 'f0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11';
      const evaluationId = 'd0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11';

      // 1. Insert agent
      await supabase.from('agents').upsert([{
        agent_id: agentId,
        name: 'Emily Davis',
        team: 'Customer Support'
      }]);

      // 2. Insert call
      await supabase.from('calls').upsert([{
        call_id: callId,
        agent_id: agentId,
        audio_path: '/call_1.mp3',
        call_date: new Date().toISOString(),
        duration_seconds: Math.floor(mockCallData.duration || 636)
      }]);

      // 3. Insert job
      await supabase.from('jobs').upsert([{
        job_id: jobId,
        call_id: callId,
        status: 'complete',
        stage: 'evaluation',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }]);

      // 4. Insert transcripts
      const dbTurns = mockCallData.turns.map((t, idx) => ({
        transcript_id: `turn-id-uuid-${idx}`,
        call_id: callId,
        turn_id: idx,
        speaker: t.speaker,
        text: t.text,
        start_time: t.start,
        end_time: t.end
      }));
      await supabase.from('transcripts').upsert(dbTurns);

      // 5. Insert evaluations
      await supabase.from('evaluations').upsert([{
        evaluation_id: evaluationId,
        call_id: callId,
        agent_id: agentId,
        evaluation_data: mockCallData.evaluation,
        created_at: new Date().toISOString()
      }]);

      // 6. Insert sentiment segments
      const dbSentiment = mockSentimentData.segments.map((s, idx) => ({
        id: idx + 1,
        call_id: callId,
        seq_id: s.seq_id,
        sentiment: s.sentiment,
        emotion: s.emotion,
        confidence: s.confidence
      }));
      await supabase.from('sentiment_segments').upsert(dbSentiment);

      alert("Successfully seeded Supabase with sample banking call data!");
      fetchCallsAndJobs();
    } catch (err) {
      alert("Error seeding database: " + err.message);
      console.error(err);
    } finally {
      setPopulating(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
      {/* Header */}
      <header className="px-6 py-4 border-b border-slate-800 bg-slate-950 flex flex-wrap justify-between items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-indigo-400">Call Compliance Dashboard</h1>
            <p className="text-xs text-slate-500 mt-0.5">
              {dbMode === 'live' ? '🟢 Connected to Supabase Live DB' : '🟡 Offline Mode (Showing Local Mock Files)'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {dbMode === 'mock' && (
            <button 
              onClick={populateDatabase}
              disabled={populating}
              className="px-4 py-2 text-xs font-semibold rounded-lg bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white transition shadow shadow-indigo-900/40 disabled:opacity-50"
            >
              {populating ? 'Seeding...' : '🔌 Seed Supabase with Sample Call'}
            </button>
          )}
          <span className="px-3 py-1.5 rounded-full bg-slate-800 text-slate-300 text-xs font-mono border border-slate-700">
            {selectedCall ? selectedCall.filename : 'No Call Selected'}
          </span>
        </div>
      </header>

      {/* Main Grid */}
      <div className="flex-grow flex flex-col md:flex-row min-h-0">
        {/* Sidebar - Call List */}
        <aside className="w-full md:w-80 border-r border-slate-800 bg-slate-950 p-4 flex flex-col gap-4">
          <h2 className="text-xs uppercase tracking-wider font-semibold text-slate-500 px-1">Recorded Calls</h2>
          <div className="flex-grow overflow-y-auto space-y-2 max-h-[30vh] md:max-h-none">
            {calls.map((c) => {
              const isActive = c.call_id === selectedCallId;
              const jobStatus = jobs[c.call_id]?.status || 'complete';
              
              let statusBadge = 'bg-slate-800 text-slate-400 border-slate-700';
              if (jobStatus === 'processing') statusBadge = 'bg-amber-950/50 text-amber-400 border-amber-800/50 animate-pulse';
              if (jobStatus === 'complete') statusBadge = 'bg-emerald-950/40 text-emerald-400 border-emerald-800/40';
              if (jobStatus === 'failed') statusBadge = 'bg-red-950/50 text-red-400 border-red-800/50';
              if (jobStatus === 'queued') statusBadge = 'bg-indigo-950/50 text-indigo-400 border-indigo-800/50';

              return (
                <button
                  key={c.call_id}
                  onClick={() => setSelectedCallId(c.call_id)}
                  className={`w-full text-left p-3 rounded-xl border transition-all duration-200 flex flex-col gap-1.5
                    ${isActive 
                      ? 'bg-slate-800/80 border-indigo-500/50 shadow-md shadow-indigo-950/10' 
                      : 'bg-slate-900/40 border-slate-800/60 hover:bg-slate-900/80'}`}
                >
                  <div className="flex justify-between items-start w-full">
                    <span className="font-semibold text-xs text-slate-200 truncate pr-2 max-w-[140px]">{c.filename}</span>
                    <span className={`text-[9px] font-mono uppercase px-1.5 py-0.5 rounded border ${statusBadge}`}>
                      {jobStatus}
                    </span>
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>{fmt(c.duration_seconds || 0)}</span>
                    <span>{new Date(c.call_date).toLocaleDateString()}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </aside>

        {/* Dynamic Detail Dashboard */}
        <main className="flex-grow p-6 flex flex-col gap-5 overflow-y-auto">
          {loading ? (
            <div className="flex-grow flex items-center justify-center flex-col gap-3">
              <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-sm text-slate-400">Loading call details from Supabase...</span>
            </div>
          ) : !selectedCallId ? (
            <div className="flex-grow flex items-center justify-center flex-col text-slate-500">
              <svg className="w-16 h-16 mb-4 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
              <h3 className="text-base font-semibold">Select a Call to Analyze</h3>
              <p className="text-xs text-slate-600 mt-1">Pick a file from the recorded list sidebar to see compliance metrics.</p>
            </div>
          ) : (
            <>
              {/* Audio player card */}
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
                    {/* scrubber */}
                    <div ref={scrubberRef}
                      className="relative h-3 bg-slate-800 rounded-full cursor-pointer select-none"
                      onMouseDown={onScrubStart}
                      onTouchStart={onScrubStart}>
                      <BufferedRanges audio={audioRef} duration={duration} currentTime={currentTime} />
                      <div className="absolute inset-y-0 left-0 bg-indigo-500 rounded-full pointer-events-none"
                        style={{ width: `${progress}%` }} />
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

                {/* Chapter strip */}
                {chapters.length > 0 && (
                  <div className="mt-5">
                    <div className="relative h-9 rounded-lg overflow-hidden border border-slate-800 bg-slate-900">
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
                              ${active ? 'text-white' : 'text-slate-400 group-hover:text-slate-200'}`}>
                              {w > 5 ? c.index : ''}
                            </span>
                          </button>
                        );
                      })}
                      <div className="absolute top-0 bottom-0 w-0.5 bg-white pointer-events-none shadow-[0_0_8px_rgba(255,255,255,0.8)]"
                        style={{ left: `${progress}%` }} />
                    </div>

                    {activeChapter && (
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
                    )}
                  </div>
                )}
              </section>

              {/* Transcript & QA grid */}
              <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 flex-grow min-h-0">
                {/* Conversation Box */}
                <section className="lg:col-span-3 bg-slate-950 rounded-xl border border-slate-800 shadow-xl flex flex-col min-h-0">
                  <div className="px-5 py-3 border-b border-slate-800 flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-slate-300">Conversation</h2>
                    <div className="flex gap-4 text-xs items-center">
                      <span className="flex items-center gap-1.5 text-indigo-400"><span className="w-2 h-2 rounded-full bg-indigo-500" />Agent</span>
                      <span className="flex items-center gap-1.5 text-emerald-400"><span className="w-2 h-2 rounded-full bg-emerald-500" />Customer</span>
                    </div>
                  </div>

                  <div ref={scrollRef} className="overflow-y-auto p-5 space-y-3" style={{ maxHeight: '46vh' }}>
                    {transcripts.map((t, i) => {
                      const isAgent = t.speaker === 'AGENT';
                      const active  = i === activeTurnIdx;
                      
                      // Match sentiment segment using turn_id or sequence mapping
                      const sent = sentimentMap.get(t.turn_id) || sentimentMap.get(t.transcript_id);
                      const bg = sent?.sentiment === 'Positive' ? 'bg-emerald-500/20'
                               : sent?.sentiment === 'Negative' ? 'bg-red-500/25'
                               : '';

                      return (
                        <div key={t.transcript_id || i} ref={active ? activeTurnRef : null}
                          className={`flex ${isAgent ? 'justify-end' : 'justify-start'}`}>
                          <button onClick={() => seek(t.start_time + 0.02)}
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
                              <span className="text-slate-600">{fmt(t.start_time)}</span>
                            </div>
                            <div className="text-sm text-slate-200 leading-relaxed">
                              <span className={`${bg} rounded px-0.5`}>
                                {t.text}{' '}
                              </span>
                            </div>
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </section>

                {/* Scorecard Panel */}
                <CompliancePanel evaluation={evaluation} time={displayTime} seek={seek} fmt={fmt} />
              </div>
            </>
          )}
        </main>
      </div>

      <audio ref={audioRef} src={selectedCall ? selectedCall.audio_path : ''} preload="auto" />
    </div>
  );
}
