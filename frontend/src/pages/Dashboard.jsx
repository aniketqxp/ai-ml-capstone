import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  AudioLines,
  BrainCircuit,
  Check,
  CircleCheck,
  CircleHelp,
  ClipboardCheck,
  Clock3,
  CloudUpload,
  FileAudio,
  LoaderCircle,
  Minus,
  Play,
  ScanText,
  ShieldCheck,
  TriangleAlert,
} from 'lucide-react';
import { apiUrl } from '../api';
import { ThemeToggle } from '../components/ThemeToggle';
import { fmt } from '../lib/format';

const PAGE_SIZE = 10;
const ACTIVE_STATUSES = new Set(['queued', 'processing']);
const ASPECTS = [
  ['request', 'Request'],
  ['process', 'Process'],
  ['experience', 'Experience'],
  ['outcome', 'Outcome'],
];
const STAGE_ICONS = {
  queued: Clock3,
  transcript: FileAudio,
  segments: ScanText,
  acoustic: AudioLines,
  evaluation: BrainCircuit,
  evidence: ScanText,
  requirements: ClipboardCheck,
  findings: ScanText,
  decision: ShieldCheck,
  presentation: BrainCircuit,
  publish: CloudUpload,
  ready: CircleCheck,
};
const RESULT_ORDER = {
  review: 0,
  uncertain: 1,
  processing: 2,
  ok: 3,
  ready: 4,
  unsupported: 5,
};

function callResult(call) {
  if (ACTIVE_STATUSES.has(call.status)) return 'processing';
  if (call.result) return call.result;
  if (!call.evaluation_supported) return 'unsupported';
  return 'ready';
}

function AspectMark({ aspect, label }) {
  const state = aspect?.state || 'uncertain';
  const config = {
    ok: {
      Icon: Check,
      label: `${label}: satisfactory`,
      cls: 'border-emerald-500 bg-emerald-50 text-emerald-700 dark:border-emerald-600 dark:bg-emerald-950 dark:text-emerald-300',
    },
    concern: {
      Icon: TriangleAlert,
      label: `${label}: concern`,
      cls: 'border-red-400 bg-red-50 text-red-700 dark:border-red-700 dark:bg-red-950 dark:text-red-300',
    },
    uncertain: {
      Icon: CircleHelp,
      label: `${label}: insufficient evidence`,
      cls: 'border-amber-400 bg-amber-50 text-amber-700 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-300',
    },
  }[state];
  const { Icon } = config;
  return (
    <span
      className={`mx-auto flex h-7 w-7 items-center justify-center rounded-full border ${config.cls}`}
      title={aspect?.summary || config.label}
      aria-label={config.label}
    >
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
    </span>
  );
}

function SignalLegend() {
  const items = [
    ['Clear', CircleCheck, 'text-emerald-600 dark:text-emerald-400'],
    ['Review', TriangleAlert, 'text-red-600 dark:text-red-400'],
    ['Uncertain', CircleHelp, 'text-amber-600 dark:text-amber-400'],
    ['Processing', LoaderCircle, 'text-[#1557ff] dark:text-blue-300'],
  ];
  return (
    <div className="flex items-center justify-center gap-5 normal-case sm:gap-7" aria-label="Signal legend">
      {items.map(([label, Icon, tone]) => (
        <span key={label} className="inline-flex items-center gap-1.5 whitespace-nowrap text-[10px] font-semibold text-slate-500 dark:text-slate-400">
          <Icon className={`h-3.5 w-3.5 ${tone}`} aria-hidden="true" />
          {label}
        </span>
      ))}
    </div>
  );
}

function ResultBadge({ result }) {
  const config = {
    review: {
      label: 'Review',
      Icon: TriangleAlert,
      cls: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300',
    },
    uncertain: {
      label: 'Uncertain',
      Icon: CircleHelp,
      cls: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300',
    },
    processing: {
      label: 'Processing',
      Icon: LoaderCircle,
      cls: 'border-indigo-200 bg-indigo-50 text-indigo-700 dark:border-indigo-900 dark:bg-indigo-950 dark:text-indigo-300',
    },
    ok: {
      label: 'OK',
      Icon: Check,
      cls: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-300',
    },
    ready: {
      label: 'Ready',
      Icon: Play,
      cls: 'border-slate-200 bg-slate-50 text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300',
    },
    unsupported: {
      label: 'Unsupported',
      Icon: Minus,
      cls: 'border-slate-200 bg-slate-50 text-slate-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400',
    },
  }[result];
  const { Icon } = config;
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-sm border px-2 py-1 text-xs font-bold ${config.cls}`}>
      <Icon className={`h-3.5 w-3.5 ${result === 'processing' ? 'animate-spin' : ''}`} aria-hidden="true" />
      {config.label}
    </span>
  );
}

function ProcessingStatus({ call }) {
  const percent = call.pipeline?.percent || 0;
  const label = call.pipeline?.current_stage_label || call.stage || 'Queued';
  return (
    <div className="w-32" title={`Processing: ${label}`}>
      <div className="mb-1 flex items-center justify-between gap-2 text-[11px]">
        <span className="truncate font-semibold text-[#1557ff] dark:text-blue-300">{label}</span>
        <span className="font-mono text-slate-500">{percent}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-sm bg-slate-200 dark:bg-slate-700">
        <div
          className="h-full bg-[#1557ff] transition-[width] duration-500"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

function StatusMark({ call }) {
  if (ACTIVE_STATUSES.has(call.status)) return <ProcessingStatus call={call} />;
  if (call.status === 'failed') {
    return (
      <TriangleAlert
        className="mx-auto h-5 w-5 text-red-600"
        title={call.error || 'Processing failed'}
        aria-label="Processing failed"
      />
    );
  }
  if (call.evaluation_available) {
    return (
      <CircleCheck
        className="mx-auto h-5 w-5 text-emerald-600"
        title="Evaluation ready"
        aria-label="Evaluation ready"
      />
    );
  }
  if (call.evaluation_supported) {
    return (
      <Clock3
        className="mx-auto h-5 w-5 text-slate-400"
        title="Ready to evaluate"
        aria-label="Ready to evaluate"
      />
    );
  }
  return (
    <Minus
      className="mx-auto h-5 w-5 text-slate-300"
      title="Outside current evaluator coverage"
      aria-label="Outside current evaluator coverage"
    />
  );
}

function PipelineStep({ step }) {
  const StageIcon = STAGE_ICONS[step.id] || Clock3;
  const completed = step.state === 'completed';
  const active = step.state === 'active';
  const failed = step.state === 'failed';
  const skipped = step.state === 'skipped';
  const Icon = completed ? Check
    : active ? LoaderCircle
      : failed ? TriangleAlert
        : skipped ? Minus
          : StageIcon;
  return (
    <li className="min-w-0">
      <div className="flex items-center gap-2">
        <span className={[
          'flex h-7 w-7 shrink-0 items-center justify-center rounded-sm border',
          completed && 'border-emerald-600 bg-emerald-600 text-white',
          active && 'border-[#1557ff] bg-blue-50 text-[#1557ff] ring-2 ring-blue-100 dark:bg-blue-950 dark:text-blue-300 dark:ring-blue-900',
          failed && 'border-red-600 bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300',
          skipped && 'border-slate-200 bg-slate-100 text-slate-400 dark:border-slate-700 dark:bg-slate-800',
          step.state === 'pending' && 'border-slate-300 bg-white text-slate-400 dark:border-slate-700 dark:bg-slate-950',
        ].filter(Boolean).join(' ')}>
          <Icon className={`h-3.5 w-3.5 ${active ? 'animate-spin' : ''}`} aria-hidden="true" />
        </span>
        <span className={`truncate text-[11px] font-medium ${active || completed ? 'text-slate-800 dark:text-slate-100' : 'text-slate-400'}`}>
          {step.label}
        </span>
        {skipped && <span className="sr-only">skipped</span>}
      </div>
    </li>
  );
}

function PipelineActivity({ calls }) {
  if (!calls.length) return null;
  return (
    <section className="mb-5" aria-label="Pipeline activity">
      <div className="mb-2 flex items-center justify-between">
        <h2 id="pipeline-activity-title" className="text-sm font-semibold text-slate-900 dark:text-white">
          Processing
        </h2>
        <span className="font-mono text-xs text-slate-500">{calls.length} active</span>
      </div>
      <div className="space-y-2">
        {calls.map((call) => (
          <article key={call.db_call_id} className="border border-slate-300 border-l-4 border-l-[#1557ff] bg-white p-4 dark:border-slate-700 dark:border-l-blue-400 dark:bg-slate-950">
            <div className="flex items-center justify-between gap-4">
              <p className="truncate font-mono text-sm font-semibold">{call.call_id}</p>
              <p className="shrink-0 text-xs font-bold text-[#1557ff] dark:text-blue-300">
                <span>{call.pipeline?.current_stage_label || call.stage || 'Queued'}</span>
                <span aria-hidden="true"> / </span>
                <span>{call.pipeline?.percent || 0}%</span>
              </p>
            </div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-sm bg-slate-200 dark:bg-slate-800">
              <div
                className="h-full bg-[#1557ff] transition-[width] duration-500"
                style={{ width: `${call.pipeline?.percent || 0}%` }}
              />
            </div>
            {call.pipeline?.stages?.length > 0 && (
              <ol className="mt-4 grid grid-cols-2 gap-x-3 gap-y-3 sm:grid-cols-4 xl:grid-cols-6">
                {call.pipeline.stages.map((step) => <PipelineStep key={step.id} step={step} />)}
              </ol>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

export default function Dashboard() {
  const [calls, setCalls] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [selected, setSelected] = useState(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [view, setView] = useState('evaluated');
  const [resultFilter, setResultFilter] = useState('all');
  const [page, setPage] = useState(1);

  const activeCallKey = calls
    ?.filter((call) => ACTIVE_STATUSES.has(call.status))
    .map((call) => call.db_call_id)
    .sort()
    .join('|') || '';

  const loadCatalog = useCallback(async () => {
    try {
      const response = await fetch(apiUrl('/calls/catalog'));
      if (!response.ok) throw new Error('Could not load the call catalog.');
      setCalls(await response.json());
      setLoadError(null);
    } catch (error) {
      setLoadError(error.message);
    }
  }, []);

  useEffect(() => {
    loadCatalog();
  }, [loadCatalog]);

  useEffect(() => {
    if (!activeCallKey) return undefined;
    let cancelled = false;
    const poll = async () => {
      const updates = await Promise.all(activeCallKey.split('|').map(async (callId) => {
        try {
          const response = await fetch(apiUrl(`/calls/${callId}/status`));
          return response.ok ? await response.json() : null;
        } catch {
          return null;
        }
      }));
      if (cancelled) return;
      const byId = new Map(updates.filter(Boolean).map((update) => [update.call_id, update]));
      const reachedTerminal = updates.some((update) => update && ['succeeded', 'failed', 'complete'].includes(update.status));
      setCalls((current) => current.map((call) => {
        const update = byId.get(call.db_call_id);
        return update ? { ...call, status: update.status, stage: update.stage, pipeline: update.pipeline, error: update.error } : call;
      }));
      if (reachedTerminal) await loadCatalog();
    };
    poll();
    const timer = window.setInterval(poll, 4000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeCallKey, loadCatalog]);

  useEffect(() => {
    setPage(1);
  }, [view, resultFilter]);

  const toggleSelected = (dbCallId) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(dbCallId)) next.delete(dbCallId);
      else next.add(dbCallId);
      return next;
    });
  };

  const analyzeSelected = async () => {
    setSubmitting(true);
    setLoadError(null);
    try {
      const response = await fetch(apiUrl('/calls/analyze'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ call_ids: [...selected], force: true }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail?.message || body.detail || 'Could not queue calls.');
      }
      const { jobs } = await response.json();
      const byId = new Map(jobs.map((job) => [job.call_id, job]));
      setCalls((current) => current.map((call) => {
        const job = byId.get(call.db_call_id);
        return job ? { ...call, status: job.status, stage: 'uploaded', pipeline: null, job_id: job.job_id } : call;
      }));
      setSelected(new Set());
    } catch (error) {
      setLoadError(error.message);
    } finally {
      setSubmitting(false);
    }
  };

  const activeCalls = calls?.filter((call) => ACTIVE_STATUSES.has(call.status)) || [];
  const evaluated = calls?.filter((call) => call.evaluation_available) || [];
  const counts = {
    review: evaluated.filter((call) => call.result === 'review').length,
    uncertain: evaluated.filter((call) => call.result === 'uncertain').length,
    ok: evaluated.filter((call) => call.result === 'ok').length,
    processing: activeCalls.length,
  };

  const sorted = useMemo(() => {
    if (!calls) return [];
    return calls
      .filter((call) => view === 'all' || call.evaluation_available)
      .filter((call) => resultFilter === 'all' || callResult(call) === resultFilter)
      .sort((a, b) => {
        const resultDifference = RESULT_ORDER[callResult(a)] - RESULT_ORDER[callResult(b)];
        if (resultDifference) return resultDifference;
        const concernDifference = (b.concern_count || 0) - (a.concern_count || 0);
        if (concernDifference) return concernDifference;
        return String(a.call_id).localeCompare(String(b.call_id));
      });
  }, [calls, resultFilter, view]);
  const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const pageSafe = Math.min(page, totalPages);
  const paged = sorted.slice((pageSafe - 1) * PAGE_SIZE, pageSafe * PAGE_SIZE);

  return (
    <div className="flex min-h-screen flex-col bg-white text-slate-950 dark:bg-slate-950 dark:text-slate-100">
      <header className="border-b border-slate-300 bg-white px-4 py-4 sm:px-6 dark:border-slate-700 dark:bg-slate-950">
        <div className="flex w-full items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="flex h-11 w-11 items-center justify-center rounded bg-[#1557ff] text-white" aria-hidden="true">
              <AudioLines size={22} />
            </span>
            <div>
              <h1 className="text-xl font-bold text-slate-950 dark:text-white">Call evaluation</h1>
              <p className="mt-0.5 text-sm text-slate-500">Review queue and evidence status</p>
            </div>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="w-full flex-grow p-4 sm:p-5">
        {loadError && <p className="mb-4 text-sm text-red-600 dark:text-red-400">{loadError}</p>}
        {!calls && !loadError && <p className="text-sm text-slate-500">Loading calls...</p>}

        {calls && (
          <>
            <PipelineActivity calls={activeCalls} />

            <section className="mb-4 grid grid-cols-2 border border-slate-300 text-sm sm:grid-cols-5 dark:border-slate-700" aria-label="Evaluation summary">
              {[
                [evaluated.length, 'Evaluated', 'text-slate-950 dark:text-white'],
                [counts.review, 'Review', 'text-red-700 dark:text-red-300'],
                [counts.uncertain, 'Uncertain', 'text-amber-700 dark:text-amber-300'],
                [counts.ok, 'Clear', 'text-emerald-700 dark:text-emerald-300'],
                [counts.processing, 'Processing', 'text-[#1557ff] dark:text-blue-300'],
              ].map(([value, label, tone], index) => (
                <span key={label} className={`px-4 py-4 ${index ? 'border-l border-slate-300 dark:border-slate-700' : ''}`}>
                  <strong className={`block font-mono text-2xl font-semibold ${tone}`}>{value}</strong>
                  <span className="mt-0.5 block text-[10px] font-bold uppercase text-slate-500">{label}</span>
                </span>
              ))}
            </section>

            <div className="mb-3 flex min-h-10 flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="inline-flex rounded-sm border border-slate-300 bg-white p-0.5 dark:border-slate-700 dark:bg-slate-950" aria-label="Call visibility">
                  {[
                    ['evaluated', 'Evaluated'],
                    ['all', 'All calls'],
                  ].map(([value, label]) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => setView(value)}
                      className={`rounded-sm px-3 py-1.5 text-xs font-bold ${view === value ? 'bg-slate-950 text-white dark:bg-white dark:text-slate-950' : 'text-slate-500 hover:text-slate-950 dark:hover:text-white'}`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <select
                  value={resultFilter}
                  onChange={(event) => setResultFilter(event.target.value)}
                  className="h-8 rounded-sm border border-slate-300 bg-white px-2 text-xs font-semibold text-slate-600 outline-none focus:border-[#1557ff] dark:border-slate-700 dark:bg-slate-950 dark:text-slate-300"
                  aria-label="Filter by result"
                >
                  <option value="all">All results</option>
                  <option value="review">Review</option>
                  <option value="uncertain">Uncertain</option>
                  <option value="ok">OK</option>
                  <option value="processing">Processing</option>
                </select>
              </div>
              <button
                type="button"
                onClick={analyzeSelected}
                disabled={!selected.size || submitting}
                className="inline-flex h-9 min-w-36 items-center justify-center gap-2 rounded bg-[#1557ff] px-4 text-sm font-bold text-white transition hover:bg-blue-700 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40"
              >
                {submitting ? <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Play className="h-4 w-4" aria-hidden="true" />}
                {submitting
                  ? 'Queueing'
                  : selected.size
                    ? `Analyze ${selected.size} ${selected.size === 1 ? 'call' : 'calls'}`
                    : 'Analyze'}
              </button>
            </div>

            <div className="overflow-x-auto border border-slate-300 bg-white dark:border-slate-700 dark:bg-slate-950">
              <table className="w-full min-w-[1120px] table-fixed text-sm">
                <thead>
                  <tr className="bg-slate-100 text-left text-[11px] uppercase text-slate-600 dark:bg-slate-900 dark:text-slate-300">
                    <th rowSpan={2} className="w-[4%] border-b border-r border-slate-300 px-3 py-3 dark:border-slate-700"><span className="sr-only">Select</span></th>
                    <th rowSpan={2} className="w-[30%] border-b border-r border-slate-300 px-4 py-3 font-semibold dark:border-slate-700">Call</th>
                    {ASPECTS.map(([key, label]) => (
                      <th key={key} className="w-[10%] px-3 pb-1 pt-3 text-center font-semibold">{label}</th>
                    ))}
                    <th rowSpan={2} className="w-[8%] border-b border-l border-slate-300 px-3 py-3 text-center font-semibold dark:border-slate-700">Concerns</th>
                    <th rowSpan={2} className="w-[9%] border-b border-l border-slate-300 px-3 py-3 font-semibold dark:border-slate-700">Result</th>
                    <th rowSpan={2} className="w-[9%] border-b border-l border-slate-300 px-3 py-3 text-center font-semibold dark:border-slate-700">Status</th>
                  </tr>
                  <tr className="border-b border-slate-300 bg-slate-100 dark:border-slate-700 dark:bg-slate-900">
                    <th colSpan={4} className="px-3 pb-3 pt-1">
                      <SignalLegend />
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {!paged.length && (
                    <tr>
                      <td colSpan={9} className="px-4 py-8 text-center text-sm text-slate-500">No calls match this view.</td>
                    </tr>
                  )}
                  {paged.map((call) => {
                    const active = ACTIVE_STATUSES.has(call.status);
                    const selectable = !active && call.evaluation_supported;
                    const result = callResult(call);
                    return (
                      <tr key={call.call_id} className="border-b border-slate-200 transition-colors last:border-0 hover:bg-blue-50/60 dark:border-slate-800 dark:hover:bg-blue-950/20">
                        <td className={`border-l-4 px-3 py-3 text-center ${result === 'review' ? 'border-l-red-500' : 'border-l-transparent'}`}>
                          {selectable && (
                            <input
                              type="checkbox"
                              checked={selected.has(call.db_call_id)}
                              onChange={() => toggleSelected(call.db_call_id)}
                              aria-label={`Select ${call.call_id}`}
                              className="h-4 w-4 accent-[#1557ff]"
                            />
                          )}
                        </td>
                        <td className="border-r border-slate-200 px-4 py-3 dark:border-slate-800">
                          {call.evaluation_available ? (
                            <Link to={`/calls/${call.call_id}`} className="font-mono text-xs font-bold text-[#1557ff] hover:underline dark:text-blue-300">
                              {call.call_id}
                            </Link>
                          ) : (
                            <span className="font-mono text-xs font-semibold text-slate-700 dark:text-slate-300">{call.call_id}</span>
                          )}
                          <p className="mt-1 text-[11px] capitalize text-slate-400">
                            {call.domain} / {call.accent} / {fmt(call.duration || 0)}
                          </p>
                        </td>
                        {ASPECTS.map(([key, label]) => (
                          <td key={key} className="px-3 py-3 text-center">
                            {call.evaluation_available ? <AspectMark aspect={call.aspects?.[key]} label={label} /> : <span className="text-slate-300">-</span>}
                          </td>
                        ))}
                        <td className="border-l border-slate-200 px-3 py-3 text-center dark:border-slate-800">
                          {call.evaluation_available ? (
                            <span className={`inline-flex items-center justify-center gap-1.5 font-mono font-semibold ${(call.concern_count || 0) > 0 ? 'text-red-700 dark:text-red-300' : 'text-slate-500'}`}>
                              {call.concern_count || 0}
                              {call.audio_warning && (
                                <AudioLines className="h-4 w-4 text-[#1557ff]" title="Audio supports a surfaced warning" aria-label="Audio supports a surfaced warning" />
                              )}
                            </span>
                          ) : <span className="text-slate-300">-</span>}
                        </td>
                        <td className="border-l border-slate-200 px-3 py-3 dark:border-slate-800"><ResultBadge result={result} /></td>
                        <td className="border-l border-slate-200 px-3 py-3 text-center dark:border-slate-800"><StatusMark call={call} /></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {sorted.length > PAGE_SIZE && (
                <div className="flex items-center justify-between border-t border-slate-200 px-4 py-3 text-xs dark:border-slate-800">
                  <span className="text-slate-500">
                    {(pageSafe - 1) * PAGE_SIZE + 1}-{Math.min(pageSafe * PAGE_SIZE, sorted.length)} of {sorted.length}
                  </span>
                  <div className="flex items-center gap-2">
                    <button type="button" onClick={() => setPage((value) => Math.max(1, value - 1))} disabled={pageSafe === 1} className="rounded-sm border border-slate-300 px-2.5 py-1.5 font-semibold disabled:opacity-40 dark:border-slate-700">Previous</button>
                    <span className="font-mono text-slate-500">{pageSafe}/{totalPages}</span>
                    <button type="button" onClick={() => setPage((value) => Math.min(totalPages, value + 1))} disabled={pageSafe === totalPages} className="rounded-sm border border-slate-300 px-2.5 py-1.5 font-semibold disabled:opacity-40 dark:border-slate-700">Next</button>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
