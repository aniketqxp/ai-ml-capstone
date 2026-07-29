import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  AudioLines,
  BrainCircuit,
  Check,
  CircleCheck,
  ClipboardCheck,
  Clock3,
  CloudUpload,
  FileAudio,
  LoaderCircle,
  Minus,
  SearchCheck,
  ScanText,
  ShieldCheck,
  TriangleAlert,
} from 'lucide-react';
import { apiUrl } from '../api';
import { ThemeToggle } from '../components/ThemeToggle';
import { fmt } from '../lib/format';

const PAGE_SIZE = 10;
const ACTIVE_STATUSES = new Set(['queued', 'processing']);
const STAGE_ICONS = {
  queued: Clock3,
  transcript: FileAudio,
  segments: ScanText,
  acoustic: AudioLines,
  evaluation: BrainCircuit,
  evidence: SearchCheck,
  requirements: ClipboardCheck,
  findings: ScanText,
  decision: ShieldCheck,
  presentation: BrainCircuit,
  publish: CloudUpload,
  ready: CircleCheck,
};

const DISPOSITION = {
  needs_attention: {
    label: 'Needs attention',
    cls: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300',
  },
  no_attention_finding: {
    label: 'No attention finding',
    cls: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-300',
  },
  evaluation_incomplete: {
    label: 'Evaluation incomplete',
    cls: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300',
  },
  setup_required: {
    label: 'Setup required',
    cls: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300',
  },
  unsupported: {
    label: 'Not supported',
    cls: 'border-slate-200 bg-slate-50 text-slate-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400',
  },
  not_evaluated: {
    label: 'Not evaluated',
    cls: 'border-slate-200 bg-slate-50 text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300',
  },
  processing: {
    label: 'In progress',
    cls: 'border-indigo-200 bg-indigo-50 text-indigo-700 dark:border-indigo-900 dark:bg-indigo-950 dark:text-indigo-300',
  },
};

function dispositionFor(call) {
  if (
    ACTIVE_STATUSES.has(call.status)
    && !call.evaluation_available
  ) {
    return DISPOSITION.processing;
  }
  return DISPOSITION[call.evaluation_state] || DISPOSITION.not_evaluated;
}

function checklistSummary(call) {
  const counts = call.checklist_counts || {};
  const concerns = (counts.incorrect || 0) + (counts.not_demonstrated || 0);
  const unclear = counts.unable_to_determine || 0;
  if (!call.evaluation_available) return { demonstrated: '-', concerns: '-', unclear: '-' };
  return {
    demonstrated: counts.demonstrated || 0,
    concerns,
    unclear,
  };
}

function ProcessingStatus({ call }) {
  const pipeline = call.pipeline;
  const percent = pipeline?.percent || 0;
  const label = pipeline?.current_stage_label || call.stage || 'Queued';
  return (
    <div className="w-40" aria-label={`Analysis stage: ${label}`}>
      <div className="mb-1.5 flex justify-between gap-3 text-xs">
        <span className="truncate font-medium text-indigo-700 dark:text-indigo-300">
          {label}
        </span>
        <span className="font-mono text-slate-500">{percent}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-sm bg-slate-200 dark:bg-slate-700">
        <div
          className="h-full bg-indigo-600 transition-[width] duration-500"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
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
      <div className="flex items-center gap-2.5">
        <span
          className={[
            'flex h-8 w-8 shrink-0 items-center justify-center rounded-full border',
            completed && 'border-emerald-600 bg-emerald-600 text-white',
            active && 'border-indigo-600 bg-indigo-50 text-indigo-700 ring-2 ring-indigo-100 dark:bg-indigo-950 dark:text-indigo-300 dark:ring-indigo-900',
            failed && 'border-red-600 bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300',
            skipped && 'border-slate-200 bg-slate-100 text-slate-400 dark:border-slate-700 dark:bg-slate-800',
            step.state === 'pending' && 'border-slate-300 bg-white text-slate-400 dark:border-slate-700 dark:bg-slate-950',
          ].filter(Boolean).join(' ')}
        >
          <Icon className={`h-4 w-4 ${active ? 'animate-spin' : ''}`} aria-hidden="true" />
        </span>
        <span className="min-w-0">
          <span className={[
            'block text-xs font-medium leading-4',
            active || completed
              ? 'text-slate-900 dark:text-slate-100'
              : failed
                ? 'text-red-700 dark:text-red-300'
                : 'text-slate-500',
          ].join(' ')}>
            {step.label}
          </span>
          <span className="block text-[11px] capitalize text-slate-400">
            {step.state}
          </span>
        </span>
      </div>
    </li>
  );
}

function PipelineActivity({ calls }) {
  if (!calls.length) return null;

  return (
    <section className="mb-6" aria-labelledby="pipeline-activity-title">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h2 id="pipeline-activity-title" className="text-sm font-semibold text-slate-900 dark:text-white">
            Pipeline activity
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Live worker stages from the processing service
          </p>
        </div>
        <span className="font-mono text-xs text-slate-500">
          {calls.length} active
        </span>
      </div>

      <div className="space-y-3">
        {calls.map((call) => {
          const pipeline = call.pipeline;
          return (
            <article
              key={call.db_call_id}
              className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-950"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate font-mono text-sm font-semibold text-slate-900 dark:text-white">
                    {call.call_id}
                  </p>
                  <p className="mt-1 text-xs capitalize text-slate-500">
                    {call.domain} · {call.accent}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-semibold text-indigo-700 dark:text-indigo-300">
                    {pipeline?.current_stage_label || call.stage || 'Queued'}
                  </p>
                  <p className="mt-0.5 font-mono text-xs text-slate-500">
                    {pipeline?.percent || 0}%
                  </p>
                </div>
              </div>

              <div className="mt-4 h-2 overflow-hidden rounded-sm bg-slate-200 dark:bg-slate-800">
                <div
                  className="h-full bg-indigo-600 transition-[width] duration-500"
                  style={{ width: `${pipeline?.percent || 0}%` }}
                />
              </div>

              {pipeline?.stages?.length > 0 && (
                <ol className="mt-5 grid grid-cols-2 gap-x-4 gap-y-4 sm:grid-cols-4 xl:grid-cols-6">
                  {pipeline.stages.map((step) => (
                    <PipelineStep key={step.id} step={step} />
                  ))}
                </ol>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

export default function Dashboard() {
  const [calls, setCalls] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [selected, setSelected] = useState(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [domainFilter, setDomainFilter] = useState('all');
  const [accentFilter, setAccentFilter] = useState('all');
  const [dispositionFilter, setDispositionFilter] = useState('all');
  const [sortField, setSortField] = useState(null);
  const [sortDir, setSortDir] = useState('asc');
  const [page, setPage] = useState(1);
  const activeCallKey = calls
    ?.filter((call) => ACTIVE_STATUSES.has(call.status))
    .map((call) => call.db_call_id)
    .sort()
    .join('|') || '';

  const loadCatalog = useCallback(async () => {
    try {
      const response = await fetch(apiUrl('/calls/catalog'));
      if (!response.ok) throw new Error('could not load the call catalog');
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
      const activeIds = activeCallKey.split('|');
      const updates = await Promise.all(activeIds.map(async (callId) => {
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
        return update
          ? {
            ...call,
            status: update.status,
            stage: update.stage,
            pipeline: update.pipeline,
            error: update.error,
          }
          : call;
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
  }, [domainFilter, accentFilter, dispositionFilter]);

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
        throw new Error(body.detail?.message || body.detail || 'could not queue calls');
      }
      const { jobs } = await response.json();
      const byId = new Map(jobs.map((job) => [job.call_id, job]));
      setCalls((current) => current.map((call) => {
        const job = byId.get(call.db_call_id);
        return job
          ? {
            ...call,
            status: job.status,
            stage: 'uploaded',
            pipeline: null,
            job_id: job.job_id,
          }
          : call;
      }));
      setSelected(new Set());
    } catch (error) {
      setLoadError(error.message);
    } finally {
      setSubmitting(false);
    }
  };

  const domains = calls ? [...new Set(calls.map((call) => call.domain).filter(Boolean))].sort() : [];
  const accents = calls ? [...new Set(calls.map((call) => call.accent).filter(Boolean))].sort() : [];
  const filtered = calls ? calls.filter((call) =>
    (domainFilter === 'all' || call.domain === domainFilter) &&
    (accentFilter === 'all' || call.accent === accentFilter) &&
    (
      dispositionFilter === 'all'
      || (
        dispositionFilter === 'processing'
          ? ACTIVE_STATUSES.has(call.status)
          : call.evaluation_state === dispositionFilter
      )
    )
  ) : [];
  const evaluated = calls?.filter((call) => call.evaluation_available) || [];
  const activeCalls = calls?.filter((call) => ACTIVE_STATUSES.has(call.status)) || [];

  const toggleSort = (field) => {
    if (sortField === field) setSortDir((direction) => direction === 'asc' ? 'desc' : 'asc');
    else { setSortField(field); setSortDir('asc'); }
  };
  const sortValue = (call) => {
    if (sortField === 'duration') return call.duration || 0;
    if (sortField === 'demonstrated') {
      return call.checklist_counts?.demonstrated ?? -1;
    }
    if (sortField === 'concerns') {
      return (
        (call.checklist_counts?.incorrect || 0)
        + (call.checklist_counts?.not_demonstrated || 0)
      );
    }
    return 0;
  };
  const sorted = sortField
    ? [...filtered].sort((a, b) => (sortValue(a) - sortValue(b)) * (sortDir === 'asc' ? 1 : -1))
    : [...filtered].sort((a, b) => {
      const activeDifference = Number(ACTIVE_STATUSES.has(b.status))
        - Number(ACTIVE_STATUSES.has(a.status));
      if (activeDifference) return activeDifference;
      const attentionDifference = Number(b.attention_required)
        - Number(a.attention_required);
      if (attentionDifference) return attentionDifference;
      return Number(a.evaluation_available) - Number(b.evaluation_available);
    });
  const sortArrow = (field) => sortField === field ? (sortDir === 'asc' ? ' ^' : ' v') : '';
  const needsAttention = evaluated.filter((call) => call.attention_required).length;
  const completeCount = evaluated.filter(
    (call) => call.evaluation_status === 'complete',
  ).length;
  const incompleteCount = calls?.filter(
    (call) => call.evaluation_state === 'evaluation_incomplete',
  ).length || 0;
  const readyCount = calls?.filter(
    (call) => (
      call.evaluation_supported
      && !call.evaluation_available
      && !ACTIVE_STATUSES.has(call.status)
    ),
  ).length || 0;
  const unsupportedCount = calls?.filter(
    (call) => call.evaluation_state === 'unsupported',
  ).length || 0;
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageSafe = Math.min(page, totalPages);
  const paged = sorted.slice((pageSafe - 1) * PAGE_SIZE, pageSafe * PAGE_SIZE);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100 flex flex-col">
      <header className="px-6 py-4 border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-indigo-600 dark:text-indigo-400">Call Evaluation Dashboard</h1>
          <p className="mt-0.5 text-sm text-slate-500">
            {calls
              ? `${evaluated.length} evaluated, ${readyCount} ready to run, ${unsupportedCount} outside current coverage`
              : 'Loading...'}
          </p>
        </div>
        <ThemeToggle />
      </header>

      <main className="flex-grow max-w-7xl w-full mx-auto p-6">
        {loadError && <p className="mb-4 text-sm text-red-600 dark:text-red-400">{loadError}</p>}
        {!calls && !loadError && <p className="text-sm text-slate-500">Loading calls...</p>}

        {calls && (
          <>
            <PipelineActivity calls={activeCalls} />

            <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" aria-label="Call summary">
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Evaluated</div>
                <div className="text-3xl font-bold mt-1">{evaluated.length}</div>
                <div className="text-xs text-slate-500 mt-0.5">{readyCount} ready to run</div>
              </div>
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Needs attention</div>
                <div className={`text-3xl font-bold mt-1 ${needsAttention ? 'text-red-600 dark:text-red-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                  {needsAttention}
                </div>
                <div className="text-xs text-slate-500 mt-0.5">Evidence-backed disposition</div>
              </div>
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Complete coverage</div>
                <div className="text-3xl font-bold mt-1">{completeCount}</div>
                <div className="text-xs text-slate-500 mt-0.5">All applicable checks assessed</div>
              </div>
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Incomplete</div>
                <div className={`text-3xl font-bold mt-1 ${incompleteCount ? 'text-amber-600 dark:text-amber-400' : ''}`}>
                  {incompleteCount}
                </div>
                <div className="text-xs text-slate-500 mt-0.5">{activeCalls.length} processing now</div>
              </div>
            </section>

            <div className="mb-3 flex min-h-10 items-center justify-between gap-4">
              <p className="text-sm text-slate-500">
                Select supported calls to evaluate or run again.
              </p>
              <button
                type="button"
                onClick={analyzeSelected}
                disabled={!selected.size || submitting}
                className="min-w-40 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {submitting ? 'Queueing...' : `Analyze ${selected.size || ''} call${selected.size === 1 ? '' : 's'}`}
              </button>
            </div>

            <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950">
              <table className="w-full min-w-[1060px] text-base">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-xs uppercase text-slate-500 dark:border-slate-800">
                    <th className="w-12 px-4 py-4"><span className="sr-only">Select</span></th>
                    <th className="px-4 py-4 font-semibold">Call</th>
                    <th className="px-4 py-4 font-semibold">
                      <select value={domainFilter} onChange={(event) => setDomainFilter(event.target.value)} className="bg-transparent font-semibold uppercase text-slate-500 focus:outline-none">
                        <option value="all">Domain</option>
                        {domains.map((domain) => <option key={domain} value={domain}>{domain}</option>)}
                      </select>
                    </th>
                    <th className="px-4 py-4 font-semibold">
                      <select value={accentFilter} onChange={(event) => setAccentFilter(event.target.value)} className="bg-transparent font-semibold uppercase text-slate-500 focus:outline-none">
                        <option value="all">Accent</option>
                        {accents.map((accent) => <option key={accent} value={accent}>{accent}</option>)}
                      </select>
                    </th>
                    <th className="px-4 py-4 font-semibold"><button onClick={() => toggleSort('duration')}>Duration{sortArrow('duration')}</button></th>
                    <th className="px-4 py-4 font-semibold">
                      <select value={dispositionFilter} onChange={(event) => setDispositionFilter(event.target.value)} className="bg-transparent font-semibold uppercase text-slate-500 focus:outline-none">
                        <option value="all">Disposition</option>
                        {Object.entries(DISPOSITION).map(([value, item]) => (
                          <option key={value} value={value}>{item.label}</option>
                        ))}
                      </select>
                    </th>
                    <th className="px-4 py-4 font-semibold"><button onClick={() => toggleSort('demonstrated')}>Demonstrated{sortArrow('demonstrated')}</button></th>
                    <th className="px-4 py-4 font-semibold"><button onClick={() => toggleSort('concerns')}>Concerns{sortArrow('concerns')}</button></th>
                    <th className="px-4 py-4 font-semibold">Audio support</th>
                    <th className="px-4 py-4 font-semibold">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {!filtered.length && <tr><td colSpan={10} className="px-4 py-6 text-center text-sm text-slate-500">No calls match the selected filters.</td></tr>}
                  {paged.map((call) => {
                    const active = ACTIVE_STATUSES.has(call.status);
                    const selectable = !active && call.evaluation_supported;
                    const disposition = dispositionFor(call);
                    const checks = checklistSummary(call);
                    return (
                      <tr key={call.call_id} className="border-b border-slate-100 last:border-0 hover:bg-slate-50 dark:border-slate-800/60 dark:hover:bg-slate-900/60">
                        <td className="px-4 py-4 text-center">
                          {selectable && <input type="checkbox" checked={selected.has(call.db_call_id)} onChange={() => toggleSelected(call.db_call_id)} aria-label={`Select ${call.call_id}`} className="h-4 w-4 accent-indigo-600" />}
                        </td>
                        <td className="px-4 py-4 whitespace-nowrap font-mono text-sm">
                          {call.evaluation_available
                            ? <Link to={`/calls/${call.call_id}`} className="text-indigo-600 hover:underline dark:text-indigo-400">{call.call_id}</Link>
                            : <span className="text-slate-700 dark:text-slate-300">{call.call_id}</span>}
                        </td>
                        <td className="px-4 py-4 capitalize text-slate-700 dark:text-slate-300">{call.domain}</td>
                        <td className="px-4 py-4 font-mono text-sm text-slate-500">{call.accent}</td>
                        <td className="px-4 py-4 font-mono text-sm text-slate-500">{fmt(call.duration || 0)}</td>
                        <td className="px-4 py-4">
                          <span className={`rounded border px-2 py-0.5 text-xs font-semibold ${disposition.cls}`}>
                            {disposition.label}
                          </span>
                        </td>
                        <td className="px-4 py-4 font-mono text-sm text-emerald-700 dark:text-emerald-400">
                          {checks.demonstrated}
                        </td>
                        <td className={`px-4 py-4 font-mono text-sm ${checks.concerns > 0 ? 'font-semibold text-red-700 dark:text-red-400' : 'text-slate-500'}`}>
                          {checks.concerns}
                          {checks.unclear !== '-' && checks.unclear > 0 && (
                            <span className="ml-1.5 text-xs text-amber-600" title="Unable to determine">
                              +{checks.unclear} unclear
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-4">
                          {call.evaluation_available
                            ? (
                              <span className="text-xs text-slate-600 dark:text-slate-300" title={call.acoustic_coverage || undefined}>
                                {call.acoustic_status === 'available' ? 'Available' : call.acoustic_status === 'limited' ? 'Limited' : 'Unavailable'}
                              </span>
                            )
                            : <span className="text-xs text-slate-400">-</span>}
                        </td>
                        <td className="px-4 py-4">
                          {active ? <ProcessingStatus call={call} /> : call.status === 'failed'
                            ? <span className="text-xs font-semibold text-red-600" title={call.error}>Failed</span>
                            : <span className={`text-xs font-semibold ${call.evaluation_available ? 'text-emerald-600' : 'text-slate-500'}`}>{call.evaluation_available ? 'Evaluated' : call.evaluation_supported ? 'Ready' : 'Unavailable'}</span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {filtered.length > 0 && totalPages > 1 && (
                <div className="flex items-center justify-between border-t border-slate-200 px-5 py-3 text-sm dark:border-slate-800">
                  <span className="text-slate-500">Showing {(pageSafe - 1) * PAGE_SIZE + 1}-{Math.min(pageSafe * PAGE_SIZE, filtered.length)} of {filtered.length}</span>
                  <div className="flex items-center gap-2">
                    <button onClick={() => setPage((value) => Math.max(1, value - 1))} disabled={pageSafe === 1} className="rounded border border-slate-200 px-3 py-1 disabled:opacity-40 dark:border-slate-700">Previous</button>
                    <span className="font-mono text-xs text-slate-500">{pageSafe}/{totalPages}</span>
                    <button onClick={() => setPage((value) => Math.min(totalPages, value + 1))} disabled={pageSafe === totalPages} className="rounded border border-slate-200 px-3 py-1 disabled:opacity-40 dark:border-slate-700">Next</button>
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
