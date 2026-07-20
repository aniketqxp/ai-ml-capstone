import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { apiUrl } from '../api';
import { ThemeToggle } from '../components/ThemeToggle';
import { RISK_STYLE } from '../lib/constants';
import { fmt } from '../lib/format';

const PAGE_SIZE = 10;
const ACTIVE_STATUSES = new Set(['queued', 'processing']);
const STAGES = ['Transcribe', 'Segment', 'Acoustic', 'Evaluate', 'Export', 'Ready'];

function stageIndex(stage) {
  if (['uploaded', 'starting', 'transcribing', 'registering'].includes(stage)) return 0;
  if (stage === 'segmenting') return 1;
  if (stage === 'acoustic') return 2;
  if (stage === 'evaluating') return 3;
  if (['exporting', 'uploading', 'persisting'].includes(stage)) return 4;
  if (stage === 'done') return 5;
  return 0;
}

function ProcessingStatus({ call }) {
  const current = stageIndex(call.stage);
  return (
    <div className="w-52" aria-label={`Analysis stage: ${call.stage}`}>
      <div className="mb-1 flex justify-between text-[10px] uppercase text-slate-500">
        <span>{call.stage || 'queued'}</span>
        <span>{Math.round((current / (STAGES.length - 1)) * 100)}%</span>
      </div>
      <div className="grid grid-cols-6 gap-1">
        {STAGES.map((label, index) => (
          <span
            key={label}
            title={label}
            className={`h-1.5 rounded-sm ${index <= current ? 'bg-indigo-500' : 'bg-slate-200 dark:bg-slate-700'}`}
          />
        ))}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [calls, setCalls] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [selected, setSelected] = useState(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [domainFilter, setDomainFilter] = useState('all');
  const [accentFilter, setAccentFilter] = useState('all');
  const [riskFilter, setRiskFilter] = useState('all');
  const [sortField, setSortField] = useState(null);
  const [sortDir, setSortDir] = useState('asc');
  const [page, setPage] = useState(1);

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
    if (!calls?.some((call) => ACTIVE_STATUSES.has(call.status))) return undefined;
    let cancelled = false;
    const poll = async () => {
      const active = calls.filter((call) => ACTIVE_STATUSES.has(call.status));
      const updates = await Promise.all(active.map(async (call) => {
        try {
          const response = await fetch(apiUrl(`/calls/${call.db_call_id}/status`));
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
        return update ? { ...call, status: update.status, stage: update.stage, error: update.error } : call;
      }));
      if (reachedTerminal) await loadCatalog();
    };
    const timer = window.setInterval(poll, 4000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [calls, loadCatalog]);

  useEffect(() => {
    setPage(1);
  }, [domainFilter, accentFilter, riskFilter]);

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
        body: JSON.stringify({ call_ids: [...selected] }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail?.message || body.detail || 'could not queue calls');
      }
      const { jobs } = await response.json();
      const byId = new Map(jobs.map((job) => [job.call_id, job]));
      setCalls((current) => current.map((call) => {
        const job = byId.get(call.db_call_id);
        return job ? { ...call, status: job.status, stage: 'uploaded', job_id: job.job_id } : call;
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
    (riskFilter === 'all' || call.risk_level === riskFilter)
  ) : [];
  const analyzed = calls?.filter((call) => call.analyzed) || [];

  const toggleSort = (field) => {
    if (sortField === field) setSortDir((direction) => direction === 'asc' ? 'desc' : 'asc');
    else { setSortField(field); setSortDir('asc'); }
  };
  const sortValue = (call) => {
    if (sortField === 'duration') return call.duration || 0;
    if (sortField === 'compliance') return call.compliance_applicable ? call.compliance_passed / call.compliance_applicable : -1;
    if (sortField === 'workflow') return call.workflow_total ? call.workflow_met / call.workflow_total : -1;
    return 0;
  };
  const sorted = sortField
    ? [...filtered].sort((a, b) => (sortValue(a) - sortValue(b)) * (sortDir === 'asc' ? 1 : -1))
    : [...filtered].sort((a, b) => Number(a.analyzed) - Number(b.analyzed));
  const sortArrow = (field) => sortField === field ? (sortDir === 'asc' ? ' ^' : ' v') : '';
  const ratioTone = (passed, total) => {
    if (!total) return 'text-slate-500 dark:text-slate-400';
    const ratio = passed / total;
    if (ratio >= 0.8) return 'text-emerald-700 dark:text-emerald-400';
    if (ratio >= 0.5) return 'text-amber-700 dark:text-amber-400';
    return 'text-red-700 dark:text-red-400 font-semibold';
  };
  const heroTone = (ratio) => {
    if (ratio >= 0.8) return 'text-emerald-600 dark:text-emerald-400';
    if (ratio >= 0.5) return 'text-amber-600 dark:text-amber-400';
    return 'text-red-600 dark:text-red-400';
  };

  const totals = analyzed.reduce((value, call) => ({
    compliancePassed: value.compliancePassed + (call.compliance_passed || 0),
    complianceApplicable: value.complianceApplicable + (call.compliance_applicable || 0),
    workflowMet: value.workflowMet + (call.workflow_met || 0),
    workflowTotal: value.workflowTotal + (call.workflow_total || 0),
  }), { compliancePassed: 0, complianceApplicable: 0, workflowMet: 0, workflowTotal: 0 });
  const complianceRate = totals.complianceApplicable ? totals.compliancePassed / totals.complianceApplicable : 0;
  const workflowRate = totals.workflowTotal ? totals.workflowMet / totals.workflowTotal : 0;
  const needsReview = analyzed.filter((call) => ['review', 'escalate'].includes(call.risk_level)).length;
  const availableCount = calls?.filter((call) => !call.analyzed && !ACTIVE_STATUSES.has(call.status)).length || 0;
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageSafe = Math.min(page, totalPages);
  const paged = sorted.slice((pageSafe - 1) * PAGE_SIZE, pageSafe * PAGE_SIZE);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100 flex flex-col">
      <header className="px-6 py-4 border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-indigo-600 dark:text-indigo-400">Call Compliance Dashboard</h1>
          <p className="mt-0.5 text-sm text-slate-500">
            {calls ? `${analyzed.length} analyzed, ${availableCount} available` : 'Loading...'}
          </p>
        </div>
        <ThemeToggle />
      </header>

      <main className="flex-grow max-w-7xl w-full mx-auto p-6">
        {loadError && <p className="mb-4 text-sm text-red-600 dark:text-red-400">{loadError}</p>}
        {!calls && !loadError && <p className="text-sm text-slate-500">Loading calls...</p>}

        {calls && (
          <>
            <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" aria-label="Call summary">
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Analyzed</div>
                <div className="text-3xl font-bold mt-1">{analyzed.length}</div>
                <div className="text-xs text-slate-500 mt-0.5">{availableCount} ready to run</div>
              </div>
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Avg compliance</div>
                <div className={`text-3xl font-bold mt-1 ${heroTone(complianceRate)}`}>{Math.round(complianceRate * 100)}%</div>
              </div>
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Avg workflow</div>
                <div className={`text-3xl font-bold mt-1 ${heroTone(workflowRate)}`}>{Math.round(workflowRate * 100)}%</div>
              </div>
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Needs review</div>
                <div className={`text-3xl font-bold mt-1 ${needsReview ? 'text-amber-600 dark:text-amber-400' : 'text-emerald-600 dark:text-emerald-400'}`}>{needsReview}</div>
              </div>
            </section>

            <div className="mb-3 flex min-h-10 items-center justify-between gap-4">
              <p className="text-sm text-slate-500">Select staged calls to run through the analysis pipeline.</p>
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
                    <th className="px-4 py-4 font-semibold"><button onClick={() => toggleSort('compliance')}>Compliance{sortArrow('compliance')}</button></th>
                    <th className="px-4 py-4 font-semibold"><button onClick={() => toggleSort('workflow')}>Workflow{sortArrow('workflow')}</button></th>
                    <th className="px-4 py-4 font-semibold">
                      <select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)} className="bg-transparent font-semibold uppercase text-slate-500 focus:outline-none">
                        <option value="all">Risk</option>
                        {Object.keys(RISK_STYLE).map((risk) => <option key={risk} value={risk}>{RISK_STYLE[risk]?.label || risk}</option>)}
                      </select>
                    </th>
                    <th className="px-4 py-4 font-semibold">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {!filtered.length && <tr><td colSpan={9} className="px-4 py-6 text-center text-sm text-slate-500">No calls match the selected filters.</td></tr>}
                  {paged.map((call) => {
                    const active = ACTIVE_STATUSES.has(call.status);
                    const selectable = !call.analyzed && !active;
                    const risk = RISK_STYLE[call.risk_level];
                    return (
                      <tr key={call.call_id} className="border-b border-slate-100 last:border-0 hover:bg-slate-50 dark:border-slate-800/60 dark:hover:bg-slate-900/60">
                        <td className="px-4 py-4 text-center">
                          {selectable && <input type="checkbox" checked={selected.has(call.db_call_id)} onChange={() => toggleSelected(call.db_call_id)} aria-label={`Select ${call.call_id}`} className="h-4 w-4 accent-indigo-600" />}
                        </td>
                        <td className="px-4 py-4 whitespace-nowrap font-mono text-sm">
                          {call.analyzed
                            ? <Link to={`/calls/${call.call_id}`} className="text-indigo-600 hover:underline dark:text-indigo-400">{call.call_id}</Link>
                            : <span className="text-slate-700 dark:text-slate-300">{call.call_id}</span>}
                        </td>
                        <td className="px-4 py-4 capitalize text-slate-700 dark:text-slate-300">{call.domain}</td>
                        <td className="px-4 py-4 font-mono text-sm text-slate-500">{call.accent}</td>
                        <td className="px-4 py-4 font-mono text-sm text-slate-500">{fmt(call.duration || 0)}</td>
                        <td className="px-4 py-4 font-mono text-sm">{call.analyzed ? <span className={ratioTone(call.compliance_passed, call.compliance_applicable)}>{call.compliance_passed}/{call.compliance_applicable}</span> : '-'}</td>
                        <td className="px-4 py-4 font-mono text-sm">{call.analyzed ? <span className={ratioTone(call.workflow_met, call.workflow_total)}>{call.workflow_met}/{call.workflow_total}</span> : '-'}</td>
                        <td className="px-4 py-4">
                          {call.analyzed
                            ? <span className={`rounded border px-2 py-0.5 text-xs font-semibold ${risk?.cls || ''}`}>{risk?.label || call.risk_level}</span>
                            : <span className="text-xs text-slate-400">Pending</span>}
                        </td>
                        <td className="px-4 py-4">
                          {active ? <ProcessingStatus call={call} /> : call.status === 'failed'
                            ? <span className="text-xs font-semibold text-red-600" title={call.error}>Failed</span>
                            : <span className={`text-xs font-semibold ${call.analyzed ? 'text-emerald-600' : 'text-slate-500'}`}>{call.analyzed ? 'Analyzed' : 'Available'}</span>}
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
