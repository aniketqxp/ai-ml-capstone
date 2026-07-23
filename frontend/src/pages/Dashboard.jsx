import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { apiUrl } from '../api';
import { ThemeToggle } from '../components/ThemeToggle';
import { RISK_STYLE } from '../lib/constants';
import { fmt } from '../lib/format';

const PAGE_SIZE = 10;
const ACTIVE_STATUSES = new Set(['queued', 'processing']);
const STAGES = ['Transcribe', 'Segment', 'Acoustic', 'Evaluate', 'Export', 'Ready'];
const MAX_UPLOAD_FILES = 10;

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function stageProgress(call) {
  if (Number.isFinite(call.progress_percent)) return call.progress_percent;
  if (call.status === 'queued') return 0;
  const progress = {
    uploaded: 5,
    starting: 10,
    transcribing: 20,
    registering: 35,
    segmenting: 40,
    acoustic: 55,
    evaluating: 70,
    exporting: 85,
    uploading: 90,
    persisting: 95,
    done: 100,
  };
  return progress[call.stage] || 0;
}

function formatEta(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  if (seconds < 60) return `${Math.max(1, seconds)}s left`;
  const minutes = Math.ceil(seconds / 60);
  return minutes < 60 ? `~${minutes}m left` : `~${Math.floor(minutes / 60)}h ${minutes % 60}m left`;
}

function ProcessingStatus({ call, onCancel, cancelling }) {
  const percentage = stageProgress(call);
  const current = percentage === 0 ? -1 : Math.min(
    STAGES.length - 1,
    Math.floor((percentage / 100) * STAGES.length),
  );
  const stageLabel = call.progress_message || (call.status === 'queued'
    ? 'Waiting in queue'
    : call.stage || 'Starting');
  const eta = formatEta(call.estimated_seconds_remaining);
  return (
    <div className="w-60" aria-label={`Analysis stage: ${stageLabel}`}>
      <div className="mb-1 flex justify-between text-[10px] uppercase text-slate-500">
        <span className="max-w-40 truncate normal-case" title={stageLabel}>{stageLabel}</span>
        <span>{percentage}%</span>
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
      <div className="mt-1.5 flex items-center justify-between gap-2 text-[10px] text-slate-500">
        <span>
          {call.progress_total ? `${call.progress_current || 0}/${call.progress_total}` : eta || 'Estimating time…'}
          {call.progress_total && eta ? ` · ${eta}` : ''}
        </span>
        <button
          type="button"
          disabled={cancelling || call.cancel_requested}
          onClick={() => onCancel(call)}
          className="font-semibold text-red-600 hover:underline disabled:text-slate-400 disabled:no-underline"
        >
          {cancelling || call.cancel_requested ? 'Stopping…' : 'Cancel'}
        </button>
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
  const [uploadFiles, setUploadFiles] = useState([]);
  const [uploadDomain, setUploadDomain] = useState('banking');
  const [uploadAccent, setUploadAccent] = useState('en-CA');
  const [uploading, setUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState([]);
  const [uploadInputKey, setUploadInputKey] = useState(0);
  const [cancellingIds, setCancellingIds] = useState(new Set());

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
      const reachedTerminal = updates.some((update) => update && ['succeeded', 'failed', 'complete', 'cancelled'].includes(update.status));
      setCalls((current) => current.map((call) => {
        const update = byId.get(call.db_call_id);
        return update ? { ...call, ...update } : call;
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

  const cancelAnalysis = async (call) => {
    setCancellingIds((current) => new Set(current).add(call.db_call_id));
    try {
      const response = await fetch(apiUrl(`/calls/${call.db_call_id}/cancel`), { method: 'POST' });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || 'Could not cancel analysis');
      setCalls((current) => current.map((item) => item.db_call_id === call.db_call_id
        ? { ...item, cancel_requested: true, progress_message: 'Cancellation requested', status: body.status }
        : item));
      await loadCatalog();
    } catch (error) {
      setLoadError(error.message);
    } finally {
      setCancellingIds((current) => {
        const next = new Set(current);
        next.delete(call.db_call_id);
        return next;
      });
    }
  };

  const uploadCalls = async (event) => {
    event.preventDefault();
    if (!uploadFiles.length || uploading) return;

    setUploading(true);
    setUploadResults(uploadFiles.map((file) => ({
      name: file.name,
      status: 'waiting',
    })));

    for (let index = 0; index < uploadFiles.length; index += 1) {
      const file = uploadFiles[index];
      setUploadResults((current) => current.map((result, resultIndex) =>
        resultIndex === index ? { ...result, status: 'uploading' } : result
      ));

      try {
        const form = new FormData();
        form.append('file', file);
        form.append('domain', uploadDomain.trim() || 'general');
        form.append('accent', uploadAccent.trim() || 'en-US');

        const response = await fetch(apiUrl('/calls/ingest'), {
          method: 'POST',
          body: form,
        });
        const body = await response.json().catch(() => ({}));

        if (!response.ok) {
          throw new Error(body.detail || 'Upload failed');
        }

        setUploadResults((current) => current.map((result, resultIndex) =>
          resultIndex === index
            ? { ...result, status: 'staged', callId: body.public_call_id }
            : result
        ));
        await loadCatalog();
      } catch (error) {
        setUploadResults((current) => current.map((result, resultIndex) =>
          resultIndex === index
            ? { ...result, status: 'failed', error: error.message }
            : result
        ));
      }
    }

    setUploading(false);
    setUploadFiles([]);
    setUploadInputKey((value) => value + 1);
    await loadCatalog();
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
  const needsReview = analyzed.filter((call) =>
    call.manager_review_required === true ||
    ['review', 'escalate'].includes(call.risk_level)
  ).length;
  const sentimentCount = analyzed.filter((call) => call.dominant_sentiment).length;
  const audioFeatureCount = analyzed.filter((call) => call.has_audio_features).length;
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
        {loadError && (
          <div className="mb-4 flex items-center justify-between gap-4 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300">
            <span>{loadError}</span>
            <button
              type="button"
              onClick={loadCatalog}
              className="rounded-md border border-red-300 px-3 py-1 font-semibold hover:bg-red-100 dark:border-red-800 dark:hover:bg-red-950"
            >
              Retry
            </button>
          </div>
        )}
        {!calls && !loadError && <p className="text-sm text-slate-500">Loading calls...</p>}

        {calls && (
          <>
            <section className="mb-6 rounded-xl border border-indigo-200 bg-white p-5 shadow-sm dark:border-indigo-900 dark:bg-slate-950">
              <div className="flex flex-col gap-1">
                <p className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
                  New analysis
                </p>
                <h2 className="text-lg font-bold">Upload call audio</h2>
                <p className="text-sm text-slate-500">
                  Upload one call or a batch of up to {MAX_UPLOAD_FILES}. Each file is queued automatically for transcription, sentiment, acoustic and LLM analysis.
                </p>
              </div>

              <form onSubmit={uploadCalls} className="mt-4 grid gap-4 lg:grid-cols-[1fr_180px_180px_auto] lg:items-end">
                <label className="block">
                  <span className="mb-1.5 block text-xs font-semibold text-slate-600 dark:text-slate-300">Audio files</span>
                  <input
                    key={uploadInputKey}
                    type="file"
                    accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.aac"
                    multiple
                    disabled={uploading}
                    onChange={(event) => {
                      const selectedFiles = Array.from(event.target.files || []);
                      if (selectedFiles.length > MAX_UPLOAD_FILES) {
                        setUploadResults([{ name: 'Selection', status: 'failed', error: `Choose no more than ${MAX_UPLOAD_FILES} files at once.` }]);
                        setUploadFiles([]);
                        return;
                      }
                      setUploadFiles(selectedFiles);
                      setUploadResults([]);
                    }}
                    className="block w-full rounded-lg border border-slate-300 bg-slate-50 text-sm text-slate-600 file:mr-3 file:border-0 file:bg-indigo-600 file:px-4 file:py-2.5 file:font-semibold file:text-white hover:file:bg-indigo-500 disabled:opacity-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300"
                  />
                </label>

                <label className="block">
                  <span className="mb-1.5 block text-xs font-semibold text-slate-600 dark:text-slate-300">Domain</span>
                  <input
                    value={uploadDomain}
                    onChange={(event) => setUploadDomain(event.target.value)}
                    disabled={uploading}
                    placeholder="banking"
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm dark:border-slate-700 dark:bg-slate-900"
                  />
                </label>

                <label className="block">
                  <span className="mb-1.5 block text-xs font-semibold text-slate-600 dark:text-slate-300">Accent / locale</span>
                  <input
                    value={uploadAccent}
                    onChange={(event) => setUploadAccent(event.target.value)}
                    disabled={uploading}
                    placeholder="en-CA"
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm dark:border-slate-700 dark:bg-slate-900"
                  />
                </label>

                <button
                  type="submit"
                  disabled={!uploadFiles.length || uploading}
                  className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {uploading ? 'Uploading…' : `Upload${uploadFiles.length > 1 ? ` ${uploadFiles.length} calls` : ''}`}
                </button>
              </form>

              {uploadFiles.length > 0 && !uploading && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {uploadFiles.map((file) => (
                    <span key={`${file.name}-${file.lastModified}`} className="rounded-full border border-slate-300 px-2.5 py-1 text-xs text-slate-600 dark:border-slate-700 dark:text-slate-300">
                      {file.name} · {formatBytes(file.size)}
                    </span>
                  ))}
                </div>
              )}

              {uploadResults.length > 0 && (
                <div className="mt-4 space-y-2" aria-live="polite">
                  {uploadResults.map((result, index) => (
                    <div key={`${result.name}-${index}`} className="flex flex-col justify-between gap-1 rounded-lg bg-slate-50 px-3 py-2 text-sm sm:flex-row sm:items-center dark:bg-slate-900">
                      <span className="truncate font-medium">{result.name}</span>
                      <span className={result.status === 'failed' ? 'text-red-600 dark:text-red-400' : result.status === 'staged' ? 'text-emerald-600 dark:text-emerald-400' : 'text-indigo-600 dark:text-indigo-400'}>
                        {result.status === 'waiting' && 'Waiting'}
                        {result.status === 'uploading' && 'Uploading and validating…'}
                        {result.status === 'staged' && `Ready to select as ${result.callId}`}
                        {result.status === 'failed' && `Failed: ${result.error}`}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              <p className="mt-3 text-xs text-slate-500">
                Normal mono WAV files are accepted. Stereo or separate agent/customer channels provide more accurate speaker-specific results; mono calls are analyzed with limited speaker attribution.
              </p>
            </section>

            <section className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4 mb-6" aria-label="Call summary">
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Analyzed</div>
                <div className="text-3xl font-bold mt-1">{analyzed.length}</div>
                <div className="text-xs text-slate-500 mt-0.5">{availableCount} ready to run</div>
              </div>
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Avg compliance</div>
                <div className={`text-3xl font-bold mt-1 ${totals.complianceApplicable ? heroTone(complianceRate) : 'text-slate-400'}`}>
                  {totals.complianceApplicable ? `${Math.round(complianceRate * 100)}%` : '—'}
                </div>
              </div>
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Avg workflow</div>
                <div className={`text-3xl font-bold mt-1 ${totals.workflowTotal ? heroTone(workflowRate) : 'text-slate-400'}`}>
                  {totals.workflowTotal ? `${Math.round(workflowRate * 100)}%` : '—'}
                </div>
              </div>
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Needs review</div>
                <div className={`text-3xl font-bold mt-1 ${needsReview ? 'text-amber-600 dark:text-amber-400' : 'text-emerald-600 dark:text-emerald-400'}`}>{needsReview}</div>
              </div>
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Sentiment ready</div>
                <div className="text-3xl font-bold mt-1 text-indigo-600 dark:text-indigo-400">{sentimentCount}</div>
                <div className="text-xs text-slate-500 mt-0.5">of {analyzed.length} analyzed</div>
              </div>
              <div className="bg-white rounded-lg border border-slate-200 p-4 dark:bg-slate-950 dark:border-slate-800">
                <div className="text-xs uppercase text-slate-500 font-semibold">Audio features</div>
                <div className="text-3xl font-bold mt-1 text-sky-600 dark:text-sky-400">{audioFeatureCount}</div>
                <div className="text-xs text-slate-500 mt-0.5">feature timelines ready</div>
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
              <table className="w-full min-w-[1180px] text-base">
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
                    <th className="px-4 py-4 font-semibold">Sentiment</th>
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
                  {!filtered.length && <tr><td colSpan={10} className="px-4 py-6 text-center text-sm text-slate-500">No calls match the selected filters.</td></tr>}
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
                            ? <div><Link to={`/calls/${call.call_id}`} className="font-semibold text-indigo-600 hover:underline dark:text-indigo-400">{call.call_id}</Link>{call.subject && <p className="mt-1 max-w-64 truncate font-sans text-xs text-slate-500" title={call.subject}>{call.subject}</p>}</div>
                            : <span className="text-slate-700 dark:text-slate-300">{call.call_id}</span>}
                        </td>
                        <td className="px-4 py-4 capitalize text-slate-700 dark:text-slate-300">{call.domain}</td>
                        <td className="px-4 py-4 font-mono text-sm text-slate-500">{call.accent}</td>
                        <td className="px-4 py-4 font-mono text-sm text-slate-500">{fmt(call.duration || 0)}</td>
                        <td className="px-4 py-4 font-mono text-sm">{call.analyzed ? <span className={ratioTone(call.compliance_passed, call.compliance_applicable)}>{call.compliance_passed}/{call.compliance_applicable}</span> : '-'}</td>
                        <td className="px-4 py-4 font-mono text-sm">{call.analyzed ? <span className={ratioTone(call.workflow_met, call.workflow_total)}>{call.workflow_met}/{call.workflow_total}</span> : '-'}</td>
                        <td className="px-4 py-4">
                          {call.dominant_sentiment ? (
                            <div>
                              <span className="text-sm font-semibold">{call.dominant_sentiment}</span>
                              {call.dominant_emotion && <p className="mt-0.5 text-xs capitalize text-slate-500">{call.dominant_emotion}</p>}
                            </div>
                          ) : <span className="text-slate-400">—</span>}
                        </td>
                        <td className="px-4 py-4">
                          {call.analyzed
                            ? <span className={`rounded border px-2 py-0.5 text-xs font-semibold ${risk?.cls || ''}`}>{risk?.label || call.risk_level}</span>
                            : <span className="text-xs text-slate-400">Pending</span>}
                        </td>
                        <td className="px-4 py-4">
                          {active ? <ProcessingStatus call={call} onCancel={cancelAnalysis} cancelling={cancellingIds.has(call.db_call_id)} /> : call.status === 'failed'
                            ? <span className="text-xs font-semibold text-red-600" title={call.error}>Failed</span>
                            : call.status === 'cancelled'
                              ? <span className="text-xs font-semibold text-amber-600">Cancelled</span>
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
