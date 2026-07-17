import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { fmt } from '../lib/format';
import { RISK_STYLE } from '../lib/constants';
import { ThemeToggle } from '../components/ThemeToggle';

export default function Dashboard() {
  const [calls, setCalls] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [domainFilter, setDomainFilter] = useState('all');
  const [accentFilter, setAccentFilter] = useState('all');
  const [riskFilter, setRiskFilter] = useState('all');
  const [sortField, setSortField] = useState(null);
  const [sortDir, setSortDir] = useState('asc');
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 10;

  useEffect(() => {
    fetch('/calls_index.json')
      .then((res) => {
        if (!res.ok) throw new Error('could not load calls_index.json');
        return res.json();
      })
      .then(setCalls)
      .catch((err) => setLoadError(err.message));
  }, []);

  useEffect(() => {
    setPage(1);
  }, [domainFilter, accentFilter, riskFilter]);

  const domains = calls ? [...new Set(calls.map((c) => c.domain))].sort() : [];
  const accents = calls ? [...new Set(calls.map((c) => c.accent))].sort() : [];
  const risks = Object.keys(RISK_STYLE);
  const filtered = calls
    ? calls.filter(
        (c) =>
          (domainFilter === 'all' || c.domain === domainFilter) &&
          (accentFilter === 'all' || c.accent === accentFilter) &&
          (riskFilter === 'all' || c.risk_level === riskFilter)
      )
    : [];

  const toggleSort = (field) => {
    if (sortField === field) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else { setSortField(field); setSortDir('asc'); }
  };
  const sortValue = (c) => {
    if (sortField === 'duration') return c.duration;
    if (sortField === 'compliance') return c.compliance_applicable ? c.compliance_passed / c.compliance_applicable : 0;
    if (sortField === 'workflow') return c.workflow_total ? c.workflow_met / c.workflow_total : 0;
    return 0;
  };
  const sorted = sortField
    ? [...filtered].sort((a, b) => (sortValue(a) - sortValue(b)) * (sortDir === 'asc' ? 1 : -1))
    : filtered;
  const sortArrow = (field) => (sortField === field ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '');
  const ratioTone = (passed, total) => {
    if (!total) return 'text-slate-500 dark:text-slate-400';
    const r = passed / total;
    if (r >= 0.8) return 'text-emerald-700 dark:text-emerald-400';
    if (r >= 0.5) return 'text-amber-700 dark:text-amber-400';
    return 'text-red-700 dark:text-red-400 font-semibold bg-red-50 dark:bg-red-950/50 px-1.5 py-0.5 rounded';
  };
  const heroTone = (r) => {
    if (r >= 0.8) return 'text-emerald-600 dark:text-emerald-400';
    if (r >= 0.5) return 'text-amber-600 dark:text-amber-400';
    return 'text-red-600 dark:text-red-400';
  };

  const stats = calls
    ? (() => {
        const compPassed = calls.reduce((s, c) => s + c.compliance_passed, 0);
        const compApplicable = calls.reduce((s, c) => s + c.compliance_applicable, 0);
        const wfMet = calls.reduce((s, c) => s + c.workflow_met, 0);
        const wfTotal = calls.reduce((s, c) => s + c.workflow_total, 0);
        const escalateCount = calls.filter((c) => c.risk_level === 'escalate').length;
        const reviewCount = calls.filter((c) => c.risk_level === 'review').length;
        return {
          totalCalls: calls.length,
          complianceRate: compApplicable ? compPassed / compApplicable : 0,
          workflowRate: wfTotal ? wfMet / wfTotal : 0,
          avgDuration: calls.reduce((s, c) => s + c.duration, 0) / calls.length,
          escalateCount,
          reviewCount,
          needsAttention: escalateCount + reviewCount,
        };
      })()
    : null;

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageSafe = Math.min(page, totalPages);
  const paged = sorted.slice((pageSafe - 1) * PAGE_SIZE, pageSafe * PAGE_SIZE);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100 flex flex-col">
      <header className="px-6 py-4 border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-indigo-600 dark:text-indigo-400">Call Compliance Dashboard</h1>
          <p className="text-sm text-slate-500 dark:text-slate-500 mt-0.5">
            {calls
              ? filtered.length === calls.length
                ? `${calls.length} evaluated calls`
                : `showing ${filtered.length} of ${calls.length} evaluated calls`
              : 'Loading…'}
          </p>
        </div>
        <ThemeToggle />
      </header>

      <main className="flex-grow max-w-6xl w-full mx-auto p-6">
        {loadError && <p className="text-red-600 dark:text-red-400 text-sm">{loadError}</p>}
        {!calls && !loadError && <p className="text-slate-500 text-sm">Loading calls…</p>}

        {calls && stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white rounded-xl border border-slate-200 shadow-xl p-4 dark:bg-slate-950 dark:border-slate-800">
              <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold">Total calls</div>
              <div className="text-3xl font-bold text-slate-900 dark:text-slate-100 mt-1">{stats.totalCalls}</div>
              <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">avg {fmt(stats.avgDuration)}</div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 shadow-xl p-4 dark:bg-slate-950 dark:border-slate-800">
              <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold">Avg compliance</div>
              <div className={`text-3xl font-bold mt-1 ${heroTone(stats.complianceRate)}`}>{Math.round(stats.complianceRate * 100)}%</div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 shadow-xl p-4 dark:bg-slate-950 dark:border-slate-800">
              <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold">Avg workflow</div>
              <div className={`text-3xl font-bold mt-1 ${heroTone(stats.workflowRate)}`}>{Math.round(stats.workflowRate * 100)}%</div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 shadow-xl p-4 dark:bg-slate-950 dark:border-slate-800">
              <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold">Needs review</div>
              <div className={`text-3xl font-bold mt-1 ${stats.needsAttention === 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}`}>
                {stats.needsAttention}
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                {stats.escalateCount} escalate · {stats.reviewCount} review
              </div>
            </div>
          </div>
        )}

        {calls && (
          <div className="bg-white rounded-xl border border-slate-200 shadow-xl overflow-hidden dark:bg-slate-950 dark:border-slate-800">
            <table className="w-full text-base">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 text-left text-xs uppercase tracking-wider text-slate-500">
                  <th className="px-5 py-4 font-semibold">Call</th>
                  <th className="px-5 py-4 font-semibold">
                    <select
                      value={domainFilter}
                      onChange={(e) => setDomainFilter(e.target.value)}
                      className="bg-transparent text-slate-500 dark:text-slate-500 font-semibold uppercase tracking-wider text-xs cursor-pointer focus:outline-none -ml-1"
                    >
                      <option value="all">Domain</option>
                      {domains.map((d) => (
                        <option key={d} value={d} className="capitalize">{d}</option>
                      ))}
                    </select>
                  </th>
                  <th className="px-5 py-4 font-semibold">
                    <select
                      value={accentFilter}
                      onChange={(e) => setAccentFilter(e.target.value)}
                      className="bg-transparent text-slate-500 dark:text-slate-500 font-semibold uppercase tracking-wider text-xs cursor-pointer focus:outline-none -ml-1"
                    >
                      <option value="all">Accent</option>
                      {accents.map((a) => (
                        <option key={a} value={a}>{a}</option>
                      ))}
                    </select>
                  </th>
                  <th className="px-5 py-4 font-semibold">
                    <button onClick={() => toggleSort('duration')} className="hover:text-slate-700 dark:hover:text-slate-300">
                      Duration{sortArrow('duration')}
                    </button>
                  </th>
                  <th className="px-5 py-4 font-semibold">
                    <button onClick={() => toggleSort('compliance')} className="hover:text-slate-700 dark:hover:text-slate-300">
                      Compliance{sortArrow('compliance')}
                    </button>
                  </th>
                  <th className="px-5 py-4 font-semibold">
                    <button onClick={() => toggleSort('workflow')} className="hover:text-slate-700 dark:hover:text-slate-300">
                      Workflow{sortArrow('workflow')}
                    </button>
                  </th>
                  <th className="px-5 py-4 font-semibold">
                    <select
                      value={riskFilter}
                      onChange={(e) => setRiskFilter(e.target.value)}
                      className="bg-transparent text-slate-500 dark:text-slate-500 font-semibold uppercase tracking-wider text-xs cursor-pointer focus:outline-none -ml-1"
                    >
                      <option value="all">Risk</option>
                      {risks.map((r) => (
                        <option key={r} value={r}>{RISK_STYLE[r]?.label || r}</option>
                      ))}
                    </select>
                  </th>
                  <th className="px-5 py-4 font-semibold">Subject</th>
                </tr>
              </thead>
              <tbody>
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-4 py-6 text-center text-slate-500 text-sm">
                      No calls match the selected filters.
                    </td>
                  </tr>
                )}
                {paged.map((c, i) => {
                  const risk = RISK_STYLE[c.risk_level];
                  const rank = (pageSafe - 1) * PAGE_SIZE + i + 1;
                  const label = `Call ${String(rank).padStart(String(calls.length).length, '0')}`;
                  return (
                    <tr key={c.call_id} className="border-b border-slate-100 dark:border-slate-800/60 last:border-0 hover:bg-slate-50 dark:hover:bg-slate-900/60 transition-colors">
                      <td className="px-5 py-4 whitespace-nowrap">
                        <Link to={`/calls/${c.call_id}`} className="text-indigo-600 dark:text-indigo-400 hover:underline font-mono text-sm">
                          {label}
                        </Link>
                      </td>
                      <td className="px-5 py-4 text-slate-700 dark:text-slate-300 capitalize">{c.domain}</td>
                      <td className="px-5 py-4 text-slate-500 dark:text-slate-500 font-mono text-sm">{c.accent}</td>
                      <td className="px-5 py-4 text-slate-500 dark:text-slate-400 font-mono text-sm">{fmt(c.duration)}</td>
                      <td className="px-5 py-4 font-mono text-sm">
                        <span className={ratioTone(c.compliance_passed, c.compliance_applicable)}>
                          {c.compliance_passed}/{c.compliance_applicable}
                        </span>
                      </td>
                      <td className="px-5 py-4 font-mono text-sm">
                        <span className={ratioTone(c.workflow_met, c.workflow_total)}>
                          {c.workflow_met}/{c.workflow_total}
                        </span>
                      </td>
                      <td className="px-5 py-4">
                        <span className={`text-xs font-semibold px-2 py-0.5 rounded border ${risk?.cls || ''}`}>
                          {risk?.label || c.risk_level}
                        </span>
                      </td>
                      <td className="px-5 py-4 text-slate-500 dark:text-slate-400 text-sm max-w-xs truncate" title={c.subject}>
                        {c.subject}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {filtered.length > 0 && totalPages > 1 && (
              <div className="flex items-center justify-between px-5 py-3 border-t border-slate-200 dark:border-slate-800 text-sm">
                <span className="text-slate-500 dark:text-slate-400">
                  Showing {(pageSafe - 1) * PAGE_SIZE + 1}–{Math.min(pageSafe * PAGE_SIZE, filtered.length)} of {filtered.length}
                </span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={pageSafe === 1}
                    className="px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-800 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-100 dark:hover:bg-slate-900"
                  >
                    ‹ Prev
                  </button>
                  {Array.from({ length: totalPages }, (_, i) => i + 1).map((p) => (
                    <button
                      key={p}
                      onClick={() => setPage(p)}
                      className={`w-8 h-8 rounded-lg text-sm font-medium ${
                        p === pageSafe
                          ? 'bg-indigo-600 text-white'
                          : 'hover:bg-slate-100 dark:hover:bg-slate-900 text-slate-600 dark:text-slate-400'
                      }`}
                    >
                      {p}
                    </button>
                  ))}
                  <button
                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                    disabled={pageSafe === totalPages}
                    className="px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-800 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-100 dark:hover:bg-slate-900"
                  >
                    Next ›
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
