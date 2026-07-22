import React from 'react';

const ComplianceScorecard = ({ score = 85, industry = "Banking", flags = [] }) => {
  const getStatusColor = (s) => {
    if (s >= 90) return 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10';
    if (s >= 75) return 'text-amber-400 border-amber-500/20 bg-amber-500/10';
    return 'text-rose-400 border-rose-500/20 bg-rose-500/10';
  };

  return (
    <div className="glass-card p-6 flex flex-col gap-6">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Compliance Score</h3>
          <p className="text-xs text-slate-500 mt-1">Sector: {industry}</p>
        </div>
        <div className={`px-4 py-2 rounded-lg border font-bold text-2xl ${getStatusColor(score)}`}>
          {score}%
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex justify-between items-center text-sm">
          <span className="text-slate-300">Professionalism</span>
          <div className="w-32 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full bg-indigo-500" style={{ width: '92%' }}></div>
          </div>
        </div>
        <div className="flex justify-between items-center text-sm">
          <span className="text-slate-300">Resolution</span>
          <div className="w-32 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full bg-purple-500" style={{ width: '78%' }}></div>
          </div>
        </div>
        <div className="flex justify-between items-center text-sm">
          <span className="text-slate-300">Empathy</span>
          <div className="w-32 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full bg-pink-500" style={{ width: '85%' }}></div>
          </div>
        </div>
      </div>

      <div className="pt-4 border-t border-slate-800">
        <h4 className="text-xs font-semibold text-slate-500 uppercase mb-3">Critical Flags</h4>
        <div className="flex flex-wrap gap-2">
          {flags.length > 0 ? flags.map((flag, i) => (
            <span key={i} className="px-2 py-1 rounded bg-rose-500/20 border border-rose-500/30 text-rose-400 text-[10px] font-bold uppercase">
              {flag}
            </span>
          )) : (
            <span className="text-slate-400 text-xs italic">No critical violations detected</span>
          )}
        </div>
      </div>
    </div>
  );
};

export default ComplianceScorecard;
