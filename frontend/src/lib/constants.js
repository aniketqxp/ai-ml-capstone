export const SENTIMENT_STYLE = {
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

export const CHAPTER_COLORS = [
  '#6366f1', '#0ea5e9', '#10b981', '#f59e0b',
  '#ec4899', '#8b5cf6', '#14b8a6', '#ef4444',
  '#22c55e', '#eab308', '#06b6d4', '#a855f7',
];

export const COMPLIANCE_ITEMS = [
  ['name_announced', 'Name announced'],
  ['company_announced', 'Company announced'],
  ['recording_disclosure', 'Recording disclosure'],
  ['identity_verified', 'Identity verified'],
  ['resolution_provided', 'Resolution provided'],
  ['transfer_next_steps', 'Transfer next-steps'],
];

export const QUALITY_DIMS = [
  ['efficiency', 'Efficiency'],
  ['problem_resolution', 'Problem resolution'],
  ['clarity', 'Clarity'],
  ['professionalism', 'Professionalism'],
  ['empathy', 'Empathy'],
  ['customer_satisfaction', 'Customer satisfaction'],
];

export const RISK_STYLE = {
  none:     { label: 'No escalation', cls: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950 dark:text-emerald-300 dark:border-emerald-800' },
  review:   { label: 'Review',        cls: 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:border-amber-800' },
  escalate: { label: 'Escalate',      cls: 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-800' },
};

// Where each agent grounded its judgments on THIS call. One lane per agent;
// every marker is a piece of evidence the agent cited, anchored to its real
// moment in the audio. Markers light up as playback passes them.
export const LANE_STYLE = {
  compliance: {
    label: 'Compliance',
    on:  'bg-indigo-400 border-indigo-300 shadow-[0_0_6px_rgba(129,140,248,0.8)]',
    off: 'bg-indigo-200 border-indigo-300 dark:bg-indigo-900 dark:border-indigo-700',
    fail: 'bg-red-500 border-red-400',
  },
  quality: {
    label: 'Quality',
    on:  'bg-emerald-400 border-emerald-300 shadow-[0_0_6px_rgba(52,211,153,0.8)]',
    off: 'bg-emerald-200 border-emerald-300 dark:bg-emerald-900 dark:border-emerald-700',
    fail: 'bg-red-500 border-red-400',
  },
  workflow: {
    label: 'Workflow',
    on:  'bg-sky-400 border-sky-300 shadow-[0_0_6px_rgba(56,189,248,0.8)]',
    off: 'bg-sky-200 border-sky-300 dark:bg-sky-900 dark:border-sky-700',
    fail: 'bg-red-500 border-red-400',
  },
  escalation: {
    label: 'Escalation',
    on:  'bg-amber-400 border-amber-300 shadow-[0_0_6px_rgba(251,191,36,0.8)]',
    off: 'bg-amber-200 border-amber-300 dark:bg-amber-900 dark:border-amber-700',
    fail: 'bg-red-500 border-red-400',
  },
};
