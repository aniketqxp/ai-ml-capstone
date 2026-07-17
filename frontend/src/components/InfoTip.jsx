// Small hover tooltip — used to explain HOW a score was computed without
// cluttering the scorecard with modality tags.
export function InfoTip({ tip, children }) {
  return (
    <span className="relative group/tip inline-flex items-center">
      {children}
      <span className="pointer-events-none absolute z-30 hidden group-hover/tip:block bottom-full left-1/2 -translate-x-1/2 mb-1.5 w-60 bg-white border border-slate-200 text-slate-600 dark:bg-slate-900 dark:border-slate-700 dark:text-slate-300 rounded-lg px-2.5 py-2 text-[10px] leading-relaxed shadow-xl normal-case tracking-normal font-normal">
        {tip}
      </span>
    </span>
  );
}

export function InfoDot() {
  return (
    <span className="text-slate-400 hover:text-slate-600 border border-slate-300 dark:text-slate-500 dark:hover:text-slate-300 dark:border-slate-700 rounded-full w-3.5 h-3.5 inline-flex items-center justify-center text-[8px] cursor-help select-none">i</span>
  );
}
