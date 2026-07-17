export function ScoreDots({ score }) {
  const color = score >= 4 ? 'bg-emerald-500' : score === 3 ? 'bg-amber-500' : 'bg-red-500';
  return (
    <span className="flex gap-1">
      {[1, 2, 3, 4, 5].map((n) => (
        <span key={n} className={`w-1.5 h-1.5 rounded-full ${n <= score ? color : 'bg-slate-300 dark:bg-slate-700'}`} />
      ))}
    </span>
  );
}
