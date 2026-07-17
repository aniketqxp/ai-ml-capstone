import { useState, useEffect } from 'react';

// Shows how much of the audio has buffered (grey bar behind the playhead)
export function BufferedRanges({ audio, duration }) {
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
    return () => { a.removeEventListener('progress', update); a.removeEventListener('timeupdate', update); };
  }, [audio]);
  if (!duration) return null;
  return <>
    {ranges.map(([s, e], i) => (
      <div key={i} className="absolute inset-y-0 bg-slate-400 dark:bg-slate-600 rounded-full pointer-events-none"
        style={{ left: `${(s / duration) * 100}%`, width: `${((e - s) / duration) * 100}%` }} />
    ))}
  </>;
}
