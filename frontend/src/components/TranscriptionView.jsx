import React from 'react';

const TranscriptionView = ({ segments = [] }) => {
  const mockSegments = [
    { speaker: "Agent", time: "0:02", text: "Hello, thank you for calling Sheridan Bank support, my name is Alex. How can I help you today?", sentiment: "Positive" },
    { speaker: "Customer", time: "0:12", text: "Hi Alex, I'm having trouble with a transaction that was declined this morning, and I really need to get it sorted.", sentiment: "Neutral" },
    { speaker: "Agent", time: "0:25", text: "I'm very sorry to hear that. I completely understand how frustrating that can be. Let me look into this right away for you.", sentiment: "Empathetic" },
  ];

  const displaySegments = segments.length > 0 ? segments : mockSegments;

  return (
    <div className="glass-card flex flex-col h-full bg-slate-950/40">
      <div className="p-6 border-b border-slate-800 flex justify-between items-center bg-slate-900/20">
        <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse"></span>
          Live Transcription
        </h3>
        <div className="flex gap-2">
          <div className="text-[10px] bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 px-2 py-0.5 rounded">Diarization Active</div>
        </div>
      </div>

      <div className="flex-grow overflow-y-auto p-6 space-y-6 max-h-[500px]">
        {displaySegments.map((seg, i) => (
          <div key={i} className={`flex flex-col gap-2 ${seg.speaker === 'Agent' ? 'items-start' : 'items-end'}`}>
            <div className="flex items-center gap-2 px-1">
              <span className={`text-[10px] font-bold uppercase tracking-widest ${seg.speaker === 'Agent' ? 'text-indigo-400' : 'text-purple-400'}`}>
                {seg.speaker}
              </span>
              <span className="text-[10px] text-slate-600 font-mono">{seg.time}</span>
            </div>
            <div className={`max-w-[85%] p-4 rounded-2xl text-sm leading-relaxed ${
              seg.speaker === 'Agent' 
                ? 'bg-slate-900 border border-slate-800 text-slate-200 rounded-tl-none' 
                : 'bg-indigo-600/10 border border-indigo-500/20 text-indigo-100 rounded-tr-none'
            }`}>
              {seg.text}
              {seg.sentiment && (
                <div className="mt-2 pt-2 border-t border-slate-800/50 flex justify-between items-center">
                  <span className="text-[10px] text-slate-500">Sentiment: <span className="text-slate-300 font-medium">{seg.sentiment}</span></span>
                  <div className={`w-2 h-2 rounded-full ${
                    seg.sentiment === 'Positive' || seg.sentiment === 'Empathetic' ? 'bg-emerald-500 shadow-sm shadow-emerald-500/50' : 'bg-slate-600'
                  }`}></div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TranscriptionView;
