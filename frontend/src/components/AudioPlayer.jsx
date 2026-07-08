import React from 'react';

const AudioPlayer = ({ src, fileName = "call_recording_001.wav" }) => {
  return (
    <div className="glass-card p-6 bg-slate-900/80">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-indigo-500/20 flex items-center justify-center border border-indigo-500/30">
            <svg className="w-5 h-5 text-indigo-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.984 5.984 0 01-1.757 4.243 1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.983 3.983 0 00-1.172-2.828a1 1 0 010-1.415z" clipRule="evenodd" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-200">{fileName}</h3>
            <p className="text-[10px] text-slate-500 uppercase tracking-widest">Wav Format • 44.1kHz</p>
          </div>
        </div>
        <button className="text-slate-400 hover:text-indigo-400 transition-colors">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
          </svg>
        </button>
      </div>

      <div className="relative h-16 bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center px-4">
        {/* Mock Waveform */}
        <div className="flex gap-[2px] items-center w-full">
          {[...Array(60)].map((_, i) => (
            <div 
              key={i} 
              className={`flex-grow rounded-full transition-all duration-300 ${i < 25 ? 'bg-indigo-500 h-[60%]' : 'bg-slate-800 h-[30%]'}`}
              style={{ height: `${Math.random() * 80 + 20}%` }}
            ></div>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between mt-4">
        <span className="text-xs font-mono text-slate-500">00:42 / 02:15</span>
        <div className="flex items-center gap-4">
          <button className="p-2 text-slate-400 hover:text-indigo-400">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M8.445 14.832A1 1 0 0010 14V6a1 1 0 00-1.555-.832l-6 4a1 1 0 000 1.664l6 4z" />
            </svg>
          </button>
          <button className="w-12 h-12 rounded-full bg-indigo-500 text-white flex items-center justify-center hover:bg-indigo-600 shadow-lg shadow-indigo-500/20">
            <svg className="w-6 h-6 ml-1" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
            </svg>
          </button>
          <button className="p-2 text-slate-400 hover:text-indigo-400">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M4.555 5.168A1 1 0 003 6v8a1 1 0 001.555.832l6-4a1 1 0 000-1.664l-6-4z" />
            </svg>
          </button>
        </div>
        <div className="w-20"></div>
      </div>
    </div>
  );
};

export default AudioPlayer;
