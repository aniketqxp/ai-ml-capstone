import React, { useState, useEffect } from 'react';

function App() {
  const [status, setStatus] = useState("Connecting...");

  useEffect(() => {
    fetch("http://localhost:8000/")
      .then(res => res.json())
      .then(data => setStatus(`Backend is ${data.status}`))
      .catch(() => setStatus("Offline"));
  }, []);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col justify-between">
      <header className="p-6 border-b border-slate-800 bg-slate-950 flex justify-between items-center">
        <h1 className="text-2xl font-bold tracking-tight text-indigo-400">AI/ML Compliance Dashboard</h1>
        <span className="px-3 py-1 rounded bg-indigo-950 text-indigo-300 border border-indigo-800 text-sm font-medium">{status}</span>
      </header>

      <main className="flex-grow p-8 max-w-7xl mx-auto w-full grid grid-cols-1 md:grid-cols-3 gap-8">
        <section className="bg-slate-950 rounded-xl border border-slate-800 p-6 shadow-xl flex flex-col justify-between">
          <h2 className="text-xl font-semibold mb-4 text-slate-200">Compliance Audits</h2>
          <p className="text-slate-400 flex-grow">Track and trace compliance scores, Llama structural evidence, and emotional classifications.</p>
        </section>
      </main>

      <footer className="p-6 border-t border-slate-800 bg-slate-950 text-center text-slate-500 text-sm">
        © 2026 AI/ML Capstone Infrastructure
      </footer>
    </div>
  );
}

export default App;
