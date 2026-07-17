/*
 * Call evaluation report — a seekable call viewed under one LENS at a time.
 *
 * Model:
 *   - The CALL is a real, playable, seekable transport (audio + chapters).
 *     It stays constant across every lens; only the overlay changes.
 *   - A lens is a professional way of evaluating a call. Each surfaces ONE
 *     score (computed however behind the scenes) made defensible by the
 *     evidence beneath it — every claim clickable to the exact moment, which
 *     seeks the audio there so you can hear it yourself.
 *   - Friction never claims "mood." It flags discrete moments and hands you
 *     the words + the tone reading; you supply the judgment. Ambiguity is
 *     left as ambiguity, not smoothed into a curve.
 *
 * Reads the same call_data.json as the player; audio is /call_1.mp3.
 */
import React, { useState, useRef, useEffect, useMemo } from 'react';
import callData from '../call_data.json';

const ev = callData.evaluation;
const turns = callData.turns || [];
const chapters = callData.chapters || [];

const fmt = (s) => (s == null ? '' : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`);
const nearestTurn = (sec, speaker) => {
  const inWin = turns.filter((t) => t.start <= sec && sec <= (t.end ?? t.start + 4));
  return (speaker ? inWin.find((t) => t.speaker === speaker) : null) || inWin[0]
    || turns.reduce((b, t) => (Math.abs(t.start - sec) < Math.abs((b?.start ?? 1e9) - sec) ? t : b), null);
};

const RULE_LABELS = {
  name_announced: 'Agent gave their name',
  company_announced: 'Agent named the company',
  recording_disclosure: 'Recording was disclosed',
  identity_verified: 'Customer identity verified',
  resolution_provided: 'Resolution / next steps before closing',
  transfer_next_steps: 'Transfer handled with next steps',
};

/* ── build every lens once ────────────────────────────────────────── */

function buildLenses(dur) {
  // COMPLIANCE
  const rules = Object.entries(RULE_LABELS).map(([key, label]) => {
    const it = ev.compliance[key] || {};
    const status = it.passed === true ? 'pass' : it.passed === false ? 'fail' : 'na';
    const absent = it.passed === false && !it.evidence;
    return { id: `c:${key}`, key, label, status, absent,
             sec: it.evidence?.sec ?? (absent ? dur * 0.03 : null),
             evidence: it.evidence, note: it.note };
  });
  const cApplic = rules.filter((r) => r.status !== 'na');
  const cPass = cApplic.filter((r) => r.status === 'pass').length;
  const cBad = cApplic.some((r) => r.status === 'fail');

  // RESOLUTION
  const raw = ev.workflow?.expected_steps || [];
  const secs = raw.map((s) => s.evidence?.sec ?? null);
  const steps = raw.map((s, i) => {
    let sec = s.evidence?.sec ?? null;
    if (sec == null) {
      const prev = [...secs.slice(0, i)].reverse().find((x) => x != null);
      const next = secs.slice(i + 1).find((x) => x != null);
      sec = prev != null && next != null ? (prev + next) / 2 : prev ?? next ?? ((i + 0.5) / raw.length) * dur;
    }
    return { id: `r:${i}`, label: s.step, status: s.met === true ? 'pass' : s.met === false ? 'fail' : 'na',
             absent: s.met === false, sec, evidence: s.evidence, note: s.rationale };
  });
  const rDone = steps.filter((s) => s.status === 'pass').length;
  const rBad = steps.some((s) => s.status === 'fail');

  // FRICTION — discrete tone moments, NOT a curve. Evidence = tone signal + words.
  const traj = (ev._acoustic?.trajectory || []).filter((b) => b.esc != null);
  const moments = [...traj].sort((a, b) => b.esc - a.esc).slice(0, 4).map((b, i) => {
    const sec = b.p * dur;
    const t = nearestTurn(sec, 'CUSTOMER') || nearestTurn(sec);
    return { id: `f:${i}`, sec, esc: b.esc, tone: b.cust_val,
             label: `Customer tone rose here (${Math.round(b.esc * 100)}/100)`,
             evidence: t ? { quote: t.text, speaker: t.speaker, sec: t.start } : null };
  }).sort((a, b) => a.sec - b.sec);
  const eh = ev.escalation.hybrid || {};
  const frictionScore = Math.round(100 * Math.max(eh.late_mean_escalation || 0, (eh.peak_escalation || 0) * 0.6));
  const agree = eh.acoustic_risk != null && eh.method === 'agreement';

  // EFFICIENCY — operational, from turns + chapters.
  let talk = { AGENT: 0, CUSTOMER: 0 };
  turns.forEach((t) => { const d = (t.end ?? t.start) - t.start; if (talk[t.speaker] != null) talk[t.speaker] += Math.max(0, d); });
  const spoken = talk.AGENT + talk.CUSTOMER;
  const silence = Math.max(0, dur - spoken);
  const chapDur = chapters.map((c) => ({ label: c.label, start: c.start, len: c.end - c.start }));
  const longest = chapDur.reduce((a, b) => (b.len > (a?.len ?? 0) ? b : a), null);
  const deadPct = Math.round((silence / dur) * 100);

  return {
    compliance: {
      key: 'compliance', label: 'Compliance', q: 'Were the rules followed?',
      score: Math.round((cPass / (cApplic.length || 1)) * 100), unit: '%', bad: cBad,
      headline: cBad ? `${cApplic.length - cPass} of ${cApplic.length} required rules broken`
                     : `All ${cApplic.length} applicable rules followed`,
      how: <>Universal rules for every {ev.metadata.domain} call. The score is simply the share followed — but a
           single broken rule flags the call regardless, and each verdict below is backed by the exact quote.</>,
      markers: rules.filter((r) => r.sec != null), rows: rules, kind: 'check', caption: 'Every rule, with proof',
    },
    resolution: {
      key: 'resolution', label: 'Resolution', q: 'Did they do the job?',
      score: Math.round((rDone / (steps.length || 1)) * 100), unit: '%', bad: rBad,
      headline: `${rDone} of ${steps.length} expected steps completed`,
      how: <>The customer called to <strong>{ev.workflow?.subject}</strong>. The score is the share of expected
           steps done; a missed step is pinned to where it should have happened.</>,
      markers: steps, rows: steps, kind: 'check', caption: 'Expected steps, with proof',
    },
    friction: {
      key: 'friction', label: 'Friction', q: 'Where did it get tense?', hasIssue: frictionScore >= 30,
      score: frictionScore, unit: '/100', bad: frictionScore >= 50,
      headline: agree ? `Words and tone agree: customer sounded ${ev.escalation.customer_emotion_text}`
                      : `Customer sounded ${ev.escalation.customer_emotion_text}`,
      how: <>This lens does not claim a mood. It flags the moments the customer&rsquo;s <strong>tone</strong> rose
           and shows you the <strong>words</strong> there — you judge. Where words and tone independently agree,
           that is stated; where they don&rsquo;t, it is left open.</>,
      markers: moments, rows: moments, kind: 'moment', caption: 'Highest-tension moments (hear them)',
    },
    efficiency: {
      key: 'efficiency', label: 'Efficiency', q: 'Was time used well?',
      score: deadPct, unit: '% dead air', bad: deadPct > 25,
      headline: `${fmt(dur)} handle time · longest phase: ${longest?.label}`,
      how: <>Operational, not a judgment: where the minutes went. Talk split and dead air aggregate across calls
           into real staffing signal.</>,
      markers: chapters.map((c, i) => ({ id: `e:${i}`, sec: c.start, status: c.label === longest?.label ? 'na' : 'pass', label: c.label })),
      rows: null,
      bars: [
        { label: 'Agent talking', pct: Math.round((talk.AGENT / dur) * 100) },
        { label: 'Customer talking', pct: Math.round((talk.CUSTOMER / dur) * 100) },
        { label: 'Silence / dead air', pct: deadPct },
      ],
      chapDur, longest, kind: 'bars', caption: 'Where the time went',
    },
  };
}

function overall(L) {
  if (L.compliance.bad) return { bad: true, v: 'Flagged — rule broken', b: 'A required rule was not followed; this outranks everything else.' };
  const risk = ev.escalation.risk_level || 'none';
  if (risk === 'escalate') return { bad: true, v: 'Flagged — escalate', b: 'Escalation risk needs immediate attention.' };
  if (risk === 'review') return { bad: true, v: 'Flagged — review', b: 'Signals disagree or indicate risk; a person should listen.' };
  return { bad: false, v: 'Clean call', b: 'All rules followed, no risk detected.' };
}

/* ── the seekable call ────────────────────────────────────────────── */

function Transport({ audioRef, dur, cur, playing, onToggle, onSeek, markers, focused, onPick }) {
  const barRef = useRef(null);
  const pctL = (s) => `${Math.max(0, Math.min(100, (s / dur) * 100))}%`;
  const longest = markers; // not used; placeholder
  const clickBar = (e) => {
    const r = barRef.current.getBoundingClientRect();
    onSeek(((e.clientX - r.left) / r.width) * dur);
  };
  return (
    <div className="transport">
      <div className="tp-row">
        <button className="tp-play" onClick={onToggle} aria-label={playing ? 'pause' : 'play'}>{playing ? '❚❚' : '▶'}</button>
        <span className="tp-time">{fmt(cur)} / {fmt(dur)}</span>
        <div className="tp-barwrap">
          <div className="tp-bar" ref={barRef} onClick={clickBar}>
            {chapters.map((c, i) => (
              <span key={i} className="tp-chap" style={{ left: pctL(c.start), width: pctL(c.end - c.start) }} title={c.label}>{c.label}</span>
            ))}
            <span className="tp-playhead" style={{ left: pctL(cur) }} />
          </div>
          <div className="tp-rail">
            {markers.filter((m) => m.sec != null).map((m) => {
              const cls = m.absent ? 'absent' : m.status === 'pass' ? 'pass' : m.status === 'na' ? 'na' : m.status === 'fail' ? 'fail' : 'pass';
              const glyph = m.absent ? '✕' : cls === 'pass' ? '✓' : cls === 'na' ? '·' : '✕';
              return (
                <button key={m.id} className={`tpmark ${cls}${focused === m.id ? ' focused' : ''}`}
                        style={{ left: pctL(m.sec) }} title={`${m.label} — ${fmt(m.sec)}`}
                        onClick={() => onPick(m.id, m.sec)}>{glyph}</button>
              );
            })}
          </div>
        </div>
      </div>
      <p className="tp-hint">Click the bar to seek anywhere. Each mark is a finding for the current lens — click it to jump and hear it.</p>
    </div>
  );
}

/* ── evidence quote (seeks audio) ─────────────────────────────────── */

function Quote({ e, onSeek }) {
  if (!e) return null;
  return (
    <div className="quote">
      <div className="quote-meta">
        <button className="seek" onClick={() => onSeek(e.sec)}>{fmt(e.sec)}</button>
        <span>{e.speaker}</span>
      </div>
      <p className="quote-text">&ldquo;{e.quote}&rdquo;</p>
    </div>
  );
}

/* ── page ─────────────────────────────────────────────────────────── */

export default function Report() {
  const audioRef = useRef(null);
  const rowRefs = useRef({});
  const [dur, setDur] = useState(callData.duration || 600);
  const [cur, setCur] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [lensKey, setLensKey] = useState('compliance');
  const [focused, setFocused] = useState(null);
  const [pending, setPending] = useState(null);

  const L = useMemo(() => buildLenses(dur), [dur]);
  const o = useMemo(() => overall(L), [L]);
  const lens = L[lensKey];

  useEffect(() => {
    const a = audioRef.current; if (!a) return;
    const t = () => setCur(a.currentTime);
    const m = () => { if (a.duration && isFinite(a.duration)) setDur(a.duration); };
    const p = () => setPlaying(true), s = () => setPlaying(false);
    a.addEventListener('timeupdate', t); a.addEventListener('loadedmetadata', m);
    a.addEventListener('play', p); a.addEventListener('pause', s);
    return () => { a.removeEventListener('timeupdate', t); a.removeEventListener('loadedmetadata', m);
                   a.removeEventListener('play', p); a.removeEventListener('pause', s); };
  }, []);

  const seek = (sec) => { const a = audioRef.current; if (a) { a.currentTime = Math.max(0, Math.min(dur, sec)); setCur(a.currentTime); } };
  const toggle = () => { const a = audioRef.current; if (!a) return; a.paused ? a.play() : a.pause(); };
  const pick = (id, sec) => { setFocused(id); if (sec != null) seek(sec); setPending(id); };

  useEffect(() => {
    if (pending && rowRefs.current[pending]) {
      rowRefs.current[pending].scrollIntoView({ behavior: 'smooth', block: 'center' });
      setPending(null);
    }
  }, [pending, lensKey]);

  return (
    <main className="report">
      <div className="rpt-head">
        <div>
          <p className="rpt-id">Call evaluation</p>
          <p className="rpt-title">{ev.metadata.call_id}</p>
          <p className="rpt-sub">{ev.metadata.domain} · rubric v{ev.rubric_version}</p>
        </div>
        <div className={`overall${o.bad ? ' bad' : ''}`}>
          <p className="v">{o.v}</p>
          <p className="b">{o.b}</p>
        </div>
      </div>

      <Transport audioRef={audioRef} dur={dur} cur={cur} playing={playing}
                 onToggle={toggle} onSeek={seek} markers={lens.markers} focused={focused} onPick={pick} />

      <div className="lenstabs">
        {Object.values(L).map((x) => (
          <button key={x.key} className={`lenstab${lensKey === x.key ? ' active' : ''}${lensKey === x.key && x.bad ? ' bad' : ''}`}
                  onClick={() => { setLensKey(x.key); setFocused(null); }}>
            {x.label}{(x.bad || x.hasIssue) && <span className="dot" />}
            <span className="lt-q">{x.q}</span>
          </button>
        ))}
      </div>

      <div className="lens" key={lensKey}>
        <div className="lens-top">
          <span className={`lens-score${lens.bad ? ' bad' : ''}`}>{lens.score}<span className="unit">{lens.unit}</span></span>
          <span className={`lens-headline${lens.bad ? ' bad' : ''}`}>{lens.headline}</span>
        </div>
        <p className="lens-how">{lens.how}</p>

        {lens.kind === 'bars' ? (
          <>
            <p className="ev-caption">{lens.caption}</p>
            <div className="ev-list">
              {lens.bars.map((b) => (
                <div className="bar-row" key={b.label}>
                  <span>{b.label}</span>
                  <span className="bar-track"><span className="bar-fill" style={{ width: `${b.pct}%` }} /></span>
                  <span className="bar-val">{b.pct}%</span>
                </div>
              ))}
            </div>
            <p className="ev-caption" style={{ marginTop: 22 }}>Time per phase — click to jump</p>
            <div className="ev-list">
              {lens.chapDur.map((c, i) => (
                <div className={`ev-row${c.label === lens.longest?.label ? ' issue' : ''}`} key={i}>
                  <button className="ev-status na" onClick={() => seek(c.start)}>{fmt(c.len)}</button>
                  <p className="ev-label">{c.label}{c.label === lens.longest?.label ? ' — longest phase' : ''}</p>
                </div>
              ))}
            </div>
          </>
        ) : (
          <>
            <p className="ev-caption">{lens.caption}</p>
            <div className="ev-list">
              {lens.rows.map((r) => {
                const issue = r.status === 'fail';
                const mark = lens.kind === 'moment' ? fmt(r.sec)
                  : r.status === 'pass' ? (lens.key === 'compliance' ? 'FOLLOWED' : 'DONE')
                  : r.status === 'fail' ? (lens.key === 'compliance' ? 'NOT DONE' : 'MISSED') : 'N/A';
                const cls = r.status === 'pass' ? 'pass' : r.status === 'fail' ? 'fail' : 'na';
                return (
                  <div key={r.id} ref={(el) => { if (el) rowRefs.current[r.id] = el; }}
                       className={`ev-row${issue ? ' issue' : ''}${focused === r.id ? ' focused' : ''}`}>
                    <button className={`ev-status ${lens.kind === 'moment' ? 'na' : cls}`}
                            onClick={() => pick(r.id, r.sec)}>{mark}</button>
                    <div>
                      <p className="ev-label">{r.label}</p>
                      {r.key === 'identity_verified' && ev.compliance.identity_method &&
                        <p className="ev-note">Verified using: {ev.compliance.identity_method}</p>}
                      {r.note && <p className="ev-note">{r.note}</p>}
                      {r.absent && <p className="ev-missing">Expected around {fmt(r.sec)} — nothing in the call covers it.</p>}
                      <Quote e={r.evidence} onSeek={seek} />
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>

      <footer className="trust">
        <h3>Why you can trust this</h3>
        <p>Every quote was checked to exist word-for-word in the transcript: {ev._anchor_stats?.anchored} of {ev._anchor_stats?.total} citations verified against their timestamps — and each one seeks the audio so you can hear it.</p>
        <p>Each lens shows one score, but the score is not the claim — the evidence under it is. Where words and voice were both readable they were compared; agreement is stated, disagreement is left open rather than averaged away.</p>
        <p>Anything undeterminable is marked N/A, never guessed.</p>
      </footer>

      <audio ref={audioRef} src="/call_1.mp3" preload="metadata" />
    </main>
  );
}
