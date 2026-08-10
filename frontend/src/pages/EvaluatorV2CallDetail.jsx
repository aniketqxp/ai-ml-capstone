import {
  AlertTriangle,
  AudioLines,
  ArrowLeft,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  FileQuestion,
  GripVertical,
  ListChecks,
  Mail,
  Minus,
  Pause,
  PanelRightClose,
  PanelRightOpen,
  Play,
  Send,
  ShieldCheck,
  Volume2,
  X,
} from 'lucide-react';
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Link, useParams } from 'react-router-dom';
import WaveSurfer from 'wavesurfer.js';

import { apiUrl } from '../api';
import { ThemeToggle } from '../components/ThemeToggle';
import { fmt } from '../lib/format';

const STATE_COPY = {
  needs_attention: {
    label: 'Needs attention',
    icon: AlertTriangle,
    tone: 'border-red-800 bg-red-700 text-white dark:border-red-500 dark:bg-red-700 dark:text-white',
    iconTone: 'text-white',
  },
  no_attention_finding: {
    label: 'No review finding identified',
    icon: ShieldCheck,
    tone: 'border-emerald-800 bg-emerald-700 text-white dark:border-emerald-500 dark:bg-emerald-700 dark:text-white',
    iconTone: 'text-white',
  },
  evaluation_incomplete: {
    label: 'Evaluation incomplete',
    icon: FileQuestion,
    tone: 'border-amber-600 bg-amber-400 text-slate-950 dark:border-amber-400 dark:bg-amber-400 dark:text-slate-950',
    iconTone: 'text-slate-950',
  },
  evaluation_unavailable: {
    label: 'Evaluation unavailable',
    icon: FileQuestion,
    tone: 'border-slate-800 bg-slate-800 text-white dark:border-slate-600 dark:bg-slate-800 dark:text-white',
    iconTone: 'text-white',
  },
};

function classNames(...values) {
  return values.filter(Boolean).join(' ');
}

function useResizableWidth({ initial, min, max, storageKey, invert = false }) {
  const resolveMax = useCallback(
    () => Math.max(min, typeof max === 'function' ? max() : max),
    [max, min],
  );
  const clamp = useCallback(
    (value) => Math.min(resolveMax(), Math.max(min, value)),
    [min, resolveMax],
  );
  const [width, setWidth] = useState(() => {
    try {
      const saved = Number(window.localStorage.getItem(storageKey));
      return clamp(Number.isFinite(saved) && saved > 0 ? saved : initial);
    } catch {
      return clamp(initial);
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, String(width));
    } catch {
      // Resizing still works when browser storage is unavailable.
    }
  }, [storageKey, width]);

  useEffect(() => {
    const constrain = () => setWidth((value) => clamp(value));
    window.addEventListener('resize', constrain);
    return () => window.removeEventListener('resize', constrain);
  }, [clamp]);

  const startResize = useCallback((event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const originX = event.clientX;
    const originWidth = width;
    const previousCursor = document.body.style.cursor;
    const previousSelection = document.body.style.userSelect;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    const move = (moveEvent) => {
      const delta = (moveEvent.clientX - originX) * (invert ? -1 : 1);
      setWidth(clamp(originWidth + delta));
    };
    const stop = () => {
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', stop);
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousSelection;
    };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', stop);
  }, [clamp, invert, width]);

  const handleKeyDown = useCallback((event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home'].includes(event.key)) return;
    event.preventDefault();
    if (event.key === 'Home') {
      setWidth(clamp(initial));
      return;
    }
    const physicalDelta = event.key === 'ArrowRight' ? 16 : -16;
    setWidth((value) => clamp(
      value + physicalDelta * (invert ? -1 : 1),
    ));
  }, [clamp, initial, invert]);

  return {
    width,
    min,
    max: resolveMax(),
    startResize,
    handleKeyDown,
    reset: () => setWidth(clamp(initial)),
  };
}

function ResizeHandle({ label, controls, resize }) {
  return (
    <div
      role="separator"
      aria-label={label}
      aria-controls={controls}
      aria-orientation="vertical"
      aria-valuemin={resize.min}
      aria-valuemax={resize.max}
      aria-valuenow={Math.round(resize.width)}
      tabIndex={0}
      onPointerDown={resize.startResize}
      onKeyDown={resize.handleKeyDown}
      onDoubleClick={resize.reset}
      title="Drag to resize. Double-click to reset."
      className="group relative hidden cursor-col-resize items-center justify-center bg-slate-100 text-slate-400 outline-none transition hover:bg-blue-50 hover:text-[#1557ff] focus:bg-blue-50 focus:text-[#1557ff] lg:flex dark:bg-slate-900 dark:hover:bg-blue-950 dark:focus:bg-blue-950"
    >
      <span className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-slate-300 group-hover:bg-[#1557ff] group-focus:bg-[#1557ff] dark:bg-slate-700" />
      <GripVertical size={13} className="relative z-10" aria-hidden="true" />
    </div>
  );
}

function usefulSupportingCopy(value) {
  const text = String(value || '').trim();
  if (!text) return null;
  const implementationCopy = [
    /evidence-backed findings require review/i,
    /segments? matched a high-precision explicit-text rule/i,
    /applicable process checks were supported by transcript evidence/i,
    /supporting evidence is limited/i,
    /provisional support gate/i,
  ];
  return implementationCopy.some((pattern) => pattern.test(text)) ? null : text;
}

async function fetchOptionalJson(path) {
  try {
    const response = await fetch(apiUrl(path));
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

async function fetchEvaluatorArtifact(path) {
  const candidates = [...new Set([path, apiUrl(path)])];
  for (const candidate of candidates) {
    try {
      const response = await fetch(candidate);
      if (!response.ok) continue;
      const contentType = response.headers.get('content-type') || '';
      if (!contentType.includes('application/json')) continue;
      return await response.json();
    } catch {
      // Try the processing API after a missing same-origin artifact.
    }
  }
  return null;
}

function speakerLabel(speaker) {
  if (String(speaker).toUpperCase() === 'AGENT') return 'Agent';
  if (String(speaker).toUpperCase() === 'CUSTOMER') return 'Customer';
  return 'Unknown';
}

function threeClassSentiment(segment) {
  const value = String(
    segment?.sentiment_class || segment?.sentiment || '',
  ).toLowerCase();
  if (['positive', 'negative', 'neutral'].includes(value)) return value;
  const emotion = String(segment?.dominant_emotion || '').toLowerCase();
  if (['happy', 'happiness', 'calm'].includes(emotion)) return 'positive';
  if (
    ['anger', 'angry', 'fear', 'sad', 'sadness', 'disgust', 'anxiety', 'stress']
      .includes(emotion)
  ) return 'negative';
  return 'neutral';
}

function attachSentiment(turns, sentiment) {
  const normalized = (sentiment?.segments || [])
    .map((segment) => ({
      ...segment,
      start: Number(segment.start_time ?? segment.start),
      end: Number(segment.end_time ?? segment.end),
      className: segment.processing_status === 'success'
        ? threeClassSentiment(segment)
        : null,
    }))
    .filter((segment) => Number.isFinite(segment.start))
    .sort((a, b) => a.start - b.start);

  return (turns || []).map((turn, index, allTurns) => {
    const start = Number(turn.start);
    const wordEnd = turn.words?.length
      ? Number(turn.words[turn.words.length - 1].end)
      : null;
    const nextStart = Number(allTurns[index + 1]?.start);
    const end = Number.isFinite(Number(turn.end))
      ? Number(turn.end)
      : Number.isFinite(wordEnd)
        ? wordEnd
        : Number.isFinite(nextStart)
          ? nextStart
          : start + 30;
    const speaker = String(turn.speaker || '').toUpperCase();
    const matches = normalized.filter((segment) => (
      String(segment.speaker || '').toUpperCase() === speaker
      && segment.start < end + 0.2
      && (Number.isFinite(segment.end) ? segment.end : segment.start) > start - 0.2
    ));
    return matches.length ? { ...turn, sentimentSegments: matches } : turn;
  });
}

function unavailableRun(call, shadow) {
  const unsupported = shadow?.status === 'unsupported_domain';
  const failed = shadow?.status === 'failed';
  const summary = unsupported
    ? `Evaluator v2 does not yet have a ${call.evaluation?.metadata?.domain || 'matching'} domain profile for this call.`
    : failed
      ? 'The evaluator could not complete this run. The call and transcript remain available.'
      : 'No evaluator v2 artifact is available for this call.';
  return {
    mode: 'shadow',
    status: shadow?.status || 'unavailable',
    evaluator_version: shadow?.evaluator_version || 'v2',
    decision_sha256: null,
    presentation: {
      state: 'evaluation_unavailable',
      evaluation_status: shadow?.status || 'unavailable',
      attention_required: false,
      headline: 'Evaluation unavailable',
      summary,
      manager_questions: [],
      checklist: [],
      acoustic_context: {
        status: 'unavailable',
        coverage_label: 'Audio support unavailable',
        conclusion: 'No acoustic evaluation artifact is available.',
        observations: [],
      },
      primary_reasons: [],
      additional_reason_count: 0,
      positive_highlights: [],
      additional_positive_count: 0,
      recommended_action: null,
      evidence: [],
      completeness_notice: null,
      details: {
        additional_findings: [],
        additional_positive_findings: [],
        evidence: [],
        uncertainty_messages: shadow?.failure_reason
          ? [shadow.failure_reason]
          : [],
        evaluator_version: shadow?.evaluator_version || 'v2',
        domain_profile_id: shadow?.profile_id || 'not available',
        decision_policy_id: 'not run',
        decision_policy_version: '',
      },
    },
  };
}

function mergeWordTimings(call, timingArtifact) {
  if (!timingArtifact?.turns?.length) return call;
  const timedTurns = timingArtifact.turns;
  return {
    ...call,
    turns: (call.turns || []).map((turn, index) => {
      if (turn.words?.length) return turn;
      const timed = timedTurns[index];
      if (
        !timed
        || String(timed.speaker).toUpperCase()
          !== String(turn.speaker).toUpperCase()
        || Math.abs(Number(timed.start) - Number(turn.start)) > 0.25
      ) {
        return turn;
      }
      return { ...turn, words: timed.words || [] };
    }),
  };
}

function AudioPlayer({
  callId,
  audioRef,
  isPlaying,
  currentTime,
  duration,
  onToggle,
  onSeek,
}) {
  const waveformRef = useRef(null);

  useEffect(() => {
    const audio = audioRef.current;
    const container = waveformRef.current;
    if (!audio || !container) return undefined;

    const wavesurfer = WaveSurfer.create({
      container,
      media: audio,
      height: 52,
      waveColor: '#cbd5e1',
      progressColor: '#1557ff',
      cursorColor: '#1557ff',
      cursorWidth: 2,
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      normalize: true,
      interact: false,
    });

    return () => wavesurfer.destroy();
  }, [audioRef, callId]);

  return (
    <section
      aria-label="Call audio"
      className="border-y border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950"
    >
      <div className="flex w-full items-center gap-3 px-4 py-3 sm:gap-4 sm:px-5">
        <button
          type="button"
          onClick={onToggle}
          className="flex h-12 w-12 shrink-0 items-center justify-center rounded bg-[#1557ff] text-white transition duration-200 hover:bg-blue-700 active:scale-[0.97] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#1557ff]"
          title={isPlaying ? 'Pause call' : 'Play call'}
          aria-label={isPlaying ? 'Pause call' : 'Play call'}
        >
          {isPlaying ? (
            <Pause size={18} fill="currentColor" />
          ) : (
            <Play size={18} fill="currentColor" className="ml-0.5" />
          )}
        </button>
        <div className="hidden items-center gap-3 sm:flex">
          <Volume2 size={18} className="text-slate-700 dark:text-slate-300" aria-hidden="true" />
          <span className="w-20 font-mono text-xs font-semibold text-slate-700 dark:text-slate-300">
            {fmt(currentTime)} / {fmt(duration)}
          </span>
        </div>
        <div className="relative min-w-0 flex-1 py-1">
          <div ref={waveformRef} className="pointer-events-none h-[52px] w-full" aria-hidden="true" />
          <input
            type="range"
            min="0"
            max={Math.max(duration, 0)}
            step="0.1"
            value={Math.min(currentTime, duration || 0)}
            onChange={(event) => onSeek(Number(event.target.value))}
            className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
            aria-label="Seek through call"
          />
          <div className="pointer-events-none absolute inset-x-0 -bottom-1 flex justify-between font-mono text-[10px] text-slate-400 sm:hidden">
            <span>{fmt(currentTime)}</span>
            <span>{fmt(duration)}</span>
          </div>
        </div>
        <audio
          ref={audioRef}
          src={apiUrl(`/audio/${encodeURIComponent(callId)}.mp3`)}
          preload="metadata"
        />
      </div>
    </section>
  );
}

function StatusSection({ view }) {
  const config = STATE_COPY[view.state] || STATE_COPY.evaluation_incomplete;
  const Icon = config.icon;
  return (
    <section className={classNames('verdict-reveal border-b px-5 py-4', config.tone)}>
      <div className="flex items-center gap-3">
        <Icon
          size={22}
          className={classNames('shrink-0', config.iconTone)}
          aria-hidden="true"
        />
        <p className="text-sm font-bold">{config.label}</p>
      </div>
    </section>
  );
}

function EvidenceLink({
  evidenceIds,
  evidenceById,
  onSeek,
  compact = false,
}) {
  const evidence = evidenceIds
    .map((evidenceId) => evidenceById.get(evidenceId))
    .find((item) => item?.seekable);
  if (!evidence) return null;
  return (
    <button
      type="button"
      onClick={() => onSeek(evidence.start_seconds)}
      className={classNames(
        'inline-flex shrink-0 items-center gap-1.5 text-xs font-bold text-[#1557ff] transition hover:text-blue-800 hover:underline dark:text-blue-300',
        compact ? 'mt-0' : 'mt-2',
      )}
    >
      <Play size={12} fill="currentColor" aria-hidden="true" />
      {compact ? fmt(evidence.start_seconds) : `Evidence at ${fmt(evidence.start_seconds)}`}
    </button>
  );
}

const ANSWER_TONES = {
  yes: 'text-emerald-700 dark:text-emerald-400',
  no: 'text-red-700 dark:text-red-400',
  partly: 'text-amber-700 dark:text-amber-400',
  unclear: 'text-slate-500 dark:text-slate-400',
};

function ManagerQuestions({ questions, evidenceById, onSeek }) {
  if (!questions.length) return null;
  return (
    <section className="border-b border-slate-300 px-5 py-5 dark:border-slate-700">
      <h2 className="text-xs font-bold uppercase text-slate-950 dark:text-white">
        Call assessment
      </h2>
      <div className="mt-3 divide-y divide-slate-300 dark:divide-slate-700">
        {questions.map((item, index) => (
          <article key={item.question_id} className="grid grid-cols-[2rem_minmax(0,1fr)_auto] gap-x-3 py-4 first:pt-1">
            <span className="flex h-8 w-8 items-center justify-center rounded-sm bg-slate-950 font-mono text-sm font-bold text-white dark:bg-white dark:text-slate-950">
              {index + 1}
            </span>
            <div className="min-w-0">
              <p className="text-sm font-semibold leading-5 text-slate-950 dark:text-white">
                {item.question}
              </p>
              {usefulSupportingCopy(item.summary) && (
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  {usefulSupportingCopy(item.summary)}
                </p>
              )}
              <EvidenceLink
                evidenceIds={item.evidence_ids}
                evidenceById={evidenceById}
                onSeek={onSeek}
              />
            </div>
            <span className={classNames('pt-1 text-xs font-bold uppercase', ANSWER_TONES[item.answer])}>
              {item.answer_label}
            </span>
          </article>
        ))}
      </div>
    </section>
  );
}

function FindingSection({
  title,
  findings,
  evidenceById,
  onSeek,
  onFeedback,
  feedbackBusy,
  tone,
}) {
  if (!findings.length) return null;
  const toneClass = {
    incorrect: 'text-red-700 dark:text-red-400',
    missed: 'text-amber-700 dark:text-amber-400',
    concern: 'text-rose-700 dark:text-rose-400',
  }[tone];
  return (
    <section className="border-b border-slate-200 px-5 py-5 dark:border-slate-800">
      <h2 className={classNames('text-xs font-semibold uppercase', toneClass)}>
        {title}
      </h2>
      <div className="mt-3 divide-y divide-slate-200 dark:divide-slate-800">
        {findings.map((finding) => {
          return (
            <article key={finding.finding_id} className="py-4 first:pt-1">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-xs font-medium text-slate-500">
                    {finding.category_label}
                  </p>
                  <h3 className="mt-1 text-sm font-semibold text-slate-950 dark:text-white">
                    {finding.title}
                  </h3>
                </div>
                <button
                  type="button"
                  onClick={() =>
                    onFeedback('dismiss_finding', {
                      finding_id: finding.finding_id,
                    })
                  }
                  disabled={feedbackBusy}
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-slate-400 transition hover:bg-red-50 hover:text-red-700 disabled:opacity-50 dark:hover:bg-red-950/40 dark:hover:text-red-300"
                  title="Dismiss this finding"
                  aria-label={`Dismiss ${finding.title}`}
                >
                  <X size={16} aria-hidden="true" />
                </button>
              </div>
              <EvidenceLink
                evidenceIds={finding.evidence_ids}
                evidenceById={evidenceById}
                onSeek={onSeek}
              />
            </article>
          );
        })}
      </div>
    </section>
  );
}

function EmailActionPopover({ action, notification, busy, error, onSend }) {
  const [open, setOpen] = useState(false);
  const popoverRef = useRef(null);
  const triggerRef = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') setOpen(false);
    };
    const closeOnOutsideClick = (event) => {
      if (
        !popoverRef.current?.contains(event.target)
        && !triggerRef.current?.contains(event.target)
      ) setOpen(false);
    };
    document.addEventListener('keydown', closeOnEscape);
    document.addEventListener('pointerdown', closeOnOutsideClick);
    return () => {
      document.removeEventListener('keydown', closeOnEscape);
      document.removeEventListener('pointerdown', closeOnOutsideClick);
    };
  }, [open]);

  if (!action) return null;
  const sent = notification?.status === 'sent';
  const awaiting = notification?.status === 'awaiting_approval';
  const statusLabel = {
    sent: 'Email sent',
    awaiting_approval: 'Awaiting approval',
    configuration_required: 'Email setup required',
    failed: 'Delivery failed',
    disabled: 'Email disabled',
  }[notification?.status];
  const statusTone = sent
    ? 'bg-emerald-400'
    : notification?.status === 'failed'
      ? 'bg-red-400'
      : notification?.status
        ? 'bg-amber-300'
        : 'bg-white/80';
  const recipient = notification?.recipient || action.audience;
  return (
    <div className="fixed bottom-5 right-5 z-50 sm:bottom-6 sm:right-6">
      {open && (
        <section
          ref={popoverRef}
          id="email-action-popover"
          role="dialog"
          aria-label="Email action"
          className="absolute bottom-16 right-0 w-[min(22rem,calc(100vw-2.5rem))] rounded-lg border border-slate-300 bg-white p-4 shadow-2xl dark:border-slate-700 dark:bg-slate-950"
        >
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-sm font-bold text-slate-950 dark:text-white">
              Email action
            </h2>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="flex h-8 w-8 items-center justify-center rounded text-slate-400 transition hover:bg-slate-100 hover:text-slate-950 dark:hover:bg-slate-800 dark:hover:text-white"
              aria-label="Close email action"
              title="Close"
            >
              <X size={16} aria-hidden="true" />
            </button>
          </div>
          <p className="mt-3 text-sm font-semibold text-slate-950 dark:text-white">
            {action.label}
          </p>
          <p className="mt-1 truncate text-xs text-slate-500" title={recipient}>
            To: <span className="capitalize">{recipient}</span>
          </p>
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 pt-3 dark:border-slate-800">
            {statusLabel && (
              <span className={classNames(
                'text-xs font-semibold',
                sent ? 'text-emerald-700 dark:text-emerald-400' : 'text-amber-700 dark:text-amber-400',
              )}>
                {statusLabel}
              </span>
            )}
            {!sent && (
              <button
                type="button"
                onClick={onSend}
                disabled={busy}
                className="ml-auto inline-flex h-9 items-center gap-1.5 rounded bg-[#1557ff] px-3 text-xs font-bold text-white transition hover:bg-blue-700 disabled:opacity-50"
              >
                <Send size={13} aria-hidden="true" />
                {busy
                  ? 'Sending...'
                  : awaiting || action.requires_human_approval
                    ? 'Approve and send'
                    : 'Send email'}
              </button>
            )}
          </div>
          {(notification?.error || error) && (
            <p className="mt-3 border-t border-slate-200 pt-3 text-xs leading-5 text-red-600 dark:border-slate-800 dark:text-red-400">
              {notification?.error || error}
            </p>
          )}
        </section>
      )}
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="relative flex h-14 w-14 items-center justify-center rounded-full bg-[#1557ff] text-white shadow-lg transition hover:bg-blue-700 hover:shadow-xl active:scale-95"
        aria-label="Open email action"
        title="Email action"
        aria-expanded={open}
        aria-controls="email-action-popover"
      >
        <Mail size={21} aria-hidden="true" />
        <span
          className={classNames(
            'absolute right-0.5 top-0.5 h-3 w-3 rounded-full border-2 border-[#1557ff]',
            statusTone,
          )}
          aria-hidden="true"
        />
      </button>
    </div>
  );
}

function PositiveSection({ findings, evidenceById, onSeek }) {
  if (!findings.length) return null;
  return (
    <section className="border-b border-slate-200 px-5 py-5 dark:border-slate-800">
      <h2 className="text-xs font-semibold uppercase text-slate-500">
        Effective moments
      </h2>
      <ul className="relative mt-3 space-y-4 before:absolute before:bottom-3 before:left-[8px] before:top-3 before:w-px before:bg-emerald-200 dark:before:bg-emerald-900">
        {findings.map((finding) => {
          const evidence = finding.evidence_ids
            .map((evidenceId) => evidenceById.get(evidenceId))
            .find((item) => item?.seekable);
          return (
            <li key={finding.finding_id} className="relative grid grid-cols-[1.1rem_3rem_minmax(0,1fr)] items-start gap-2.5">
              <CheckCircle2
                size={18}
                className="relative z-10 mt-0.5 shrink-0 bg-white text-emerald-600 dark:bg-slate-950 dark:text-emerald-400"
                aria-hidden="true"
              />
              <span className="pt-0.5 font-mono text-[11px] font-semibold text-slate-600 dark:text-slate-300">
                {evidence ? fmt(evidence.start_seconds) : '--:--'}
              </span>
              <div>
                <p className="text-sm font-semibold text-slate-950 dark:text-white">
                  {finding.title}
                </p>
                <EvidenceLink
                  evidenceIds={finding.evidence_ids}
                  evidenceById={evidenceById}
                  onSeek={onSeek}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

const CHECK_STATUS = {
  demonstrated: {
    label: 'Demonstrated',
    icon: Check,
    tone: 'text-emerald-700 dark:text-emerald-400',
  },
  incorrect: {
    label: 'Incorrect',
    icon: X,
    tone: 'text-red-700 dark:text-red-400',
  },
  not_demonstrated: {
    label: 'Not demonstrated',
    icon: Minus,
    tone: 'text-amber-700 dark:text-amber-400',
  },
  unable_to_determine: {
    label: 'Unable to determine',
    icon: CircleHelp,
    tone: 'text-slate-500 dark:text-slate-400',
  },
};

function ChecklistSection({ checks, evidenceById, onSeek }) {
  if (!checks.length) return null;
  return (
    <section className="border-b border-slate-200 px-5 py-5 dark:border-slate-800">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase text-slate-500">
        <ListChecks size={15} aria-hidden="true" />
        Applicable checks
      </div>
      <div className="mt-3 divide-y divide-slate-200 dark:divide-slate-800">
        {checks.map((item) => {
          const config = CHECK_STATUS[item.status] || CHECK_STATUS.unable_to_determine;
          const Icon = config.icon;
          return (
            <article key={item.requirement_id} className="py-2.5 first:pt-1">
              <div className="flex items-start gap-2.5">
                <Icon
                  size={16}
                  className={classNames('mt-0.5 shrink-0', config.tone)}
                  aria-hidden="true"
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-start gap-3">
                    <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
                      {item.title}
                    </p>
                    <span className={classNames('ml-auto shrink-0 text-[11px] font-semibold', config.tone)}>
                      {config.label}
                    </span>
                    <EvidenceLink
                      evidenceIds={item.evidence_ids}
                      evidenceById={evidenceById}
                      onSeek={onSeek}
                      compact
                    />
                  </div>
                  {item.status !== 'demonstrated'
                    && !item.promoted
                    && usefulSupportingCopy(item.summary) && (
                    <p className="mt-1 text-xs leading-5 text-slate-500">
                      {usefulSupportingCopy(item.summary)}
                    </p>
                  )}
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

const SIGNAL_MODES = {
  escalation: {
    label: 'Escalation',
    key: 'escalation_score',
    format: (value) => `${Math.round(value * 100)}%`,
  },
  volume: {
    label: 'Volume',
    key: 'volume_db_mean',
    format: (value) => `${value.toFixed(1)} dB`,
  },
  pitch: {
    label: 'Pitch',
    key: 'pitch_mean_hz',
    format: (value) => `${Math.round(value)} Hz`,
  },
  pace: {
    label: 'Pace',
    key: 'speech_rate_words_per_minute',
    format: (value) => `${Math.round(value)} wpm`,
  },
};

function signalValue(point, key) {
  const direct = point?.[key];
  const nested = point?.audio_features?.[key];
  const value = Number(direct ?? nested);
  return Number.isFinite(value) ? value : null;
}

function VoiceSignalSection({ context, sentiment, currentTime, onSeek }) {
  const [mode, setMode] = useState('escalation');
  const summary = sentiment?.audio_feature_summary || {};
  const callSummary = sentiment?.call_summary || {};
  const series = sentiment?.dashboard_audio_feature_series || [];
  if (!context && !series.length) return null;

  const metrics = [
    ['Sentiment', callSummary.dominant_sentiment, ''],
    ['Emotion', callSummary.dominant_emotion, ''],
    ['Peak escalation', callSummary.max_escalation_score != null
      ? Math.round(Number(callSummary.max_escalation_score) * 100) : null, '%'],
    ['Pace', summary.average_speech_rate_wpm, ' wpm', 0],
    ['Pauses', summary.average_pause_ratio != null
      ? Number(summary.average_pause_ratio) * 100 : null, '%', 1],
    ['Pitch', summary.average_pitch_hz, ' Hz', 0],
    ['Volume', summary.average_volume_db, ' dB', 1],
    ['Energy', summary.average_energy, '', 3],
  ].filter(([, value]) => value !== null && value !== undefined && value !== '');
  const signal = SIGNAL_MODES[mode];
  const plotted = series
    .map((point) => ({
      ...point,
      start: Number(point.start_time ?? point.start),
      value: signalValue(point, signal.key),
      sentimentClass: threeClassSentiment(point),
    }))
    .filter((point) => Number.isFinite(point.start) && point.value !== null);
  const values = plotted.map((point) => point.value);
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const range = max - min || 1;
  const distribution = callSummary.sentiment_distribution || {};
  const distributionTotal = Object.values(distribution)
    .reduce((total, value) => total + Number(value || 0), 0);

  return (
    <section className="border-b border-slate-300 px-5 py-5 dark:border-slate-700">
      <div className="flex items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 text-xs font-bold uppercase text-slate-950 dark:text-white">
          <AudioLines size={15} aria-hidden="true" />
          Voice &amp; sentiment
        </h2>
        <span className="font-mono text-[10px] text-slate-400">
          {summary.total_segments_with_audio_features || series.length} segments
        </span>
      </div>
      {metrics.length > 0 && (
        <dl className="mt-3 grid grid-cols-2 border-y border-slate-200 dark:border-slate-800">
          {metrics.map(([label, value, unit, digits = null], index) => (
            <div key={label} className={classNames(
              'px-3 py-2.5',
              index % 2 === 0 && 'border-r border-slate-200 dark:border-slate-800',
              index < metrics.length - 2 && 'border-b border-slate-200 dark:border-slate-800',
            )}>
              <dt className="text-[10px] font-semibold uppercase text-slate-400">{label}</dt>
              <dd className="mt-0.5 truncate text-xs font-bold capitalize text-slate-800 dark:text-slate-100">
                {digits === null ? value : Number(value).toFixed(digits)}{unit}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {distributionTotal > 0 && (
        <div className="mt-4">
          <div className="flex h-2 overflow-hidden rounded-sm" aria-label="Sentiment distribution">
            {['Positive', 'Neutral', 'Negative'].map((label) => {
              const value = Number(distribution[label] || 0);
              if (!value) return null;
              return (
                <span
                  key={label}
                  title={`${label}: ${value}`}
                  className={{
                    Positive: 'bg-emerald-500',
                    Neutral: 'bg-slate-400',
                    Negative: 'bg-rose-500',
                  }[label]}
                  style={{ width: `${(value / distributionTotal) * 100}%` }}
                />
              );
            })}
          </div>
          <div className="mt-1.5 flex justify-between text-[10px] font-semibold text-slate-500">
            {['Positive', 'Neutral', 'Negative'].map((label) => (
              <span key={label}>{label} {distribution[label] || 0}</span>
            ))}
          </div>
        </div>
      )}
      {plotted.length > 0 && (
        <div className="mt-4">
          <div className="flex items-center justify-between gap-3">
            <div className="inline-flex border border-slate-200 p-0.5 dark:border-slate-700" aria-label="Voice signal timeline">
              {Object.entries(SIGNAL_MODES).map(([key, option]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setMode(key)}
                  className={classNames(
                    'px-2 py-1 text-[10px] font-bold transition',
                    mode === key
                      ? 'bg-slate-950 text-white dark:bg-white dark:text-slate-950'
                      : 'text-slate-500 hover:text-slate-950 dark:hover:text-white',
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <span className="text-[10px] text-slate-400">Click to play</span>
          </div>
          <div className="mt-2 flex h-16 items-end gap-px border-b border-slate-300 dark:border-slate-700">
            {plotted.map((point, index) => (
              <button
                key={`${point.start}-${index}`}
                type="button"
                onClick={() => onSeek(point.start)}
                title={`${fmt(point.start)} | ${signal.format(point.value)} | ${point.sentimentClass}`}
                aria-label={`${signal.label} at ${fmt(point.start)}: ${signal.format(point.value)}`}
                className={classNames(
                  'min-w-px flex-1 transition-opacity hover:opacity-60',
                  point.sentimentClass === 'positive' && 'bg-emerald-500',
                  point.sentimentClass === 'neutral' && 'bg-slate-400',
                  point.sentimentClass === 'negative' && 'bg-rose-500',
                  currentTime >= point.start
                    && currentTime < Number(plotted[index + 1]?.start ?? point.start + 1)
                    && 'ring-2 ring-[#1557ff] ring-offset-1',
                )}
                style={{ height: `${18 + ((point.value - min) / range) * 82}%` }}
              />
            ))}
          </div>
        </div>
      )}
      {context?.observations?.length > 0 && (
        <div className="mt-3 divide-y divide-slate-200 dark:divide-slate-800">
          {context.observations.map((item) => (
            <button
              key={item.observation_id}
              type="button"
              onClick={() => item.start_seconds != null && onSeek(item.start_seconds)}
              disabled={item.start_seconds == null}
              className="block w-full py-3 text-left disabled:cursor-default"
            >
              <span className="text-xs font-semibold text-violet-700 dark:text-violet-400">
                {item.label}
              </span>
              {usefulSupportingCopy(item.summary) && (
                <span className="mt-1 block text-xs leading-5 text-slate-500">
                  {usefulSupportingCopy(item.summary)}
                </span>
              )}
              {item.start_seconds != null && (
                <span className="mt-1.5 inline-flex items-center gap-1 font-mono text-[11px] text-sky-700 dark:text-sky-400">
                  <Play size={11} fill="currentColor" aria-hidden="true" />
                  {fmt(item.start_seconds)}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function DetailsSection({ view, run }) {
  const details = view.details;
  return (
    <details className="group px-5 py-4">
      <summary className="flex cursor-pointer list-none items-center justify-between text-sm font-medium text-slate-700 dark:text-slate-200">
        Evaluation details
        <ChevronDown
          size={17}
          className="transition group-open:rotate-180"
          aria-hidden="true"
        />
      </summary>
      <div className="mt-4 space-y-4 border-t border-slate-200 pt-4 text-xs text-slate-500 dark:border-slate-800">
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2">
          <dt>Evaluator</dt>
          <dd className="font-mono text-slate-700 dark:text-slate-300">
            {details.evaluator_version}
          </dd>
          <dt>Policy</dt>
          <dd className="font-mono text-slate-700 dark:text-slate-300">
            {details.decision_policy_id} {details.decision_policy_version}
          </dd>
          <dt>Mode</dt>
          <dd className="text-slate-700 dark:text-slate-300">{run.mode}</dd>
        </dl>
        {view.completeness_notice?.affected_requirements?.length > 0 && (
          <div>
            <p className="font-medium text-slate-700 dark:text-slate-300">
              Unassessed requirements
            </p>
            <ul className="mt-2 space-y-1.5">
              {view.completeness_notice.affected_requirements.map((message) => (
                <li key={message}>{message}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </details>
  );
}

function ChapterSidebar({
  chapters,
  currentTime,
  onSeek,
  open,
  onToggle,
}) {
  const activeIndex = useMemo(() => {
    let active = 0;
    for (let index = 0; index < chapters.length; index += 1) {
      if (currentTime >= chapters[index].start) active = index;
      else break;
    }
    return active;
  }, [chapters, currentTime]);

  if (!chapters.length) return null;
  return (
    <nav
      id="chapters-panel"
      aria-label="Call chapters"
      className="order-first w-full min-w-0 max-w-full overflow-hidden border-b border-slate-200 bg-white lg:order-none lg:h-full lg:border-b-0 lg:border-l dark:border-slate-800 dark:bg-slate-950"
    >
      <div className="flex h-12 items-center justify-between border-b border-slate-200 px-3 dark:border-slate-800">
        {open && (
          <span className="text-xs font-bold uppercase text-slate-950 dark:text-white">
            Chapters
          </span>
        )}
        <button
          type="button"
          onClick={onToggle}
          className="ml-auto flex h-8 w-8 items-center justify-center rounded text-slate-500 transition hover:bg-slate-100 hover:text-[#1557ff] dark:hover:bg-slate-900"
          title={open ? 'Collapse chapters' : 'Expand chapters'}
          aria-label={open ? 'Collapse chapters' : 'Expand chapters'}
          aria-expanded={open}
        >
          {open ? <PanelRightClose size={17} /> : <PanelRightOpen size={17} />}
        </button>
      </div>
      {open && (
        <ol className="chapter-list flex gap-1 overflow-x-auto p-2 lg:block lg:h-[calc(100%-3rem)] lg:overflow-y-auto lg:overflow-x-hidden lg:p-0">
          {chapters.map((chapter, index) => {
            const selected = index === activeIndex;
            return (
              <li key={`${chapter.index}-${chapter.start}`} className="min-w-44 lg:min-w-0">
                <button
                  type="button"
                  onClick={() => onSeek(chapter.start)}
                  aria-current={selected ? 'step' : undefined}
                  className={classNames(
                    'relative flex min-h-16 w-full items-start gap-3 border-l-2 px-3 py-3 text-left transition duration-200',
                    selected
                      ? 'border-[#1557ff] bg-blue-50 text-[#1557ff] dark:bg-blue-950/30 dark:text-blue-300'
                      : 'border-transparent text-slate-600 hover:bg-slate-50 hover:text-slate-950 dark:text-slate-300 dark:hover:bg-slate-900 dark:hover:text-white',
                  )}
                >
                  <span className={classNames(
                    'flex h-6 w-6 shrink-0 items-center justify-center rounded-full border font-mono text-[11px] font-bold',
                    selected
                      ? 'border-[#1557ff] bg-[#1557ff] text-white'
                      : 'border-slate-300 text-slate-500 dark:border-slate-700',
                  )}>
                    {index + 1}
                  </span>
                  <span className="min-w-0">
                    <span className="block text-xs font-semibold leading-4">
                      {chapter.label}
                    </span>
                    <span className="mt-1 block font-mono text-[10px] opacity-70">
                      {fmt(chapter.start)}
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </nav>
  );
}

function TimedWords({ words, text, currentTime, agent }) {
  if (!words?.length) return text;
  return words.map((word, wordIndex) => {
    const wordActive = currentTime >= word.start && currentTime < word.end;
    return (
      <span key={`${word.start}-${wordIndex}`}>
        {wordIndex > 0 ? ' ' : ''}
        <span
          data-word-start={word.start}
          data-active={wordActive ? 'true' : 'false'}
          className={classNames(
            'rounded-sm transition-colors duration-150',
            wordActive && (agent
              ? 'bg-blue-200 px-0.5 text-blue-950 dark:bg-blue-700 dark:text-white'
              : 'bg-emerald-200 px-0.5 text-emerald-950 dark:bg-emerald-700 dark:text-white'),
          )}
        >
          {word.word}
        </span>
      </span>
    );
  });
}

function Transcript({
  turns,
  chapters,
  currentTime,
  onSeek,
  activeTurnRef,
  scrollRef,
}) {
  const [chaptersOpen, setChaptersOpen] = useState(true);
  const chapterResize = useResizableWidth({
    initial: 224,
    min: 176,
    max: 300,
    storageKey: 'evaluator-v2:chapters-width',
    invert: true,
  });
  const activeIndex = useMemo(() => {
    let active = -1;
    for (let index = 0; index < turns.length; index += 1) {
      if (currentTime >= turns[index].start) active = index;
      else break;
    }
    return active;
  }, [currentTime, turns]);

  useEffect(() => {
    activeTurnRef.current?.scrollIntoView({
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches
        ? 'auto'
        : 'smooth',
      block: 'nearest',
    });
  }, [activeIndex, activeTurnRef]);

  const activeChapter = useMemo(() => {
    let current = chapters[0];
    for (const chapter of chapters) {
      if (currentTime >= chapter.start) current = chapter;
      else break;
    }
    return current;
  }, [chapters, currentTime]);

  return (
    <section className="min-w-0 bg-white lg:h-full lg:overflow-hidden dark:bg-slate-950">
      <div
        style={{ '--chapters-width': `${chapterResize.width}px` }}
        className={classNames(
          'grid grid-cols-[minmax(0,1fr)] lg:h-full',
          chaptersOpen
            ? 'lg:grid-cols-[minmax(0,1fr)_10px_var(--chapters-width)]'
            : 'lg:grid-cols-[minmax(0,1fr)_3.5rem]',
        )}
      >
        <div id="conversation-panel" className="min-w-0 lg:flex lg:min-h-0 lg:flex-col">
          <div className="flex min-h-12 items-center justify-between border-b border-slate-200 px-5 dark:border-slate-800">
            <div className="min-w-0">
              <h2 className="text-sm font-bold text-slate-950 dark:text-white">
                Conversation
              </h2>
              {activeChapter?.summary && (
                <p className="mt-0.5 truncate text-[11px] text-slate-500">
                  {activeChapter.summary}
                </p>
              )}
            </div>
            <span className="ml-4 shrink-0 font-mono text-[11px] text-slate-500">
              {turns.length} turns
            </span>
          </div>
          <div
            ref={scrollRef}
            className="bg-slate-50 px-4 py-5 sm:px-6 lg:min-h-0 lg:flex-1 lg:overflow-y-auto dark:bg-slate-900"
          >
            <div className="mx-auto max-w-4xl space-y-3">
              {turns.map((turn, index) => {
                const active = index === activeIndex;
                const agent = String(turn.speaker).toUpperCase() === 'AGENT';
                const sentenceSegments = turn.sentimentSegments || [];
                return (
                  <div
                    key={`${turn.start}-${index}`}
                    className={classNames('flex', agent ? 'justify-start' : 'justify-end')}
                  >
                    <div
                      ref={active ? activeTurnRef : null}
                      className={classNames(
                        'turn-bubble group block w-fit max-w-[92%] border px-4 py-3 text-left transition duration-200 sm:max-w-[82%]',
                        agent
                          ? 'rounded-[8px_8px_8px_2px] border-blue-200 bg-blue-50 text-slate-900 dark:border-blue-900 dark:bg-blue-950/35 dark:text-slate-100'
                          : 'rounded-[8px_8px_2px_8px] border-emerald-200 bg-emerald-50 text-slate-900 dark:border-emerald-900 dark:bg-emerald-950/30 dark:text-slate-100',
                        active && (agent
                          ? 'turn-focus-agent border-[#1557ff] ring-1 ring-[#1557ff]'
                          : 'turn-focus-customer border-emerald-600 ring-1 ring-emerald-600'),
                      )}
                    >
                      <span className="flex items-center gap-2">
                        <span className={classNames(
                          'text-xs font-bold',
                          agent ? 'text-[#1557ff] dark:text-blue-300' : 'text-emerald-700 dark:text-emerald-300',
                        )}>
                          {speakerLabel(turn.speaker)}
                        </span>
                        <span className="font-mono text-[10px] text-slate-400">
                          {fmt(turn.start)}
                        </span>
                      </span>
                      {sentenceSegments.length ? (
                        <span className="mt-1.5 block space-y-1">
                          {sentenceSegments.map((segment, segmentIndex) => {
                            const audio = segment.audio_features || {};
                            const sentimentOverlay = {
                              positive: 'bg-emerald-200/70 ring-1 ring-inset ring-emerald-300/80 dark:bg-emerald-500/20 dark:ring-emerald-500/35',
                              neutral: 'bg-slate-200/80 ring-1 ring-inset ring-slate-300/80 dark:bg-slate-500/20 dark:ring-slate-500/35',
                              negative: 'bg-rose-200/75 ring-1 ring-inset ring-rose-300/80 dark:bg-rose-500/20 dark:ring-rose-500/35',
                            }[segment.className];
                            const sentenceWords = (turn.words || []).filter((word) => (
                              Number(word.start) < Number(segment.end) + 0.08
                              && Number(word.end) > Number(segment.start) - 0.08
                            ));
                            const sentimentTitle = [
                              segment.dominant_emotion
                                ? `Emotion: ${segment.dominant_emotion}` : null,
                              segment.escalation_score != null
                                ? `Escalation: ${Math.round(segment.escalation_score * 100)}%` : null,
                              audio.speech_rate_words_per_minute != null
                                ? `Pace: ${Math.round(audio.speech_rate_words_per_minute)} wpm` : null,
                              audio.pause_ratio != null
                                ? `Pauses: ${Math.round(audio.pause_ratio * 100)}%` : null,
                              audio.pitch_mean_hz != null
                                ? `Pitch: ${Math.round(audio.pitch_mean_hz)} Hz` : null,
                              audio.volume_db_mean != null
                                ? `Volume: ${Number(audio.volume_db_mean).toFixed(1)} dB` : null,
                            ].filter(Boolean).join(' | ');
                            return (
                              <button
                                key={`${segment.segment_index ?? segment.start}-${segmentIndex}`}
                                type="button"
                                onClick={() => onSeek(segment.start)}
                                data-sentiment={segment.className || undefined}
                                aria-label={segment.className
                                  ? `${segment.className} sentiment: ${segment.text}`
                                  : `Play sentence: ${segment.text}`}
                                className={classNames(
                                  'block w-full rounded-md px-2.5 py-2 text-left transition hover:brightness-95 dark:hover:brightness-110',
                                  sentimentOverlay,
                                )}
                                title={sentimentTitle || 'Play this sentence'}
                              >
                                <span className="block text-sm leading-6 text-slate-700 dark:text-slate-200">
                                  <TimedWords
                                    words={sentenceWords}
                                    text={segment.text}
                                    currentTime={currentTime}
                                    agent={agent}
                                  />
                                </span>
                              </button>
                            );
                          })}
                        </span>
                      ) : (
                        <button
                          type="button"
                          onClick={() => onSeek(turn.start)}
                          className="mt-1.5 block w-full text-left text-sm leading-6 text-slate-700 dark:text-slate-200"
                        >
                          <TimedWords
                            words={turn.words}
                            text={turn.text}
                            currentTime={currentTime}
                            agent={agent}
                          />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
        {chaptersOpen && (
          <ResizeHandle
            label="Resize chapters panel"
            controls="chapters-panel"
            resize={chapterResize}
          />
        )}
        <ChapterSidebar
          chapters={chapters}
          currentTime={currentTime}
          onSeek={onSeek}
          open={chaptersOpen}
          onToggle={() => setChaptersOpen((value) => !value)}
        />
      </div>
    </section>
  );
}

export default function EvaluatorV2CallDetail() {
  const { callId } = useParams();
  const audioRef = useRef(null);
  const activeTurnRef = useRef(null);
  const scrollRef = useRef(null);
  const [callData, setCallData] = useState(null);
  const [run, setRun] = useState(null);
  const [sentimentData, setSentimentData] = useState(null);
  const [emailNotification, setEmailNotification] = useState(null);
  const [loadState, setLoadState] = useState('loading');
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [feedbackBusy, setFeedbackBusy] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const [emailBusy, setEmailBusy] = useState(false);
  const [emailError, setEmailError] = useState('');
  const analysisMax = useCallback(() => {
    if (window.innerWidth < 1024) return 620;
    return Math.min(620, window.innerWidth - 650);
  }, []);
  const analysisResize = useResizableWidth({
    initial: 500,
    min: 340,
    max: analysisMax,
    storageKey: 'evaluator-v2:analysis-width',
  });

  useEffect(() => {
    let cancelled = false;
    setLoadState('loading');
    Promise.all([
      fetchOptionalJson(`/calls/${encodeURIComponent(callId)}.json`),
      fetchEvaluatorArtifact(
        `/evaluation-v2/${encodeURIComponent(callId)}.json`,
      ),
      fetchEvaluatorArtifact(
        `/word-timings/${encodeURIComponent(callId)}.json`,
      ),
      fetchOptionalJson(`/sentiment/${encodeURIComponent(callId)}.json`),
      fetchOptionalJson(`/calls/${encodeURIComponent(callId)}/email`),
    ]).then(([rawCall, shadow, timingArtifact, sentiment, emailState]) => {
      if (cancelled) return;
      if (!rawCall) {
        setLoadState('error');
        return;
      }
      const timedCall = mergeWordTimings(rawCall, timingArtifact);
      const call = {
        ...timedCall,
        turns: attachSentiment(timedCall.turns, sentiment),
      };
      setCallData(call);
      setSentimentData(sentiment);
      setEmailNotification(emailState?.notifications?.[0] || null);
      setDuration(call.duration || 0);
      const embedded = call.evaluation_v2 || shadow;
      if (
        embedded?.status !== 'succeeded' ||
        !embedded?.presentation
      ) {
        setRun(unavailableRun(call, embedded));
        setLoadState('ready');
        return;
      }
      setRun(embedded);
      setLoadState('ready');
    });
    return () => {
      cancelled = true;
    };
  }, [callId]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return undefined;
    const time = () => setCurrentTime(audio.currentTime || 0);
    const metadata = () =>
      setDuration(audio.duration || callData?.duration || 0);
    const play = () => setIsPlaying(true);
    const pause = () => setIsPlaying(false);
    audio.addEventListener('timeupdate', time);
    audio.addEventListener('loadedmetadata', metadata);
    audio.addEventListener('play', play);
    audio.addEventListener('pause', pause);
    return () => {
      audio.removeEventListener('timeupdate', time);
      audio.removeEventListener('loadedmetadata', metadata);
      audio.removeEventListener('play', play);
      audio.removeEventListener('pause', pause);
    };
  }, [callData]);

  const seek = useCallback(
    (seconds) => {
      const audio = audioRef.current;
      const next = Math.max(0, Math.min(seconds, duration || seconds));
      if (audio) audio.currentTime = next;
      setCurrentTime(next);
    },
    [duration],
  );

  const toggleAudio = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) audio.play();
    else audio.pause();
  }, []);

  const submitFeedback = useCallback(
    async (feedbackType, target = {}) => {
      if (!run?.decision_sha256) return;
      setFeedbackBusy(true);
      setFeedbackStatus('');
      const payload = {
        feedback_type: feedbackType,
        decision_sha256: run.decision_sha256,
        ...target,
      };
      try {
        const response = await fetch(
          apiUrl(`/calls/${encodeURIComponent(callId)}/feedback`),
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          },
        );
        if (!response.ok) throw new Error('Feedback API unavailable');
        const savedFeedback = await response.json();
        if (!savedFeedback.feedback_id) {
          throw new Error('Feedback API returned an invalid response');
        }
        setFeedbackStatus('Feedback recorded.');
      } catch {
        if (import.meta.env.DEV) {
          const key = `evaluation-feedback:${callId}`;
          const saved = JSON.parse(localStorage.getItem(key) || '[]');
          saved.push({ ...payload, created_at: new Date().toISOString() });
          localStorage.setItem(key, JSON.stringify(saved));
          setFeedbackStatus('Feedback saved in this local preview.');
        } else {
          setFeedbackStatus('Feedback could not be recorded.');
        }
      } finally {
        setFeedbackBusy(false);
      }
    },
    [callId, run],
  );

  const sendEmail = useCallback(async () => {
    if (!run?.decision_sha256) return;
    setEmailBusy(true);
    setEmailError('');
    try {
      const response = await fetch(
        apiUrl(`/calls/${encodeURIComponent(callId)}/email`),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            decision_sha256: run.decision_sha256,
            approve: true,
          }),
        },
      );
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'Email could not be sent');
      setEmailNotification(result);
    } catch (error) {
      setEmailError(error.message || 'Email could not be sent.');
    } finally {
      setEmailBusy(false);
    }
  }, [callId, run]);

  if (loadState === 'loading') {
    return (
      <div className="flex min-h-screen items-center justify-center bg-white text-sm text-slate-500 dark:bg-slate-950">
        Loading evaluation...
      </div>
    );
  }

  if (loadState === 'error' || !callData || !run) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-3 bg-white text-slate-900 dark:bg-slate-950 dark:text-white">
        <FileQuestion size={24} className="text-slate-400" />
        <p className="text-sm">This call could not be loaded.</p>
        <Link to="/" className="text-sm text-sky-700 hover:underline">
          Return to calls
        </Link>
      </div>
    );
  }

  const view = run.presentation;
  const evidenceById = new Map(
    [...view.evidence, ...(view.details.evidence || [])]
      .map((item) => [item.evidence_id, item]),
  );
  const incorrect = view.primary_reasons
    .filter((finding) => finding.outcome === 'incorrect');
  const missed = view.primary_reasons
    .filter((finding) => finding.outcome === 'missed');
  const concerns = view.primary_reasons
    .filter((finding) => !['incorrect', 'missed'].includes(finding.outcome));

  return (
    <div className="min-h-screen bg-white text-slate-950 dark:bg-slate-950 dark:text-slate-100">
      <header className="border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950">
        <div className="flex min-h-[68px] w-full items-center justify-between gap-4 px-4 py-3 sm:px-5">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              to="/"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded bg-[#1557ff] text-white transition duration-200 hover:bg-blue-700 active:scale-[0.97]"
              title="All calls"
              aria-label="All calls"
            >
              <ArrowLeft size={18} />
            </Link>
            <div className="min-w-0">
              <h1 className="truncate text-base font-bold sm:text-lg">{callData.call}</h1>
              <p className="mt-0.5 flex items-center gap-2 text-xs text-slate-500">
                <span>{fmt(callData.duration || duration)}</span>
                <span aria-hidden="true">&bull;</span>
                <span>Call evaluation</span>
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span
              className="hidden items-center gap-1.5 text-xs text-slate-500 sm:inline-flex"
              title="Evaluator v2 is running alongside the current evaluator"
            >
              <CircleHelp size={14} aria-hidden="true" />
              Evaluator v2
            </span>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <AudioPlayer
        callId={callId}
        audioRef={audioRef}
        isPlaying={isPlaying}
        currentTime={currentTime}
        duration={duration}
        onToggle={toggleAudio}
        onSeek={seek}
      />

      <main
        style={{ '--analysis-width': `${analysisResize.width}px` }}
        className="grid w-full grid-cols-1 lg:h-[calc(100vh-9.25rem)] lg:grid-cols-[var(--analysis-width)_10px_minmax(0,1fr)] lg:overflow-hidden"
      >
        <aside
          id="analysis-panel"
          className="border-b border-slate-300 bg-white lg:overflow-y-auto lg:border-b-0 lg:border-r dark:border-slate-700 dark:bg-slate-950"
        >
          <StatusSection view={view} />
          <ManagerQuestions
            questions={view.manager_questions || []}
            evidenceById={evidenceById}
            onSeek={seek}
          />
          <VoiceSignalSection
            context={view.acoustic_context}
            sentiment={sentimentData}
            currentTime={currentTime}
            onSeek={seek}
          />
          <FindingSection
            title="Incorrect handling"
            tone="incorrect"
            findings={incorrect}
            evidenceById={evidenceById}
            onSeek={seek}
            onFeedback={submitFeedback}
            feedbackBusy={feedbackBusy}
          />
          <FindingSection
            title="Missing steps"
            tone="missed"
            findings={missed}
            evidenceById={evidenceById}
            onSeek={seek}
            onFeedback={submitFeedback}
            feedbackBusy={feedbackBusy}
          />
          <FindingSection
            title="Observed concerns"
            tone="concern"
            findings={concerns}
            evidenceById={evidenceById}
            onSeek={seek}
            onFeedback={submitFeedback}
            feedbackBusy={feedbackBusy}
          />
          <PositiveSection
            findings={view.positive_highlights}
            evidenceById={evidenceById}
            onSeek={seek}
          />
          <ChecklistSection
            checks={view.checklist || []}
            evidenceById={evidenceById}
            onSeek={seek}
          />
          {feedbackStatus && (
            <p
              className="border-b border-slate-200 px-5 py-3 text-xs text-slate-500 dark:border-slate-800"
              role="status"
              aria-live="polite"
            >
              {feedbackStatus}
            </p>
          )}
          <DetailsSection view={view} run={run} />
        </aside>
        <ResizeHandle
          label="Resize analysis panel"
          controls="analysis-panel"
          resize={analysisResize}
        />
        <Transcript
          turns={callData.turns || []}
          chapters={callData.chapters || []}
          currentTime={currentTime}
          onSeek={seek}
          activeTurnRef={activeTurnRef}
          scrollRef={scrollRef}
        />
      </main>
      <EmailActionPopover
        action={view.recommended_action}
        notification={emailNotification}
        busy={emailBusy}
        error={emailError}
        onSend={sendEmail}
      />
    </div>
  );
}
