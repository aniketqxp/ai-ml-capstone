import {
  AlertTriangle,
  ArrowLeft,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  FileQuestion,
  Pause,
  Play,
  ShieldCheck,
  ThumbsDown,
  ThumbsUp,
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

import { apiUrl } from '../api';
import { ThemeToggle } from '../components/ThemeToggle';
import { fmt } from '../lib/format';
import LegacyCallDetail from './LegacyCallDetail';

const STATE_COPY = {
  needs_attention: {
    label: 'Needs attention',
    icon: AlertTriangle,
    tone: 'border-red-300 bg-red-50 text-red-900 dark:border-red-900 dark:bg-red-950/40 dark:text-red-100',
    iconTone: 'text-red-600 dark:text-red-400',
  },
  no_attention_finding: {
    label: 'No review finding identified',
    icon: ShieldCheck,
    tone: 'border-emerald-300 bg-emerald-50 text-emerald-950 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-100',
    iconTone: 'text-emerald-600 dark:text-emerald-400',
  },
  evaluation_incomplete: {
    label: 'Evaluation incomplete',
    icon: FileQuestion,
    tone: 'border-amber-300 bg-amber-50 text-amber-950 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-100',
    iconTone: 'text-amber-600 dark:text-amber-400',
  },
};

function classNames(...values) {
  return values.filter(Boolean).join(' ');
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

function speakerLabel(speaker) {
  if (String(speaker).toUpperCase() === 'AGENT') return 'Agent';
  if (String(speaker).toUpperCase() === 'CUSTOMER') return 'Customer';
  return 'Unknown';
}

function evidenceText(item) {
  return item.text || 'Evidence is linked to the source call.';
}

function FeedbackButton({
  icon: Icon,
  label,
  onClick,
  disabled,
  tone = 'neutral',
}) {
  const tones = {
    neutral:
      'border-slate-300 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:hover:bg-slate-900',
    approve:
      'border-emerald-700 bg-emerald-700 text-white hover:bg-emerald-600 dark:border-emerald-500 dark:bg-emerald-600 dark:hover:bg-emerald-500',
    dismiss:
      'border-slate-300 bg-white text-slate-600 hover:border-red-300 hover:text-red-700 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-300 dark:hover:border-red-800 dark:hover:text-red-300',
  };
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={classNames(
        'inline-flex h-9 items-center justify-center gap-2 rounded-md border px-3 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50',
        tones[tone],
      )}
    >
      <Icon size={16} aria-hidden="true" />
      {label}
    </button>
  );
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
  const progress = duration ? (currentTime / duration) * 100 : 0;
  return (
    <section
      aria-label="Call audio"
      className="border-y border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950"
    >
      <div className="mx-auto flex w-full max-w-7xl items-center gap-4 px-4 py-3 sm:px-6">
        <button
          type="button"
          onClick={onToggle}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-slate-950 text-white transition hover:bg-slate-700 dark:bg-white dark:text-slate-950 dark:hover:bg-slate-200"
          title={isPlaying ? 'Pause call' : 'Play call'}
          aria-label={isPlaying ? 'Pause call' : 'Play call'}
        >
          {isPlaying ? (
            <Pause size={18} fill="currentColor" />
          ) : (
            <Play size={18} fill="currentColor" className="ml-0.5" />
          )}
        </button>
        <Volume2
          size={17}
          className="hidden shrink-0 text-slate-400 sm:block"
          aria-hidden="true"
        />
        <div className="min-w-0 flex-1">
          <input
            type="range"
            min="0"
            max={Math.max(duration, 0)}
            step="0.1"
            value={Math.min(currentTime, duration || 0)}
            onChange={(event) => onSeek(Number(event.target.value))}
            className="w-full accent-slate-900 dark:accent-white"
            aria-label="Seek through call"
          />
          <div className="mt-0.5 flex justify-between font-mono text-xs text-slate-500">
            <span>{fmt(currentTime)}</span>
            <span>{fmt(duration)}</span>
          </div>
        </div>
        <div
          className="hidden h-1.5 w-20 overflow-hidden rounded-full bg-slate-200 md:block dark:bg-slate-800"
          aria-hidden="true"
        >
          <div
            className="h-full bg-slate-800 dark:bg-slate-200"
            style={{ width: `${progress}%` }}
          />
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
    <section className={classNames('border-b px-5 py-5', config.tone)}>
      <div className="flex items-start gap-3">
        <Icon
          size={22}
          className={classNames('mt-0.5 shrink-0', config.iconTone)}
          aria-hidden="true"
        />
        <div className="min-w-0">
          <p className="text-base font-semibold">{config.label}</p>
          <p className="mt-1 max-w-2xl text-sm leading-6 opacity-80">
            {view.summary}
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs font-medium opacity-75">
            <span className="inline-flex items-center gap-1.5">
              <Clock3 size={14} aria-hidden="true" />
              {view.evaluation_status.replaceAll('_', ' ')}
            </span>
            <span>{view.details.domain_profile_id}</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function FindingRows({
  findings,
  evidenceById,
  onSeek,
  onFeedback,
  feedbackBusy,
}) {
  if (!findings.length) return null;
  return (
    <section className="border-b border-slate-200 px-5 py-5 dark:border-slate-800">
      <h2 className="text-xs font-semibold uppercase text-slate-500">
        Why it needs attention
      </h2>
      <div className="mt-3 divide-y divide-slate-200 dark:divide-slate-800">
        {findings.map((finding) => {
          const firstEvidence = evidenceById.get(finding.evidence_ids[0]);
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
                  <p className="mt-1 text-sm leading-6 text-slate-600 dark:text-slate-300">
                    {finding.summary}
                  </p>
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
              {firstEvidence?.seekable && (
                <button
                  type="button"
                  onClick={() => onSeek(firstEvidence.start_seconds)}
                  className="mt-3 inline-flex items-center gap-1.5 text-xs font-semibold text-sky-700 hover:underline dark:text-sky-400"
                >
                  <Play size={13} fill="currentColor" aria-hidden="true" />
                  Hear evidence at {fmt(firstEvidence.start_seconds)}
                </button>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

function ActionSection({
  action,
  onFeedback,
  feedbackBusy,
  feedbackStatus,
}) {
  if (!action) return null;
  const approval = action.requires_human_approval;
  return (
    <section className="border-b border-slate-200 px-5 py-5 dark:border-slate-800">
      <p className="text-xs font-semibold uppercase text-slate-500">
        Next action
      </p>
      <div className="mt-3 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <p className="text-sm font-semibold text-slate-950 dark:text-white">
            {action.label}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {action.execution_label}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {approval ? (
            <FeedbackButton
              icon={Check}
              label="Approve action"
              tone="approve"
              disabled={feedbackBusy}
              onClick={() =>
                onFeedback('approve_action', {
                  action_type: action.action_type,
                })
              }
            />
          ) : (
            <FeedbackButton
              icon={ThumbsUp}
              label="Mark action complete"
              tone="approve"
              disabled={feedbackBusy}
              onClick={() =>
                onFeedback('action_completed', {
                  action_type: action.action_type,
                })
              }
            />
          )}
          <FeedbackButton
            icon={ThumbsDown}
            label="Dismiss result"
            tone="dismiss"
            disabled={feedbackBusy}
            onClick={() => onFeedback('dismiss_decision')}
          />
        </div>
      </div>
      {feedbackStatus && (
        <p
          className="mt-3 text-xs text-slate-500"
          role="status"
          aria-live="polite"
        >
          {feedbackStatus}
        </p>
      )}
    </section>
  );
}

function PositiveSection({ findings }) {
  if (!findings.length) return null;
  return (
    <section className="border-b border-slate-200 px-5 py-5 dark:border-slate-800">
      <h2 className="text-xs font-semibold uppercase text-slate-500">
        Handled well
      </h2>
      <ul className="mt-3 space-y-3">
        {findings.map((finding) => (
          <li key={finding.finding_id} className="flex items-start gap-2.5">
            <CheckCircle2
              size={17}
              className="mt-0.5 shrink-0 text-emerald-600 dark:text-emerald-400"
              aria-hidden="true"
            />
            <div>
              <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
                {finding.title}
              </p>
              <p className="mt-0.5 text-xs leading-5 text-slate-500">
                {finding.summary}
              </p>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function EvidenceSection({ evidence, findingById, onSeek }) {
  if (!evidence.length) return null;
  return (
    <section className="border-b border-slate-200 px-5 py-5 dark:border-slate-800">
      <h2 className="text-xs font-semibold uppercase text-slate-500">
        Key evidence
      </h2>
      <div className="relative mt-4 space-y-5 before:absolute before:bottom-2 before:left-[7px] before:top-2 before:w-px before:bg-slate-200 dark:before:bg-slate-800">
        {evidence.map((item) => {
          const finding = findingById.get(item.finding_ids[0]);
          return (
            <button
              key={item.evidence_id}
              type="button"
              onClick={() => item.seekable && onSeek(item.start_seconds)}
              disabled={!item.seekable}
              className="relative flex w-full items-start gap-3 text-left disabled:cursor-default"
            >
              <span
                className={classNames(
                  'relative z-10 mt-1 h-[15px] w-[15px] shrink-0 rounded-full border-2 bg-white dark:bg-slate-950',
                  item.kind === 'audio_support'
                    ? 'border-violet-500'
                    : 'border-sky-600',
                )}
              />
              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="text-xs font-semibold text-slate-700 dark:text-slate-200">
                    {finding?.title || 'Supporting evidence'}
                  </span>
                  {item.start_seconds != null && (
                    <span className="font-mono text-[11px] text-sky-700 dark:text-sky-400">
                      {fmt(item.start_seconds)}
                    </span>
                  )}
                  {item.kind === 'audio_support' && (
                    <span className="text-[11px] text-violet-700 dark:text-violet-400">
                      Audio support
                    </span>
                  )}
                </span>
                <span className="mt-1 block text-sm leading-6 text-slate-600 dark:text-slate-300">
                  {item.speaker && (
                    <strong className="mr-1 font-medium text-slate-800 dark:text-slate-100">
                      {speakerLabel(item.speaker)}:
                    </strong>
                  )}
                  {evidenceText(item)}
                </span>
              </span>
            </button>
          );
        })}
      </div>
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

function Transcript({
  turns,
  currentTime,
  onSeek,
  activeTurnRef,
  scrollRef,
}) {
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
      behavior: 'smooth',
      block: 'nearest',
    });
  }, [activeIndex, activeTurnRef]);

  return (
    <section className="min-w-0 bg-slate-50 dark:bg-slate-900">
      <div className="flex h-14 items-center justify-between border-b border-slate-200 px-5 dark:border-slate-800">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
          Conversation
        </h2>
        <span className="text-xs text-slate-500">{turns.length} turns</span>
      </div>
      <div
        ref={scrollRef}
        className="max-h-[calc(100vh-13rem)] overflow-y-auto px-4 py-5 sm:px-6"
      >
        <div className="mx-auto max-w-3xl space-y-5">
          {turns.map((turn, index) => {
            const active = index === activeIndex;
            const agent = String(turn.speaker).toUpperCase() === 'AGENT';
            return (
              <button
                key={`${turn.start}-${index}`}
                ref={active ? activeTurnRef : null}
                type="button"
                onClick={() => onSeek(turn.start)}
                className={classNames(
                  'group block w-full border-l-2 py-1 pl-4 text-left transition',
                  active
                    ? agent
                      ? 'border-sky-600'
                      : 'border-emerald-600'
                    : 'border-transparent hover:border-slate-300 dark:hover:border-slate-700',
                )}
              >
                <span className="flex items-center gap-2">
                  <span
                    className={classNames(
                      'text-xs font-semibold',
                      agent
                        ? 'text-sky-700 dark:text-sky-400'
                        : 'text-emerald-700 dark:text-emerald-400',
                    )}
                  >
                    {speakerLabel(turn.speaker)}
                  </span>
                  <span className="font-mono text-[11px] text-slate-400">
                    {fmt(turn.start)}
                  </span>
                </span>
                <span
                  className={classNames(
                    'mt-1.5 block text-sm leading-6',
                    active
                      ? 'text-slate-950 dark:text-white'
                      : 'text-slate-600 group-hover:text-slate-900 dark:text-slate-300 dark:group-hover:text-white',
                  )}
                >
                  {turn.text}
                </span>
              </button>
            );
          })}
        </div>
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
  const [loadState, setLoadState] = useState('loading');
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [feedbackBusy, setFeedbackBusy] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState('');

  useEffect(() => {
    let cancelled = false;
    setLoadState('loading');
    Promise.all([
      fetchOptionalJson(`/calls/${encodeURIComponent(callId)}.json`),
      fetchOptionalJson(
        `/evaluation-v2/${encodeURIComponent(callId)}.json`,
      ),
    ]).then(([call, shadow]) => {
      if (cancelled) return;
      if (!call) {
        setLoadState('error');
        return;
      }
      setCallData(call);
      setDuration(call.duration || 0);
      const embedded = call.evaluation_v2 || shadow;
      if (
        embedded?.status !== 'succeeded' ||
        !embedded?.presentation
      ) {
        setLoadState('legacy');
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

  if (loadState === 'legacy') {
    return <LegacyCallDetail />;
  }

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
  const allFindings = [
    ...view.primary_reasons,
    ...view.positive_highlights,
    ...view.details.additional_findings,
    ...view.details.additional_positive_findings,
  ];
  const findingById = new Map(
    allFindings.map((finding) => [finding.finding_id, finding]),
  );
  const evidenceById = new Map(
    view.evidence.map((item) => [item.evidence_id, item]),
  );

  return (
    <div className="min-h-screen bg-white text-slate-950 dark:bg-slate-950 dark:text-slate-100">
      <header className="bg-white dark:bg-slate-950">
        <div className="mx-auto flex min-h-16 w-full max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              to="/"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-slate-500 transition hover:bg-slate-100 hover:text-slate-950 dark:hover:bg-slate-900 dark:hover:text-white"
              title="All calls"
              aria-label="All calls"
            >
              <ArrowLeft size={18} />
            </Link>
            <div className="min-w-0">
              <h1 className="truncate text-sm font-semibold">{callData.call}</h1>
              <p className="mt-0.5 flex items-center gap-2 text-xs text-slate-500">
                <span>{fmt(callData.duration || duration)}</span>
                <span aria-hidden="true">/</span>
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
              Shadow run
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

      <main className="mx-auto grid w-full max-w-7xl grid-cols-1 lg:grid-cols-[minmax(320px,420px)_minmax(0,1fr)]">
        <aside className="border-b border-slate-200 bg-white lg:border-b-0 lg:border-r dark:border-slate-800 dark:bg-slate-950">
          <StatusSection view={view} />
          <FindingRows
            findings={view.primary_reasons}
            evidenceById={evidenceById}
            onSeek={seek}
            onFeedback={submitFeedback}
            feedbackBusy={feedbackBusy}
          />
          <ActionSection
            action={view.recommended_action}
            onFeedback={submitFeedback}
            feedbackBusy={feedbackBusy}
            feedbackStatus={feedbackStatus}
          />
          <PositiveSection findings={view.positive_highlights} />
          <EvidenceSection
            evidence={view.evidence}
            findingById={findingById}
            onSeek={seek}
          />
          <DetailsSection view={view} run={run} />
        </aside>
        <Transcript
          turns={callData.turns || []}
          currentTime={currentTime}
          onSeek={seek}
          activeTurnRef={activeTurnRef}
          scrollRef={scrollRef}
        />
      </main>
    </div>
  );
}
