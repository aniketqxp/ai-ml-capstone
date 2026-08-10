import {
  AlertTriangle,
  AudioLines,
  ArrowLeft,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  FileQuestion,
  ListChecks,
  Mail,
  Minus,
  Pause,
  PanelRightClose,
  PanelRightOpen,
  Play,
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
      <div className="flex items-start gap-3">
        <Icon
          size={22}
          className={classNames('mt-0.5 shrink-0', config.iconTone)}
          aria-hidden="true"
        />
        <div className="min-w-0">
          <p className="text-sm font-bold">{config.label}</p>
          <p className="mt-1 max-w-2xl text-xs leading-5 opacity-85">
            {view.summary}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] font-semibold opacity-80">
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
              <p className="mt-1 text-xs leading-5 text-slate-500">
                {item.summary}
              </p>
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

function ActionSection({
  action,
}) {
  if (!action) return null;
  return (
    <section className="border-b border-slate-200 px-5 py-5 dark:border-slate-800">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase text-slate-500">
        <Mail size={15} aria-hidden="true" />
        Email action
      </div>
      <div className="mt-3 flex items-start gap-3">
        <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-sky-50 text-sky-700 dark:bg-sky-950/50 dark:text-sky-400">
          <Mail size={16} aria-hidden="true" />
        </span>
        <div>
          <p className="text-sm font-semibold text-slate-950 dark:text-white">
            {action.label}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {action.execution_label}. Recipient: {action.audience}.
          </p>
        </div>
      </div>
    </section>
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
                <p className="mt-0.5 text-xs leading-5 text-slate-500">
                  {finding.summary}
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
                  {item.status !== 'demonstrated' && !item.promoted && (
                    <p className="mt-1 text-xs leading-5 text-slate-500">
                      {item.summary}
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

function AcousticSection({ context, onSeek }) {
  if (!context) return null;
  return (
    <section className="border-b border-slate-200 px-5 py-5 dark:border-slate-800">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase text-slate-500">
          <AudioLines size={15} aria-hidden="true" />
          Acoustic support
        </div>
        <span className="text-[11px] text-slate-400">
          {context.coverage_label}
        </span>
      </div>
      <p className="mt-3 text-sm leading-6 text-slate-600 dark:text-slate-300">
        {context.conclusion}
      </p>
      {context.observations.length > 0 && (
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
              <span className="mt-1 block text-xs leading-5 text-slate-500">
                {item.summary}
              </span>
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
      aria-label="Call chapters"
      className={classNames(
        'order-first w-full min-w-0 max-w-full overflow-hidden border-b border-slate-200 bg-white transition-[width] duration-200 lg:order-none lg:h-full lg:border-b-0 lg:border-l dark:border-slate-800 dark:bg-slate-950',
        open ? 'lg:w-56' : 'lg:w-14',
      )}
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

function Transcript({
  turns,
  chapters,
  currentTime,
  onSeek,
  activeTurnRef,
  scrollRef,
}) {
  const [chaptersOpen, setChaptersOpen] = useState(true);
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
      <div className={classNames(
        'grid grid-cols-[minmax(0,1fr)] lg:h-full',
        chaptersOpen
          ? 'lg:grid-cols-[minmax(0,1fr)_14rem]'
          : 'lg:grid-cols-[minmax(0,1fr)_3.5rem]',
      )}>
        <div className="min-w-0 lg:flex lg:min-h-0 lg:flex-col">
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
                return (
                  <div
                    key={`${turn.start}-${index}`}
                    className={classNames('flex', agent ? 'justify-start' : 'justify-end')}
                  >
                    <button
                      ref={active ? activeTurnRef : null}
                      type="button"
                      onClick={() => onSeek(turn.start)}
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
                      <span className="mt-1.5 block text-sm leading-6 text-slate-700 dark:text-slate-200">
                        {turn.words?.length
                          ? turn.words.map((word, wordIndex) => {
                              const wordActive = (
                                currentTime >= word.start
                                && currentTime < word.end
                              );
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
                            })
                          : turn.text}
                      </span>
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
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
      fetchEvaluatorArtifact(
        `/evaluation-v2/${encodeURIComponent(callId)}.json`,
      ),
      fetchEvaluatorArtifact(
        `/word-timings/${encodeURIComponent(callId)}.json`,
      ),
    ]).then(([rawCall, shadow, timingArtifact]) => {
      if (cancelled) return;
      if (!rawCall) {
        setLoadState('error');
        return;
      }
      const call = mergeWordTimings(rawCall, timingArtifact);
      setCallData(call);
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

      <main className="grid w-full grid-cols-1 lg:h-[calc(100vh-9.25rem)] lg:grid-cols-[minmax(360px,36%)_minmax(0,1fr)] lg:overflow-hidden">
        <aside className="border-b border-slate-300 bg-white lg:overflow-y-auto lg:border-b-0 lg:border-r dark:border-slate-700 dark:bg-slate-950">
          <StatusSection view={view} />
          <ManagerQuestions
            questions={view.manager_questions || []}
            evidenceById={evidenceById}
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
          <ActionSection
            action={view.recommended_action}
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
          <AcousticSection
            context={view.acoustic_context}
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
        <Transcript
          turns={callData.turns || []}
          chapters={callData.chapters || []}
          currentTime={currentTime}
          onSeek={seek}
          activeTurnRef={activeTurnRef}
          scrollRef={scrollRef}
        />
      </main>
    </div>
  );
}
