import {
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
} from 'react';
import { useParams, Link } from 'react-router-dom';
import { fmt } from '../lib/format';
import { CHAPTER_COLORS } from '../lib/constants';
import { BufferedRanges } from '../components/BufferedRanges';
import { CompliancePanel } from '../components/CompliancePanel';
import { AgentLanes } from '../components/AgentLanes';
import { ThemeToggle } from '../components/ThemeToggle';
import { apiUrl } from '../api';
import AISupervisorPanel from '../components/AISupervisorPanel';
import { SentimentInsightsPanel } from '../components/SentimentInsightsPanel';
import { AutomationDecisionsPanel } from '../components/AutomationDecisionsPanel';

async function fetchJsonOptional(url) {
  try {
    const response = await fetch(url);

    if (!response.ok) {
      return null;
    }

    return await response.json();
  } catch {
    return null;
  }
}

function normalizeSpeaker(speaker) {
  const value = String(speaker || '').toUpperCase();

  if (
    value === 'AGENT' ||
    value === 'REPRESENTATIVE' ||
    value === 'ADVISOR'
  ) {
    return 'AGENT';
  }

  if (
    value === 'CUSTOMER' ||
    value === 'CLIENT' ||
    value === 'CALLER'
  ) {
    return 'CUSTOMER';
  }

  return value || 'UNKNOWN';
}

function getSegmentId(segment) {
  if (segment?.seq_id !== undefined && segment?.seq_id !== null) {
    return String(segment.seq_id);
  }

  if (
    segment?.segment_index !== undefined &&
    segment?.segment_index !== null
  ) {
    return `segment-${segment.segment_index}`;
  }

  if (segment?.sentence_id !== undefined && segment?.sentence_id !== null) {
    return String(segment.sentence_id);
  }

  return null;
}

function getSentimentLabel(segment) {
  return (
    segment?.sentiment ||
    segment?.overall_audio_sentiment ||
    segment?.predicted_sentiment ||
    segment?.business_sentiment ||
    segment?.sentiment_label ||
    segment?.label ||
    null
  );
}

function getEmotionLabel(segment) {
  return (
    segment?.dominant_emotion ||
    segment?.predicted_emotion ||
    segment?.emotion ||
    segment?.emotion_label ||
    null
  );
}

function getSentimentClass(segment) {
  const label = String(getSentimentLabel(segment) || '').toLowerCase();

  if (
    label.includes('positive') ||
    label.includes('happy') ||
    label.includes('calm')
  ) {
    return 'bg-emerald-500/20 border-b border-emerald-600/40';
  }

  if (
    label.includes('negative') ||
    label.includes('anger') ||
    label.includes('angry') ||
    label.includes('sad') ||
    label.includes('fear') ||
    label.includes('frustrat')
  ) {
    return 'bg-red-500/25 border-b border-red-600/40';
  }

  if (label.includes('mixed')) {
    return 'bg-amber-500/20 border-b border-amber-600/40';
  }

  return '';
}



function findSentimentForSentence(sentence, sentimentMap) {
  const possibleKeys = [
    sentence?.seq_id,
    sentence?.sentence_id,
    sentence?.segment_id,
    sentence?.segment_index !== undefined
      ? `segment-${sentence.segment_index}`
      : null,
  ]
    .filter((value) => value !== undefined && value !== null)
    .map(String);

  for (const key of possibleKeys) {
    const match = sentimentMap.get(key);

    if (match) {
      return match;
    }
  }

  return null;
}

function calculateDuration(callData, sentenceData) {
  if (
    typeof callData?.duration === 'number' &&
    Number.isFinite(callData.duration)
  ) {
    return callData.duration;
  }

  if (
    typeof callData?.duration_seconds === 'number' &&
    Number.isFinite(callData.duration_seconds)
  ) {
    return callData.duration_seconds;
  }

  const endTimes = [];

  for (const turn of callData?.turns || []) {
    if (typeof turn?.end === 'number') {
      endTimes.push(turn.end);
    }
  }

  for (const sentence of sentenceData || []) {
    if (typeof sentence?.end === 'number') {
      endTimes.push(sentence.end);
    }
  }

  for (const word of callData?.agent || []) {
    if (typeof word?.end === 'number') {
      endTimes.push(word.end);
    }
  }

  for (const word of callData?.customer || []) {
    if (typeof word?.end === 'number') {
      endTimes.push(word.end);
    }
  }

  return endTimes.length > 0 ? Math.max(...endTimes) : 0;
}

function extractEvaluation(callData, evaluationData) {
  return (
    callData?.evaluation ||
    callData?.llm_evaluation ||
    callData?.analysis?.evaluation ||
    callData?.analysis ||
    evaluationData?.evaluation ||
    evaluationData?.llm_evaluation ||
    evaluationData ||
    null
  );
}

function extractChapters(callData, evaluation) {
  if (Array.isArray(callData?.chapters)) {
    return callData.chapters;
  }

  if (Array.isArray(evaluation?.chapters)) {
    return evaluation.chapters;
  }

  if (Array.isArray(evaluation?.conversation_chapters)) {
    return evaluation.conversation_chapters;
  }

  return [];
}

function getAudioPath(callData, callId) {
  // The seeded comparison call predates runtime artifact storage and ships as
  // the bundled demo recording.
  if (callId === 'en_CA_Banking_1586889') {
    return '/call_1.mp3';
  }

  const value =
    callData?.audio_path ||
    callData?.audio_url ||
    callData?.artifacts?.audio ||
    callData?.artifact_keys?.audio ||
    (callId ? `${callId}.mp3` : null);

  if (!value) {
    return null;
  }

  if (String(value).startsWith('http')) {
    return value;
  }

  const filename = String(value).split('/').pop();

  if (!filename) {
    return null;
  }

  return apiUrl(`/audio/${filename}`);
}

// The key forces a full state reset when navigating directly between calls.
export default function CallDetail() {
  const { callId } = useParams();

  return <CallDetailInner key={callId} callId={callId} />;
}

function CallDetailInner({ callId }) {
  const audioRef = useRef(null);
  const scrollRef = useRef(null);
  const activeTurnRef = useRef(null);
  const scrubberRef = useRef(null);

  const [callData, setCallData] = useState(null);
  const [evaluationData, setEvaluationData] = useState(null);
  const [sentimentData, setSentimentData] = useState(null);
  const [sentences, setSentences] = useState([]);
  const [sentimentMap, setSentimentMap] = useState(new Map());

  const [loadError, setLoadError] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [dragTime, setDragTime] = useState(0);
  const [audioFailed, setAudioFailed] = useState(false);

  // Load transcript, sentence segments, sentiment and LLM evaluation.
  useEffect(() => {
    let cancelled = false;

    setCallData(null);
    setEvaluationData(null);
    setSentimentData(null);
    setSentences([]);
    setSentimentMap(new Map());
    setLoadError(null);
    setCurrentTime(0);
    setDuration(0);
    setAudioFailed(false);

    const loadData = async () => {
      try {
        const callResponse = await fetch(
          apiUrl(`/calls/${callId}.json`),
        );

        if (!callResponse.ok) {
          throw new Error(`call "${callId}" not found`);
        }

        const loadedCallData = await callResponse.json();

        const [
          loadedSentenceData,
          loadedSentimentData,
          loadedEvaluationData,
        ] = await Promise.all([
          fetchJsonOptional(
            apiUrl(`/sentence_segments/${callId}.json`),
          ),
          fetchJsonOptional(
            apiUrl(`/sentiment/${callId}.json`),
          ),
          fetchJsonOptional(
            apiUrl(`/evaluation/${callId}.json`),
          ),
        ]);

        if (cancelled) {
          return;
        }

        const loadedSentences =
          loadedSentenceData?.sentences ||
          loadedSentenceData?.segments ||
          [];

        const loadedSentimentSegments =
          loadedSentimentData?.segments ||
          loadedSentimentData?.sentence_sentiments ||
          loadedSentimentData?.results ||
          [];

        const nextSentimentMap = new Map();

        loadedSentimentSegments.forEach((segment, index) => {
          const primaryId = getSegmentId(segment);

          if (primaryId !== null) {
            nextSentimentMap.set(primaryId, segment);
          }

          if (
            segment?.seq_id !== undefined &&
            segment?.seq_id !== null
          ) {
            nextSentimentMap.set(String(segment.seq_id), segment);
          }

          if (
            segment?.segment_index !== undefined &&
            segment?.segment_index !== null
          ) {
            nextSentimentMap.set(
              `segment-${segment.segment_index}`,
              segment,
            );
          }

          if (
            segment?.sentence_id !== undefined &&
            segment?.sentence_id !== null
          ) {
            nextSentimentMap.set(
              String(segment.sentence_id),
              segment,
            );
          }

          // Final fallback for files where sentiment rows only preserve order.
          nextSentimentMap.set(`index-${index}`, segment);
        });

        setCallData(loadedCallData);
        setSentences(loadedSentences);
        setSentimentData(loadedSentimentData);
        setSentimentMap(nextSentimentMap);
        setEvaluationData(loadedEvaluationData);
        setDuration(
          calculateDuration(loadedCallData, loadedSentences),
        );
      } catch (error) {
        if (!cancelled) {
          setLoadError(error.message);
        }
      }
    };

    loadData();

    return () => {
      cancelled = true;
    };
  }, [callId]);

  const evaluation = useMemo(
    () => extractEvaluation(callData, evaluationData),
    [callData, evaluationData],
  );

  const chapters = useMemo(
    () => extractChapters(callData, evaluation),
    [callData, evaluation],
  );

  const turns = useMemo(() => {
    if (
      Array.isArray(callData?.turns) &&
      callData.turns.length > 0
    ) {
      return callData.turns
        .map((turn) => ({
          ...turn,
          speaker: normalizeSpeaker(turn.speaker),
          start: Number(turn.start) || 0,
          end: Number(turn.end) || Number(turn.start) || 0,
          text: turn.text || '',
        }))
        .sort((a, b) => a.start - b.start);
    }

    return sentences
      .map((sentence) => ({
        ...sentence,
        speaker: normalizeSpeaker(sentence.speaker),
        start: Number(sentence.start) || 0,
        end:
          Number(sentence.end) ||
          Number(sentence.start) ||
          0,
        text: sentence.text || '',
      }))
      .sort((a, b) => a.start - b.start);
  }, [callData, sentences]);

  const audioSource = useMemo(
    () => getAudioPath(callData, callId),
    [callData, callId],
  );

  const audioAvailable = Boolean(audioSource) && !audioFailed;

  // Match sentence-segment sentiment rows with each displayed turn.
  const turnSentences = useMemo(
    () =>
      turns.map((turn) => {
        const matching = sentences.filter((sentence) => {
          const sameSpeaker =
            normalizeSpeaker(sentence.speaker) ===
            normalizeSpeaker(turn.speaker);

          const sentenceStart = Number(sentence.start) || 0;
          const sentenceEnd =
            Number(sentence.end) || sentenceStart;

          return (
            sameSpeaker &&
            sentenceStart >= turn.start - 0.1 &&
            sentenceEnd <= turn.end + 0.1
          );
        });

        // The fallback transcript already consists of individual sentences.
        if (
          matching.length === 0 &&
          !callData?.turns?.length
        ) {
          return [turn];
        }

        return matching;
      }),
    [turns, sentences, callData],
  );

  // Audio event wiring.
  useEffect(() => {
    const audio = audioRef.current;

    if (!audio || !audioAvailable) {
      return undefined;
    }

    const onTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const onMetadata = () => {
      setDuration(
        audio.duration ||
          calculateDuration(callData, sentences),
      );
    };

    const onPlay = () => {
      setIsPlaying(true);
    };

    const onPause = () => {
      setIsPlaying(false);
    };

    const onError = () => {
      setAudioFailed(true);
      setIsPlaying(false);
    };

    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('loadedmetadata', onMetadata);
    audio.addEventListener('play', onPlay);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('error', onError);

    return () => {
      audio.removeEventListener('timeupdate', onTimeUpdate);
      audio.removeEventListener('loadedmetadata', onMetadata);
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('error', onError);
    };
  }, [audioAvailable, callData, sentences]);

  const seek = useCallback(
    (time) => {
      const clamped = Math.max(
        0,
        Math.min(Number(time) || 0, duration),
      );

      if (audioRef.current && audioAvailable) {
        audioRef.current.currentTime = clamped;
      }

      // Timeline and transcript navigation still work without audio.
      setCurrentTime(clamped);
    },
    [duration, audioAvailable],
  );

  const togglePlay = async () => {
    const audio = audioRef.current;

    if (!audio || !audioAvailable) {
      return;
    }

    try {
      if (audio.paused) {
        await audio.play();
      } else {
        audio.pause();
      }
    } catch {
      setAudioFailed(true);
      setIsPlaying(false);
    }
  };

  const changeRate = () => {
    const nextRate =
      rate === 1 ? 1.5 : rate === 1.5 ? 2 : 1;

    setRate(nextRate);

    if (audioRef.current) {
      audioRef.current.playbackRate = nextRate;
    }
  };

  const timeFromEvent = useCallback(
    (event, element) => {
      const target = element || scrubberRef.current;

      if (!target || !duration) {
        return 0;
      }

      const rectangle = target.getBoundingClientRect();
      const clientX = event.touches
        ? event.touches[0].clientX
        : event.clientX;

      const ratio =
        (clientX - rectangle.left) / rectangle.width;

      return Math.max(
        0,
        Math.min(ratio * duration, duration),
      );
    },
    [duration],
  );

  const onScrubStart = useCallback(
    (event) => {
      event.preventDefault();

      const time = timeFromEvent(event);

      setIsDragging(true);
      setDragTime(time);
    },
    [timeFromEvent],
  );

  useEffect(() => {
    if (!isDragging) {
      return undefined;
    }

    const onMove = (event) => {
      setDragTime(timeFromEvent(event));
    };

    const onUp = (event) => {
      const time = timeFromEvent(event);

      setIsDragging(false);
      seek(time);
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('touchmove', onMove, {
      passive: false,
    });
    window.addEventListener('touchend', onUp);

    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('touchend', onUp);
    };
  }, [isDragging, timeFromEvent, seek]);

  const displayTime = isDragging ? dragTime : currentTime;
  const progress = duration
    ? Math.min((displayTime / duration) * 100, 100)
    : 0;

  const activeChapter = useMemo(() => {
    for (let index = chapters.length - 1; index >= 0; index -= 1) {
      if (displayTime >= Number(chapters[index].start || 0)) {
        return chapters[index];
      }
    }

    return chapters[0] || null;
  }, [displayTime, chapters]);

  const activeTurnIdx = useMemo(() => {
    let index = -1;

    for (
      let turnIndex = 0;
      turnIndex < turns.length;
      turnIndex += 1
    ) {
      if (displayTime >= turns[turnIndex].start) {
        index = turnIndex;
      } else {
        break;
      }
    }

    return index;
  }, [displayTime, turns]);

  useEffect(() => {
    if (activeTurnRef.current && scrollRef.current) {
      activeTurnRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    }
  }, [activeTurnIdx]);

  const sentimentSummary = sentimentData?.call_summary || {};
  const calibratedSentiment =
    sentimentData?.calibrated_sentiment_summary
      ?.customer_sentiment_calibrated;

  const overallSentiment =
    calibratedSentiment || sentimentSummary.dominant_sentiment || null;

  const dominantEmotion =
    sentimentSummary.dominant_emotion || null;

  const escalationScore =
    sentimentSummary.average_escalation_score ?? null;

  const managerReview =
    sentimentData?.manager_review_recommendation?.review_required ?? null;

  const managerReviewLevel =
    sentimentData?.manager_review_recommendation?.review_level || null;

  const automationDecisions =
    evaluation?.automation_decisions ||
    evaluationData?.automation_decisions ||
    [];


  if (loadError) {
    return (
      <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100 flex items-center justify-center flex-col gap-3">
        <p className="text-sm text-red-600 dark:text-red-400">
          {loadError}
        </p>

        <Link
          to="/"
          className="text-sm text-indigo-600 hover:underline dark:text-indigo-400"
        >
          &larr; Back to dashboard
        </Link>
      </div>
    );
  }

  if (!callData) {
    return (
      <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100 flex items-center justify-center">
        <p className="text-sm text-slate-500">
          Loading call…
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100 flex flex-col">
      <header className="px-6 py-4 border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950 flex justify-between items-center">
        <div>
          <Link
            to="/"
            className="text-xs text-indigo-600 hover:underline dark:text-indigo-400"
          >
            &larr; All calls
          </Link>

          <h1 className="text-xl font-bold tracking-tight text-indigo-600 dark:text-indigo-400">
            Call Compliance Player
          </h1>

          <p className="mt-0.5 text-xs text-slate-500">
            {callData.model ||
              evaluation?.metadata?.model ||
              'Contact Center Conversation Intelligence'}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span className="px-3 py-1 rounded-full bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300 text-xs font-mono border border-slate-300 dark:border-slate-700">
            {callData.call_id || callData.call || callId}
            {' · '}
            {fmt(duration)}
            {' · '}
            {turns.length} segments
          </span>

          <ThemeToggle />
        </div>
      </header>

      <main className="flex-grow max-w-6xl w-full mx-auto p-6 flex flex-col gap-5">
        <section className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-950">
            <div className="text-[10px] font-semibold uppercase text-slate-500">
              Overall sentiment
            </div>

            <div className="mt-1 text-lg font-bold capitalize">
              {overallSentiment || 'Not available'}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-950">
            <div className="text-[10px] font-semibold uppercase text-slate-500">
              Dominant emotion
            </div>

            <div className="mt-1 text-lg font-bold capitalize">
              {dominantEmotion || 'Not available'}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-950">
            <div className="text-[10px] font-semibold uppercase text-slate-500">
              Escalation score
            </div>

            <div className="mt-1 text-lg font-bold">
              {typeof escalationScore === 'number'
          ? `${Math.round(
              escalationScore <= 1
                ? escalationScore * 100
                : escalationScore
            )}%`
          : 'Not available'}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-950">
            <div className="text-[10px] font-semibold uppercase text-slate-500">
              Manager review
            </div>

            <div className="mt-1 text-lg font-bold">
              {managerReview === true
  ? `Recommended${managerReviewLevel ? ` (${managerReviewLevel})` : ''}`
  : managerReview === false
    ? 'Not required'
    : 'Not available'}
            </div>
          </div>
                </section>

        <SentimentInsightsPanel
  sentimentData={sentimentData}
  duration={duration}
  currentTime={displayTime}
  seek={seek}
/>

        <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-xl dark:bg-slate-950 dark:border-slate-800">
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={togglePlay}
              disabled={!audioAvailable}
              title={
                audioAvailable
                  ? isPlaying
                    ? 'Pause audio'
                    : 'Play audio'
                  : 'Audio file is not available'
              }
              className={`w-12 h-12 shrink-0 rounded-full transition flex items-center justify-center text-white shadow-lg ${
                audioAvailable
                  ? 'bg-indigo-600 hover:bg-indigo-500 shadow-indigo-900/40'
                  : 'bg-slate-400 cursor-not-allowed opacity-60'
              }`}
            >
              {isPlaying ? (
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <rect
                    x="6"
                    y="5"
                    width="4"
                    height="14"
                    rx="1"
                  />
                  <rect
                    x="14"
                    y="5"
                    width="4"
                    height="14"
                    rx="1"
                  />
                </svg>
              ) : (
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  style={{ marginLeft: 2 }}
                >
                  <path d="M8 5v14l11-7z" />
                </svg>
              )}
            </button>

            <div className="flex-grow">
              <div className="flex justify-between text-xs font-mono text-slate-500 dark:text-slate-400 mb-1.5">
                <span>{fmt(displayTime)}</span>
                <span>{fmt(duration)}</span>
              </div>

              <div
                ref={scrubberRef}
                className="relative h-3 bg-slate-200 dark:bg-slate-800 rounded-full cursor-pointer select-none"
                onMouseDown={onScrubStart}
                onTouchStart={onScrubStart}
              >
                {audioAvailable && (
                  <BufferedRanges
                    audio={audioRef}
                    duration={duration}
                    currentTime={currentTime}
                  />
                )}

                <div
                  className="absolute inset-y-0 left-0 bg-indigo-500 rounded-full pointer-events-none"
                  style={{ width: `${progress}%` }}
                />

                <div
                  className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 rounded-full shadow transition-transform pointer-events-none ${
                    isDragging
                      ? 'w-5 h-5 bg-white scale-110'
                      : 'w-4 h-4 bg-white'
                  }`}
                  style={{ left: `${progress}%` }}
                />
              </div>

              {!audioAvailable && (
                <p className="mt-2 text-[11px] text-slate-500">
                  Audio is not available. Transcript, sentiment and LLM
                  analysis can still be reviewed.
                </p>
              )}
            </div>

            <button
              type="button"
              onClick={changeRate}
              disabled={!audioAvailable}
              className="shrink-0 px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-xs font-mono text-slate-700 border border-slate-300 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-slate-800 dark:hover:bg-slate-700 dark:text-slate-300 dark:border-slate-700"
            >
              {rate}×
            </button>
          </div>

          {chapters.length > 0 && (
            <div className="mt-5">
              <div className="relative h-9 rounded-lg overflow-hidden border border-slate-200 bg-slate-100 dark:border-slate-800 dark:bg-slate-900">
                {chapters.map((chapter, index) => {
                  const chapterStart =
                    Number(chapter.start) || 0;
                  const chapterEnd =
                    Number(chapter.end) || chapterStart;

                  const left = duration
                    ? (chapterStart / duration) * 100
                    : 0;

                  const width = duration
                    ? ((chapterEnd - chapterStart) / duration) *
                      100
                    : 0;

                  const active =
                    activeChapter &&
                    (chapter.index === activeChapter.index ||
                      chapter === activeChapter);

                  const color =
                    CHAPTER_COLORS[
                      index % CHAPTER_COLORS.length
                    ];

                  return (
                    <button
                      type="button"
                      key={chapter.index ?? index}
                      onClick={() => seek(chapterStart + 0.05)}
                      title={`${chapter.label || `Chapter ${index + 1}`} (${fmt(chapterStart)}–${fmt(chapterEnd)})`}
                      className="absolute top-0 h-full transition-colors duration-150 border-r border-slate-950/40 group"
                      style={{
                        left: `${left}%`,
                        width: `${width}%`,
                        backgroundColor: active
                          ? color
                          : `${color}33`,
                      }}
                    >
                      <span
                        className={`absolute inset-0 flex items-center justify-center px-1 text-[10px] font-semibold truncate ${
                          active
                            ? 'text-white'
                            : 'text-slate-500 group-hover:text-slate-700 dark:text-slate-400 dark:group-hover:text-slate-200'
                        }`}
                      >
                        {width > 5
                          ? chapter.index ?? index + 1
                          : ''}
                      </span>
                    </button>
                  );
                })}

                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-white pointer-events-none shadow-[0_0_8px_rgba(255,255,255,0.8)]"
                  style={{ left: `${progress}%` }}
                />
              </div>

              {activeChapter && (
                <div className="mt-3 flex items-start gap-3">
                  <span
                    className="mt-0.5 px-2 py-0.5 rounded text-xs font-bold text-white shrink-0"
                    style={{
                      backgroundColor:
                        CHAPTER_COLORS[
                          Math.max(
                            0,
                            (Number(activeChapter.index) || 1) - 1,
                          ) % CHAPTER_COLORS.length
                        ],
                    }}
                  >
                    {activeChapter.index || 1}/{chapters.length}
                  </span>

                  <div>
                    <div className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      {activeChapter.label ||
                        activeChapter.title ||
                        'Conversation chapter'}
                    </div>

                    <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                      {activeChapter.summary ||
                        activeChapter.description}
                    </div>
                  </div>
                </div>
              )}

              <AgentLanes
                evaluation={evaluation}
                duration={duration}
                time={displayTime}
                seek={seek}
              />
            </div>
          )}

          {!chapters.length && evaluation && (
            <div className="mt-5">
              <AgentLanes
                evaluation={evaluation}
                duration={duration}
                time={displayTime}
                seek={seek}
              />
            </div>
          )}
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 flex-grow min-h-0">
          <section className="lg:col-span-3 bg-white rounded-xl border border-slate-200 shadow-xl flex flex-col min-h-0 dark:bg-slate-950 dark:border-slate-800">
            <div className="px-5 py-3 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                Conversation
              </h2>

              <div className="flex gap-4 text-xs items-center">
                <span className="flex items-center gap-1.5 text-indigo-600 dark:text-indigo-400">
                  <span className="w-2 h-2 rounded-full bg-indigo-500" />
                  Agent
                </span>

                <span className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  Customer
                </span>

                <span className="w-px h-3 bg-slate-300 dark:bg-slate-700" />

                <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400 text-[10px]">
                  <span className="w-3 h-3 rounded-sm bg-emerald-500/30 border border-emerald-700/50" />
                  positive
                </span>

                <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400 text-[10px]">
                  <span className="w-3 h-3 rounded-sm bg-red-500/30 border border-red-700/50" />
                  negative
                </span>
              </div>
            </div>

            <div
              ref={scrollRef}
              className="overflow-y-auto p-5 space-y-3"
              style={{ maxHeight: '56vh' }}
            >
              {!turns.length && (
                <p className="text-sm text-slate-500">
                  No transcript segments were found for this call.
                </p>
              )}

              {turns.map((turn, index) => {
                const isAgent =
                  normalizeSpeaker(turn.speaker) === 'AGENT';

                const active = index === activeTurnIdx;
                const matchingSentences =
                  turnSentences[index] || [];

                return (
                  <div
                    key={
                      turn.seq_id ??
                      turn.sentence_id ??
                      `${turn.start}-${index}`
                    }
                    ref={active ? activeTurnRef : null}
                    className={`flex ${
                      isAgent
                        ? 'justify-end'
                        : 'justify-start'
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => seek(turn.start + 0.02)}
                      className={`max-w-[78%] text-left rounded-2xl px-4 py-2.5 transition-all duration-200 border ${
                        isAgent
                          ? 'bg-indigo-50 border-indigo-200 rounded-br-sm dark:bg-indigo-950/60 dark:border-indigo-900'
                          : 'bg-slate-100 border-slate-200 rounded-bl-sm dark:bg-slate-800/60 dark:border-slate-700'
                      } ${
                        active
                          ? `ring-2 ring-offset-2 ring-offset-white dark:ring-offset-slate-950 scale-[1.01] shadow-lg ${
                              isAgent
                                ? 'ring-indigo-400'
                                : 'ring-emerald-400'
                            }`
                          : 'opacity-80 hover:opacity-100'
                      }`}
                    >
                      <div
                        className={`flex items-center gap-2 mb-1 text-[10px] font-mono uppercase tracking-wide ${
                          isAgent
                            ? 'text-indigo-600 dark:text-indigo-400'
                            : 'text-emerald-600 dark:text-emerald-400'
                        }`}
                      >
                        <span>
                          {isAgent ? 'Agent' : 'Customer'}
                        </span>

                        <span className="text-slate-400 dark:text-slate-600">
                          {fmt(turn.start)}
                        </span>
                      </div>
                      {(() => {
  const firstSentence = matchingSentences[0] || turn;

  const segmentSentiment =
    findSentimentForSentence(firstSentence, sentimentMap) ||
    sentimentMap.get(`index-${index}`);

  const sentimentLabel = getSentimentLabel(segmentSentiment);
  const emotionLabel = getEmotionLabel(segmentSentiment);

  if (!sentimentLabel && !emotionLabel) {
    return null;
  }

  return (
    <div className="mb-1.5 flex flex-wrap gap-1.5">
      {sentimentLabel && (
        <span className="rounded-full border border-slate-300 px-2 py-0.5 text-[10px] font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300">
          Sentiment: {sentimentLabel}
        </span>
      )}

      {emotionLabel && (
        <span className="rounded-full border border-slate-300 px-2 py-0.5 text-[10px] font-semibold capitalize text-slate-600 dark:border-slate-700 dark:text-slate-300">
          Emotion: {emotionLabel}
        </span>
      )}
    </div>
  );
})()}

                      <div className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed">
                        {matchingSentences.length > 0
                          ? matchingSentences.map(
                              (sentence, sentenceIndex) => {
                                const sentiment =
                                  findSentimentForSentence(
                                    sentence,
                                    sentimentMap,
                                  ) ||
                                  sentimentMap.get(
                                    `index-${index}`,
                                  );

                                const sentimentClass =
                                  getSentimentClass(sentiment);

                                const sentimentLabel =
                                  getSentimentLabel(sentiment);

                                const emotionLabel =
                                  getEmotionLabel(sentiment);

                                return (
                                  <span
                                    key={
                                      sentence.seq_id ??
                                      sentence.sentence_id ??
                                      `${sentence.start}-${sentenceIndex}`
                                    }
                                    className={`${sentimentClass} rounded px-0.5`}
                                    title={[
                                      sentimentLabel
                                        ? `Sentiment: ${sentimentLabel}`
                                        : null,
                                      emotionLabel
                                        ? `Emotion: ${emotionLabel}`
                                        : null,
                                    ]
                                      .filter(Boolean)
                                      .join(' · ')}
                                  >
                                    {sentence.text}{' '}
                                  </span>
                                );
                              },
                            )
                          : turn.text}
                      </div>
                    </button>
                  </div>
                );
              })}
            </div>
          </section>

          {evaluation ? (
            <CompliancePanel
              evaluation={evaluation}
              time={displayTime}
              seek={seek}
              fmt={fmt}
            />
          ) : (
            <section className="lg:col-span-2 rounded-xl border border-slate-200 bg-white p-5 shadow-xl dark:border-slate-800 dark:bg-slate-950">
              <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                LLM Compliance Analysis
              </h2>

              <p className="mt-3 text-sm leading-relaxed text-slate-500">
                The transcript and sentiment results loaded correctly,
                but no LLM evaluation object was found in the call
                artifact or evaluation endpoint.
              </p>

              <p className="mt-3 text-xs leading-relaxed text-slate-400">
                Expected data under{' '}
                <code>callData.evaluation</code> or from{' '}
                <code>
                  /evaluation/{callId}.json
                </code>
                .
              </p>
            </section>
          )}
        </div>
        {evaluation?.ai_insights && (
          <AISupervisorPanel insights={evaluation.ai_insights} />
        )}

        {!evaluation?.ai_insights &&
          evaluation?._pipeline?.nodes?.ai_supervisor?.status === 'failed' && (
            <section className="rounded-xl border border-amber-300 bg-amber-50 p-5 dark:border-amber-800 dark:bg-amber-950/30">
              <h2 className="text-sm font-semibold text-amber-800 dark:text-amber-300">
                AI Supervisor analysis unavailable
              </h2>
              <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
                The compliance, quality, workflow and escalation LLM results below are available, but the final supervisor synthesis failed during analysis.
              </p>
              <p className="mt-2 text-xs text-slate-500">
                {evaluation._pipeline.nodes.ai_supervisor.error}
              </p>
            </section>
          )}

        <AutomationDecisionsPanel decisions={automationDecisions} />
      </main>

      {audioSource && (
        <audio
          ref={audioRef}
          src={audioSource}
          preload="metadata"
          className="hidden"
        />
      )}
    </div>
  );
}
