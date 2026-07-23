import { useState } from 'react';

function formatLabel(value) {
  if (value === null || value === undefined || value === '') {
    return 'N/A';
  }

  return String(value)
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatNumber(value, precision = 2) {
  const number = Number(value);

  return Number.isFinite(number) ? number.toFixed(precision) : 'N/A';
}

function SummaryCard({ label, value, detail }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-900/70">
      <p className="text-xs font-semibold uppercase text-slate-500">
        {label}
      </p>
      <p className="mt-1 text-xl font-semibold">{value}</p>
      {detail && (
        <p className="mt-1 text-xs text-slate-500">{detail}</p>
      )}
    </div>
  );
}

function Distribution({ title, values = {}, total = 0 }) {
  const entries = Object.entries(values);

  if (entries.length === 0) return null;

  const denominator = total || entries.reduce(
    (sum, [, count]) => sum + Number(count || 0),
    0,
  );

  return (
    <div className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
      <h3 className="text-sm font-semibold">{title}</h3>
      <div className="mt-3 space-y-2">
        {entries.map(([label, count]) => {
          const percentage = denominator
            ? (Number(count || 0) / denominator) * 100
            : 0;

          return (
            <div key={label}>
              <div className="mb-1 flex justify-between text-xs text-slate-500">
                <span>{formatLabel(label)}</span>
                <span>{count} ({Math.round(percentage)}%)</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                <div
                  className="h-full rounded-full bg-indigo-500"
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AudioFeatureTimeline({ series, duration, currentTime, seek, formatNumber }) {
  const [selectedMetric, setSelectedMetric] = useState('escalation_score');

  if (!series || series.length === 0 || !duration) return null;

  const metrics = [
    {
      key: 'escalation_score',
      label: 'Escalation',
      color: '#f97316',
      format: (v) => formatNumber(v, 3),
    },
    {
      key: 'volume_db_mean',
      label: 'Volume',
      color: '#38bdf8',
      format: (v) => `${formatNumber(v)} dB`,
    },
    {
      key: 'pitch_mean_hz',
      label: 'Pitch',
      color: '#a78bfa',
      format: (v) => `${formatNumber(v)} Hz`,
    },
    {
      key: 'speech_rate_words_per_minute',
      label: 'Speech Rate',
      color: '#22c55e',
      format: (v) => `${formatNumber(v)} WPM`,
    },
  ];

  const selectedMetricConfig = metrics.find((m) => m.key === selectedMetric) || metrics[0];

  const getTime = (point) => {
    if (point.start_time !== null && point.start_time !== undefined) return Number(point.start_time);
    if (point.start !== null && point.start !== undefined) return Number(point.start);
    return 0;
  };

  const getEndTime = (point) => {
    if (point.end_time !== null && point.end_time !== undefined) return Number(point.end_time);
    if (point.end !== null && point.end !== undefined) return Number(point.end);
    return getTime(point) + 0.5;
  };

  const getSentimentColor = (sentiment) => {
    const normalized = String(sentiment || '').toLowerCase();

    if (normalized.includes('positive')) return '#22c55e';
    if (normalized.includes('negative')) return '#ef4444';
    if (normalized.includes('neutral')) return '#64748b';
    if (normalized.includes('mixed')) return '#eab308';
    return '#334155';
  };

  const currentPoint =
    series.find((point) => {
      const start = getTime(point);
      const end = getEndTime(point);
      return currentTime >= start && currentTime <= end;
    }) || series.reduce((closest, point) => {
      const t = getTime(point);
      const closestTime = getTime(closest);
      return Math.abs(t - currentTime) < Math.abs(closestTime - currentTime) ? point : closest;
    }, series[0]);

  const playheadX = Math.max(0, Math.min((currentTime / duration) * 100, 100));

  const buildPath = (metricKey) => {
    const values = series
      .map((point) => Number(point[metricKey]))
      .filter((value) => Number.isFinite(value));

    if (values.length === 0) return '';

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    return series
      .map((point) => {
        const x = Math.max(0, Math.min((getTime(point) / duration) * 100, 100));
        const rawValue = Number(point[metricKey]);
        const safeValue = Number.isFinite(rawValue) ? rawValue : min;
        const y = 34 - ((safeValue - min) / range) * 28;
        return `${x},${y}`;
      })
      .join(' ');
  };

  return (
    <div className="mt-4 rounded-xl border border-slate-800 bg-slate-900/70 p-4">
      <div className="flex flex-col gap-1 mb-3">
        <h3 className="text-sm font-semibold text-slate-300">
          Sentiment & Audio Timeline
        </h3>
        <p className="text-xs text-slate-500">
          Choose one metric to view over time. Sentiment is shown underneath so the audio pattern can be compared with positive, negative, or neutral moments.
        </p>
      </div>

      <div className="mb-4 flex flex-wrap gap-3">
        {metrics.map((metric) => (
          <label
            key={metric.key}
            className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-xs cursor-pointer transition
              ${selectedMetric === metric.key
                ? 'border-indigo-500 bg-indigo-950/40 text-white'
                : 'border-slate-800 bg-slate-950/60 text-slate-400 hover:text-slate-200'}`}
          >
            <input
              type="checkbox"
              checked={selectedMetric === metric.key}
              onChange={() => setSelectedMetric(metric.key)}
              className="accent-indigo-500"
            />
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: metric.color }}
            />
            {metric.label}
          </label>
        ))}
      </div>

      <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="rounded-lg bg-slate-950/60 p-3 border border-slate-800">
          <p className="text-xs text-slate-500">Current metric</p>
          <div className="mt-1 flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: selectedMetricConfig.color }}
            />
            <p className="font-semibold text-slate-200">
              {selectedMetricConfig.label}: {selectedMetricConfig.format(currentPoint?.[selectedMetricConfig.key])}
            </p>
          </div>
        </div>

        <div className="rounded-lg bg-slate-950/60 p-3 border border-slate-800">
          <p className="text-xs text-slate-500">Current sentiment</p>
          <div className="mt-1 flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: getSentimentColor(currentPoint?.sentiment) }}
            />
            <p className="font-semibold text-slate-200">
              {formatLabel(currentPoint?.sentiment)}
            </p>
          </div>
        </div>

        <div className="rounded-lg bg-slate-950/60 p-3 border border-slate-800">
          <p className="text-xs text-slate-500">Current emotion</p>
          <p className="mt-1 font-semibold text-slate-200">
            {formatLabel(currentPoint?.dominant_emotion)}
          </p>
        </div>
      </div>

      <div
        className="relative rounded-lg border border-slate-800 bg-slate-950/80 p-3 cursor-pointer"
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const nextTime = (x / rect.width) * duration;
          seek(nextTime);
        }}
      >
        <svg viewBox="0 0 100 48" preserveAspectRatio="none" className="h-56 w-full">
          {[4, 12, 20, 28, 36].map((y) => (
            <line
              key={y}
              x1="0"
              x2="100"
              y1={y}
              y2={y}
              stroke="#1e293b"
              strokeWidth="0.25"
            />
          ))}

          <polyline
            points={buildPath(selectedMetricConfig.key)}
            fill="none"
            stroke={selectedMetricConfig.color}
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />

          {series.map((point) => {
            const startX = Math.max(0, Math.min((getTime(point) / duration) * 100, 100));
            const endX = Math.max(startX + 0.15, Math.min((getEndTime(point) / duration) * 100, 100));
            return (
              <rect
                key={`${point.segment_index}-${point.seq_id}`}
                x={startX}
                y="42"
                width={Math.max(endX - startX, 0.15)}
                height="4"
                fill={getSentimentColor(point.sentiment)}
                opacity="0.85"
              />
            );
          })}

          <line
            x1={playheadX}
            x2={playheadX}
            y1="0"
            y2="48"
            stroke="#ffffff"
            strokeWidth="0.6"
            vectorEffect="non-scaling-stroke"
          />
        </svg>

        <div className="mt-3 flex flex-wrap gap-4 text-xs">
          <span className="flex items-center gap-1.5 text-slate-400">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: selectedMetricConfig.color }}
            />
            {selectedMetricConfig.label}
          </span>

          <span className="flex items-center gap-1.5 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            Positive sentiment
          </span>

          <span className="flex items-center gap-1.5 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-red-500" />
            Negative sentiment
          </span>

          <span className="flex items-center gap-1.5 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-slate-500" />
            Neutral sentiment
          </span>
        </div>

        <p className="mt-2 text-[10px] text-slate-600">
          Click anywhere on the chart to jump to that moment in the call.
        </p>
      </div>
    </div>
  );
}

export function SentimentInsightsPanel({
  sentimentData,
  duration,
  currentTime,
  seek,
}) {
  if (!sentimentData) {
    return null;
  }

  const callSummary =
    sentimentData.call_summary || {};

  const audioSummary =
    sentimentData.audio_feature_summary || {};

  const calibratedSentiment =
    sentimentData.calibrated_sentiment_summary || {};

  const customerTrend =
    sentimentData.customer_escalation_trend || {};

  const speakerSummary =
    sentimentData.speaker_audio_feature_summary || {};

  const managerReview =
    sentimentData.manager_review_recommendation || {};

  const audioFeatureSeries =
    sentimentData.dashboard_audio_feature_series || [];

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-xl dark:border-slate-800 dark:bg-slate-950">
      <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
        Sentiment, Escalation & Audio Insights
      </h2>

      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-900/70">
          <p className="text-xs uppercase text-slate-500">
            Risk Level
          </p>

          <p className="mt-1 text-xl font-semibold">
            {callSummary.risk_level || 'N/A'}
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-900/70">
          <p className="text-xs uppercase text-slate-500">
            Customer Sentiment
          </p>

          <p className="mt-1 text-xl font-semibold">
            {calibratedSentiment.customer_sentiment_calibrated ||
              callSummary.dominant_sentiment || 'N/A'}
          </p>
          {calibratedSentiment.customer_sentiment_calibrated &&
            callSummary.dominant_sentiment && (
              <p className="mt-1 text-xs text-slate-500">
                Raw audio dominant: {callSummary.dominant_sentiment}
              </p>
            )}
        </div>

        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-900/70">
          <p className="text-xs uppercase text-slate-500">
            Average Volume
          </p>

          <p className="mt-1 text-xl font-semibold">
            {audioSummary.average_volume_db != null
              ? `${Number(
                  audioSummary.average_volume_db,
                ).toFixed(2)} dB`
              : 'N/A'}
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-900/70">
          <p className="text-xs uppercase text-slate-500">
            Escalation Trend
          </p>

          <p className="mt-1 text-xl font-semibold capitalize">
            {customerTrend.trend || 'N/A'}
          </p>
        </div>
      </div>

      <AudioFeatureTimeline
        series={audioFeatureSeries}
        duration={duration}
        currentTime={currentTime}
        seek={seek}
        formatNumber={formatNumber}
      />

      <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <SummaryCard
          label="Average pitch"
          value={`${formatNumber(audioSummary.average_pitch_hz)} Hz`}
        />
        <SummaryCard
          label="Average volume"
          value={`${formatNumber(audioSummary.average_volume_db)} dB`}
        />
        <SummaryCard
          label="Average energy"
          value={formatNumber(audioSummary.average_energy, 4)}
        />
        <SummaryCard
          label="Pause ratio"
          value={`${formatNumber(Number(audioSummary.average_pause_ratio) * 100)}%`}
        />
        <SummaryCard
          label="Speech rate"
          value={`${formatNumber(audioSummary.average_speech_rate_wpm)} WPM`}
        />
        <SummaryCard
          label="Max escalation"
          value={`${formatNumber(Number(callSummary.max_escalation_score) * 100, 1)}%`}
        />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Distribution
          title="Sentiment distribution"
          values={callSummary.sentiment_distribution}
          total={callSummary.successful_segments}
        />
        <Distribution
          title="Emotion distribution"
          values={callSummary.emotion_distribution}
          total={callSummary.successful_segments}
        />
      </div>

      {Object.keys(speakerSummary).length > 0 && (
        <div className="mt-4">
          <h3 className="text-sm font-semibold">Speaker comparison</h3>
          <div className="mt-3 grid gap-4 lg:grid-cols-2">
            {Object.entries(speakerSummary).map(([key, speaker]) => (
              <div
                key={key}
                className="rounded-xl border border-slate-200 p-4 dark:border-slate-800"
              >
                <div className="flex items-center justify-between">
                  <h4 className="font-semibold">{formatLabel(speaker.speaker || key)}</h4>
                  <span className="text-xs text-slate-500">
                    {speaker.total_segments || 0} segments
                  </span>
                </div>
                <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                  <div><dt className="text-slate-500">Sentiment</dt><dd className="font-medium">{formatLabel(speaker.dominant_sentiment)}</dd></div>
                  <div><dt className="text-slate-500">Emotion</dt><dd className="font-medium">{formatLabel(speaker.dominant_emotion)}</dd></div>
                  <div><dt className="text-slate-500">Escalation</dt><dd className="font-medium">{formatNumber(Number(speaker.average_escalation_score) * 100, 1)}%</dd></div>
                  <div><dt className="text-slate-500">Pitch</dt><dd className="font-medium">{formatNumber(speaker.average_pitch_hz)} Hz</dd></div>
                  <div><dt className="text-slate-500">Volume</dt><dd className="font-medium">{formatNumber(speaker.average_volume_db)} dB</dd></div>
                  <div><dt className="text-slate-500">Speech rate</dt><dd className="font-medium">{formatNumber(speaker.average_speech_rate_wpm)} WPM</dd></div>
                </dl>
              </div>
            ))}
          </div>
        </div>
      )}

      {customerTrend.trend_explanation && (
        <p className="mt-4 text-sm text-slate-600 dark:text-slate-400">
          {customerTrend.trend_explanation}
        </p>
      )}

      {Object.keys(managerReview).length > 0 && (
        <div className={`mt-4 rounded-xl border p-4 ${
          managerReview.review_required
            ? 'border-amber-300 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/30'
            : 'border-emerald-300 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/30'
        }`}>
          <h3 className="text-sm font-semibold">
            Manager review: {managerReview.review_required ? 'recommended' : 'not required'}
            {managerReview.review_level ? ` (${managerReview.review_level})` : ''}
          </h3>
          {managerReview.reasons?.length > 0 && (
            <ul className="mt-2 space-y-1 text-sm text-slate-600 dark:text-slate-300">
              {managerReview.reasons.map((reason) => <li key={reason}>• {reason}</li>)}
            </ul>
          )}
          {managerReview.notes?.length > 0 && (
            <ul className="mt-2 space-y-1 text-xs text-slate-500">
              {managerReview.notes.map((note) => <li key={note}>• {note}</li>)}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
