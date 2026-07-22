import React from "react";

const PRIORITY_STYLES = {
  Critical: "bg-red-100 text-red-800 border-red-200",
  High: "bg-orange-100 text-orange-800 border-orange-200",
  Medium: "bg-yellow-100 text-yellow-800 border-yellow-200",
  Low: "bg-green-100 text-green-800 border-green-200",
  None: "bg-slate-100 text-slate-700 border-slate-200",
};

const RISK_STYLES = {
  Critical: "bg-red-100 text-red-800",
  High: "bg-orange-100 text-orange-800",
  Medium: "bg-yellow-100 text-yellow-800",
  Low: "bg-green-100 text-green-800",
};

function StatusBadge({ active, activeText, inactiveText }) {
  return (
    <span
      className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${
        active
          ? "bg-red-100 text-red-700"
          : "bg-green-100 text-green-700"
      }`}
    >
      {active ? activeText : inactiveText}
    </span>
  );
}

function MetricCard({ label, value, description }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        {label}
      </p>

      <p className="mt-2 text-xl font-bold text-slate-900">
        {value}
      </p>

      {description && (
        <p className="mt-1 text-sm text-slate-500">
          {description}
        </p>
      )}
    </div>
  );
}

export default function AISupervisorPanel({ insights }) {
  if (!insights) {
    return null;
  }

  const executive = insights.executive_summary || {};
  const intent = insights.customer_intent || {};
  const rootCause = insights.root_cause || {};
  const csat = insights.predicted_csat || {};
  const risk = insights.business_risk || {};
  const nextAction = insights.next_best_action || {};
  const coaching = insights.coaching_recommendations || [];
  const correlations = insights.audio_text_correlations || [];
  const contradictions = insights.contradictions || [];
  const automation = insights.automation || {};

  const csatConfidence =
    typeof csat.confidence === "number"
      ? `${Math.round(csat.confidence * 100)}% confidence`
      : null;

  const priorityStyle =
    PRIORITY_STYLES[automation.priority] ||
    PRIORITY_STYLES.None;

  const riskStyle =
    RISK_STYLES[risk.level] ||
    "bg-slate-100 text-slate-700";

  return (
    <section className="space-y-5 rounded-2xl border border-indigo-200 bg-slate-50 p-5 shadow-sm">
      {/* Header */}
      <div className="flex flex-col justify-between gap-3 md:flex-row md:items-center">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600 text-lg text-white">
              AI
            </span>

            <div>
              <h2 className="text-xl font-bold text-slate-900">
                AI Contact Center Supervisor
              </h2>

              <p className="text-sm text-slate-500">
                Integrated call outcome, risk, coaching and next-action analysis
              </p>
            </div>
          </div>
        </div>

        <span
          className={`inline-flex w-fit rounded-full border px-3 py-1 text-sm font-semibold ${priorityStyle}`}
        >
          {automation.priority || "None"} priority
        </span>
      </div>

      {/* Top metrics */}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Call Health"
          value={executive.overall_call_health || "Unknown"}
          description={executive.call_outcome || "Outcome unavailable"}
        />

        <MetricCard
          label="Predicted CSAT"
          value={
            csat.score !== undefined
              ? `${csat.score}/5`
              : "Unknown"
          }
          description={csatConfidence}
        />

        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Business Risk
          </p>

          <span
            className={`mt-2 inline-flex rounded-full px-3 py-1 text-sm font-bold ${riskStyle}`}
          >
            {risk.level || "Unknown"}
          </span>

          <p className="mt-2 text-sm text-slate-500">
            {risk.reasons?.[0] || "No risk explanation provided"}
          </p>
        </div>

        <MetricCard
          label="Resolution"
          value={executive.resolution_status || "Unknown"}
          description={nextAction.priority
            ? `Next-action priority: ${nextAction.priority}`
            : null}
        />
      </div>

      {/* Executive summary */}
      <div className="rounded-xl border border-indigo-100 bg-white p-5">
        <h3 className="text-sm font-bold uppercase tracking-wide text-indigo-700">
          Executive Summary
        </h3>

        <p className="mt-3 leading-7 text-slate-700">
          {executive.summary || "No executive summary available."}
        </p>
      </div>

      {/* Intent and root cause */}
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <h3 className="font-bold text-slate-900">
            Customer Intent
          </h3>

          <p className="mt-3 text-sm font-semibold text-indigo-700">
            Primary intent
          </p>

          <p className="mt-1 text-slate-700">
            {intent.primary_intent || "Unknown"}
          </p>

          {intent.customer_goal && (
            <>
              <p className="mt-4 text-sm font-semibold text-indigo-700">
                Customer goal
              </p>

              <p className="mt-1 text-slate-700">
                {intent.customer_goal}
              </p>
            </>
          )}

          {intent.secondary_intents?.length > 0 && (
            <>
              <p className="mt-4 text-sm font-semibold text-indigo-700">
                Secondary intents
              </p>

              <ul className="mt-2 space-y-2">
                {intent.secondary_intents.map((item, index) => (
                  <li
                    key={`${item}-${index}`}
                    className="flex gap-2 text-sm text-slate-700"
                  >
                    <span className="text-indigo-500">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <h3 className="font-bold text-slate-900">
            Root Cause
          </h3>

          <p className="mt-3 text-slate-700">
            {rootCause.main_reason || "No root cause was generated."}
          </p>

          {rootCause.contributing_factors?.length > 0 && (
            <ul className="mt-4 space-y-2">
              {rootCause.contributing_factors.map((factor, index) => (
                <li
                  key={`${factor}-${index}`}
                  className="flex gap-2 text-sm text-slate-600"
                >
                  <span className="text-indigo-500">•</span>
                  <span>{factor}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* Next action */}
      <div className="rounded-xl border border-blue-200 bg-blue-50 p-5">
        <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center">
          <h3 className="font-bold text-blue-900">
            Next Best Action
          </h3>

          <span className="w-fit rounded-full bg-white px-3 py-1 text-xs font-semibold text-blue-800">
            {nextAction.priority || "None"} priority
          </span>
        </div>

        <p className="mt-3 text-lg font-semibold text-blue-950">
          {nextAction.action || "No action provided"}
        </p>

        {nextAction.reason && (
          <p className="mt-2 text-sm leading-6 text-blue-800">
            {nextAction.reason}
          </p>
        )}
      </div>

      {/* Automation */}
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <h3 className="font-bold text-slate-900">
          Automated Decisions
        </h3>

        <div className="mt-4 flex flex-wrap gap-2">
          <StatusBadge
            active={automation.manager_review_required}
            activeText="Manager review required"
            inactiveText="No manager review"
          />

          <StatusBadge
            active={automation.customer_follow_up_required}
            activeText="Customer follow-up required"
            inactiveText="No customer follow-up"
          />

          <StatusBadge
            active={automation.compliance_alert_required}
            activeText="Compliance alert"
            inactiveText="No compliance alert"
          />

          <StatusBadge
            active={automation.coaching_required}
            activeText="Coaching required"
            inactiveText="No coaching required"
          />
        </div>

        {automation.reasons?.length > 0 && (
          <ul className="mt-4 space-y-2">
            {automation.reasons.map((reason, index) => (
              <li
                key={`${reason}-${index}`}
                className="flex gap-2 text-sm text-slate-600"
              >
                <span className="text-slate-400">•</span>
                <span>{reason}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Coaching */}
      {coaching.length > 0 && (
        <div className="rounded-xl border border-orange-200 bg-orange-50 p-5">
          <h3 className="font-bold text-orange-950">
            Agent Coaching Recommendations
          </h3>

          <div className="mt-4 space-y-4">
            {coaching.map((item, index) => (
              <article
                key={`${item.category}-${index}`}
                className="rounded-xl border border-orange-200 bg-white p-4"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full bg-orange-100 px-3 py-1 text-xs font-bold text-orange-800">
                    {item.category}
                  </span>

                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700">
                    {item.priority} priority
                  </span>

                  {item.timestamp && (
                    <span className="text-xs text-slate-500">
                      At {item.timestamp}
                    </span>
                  )}
                </div>

                <p className="mt-3 text-sm font-semibold text-slate-900">
                  Observation
                </p>

                <p className="mt-1 text-sm leading-6 text-slate-600">
                  {item.observation}
                </p>

                <p className="mt-3 text-sm font-semibold text-slate-900">
                  Recommendation
                </p>

                <p className="mt-1 text-sm leading-6 text-slate-700">
                  {item.recommendation}
                </p>

                {item.suggested_phrase && (
                  <div className="mt-3 rounded-lg bg-slate-50 p-3">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                      Suggested phrase
                    </p>

                    <p className="mt-1 text-sm italic text-slate-700">
                      “{item.suggested_phrase}”
                    </p>
                  </div>
                )}
              </article>
            ))}
          </div>
        </div>
      )}

      {/* Audio-text correlation */}
      {correlations.length > 0 && (
        <div className="rounded-xl border border-purple-200 bg-purple-50 p-5">
          <h3 className="font-bold text-purple-950">
            Audio and Text Correlations
          </h3>

          <div className="mt-4 space-y-3">
            {correlations.map((item, index) => (
              <div
                key={`correlation-${index}`}
                className="rounded-lg border border-purple-200 bg-white p-4"
              >
                {item.timestamp && (
                  <p className="text-xs font-semibold text-purple-700">
                    {item.timestamp}
                  </p>
                )}

                <p className="mt-1 text-sm text-slate-700">
                  {item.interpretation}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Contradictions */}
      {contradictions.length > 0 && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-5">
          <h3 className="font-bold text-red-950">
            Detected Contradictions
          </h3>

          <div className="mt-4 space-y-3">
            {contradictions.map((item, index) => (
              <div
                key={`contradiction-${index}`}
                className="rounded-lg border border-red-200 bg-white p-4"
              >
                <p className="text-xs font-bold uppercase text-red-700">
                  {item.severity || "Unknown"} severity
                </p>

                <p className="mt-2 text-sm text-slate-700">
                  {item.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}