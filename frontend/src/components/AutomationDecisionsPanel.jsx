function formatActionLabel(value) {
  if (!value) {
    return 'N/A';
  }

  return String(value)
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function AutomationDecisionsPanel({
  decisions = [],
}) {
  if (!Array.isArray(decisions) || decisions.length === 0) {
    return null;
  }

  const decisionStyles = {
    approved:
      'border-emerald-800 bg-emerald-950/40 text-emerald-300',
    requires_approval:
      'border-amber-800 bg-amber-950/40 text-amber-300',
    blocked:
      'border-red-800 bg-red-950/40 text-red-300',
  };

  const decisionLabels = {
    approved: 'Approved Automatically',
    requires_approval: 'Requires Approval',
    blocked: 'Blocked',
  };

  const priorityStyles = {
    low: 'border-slate-700 bg-slate-900 text-slate-300',
    medium:
      'border-amber-800 bg-amber-950/30 text-amber-300',
    high:
      'border-orange-800 bg-orange-950/30 text-orange-300',
    critical:
      'border-red-800 bg-red-950/30 text-red-300',
  };

  const approvedCount = decisions.filter(
    (item) => item.decision === 'approved',
  ).length;

  const approvalCount = decisions.filter(
    (item) => item.decision === 'requires_approval',
  ).length;

  const blockedCount = decisions.filter(
    (item) => item.decision === 'blocked',
  ).length;

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-xl dark:border-slate-800 dark:bg-slate-950">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-orange-500">
            Deterministic Decision Engine
          </p>

          <h2 className="mt-1 text-lg font-semibold">
            Automation Decisions
          </h2>

          <p className="mt-1 text-xs text-slate-500">
            Operational actions generated from compliance,
            escalation and quality results.
          </p>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <div className="rounded-lg border border-emerald-800 px-3 py-2 text-center">
            <p className="text-lg font-semibold text-emerald-500">
              {approvedCount}
            </p>
            <p className="text-[9px] uppercase text-slate-500">
              Approved
            </p>
          </div>

          <div className="rounded-lg border border-amber-800 px-3 py-2 text-center">
            <p className="text-lg font-semibold text-amber-500">
              {approvalCount}
            </p>
            <p className="text-[9px] uppercase text-slate-500">
              Approval
            </p>
          </div>

          <div className="rounded-lg border border-red-800 px-3 py-2 text-center">
            <p className="text-lg font-semibold text-red-500">
              {blockedCount}
            </p>
            <p className="text-[9px] uppercase text-slate-500">
              Blocked
            </p>
          </div>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {decisions.map((item, index) => (
          <article
            key={item.action_id || index}
            className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-900/70"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="font-mono text-[10px] text-slate-500">
                  {item.action_id || `action-${index + 1}`}
                </p>

                <h3 className="mt-1 text-sm font-semibold">
                  {formatActionLabel(item.action_type)}
                </h3>
              </div>

              <span
                className={`rounded-full border px-2 py-1 text-[10px] font-semibold ${
                  priorityStyles[item.priority] ||
                  priorityStyles.low
                }`}
              >
                {formatActionLabel(item.priority)}
              </span>
            </div>

            <p className="mt-3 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
              {item.reason || 'No reason provided.'}
            </p>

            <div className="mt-4 flex flex-wrap gap-2">
              <span
                className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold ${
                  decisionStyles[item.decision] ||
                  decisionStyles.requires_approval
                }`}
              >
                {decisionLabels[item.decision] ||
                  formatActionLabel(item.decision)}
              </span>

              <span className="rounded-full border border-slate-300 px-2.5 py-1 text-[10px] text-slate-500 dark:border-slate-700">
                Confidence{' '}
                {Math.round(
                  Number(item.confidence || 0) * 100,
                )}
                %
              </span>

              <span className="rounded-full border border-slate-300 px-2.5 py-1 text-[10px] text-slate-500 dark:border-slate-700">
                {item.automation_allowed
                  ? 'Automation allowed'
                  : 'Automation not allowed'}
              </span>
            </div>

            {item.rules_triggered?.length > 0 && (
              <div className="mt-4 border-t border-slate-200 pt-3 dark:border-slate-800">
                <p className="text-[10px] uppercase text-slate-500">
                  Rules Triggered
                </p>

                <div className="mt-2 flex flex-wrap gap-2">
                  {item.rules_triggered.map((rule) => (
                    <span
                      key={rule}
                      className="rounded-md border border-cyan-800 px-2 py-1 font-mono text-[9px] text-cyan-500"
                    >
                      {rule}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {item.blocked_reasons?.length > 0 && (
              <div className="mt-4 rounded-lg border border-red-800 bg-red-950/20 p-3">
                <p className="text-[10px] uppercase text-red-400">
                  Blocked Reasons
                </p>

                <ul className="mt-2 space-y-1">
                  {item.blocked_reasons.map(
                    (reason, reasonIndex) => (
                      <li
                        key={`${reason}-${reasonIndex}`}
                        className="text-xs text-slate-400"
                      >
                        • {reason}
                      </li>
                    ),
                  )}
                </ul>
              </div>
            )}

            {item.email_notification && (
              <div className="mt-4 rounded-lg border border-indigo-800 bg-indigo-950/20 p-3">
                <p className="text-[10px] uppercase text-indigo-400">
                  Email Notification · {formatActionLabel(item.email_notification.status)}
                </p>
                <p className="mt-1 text-xs text-slate-300">
                  To {formatActionLabel(item.email_notification.audience)}: {item.email_notification.recipient}
                </p>
                {item.email_notification.subject && (
                  <p className="mt-1 text-xs text-slate-400">
                    {item.email_notification.subject}
                  </p>
                )}
                {item.email_notification.status === 'configuration_required' && (
                  <p className="mt-2 text-xs text-amber-400">
                    Add the Gmail App Password to SMTP_PASSWORD to enable delivery.
                  </p>
                )}
              </div>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}
