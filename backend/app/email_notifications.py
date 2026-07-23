"""Decision-aware, idempotent email notifications for completed analyses."""
from __future__ import annotations

import hashlib
import json
import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from app.models import EmailNotification


DEMO_EMAIL = "clarayoussef01@gmail.com"


def _enabled():
    return os.environ.get("EMAIL_AUTOMATION_ENABLED", "true").lower() not in {
        "0", "false", "no", "off"
    }


def _route(decision):
    """Return (audience, purpose); None means the decision sends no email."""
    action_type = decision.get("action_type")
    state = decision.get("decision")
    if action_type == "no_action":
        return None
    if state == "blocked":
        return "manager", "blocked_action_alert"
    if state == "requires_approval":
        return "manager", "approval_request"
    if action_type == "customer_follow_up" and decision.get("automation_allowed"):
        return "customer", "customer_follow_up"
    return "manager", {
        "agent_coaching": "coaching_notice",
        "compliance_alert": "compliance_alert",
        "manager_review": "manager_review",
        "case_creation": "case_review",
    }.get(action_type, "automation_notice")


def _key(public_call_id, decision, audience, purpose):
    stable = json.dumps({
        "call": public_call_id,
        "action_id": decision.get("action_id"),
        "action_type": decision.get("action_type"),
        "decision": decision.get("decision"),
        "reason": decision.get("reason"),
        "audience": audience,
        "purpose": purpose,
    }, sort_keys=True)
    return hashlib.sha256(stable.encode()).hexdigest()


def _fallback_content(public_call_id, decision, audience, purpose, evaluation):
    action = (decision.get("action_type") or "automation action").replace("_", " ")
    reason = decision.get("reason") or "The analysis triggered this action."
    priority = (decision.get("priority") or "normal").title()
    if audience == "customer":
        return {
            "subject": f"Follow-up regarding your recent support call",
            "body": (
                "Hello,\n\nWe are following up regarding your recent support call. "
                f"Our quality review identified that additional assistance may be helpful: {reason}\n\n"
                "A support representative will review the call and follow up as appropriate. "
                "Please do not reply with passwords, PINs, or full account numbers.\n\n"
                "Regards,\nCustomer Support"
            ),
        }
    summary = ((evaluation.get("ai_insights") or {}).get("executive_summary") or {}).get("summary")
    return {
        "subject": f"[{priority}] {action.title()} — {public_call_id}",
        "body": (
            f"Hello Manager,\n\nAn automation decision requires your attention.\n\n"
            f"Call: {public_call_id}\nAction: {action.title()}\nDecision: {decision.get('decision')}\n"
            f"Priority: {priority}\nReason: {reason}\n"
            + (f"Summary: {summary}\n" if summary else "")
            + f"\nNotification type: {purpose.replace('_', ' ')}. Review the call dashboard before taking action.\n\n"
              "Regards,\nAI Call Quality Supervisor"
        ),
    }


def _llm_content(public_call_id, decision, audience, purpose, evaluation):
    fallback = _fallback_content(public_call_id, decision, audience, purpose, evaluation)
    try:
        from app.pipeline_bridge import install
        install()
        from llm_client import chat_json
        context = {
            "call_id": public_call_id,
            "audience": audience,
            "notification_purpose": purpose,
            "automation_decision": decision,
            "executive_summary": (evaluation.get("ai_insights") or {}).get("executive_summary"),
            "customer_intent": (evaluation.get("ai_insights") or {}).get("customer_intent"),
            "predicted_csat": (evaluation.get("ai_insights") or {}).get("predicted_csat"),
            "business_risk": (evaluation.get("ai_insights") or {}).get("business_risk"),
        }
        raw = chat_json(
            os.environ.get("EMAIL_LLM_PROVIDER", "mistral"),
            "You write concise call-center notification emails. Use only supplied facts. "
            "Never include passwords, PINs, full account numbers, diagnoses, or invented names. "
            "For managers, state the decision and requested review. For customers, be reassuring "
            "and do not expose internal scores or compliance findings. Return JSON with exactly "
            "two strings: subject and body.",
            json.dumps(context), max_tokens=700,
        )
        content = json.loads(raw)
        subject = str(content.get("subject") or "").replace("\r", " ").replace("\n", " ").strip()
        body = str(content.get("body") or "").strip()
        if not subject or not body or len(subject) > 300 or len(body) > 10000:
            raise ValueError("invalid LLM email content")
        return {"subject": subject, "body": body}, True
    except (Exception, SystemExit) as exc:
        print(f"[email] LLM drafting fallback: {type(exc).__name__}: {exc}")
        return fallback, False


def _send(recipient, subject, body):
    sender = os.environ.get("EMAIL_FROM", DEMO_EMAIL)
    username = os.environ.get("SMTP_USERNAME", sender)
    password = os.environ.get("SMTP_PASSWORD", "")
    if not password:
        raise RuntimeError("SMTP_PASSWORD is not configured")
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as smtp:
        smtp.login(username, password)
        smtp.send_message(message)


def process_email_notifications(db, call, public_call_id, artifacts):
    path = Path(artifacts["paths"]["evaluation"])
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    decisions = evaluation.get("automation_decisions") or []
    results = []
    for decision in decisions:
        route = _route(decision)
        if not route:
            continue
        audience, purpose = route
        recipient = os.environ.get(
            "DEMO_CUSTOMER_EMAIL" if audience == "customer" else "DEMO_MANAGER_EMAIL",
            DEMO_EMAIL,
        )
        sender = os.environ.get("EMAIL_FROM", DEMO_EMAIL)
        notification_key = _key(public_call_id, decision, audience, purpose)
        record = db.query(EmailNotification).filter(
            EmailNotification.notification_key == notification_key).first()
        if record and record.status == "sent":
            result = {"status": "sent", "audience": audience, "recipient": recipient,
                      "subject": record.subject, "sent_at": record.sent_at.isoformat()}
            decision["email_notification"] = result
            results.append(result)
            continue

        content, llm_generated = _llm_content(
            public_call_id, decision, audience, purpose, evaluation)
        if not record:
            record = EmailNotification(
                notification_key=notification_key, call_id=call.call_id,
                public_call_id=public_call_id, action_id=decision.get("action_id") or "unknown",
                action_type=decision.get("action_type") or "unknown", audience=audience,
                recipient=recipient, sender=sender, status="pending")
            db.add(record)
        record.subject = content["subject"]
        record.body = content["body"]
        record.llm_generated = llm_generated
        try:
            if not _enabled():
                record.status = "disabled"
                record.error = "EMAIL_AUTOMATION_ENABLED is false"
            else:
                _send(recipient, record.subject, record.body)
                record.status = "sent"
                record.error = None
                record.sent_at = datetime.utcnow()
        except Exception as exc:
            record.status = "configuration_required" if "SMTP_PASSWORD" in str(exc) else "failed"
            record.error = str(exc)[:1000]
        db.commit()
        result = {
            "status": record.status, "audience": audience, "recipient": recipient,
            "subject": record.subject, "llm_generated": llm_generated,
            "sent_at": record.sent_at.isoformat() if record.sent_at else None,
            "error": record.error,
        }
        decision["email_notification"] = result
        results.append(result)

    evaluation["automation_decisions"] = decisions
    evaluation["email_notifications"] = results
    path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return results
