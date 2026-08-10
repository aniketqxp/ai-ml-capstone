"""Idempotent SMTP delivery for evaluator-recommended email actions."""
from __future__ import annotations

import hashlib
import json
import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage

from app.models import EmailNotification


def _enabled():
    return os.environ.get("EMAIL_AUTOMATION_ENABLED", "true").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _recipient(audience):
    variable = (
        "DEMO_CUSTOMER_EMAIL"
        if audience == "customer"
        else "DEMO_MANAGER_EMAIL"
    )
    return (
        os.environ.get(variable)
        or os.environ.get("SMTP_USERNAME")
        or os.environ.get("EMAIL_FROM")
        or "clarayoussef01@gmail.com"
    )


def _notification_key(public_call_id, decision_sha256, action):
    payload = {
        "call_id": public_call_id,
        "decision_sha256": decision_sha256,
        "action_type": action.get("action_type"),
        "execution": action.get("execution"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def _content(public_call_id, evaluation, action, audience):
    presentation = evaluation.get("presentation") or {}
    label = action.get("label") or str(
        action.get("action_type") or "call review"
    ).replace("_", " ").title()
    reason = action.get("reason") or presentation.get("summary") or (
        "The evaluator identified a call that requires attention."
    )
    findings = presentation.get("primary_reasons") or []
    finding_lines = "\n".join(
        f"- {item.get('title')}: {item.get('summary')}"
        for item in findings[:3]
    )
    if audience == "customer":
        return {
            "subject": "Follow-up regarding your recent support call",
            "body": (
                "Hello,\n\nWe are following up regarding your recent support "
                "call. A support representative will review the interaction "
                "and follow up as appropriate. Please do not reply with "
                "passwords, PINs, or full account numbers.\n\nRegards,\n"
                "Customer Support"
            ),
        }
    details = f"\n\nReview findings:\n{finding_lines}" if finding_lines else ""
    return {
        "subject": f"{label}: {public_call_id}",
        "body": (
            "Hello Manager,\n\nThe call evaluator recommends your attention."
            f"\n\nCall: {public_call_id}\nAction: {label}\nReason: {reason}"
            f"{details}\n\nOpen the call evaluator for the transcript and "
            "timestamped evidence.\n\nRegards,\nCall Quality Analysis"
        ),
    }


def _send(recipient, subject, body):
    sender = os.environ.get("EMAIL_FROM") or os.environ.get("SMTP_USERNAME")
    password = os.environ.get("SMTP_PASSWORD", "")
    if not sender or not password:
        raise RuntimeError("SMTP credentials are not configured")
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(
        host,
        port,
        context=ssl.create_default_context(),
        timeout=30,
    ) as smtp:
        smtp.login(os.environ.get("SMTP_USERNAME", sender), password)
        smtp.send_message(message)


def serialize_notification(record):
    return {
        "notification_id": str(record.notification_id),
        "status": record.status,
        "action_type": record.action_type,
        "audience": record.audience,
        "recipient": record.recipient,
        "subject": record.subject,
        "sent_at": record.sent_at.isoformat() if record.sent_at else None,
        "error": record.error,
    }


def process_recommended_email(
    db,
    call,
    public_call_id,
    evaluation,
    *,
    approved=False,
):
    """Create the action record and send when automatic or approved."""
    decision_sha256 = str(evaluation.get("decision_sha256") or "")
    decision = evaluation.get("decision") or {}
    action = decision.get("recommended_action") or {}
    action_type = str(action.get("action_type") or "none")
    if action_type == "none":
        return None

    audience = "customer" if action_type == "customer_follow_up" else "manager"
    key = _notification_key(public_call_id, decision_sha256, action)
    record = (
        db.query(EmailNotification)
        .filter(EmailNotification.notification_key == key)
        .first()
    )
    if record and record.status == "sent":
        return serialize_notification(record)

    content = _content(public_call_id, evaluation, action, audience)
    recipient = _recipient(audience)
    sender = os.environ.get("EMAIL_FROM") or os.environ.get("SMTP_USERNAME") or ""
    if not record:
        record = EmailNotification(
            notification_key=key,
            call_id=call.call_id,
            public_call_id=public_call_id,
            decision_sha256=decision_sha256,
            action_type=action_type,
            audience=audience,
            recipient=recipient,
            sender=sender,
            status="pending",
        )
        db.add(record)

    record.recipient = recipient
    record.sender = sender
    record.subject = content["subject"]
    record.body = content["body"]
    requires_approval = action.get("execution") == "requires_approval"
    if requires_approval and not approved:
        record.status = "awaiting_approval"
        record.error = None
        db.flush()
        return serialize_notification(record)

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
        record.status = (
            "configuration_required"
            if "configured" in str(exc).lower()
            else "failed"
        )
        record.error = str(exc)[:1000]
    db.flush()
    return serialize_notification(record)
