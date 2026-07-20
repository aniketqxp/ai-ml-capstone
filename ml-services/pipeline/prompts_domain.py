"""
Whisper decode hints per call domain.

Shared by the offline batch transcriber (scripts/run_batch.py) and the
single-call orchestrator (pipeline/orchestrator.py) so both feed identical
initial_prompt context to faster-whisper. Domains without an explicit entry
fall back to FALLBACK_PROMPT -- the prompt is only a soft decoding bias, so a
generic hint is safe for the domains outside the original three.
"""

DOMAIN_PROMPTS = {
    "banking": (
        "Banking and customer service call between an agent and a customer. "
        "Topics include accounts, transfers, payments, balances, credit cards, "
        "loans, account numbers, and online banking."
    ),
    "health": (
        "Healthcare customer service call between an agent and a patient or caller. "
        "Topics include appointments, medical records, prescriptions, insurance "
        "coverage, billing, referrals, and health plan benefits."
    ),
    "telecom": (
        "Telecommunications customer service call between an agent and a customer. "
        "Topics include mobile plans, internet service, data usage, billing, "
        "account management, and technical support."
    ),
}

FALLBACK_PROMPT = "Customer service call between an agent and a customer."


def prompt_for(domain):
    """Decode hint for a domain (case-insensitive), FALLBACK_PROMPT if unknown."""
    return DOMAIN_PROMPTS.get((domain or "").lower(), FALLBACK_PROMPT)
