class WorkflowComplianceAgent:
    """
    Deterministic compliance checker for call-center transcripts.
    Scans the agent's speech for required scripted elements (greeting,
    legal disclosure, closing statement, etc.) using simple keyword/phrase
    matching -- no LLM call needed for this deterministic layer, matching
    the "deterministic compliance verification agent" described in D1/D2.
    """

    def __init__(self):
        self.rules = [
            {
                "rule_id": "greeting",
                "description": "Agent opened the call with a proper greeting/brand name",
                "keywords": ["thank you for calling", "how can i help you", "how may i assist you"],
                "required": True,
            },
            {
                "rule_id": "identity_verification",
                "description": "Agent verified the customer's identity before discussing account details",
                "keywords": ["can you verify", "confirm your account", "date of birth", "verify your identity"],
                "required": True,
            },
            {
                "rule_id": "legal_disclosure",
                "description": "Agent stated the required legal/recording disclosure",
                "keywords": ["this call may be recorded", "this call is being recorded", "for quality assurance"],
                "required": True,
            },
            {
                "rule_id": "closing_statement",
                "description": "Agent closed the call properly (thanked the customer / offered further help)",
                "keywords": ["thank you for your time", "is there anything else", "have a great day"],
                "required": True,
            },
        ]

    def _extract_text(self, segment):
        if isinstance(segment, str):
            return segment
        if isinstance(segment, dict):
            for key in ("text", "transcript", "content", "utterance"):
                if key in segment and segment[key]:
                    return str(segment[key])
        return ""

    def _extract_speaker(self, segment):
        if isinstance(segment, dict):
            for key in ("speaker", "role", "channel"):
                if key in segment and segment[key]:
                    return str(segment[key]).lower()
        return None

    def evaluate_rules(self, transcript_segments: list) -> dict:
        agent_text_parts = []
        any_speaker_labels = False

        for segment in transcript_segments:
            speaker = self._extract_speaker(segment)
            text = self._extract_text(segment)
            if not text:
                continue
            if speaker is not None:
                any_speaker_labels = True
                if speaker in ("agent", "rep", "representative"):
                    agent_text_parts.append(text)
            else:
                agent_text_parts.append(text)

        combined_text = " ".join(agent_text_parts).lower()

        results = []
        for rule in self.rules:
            matched_phrase = next((kw for kw in rule["keywords"] if kw in combined_text), None)
            results.append({
                "rule_id": rule["rule_id"],
                "description": rule["description"],
                "required": rule["required"],
                "passed": matched_phrase is not None,
                "matched_phrase": matched_phrase,
            })

        required_rules = [r for r in results if r["required"]]
        passed_required = [r for r in required_rules if r["passed"]]
        compliance_score = (
            round(len(passed_required) / len(required_rules), 3) if required_rules else None
        )

        return {
            "results": results,
            "compliance_score": compliance_score,
            "segments_evaluated": len(transcript_segments),
            "used_speaker_labels": any_speaker_labels,
        }
    # --- quick manual test, so you can see it produce a real result ---
if __name__ == "__main__":
    sample_segments = [
        {"speaker": "agent", "text": "Thank you for calling Acme Support, how can I help you today?"},
        {"speaker": "customer", "text": "Hi, I need help with my account."},
        {"speaker": "agent", "text": "Sure, can you verify your date of birth for me first?"},
        {"speaker": "customer", "text": "Sure, it's January 5th."},
        {"speaker": "agent", "text": "Great, thank you. Is there anything else I can help you with today?"},
    ]

    agent = WorkflowComplianceAgent()
    output = agent.evaluate_rules(sample_segments)

    print(f"Compliance score: {output['compliance_score']}")
    for r in output["results"]:
        status = "PASSED" if r["passed"] else "MISSING"
        print(f"  [{status}] {r['rule_id']}: {r['description']}")