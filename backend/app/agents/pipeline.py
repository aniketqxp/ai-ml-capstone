from app.agents.workflow_agent import WorkflowComplianceAgent


class CentralPipelineOrchestrator:
    def __init__(self):
        self.compliance_agent = WorkflowComplianceAgent()

    async def execute(self, audio_path: str, transcripts: list | None = None):
        # Coordinates ML model inference and multi-agent compliance evaluation.
        # `transcripts` defaults to an empty list until the transcription step
        # (owned by Aniket's WP2 pipeline) is wired in here.
        transcripts = transcripts or []

        compliance_result = self.compliance_agent.evaluate_rules(transcripts)

        return {
            "status": "success",
            "audio": audio_path,
            "transcripts": transcripts,
            "compliance_checks": compliance_result,
        }