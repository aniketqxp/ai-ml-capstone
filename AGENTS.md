# Project Context

## Git Commit Rules

- **No meta-commentary.** Never use "portfolio", "recruiter", "sleek", "professional", "hireable".
- **Technical objectivity.** Describe what changed in code logic, not why it helps a career.
  - Good: `feat: add affine transform`
  - Bad: `optimized for sleekness`
- **No "finals".** Never use "final", "overhaul", or "classic" in commit messages.
- **Atomic staging.** NEVER use `git add .` — stage files selectively.
- **Prefix rules:**
  - `feat:` — code changes (new features, logic changes)
  - `fix:` — bug fixes
  - `docs:` — documentation only
  - `chore:` — assets, configs, licenses, non-code
- **The Mirror Test.** If a commit message looks like a marketing pitch, rewrite it to sound like a compiler log.

## Stack

- **Backend:** FastAPI + faster-whisper + pyannote + Ollama
- **Frontend:** React + Vite + Tailwind
- **Evaluation:** Pydantic v2 rubric, LiteLLM router (Mistral primary → Llama-3.3-70B → GPT-4o-mini)
- **Data:** apptek HF dataset, na_testset (126 calls, banking/health/agriculture/aviation)

## Pipeline Stages

1. **Transcribe** — per-channel faster-whisper (small.en), stereo attribution
2. **Chapter** — LLM segments call into 3–8 structural phases (anchored to turn indices)
3. **Evaluate** — monolithic LLM extraction against versioned Pydantic rubric (compliance + quality + escalation)
4. **Visualize** — React call player with synced chapter strip, transcript, live compliance scorecard
