"""
Provider-agnostic LLM client (OpenAI SDK-compatible for all providers).

Swap provider with a single string -- the rest of the pipeline never changes.
Used by extract.py via: chat_json(provider, system, user)

Provider notes
--------------
github     : GitHub Models free tier -- GPT-4o-mini, zero cost, recommended default
openai     : OpenAI direct -- costs credits, use for prod/comparison only
gemini     : Google Gemini via OpenAI-compat endpoint -- free tier, 15 RPM daily cap
mistral    : Mistral AI -- mistral-small is strong at structured JSON
cohere     : Cohere -- command-r7b fastest (326ms), good at extraction
nvidia     : NVIDIA NIM -- llama-3.1-8b, solid and fast
cerebras   : Cerebras -- zai-glm-4.7 is a reasoning model (needs higher token budget)
cloudflare : Cloudflare Workers AI -- needs CLOUDFLARE_ID in .env
huggingface: HF Inference via featherless-ai -- slowest, cold-start ~14s
sambanova  : SambaNova -- has DeepSeek-V3 + Llama-4, fast on warm runs
openrouter : OpenRouter -- free models available, quality varies
deepseek   : DeepSeek -- needs paid balance (402 currently)

JSON mode support
-----------------
Providers that natively support response_format=json_object are flagged
json_mode=True. Others receive a JSON-enforcement suffix in the system prompt
and the response is extracted via regex fallback if needed.
"""

from __future__ import annotations
import os, re, json
from typing import Optional
from openai import OpenAI
from env_util import load_env

load_env()


# ── Provider registry ─────────────────────────────────────────────────────────
# Each entry: base_url, key_env, default_model, json_mode, extra_kwargs
PROVIDERS: dict[str, dict] = {
    "github": {
        "base_url":      "https://models.github.ai/inference",
        "key_env":       "GITHUB_TOKEN",
        "default_model": "openai/gpt-4o-mini",
        "json_mode":     True,
    },
    "openai": {
        "base_url":      None,
        "key_env":       "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "json_mode":     True,
    },
    "gemini": {
        "base_url":      "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env":       "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
        "json_mode":     True,
    },
    "mistral": {
        "base_url":      "https://api.mistral.ai/v1",
        "key_env":       "MISTRAL_API_KEY",
        "default_model": "mistral-small-latest",
        "json_mode":     True,
    },
    "cohere": {
        "base_url":      "https://api.cohere.com/compatibility/v1",
        "key_env":       "COHERE_API_KEY",
        "default_model": "command-r7b-12-2024",
        "json_mode":     False,   # prompt-based JSON
    },
    "nvidia": {
        "base_url":      "https://integrate.api.nvidia.com/v1",
        "key_env":       "NVIDIA_API_KEY",
        "default_model": "meta/llama-3.1-8b-instruct",
        "json_mode":     False,
    },
    "cerebras": {
        "base_url":      "https://api.cerebras.ai/v1",
        "key_env":       "CEREBRAS_API_KEY",
        "default_model": "zai-glm-4.7",
        "json_mode":     False,
        "reasoning":     True,    # needs higher token budget for chain-of-thought
    },
    "cloudflare": {
        "base_url":      None,    # built dynamically using CLOUDFLARE_ID
        "key_env":       "CLOUDFLARE_API_TOKEN",
        "default_model": "@cf/meta/llama-3.1-8b-instruct",
        "json_mode":     False,
    },
    "huggingface": {
        "base_url":      "https://router.huggingface.co/featherless-ai/v1",
        "key_env":       "HUGGINGFACE_API_TOKEN",
        "default_model": "meta-llama/Llama-3.1-8B-Instruct",
        "json_mode":     False,
    },
    "sambanova": {
        "base_url":      "https://api.sambanova.ai/v1",
        "key_env":       "SAMBANOVA_API_KEY",
        "default_model": "Meta-Llama-3.3-70B-Instruct",
        "json_mode":     False,
    },
    "openrouter": {
        "base_url":      "https://openrouter.ai/api/v1",
        "key_env":       "OPENROUTER_API_KEY",
        "default_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "json_mode":     False,
    },
    "deepseek": {
        "base_url":      "https://api.deepseek.com/v1",
        "key_env":       "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "json_mode":     True,
    },
}

JSON_ENFORCE_SUFFIX = (
    "\n\nCRITICAL: Your response must be a single valid JSON object. "
    "No markdown fences, no commentary, no preamble. Start with { and end with }."
)


def get_client(provider: str) -> tuple[OpenAI, dict]:
    """Return (OpenAI client, provider_config)."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]

    key_env = cfg.get("key_env")
    api_key = os.environ.get(key_env, "") if key_env else "local"
    if key_env and not api_key:
        raise SystemExit(f"Missing {key_env} in .env for provider '{provider}'")

    base_url = cfg["base_url"]
    if provider == "cloudflare":
        account_id = os.environ.get("CLOUDFLARE_ID", "")
        if not account_id:
            raise SystemExit("Missing CLOUDFLARE_ID in .env")
        base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"

    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs), cfg


def _extract_json(text: str) -> str:
    """Strip markdown fences and extract the first {...} block."""
    text = text.strip()
    # strip ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    # find first { ... } spanning the whole string
    start = text.find("{")
    if start == -1:
        return text
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    return text[start:end + 1] if end != -1 else text[start:]


def chat_json(
    provider: str,
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 4000,
) -> str:
    """
    Single chat completion that returns a JSON string.

    For providers with native json_mode, passes response_format=json_object.
    For others, appends a JSON-enforcement instruction and strips fences from
    the response. Either way the caller gets a raw JSON string to parse.
    """
    client, cfg = get_client(provider)
    chosen_model = model or cfg["default_model"]
    use_json_mode = cfg.get("json_mode", False)

    # reasoning models need a much larger token budget
    if cfg.get("reasoning") and max_tokens < 2048:
        max_tokens = 2048

    sys_msg = system if use_json_mode else system + JSON_ENFORCE_SUFFIX

    kwargs: dict = {
        "model":       chosen_model,
        "messages":    [{"role": "system", "content": sys_msg},
                        {"role": "user",   "content": user}],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content or ""
    return _extract_json(raw) if not use_json_mode else raw.strip()


def list_providers() -> list[str]:
    return list(PROVIDERS.keys())


def default_model(provider: str) -> str:
    return PROVIDERS[provider]["default_model"]


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time, argparse
    ap = argparse.ArgumentParser(description="Ping all providers or a specific one.")
    ap.add_argument("--provider", default=None, help="test one provider only")
    ap.add_argument("--prompt", default="Reply with valid JSON: {\"status\": \"ok\", \"msg\": \"hello\"}")
    args = ap.parse_args()

    targets = [args.provider] if args.provider else list(PROVIDERS.keys())
    print(f"\n  {'Provider':<14} {'Model':<42} {'ms':>6}  {'Result'}")
    print("  " + "-" * 90)

    for p in targets:
        cfg = PROVIDERS[p]
        mod = cfg["default_model"]
        try:
            t0 = time.time()
            raw = chat_json(p, "You are a helpful assistant.", args.prompt)
            ms = int((time.time() - t0) * 1000)
            # try to parse to confirm it's valid JSON
            parsed = json.loads(raw)
            print(f"  [OK] {p:<12} {mod:<42} {ms:>6}ms  {str(parsed)[:60]}")
        except SystemExit as e:
            print(f"  [--] {p:<12} {mod:<42}  {'missing key -- skipped'}")
        except Exception as e:
            print(f"  [XX] {p:<12} {mod:<42}  {str(e)[:70]}")
