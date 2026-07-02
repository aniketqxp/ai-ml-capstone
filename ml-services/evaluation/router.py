"""
LiteLLM-based priority router for QA extraction.

Routing policy is DERIVED FROM benchmark.py results, not guessed:
  - primary  = mistral/mistral-small-latest
       only model that DISCRIMINATED scores between calls ([5,4,3,5,2] vs
       [4,4,3,4,2]); high evidence fidelity; conservative.
  - fallback = sambanova/Meta-Llama-3.3-70B-Instruct
       fastest (2.6s), best raw fidelity (75%), perfectly conservative.
  - safety   = github/gpt-4o-mini
       always-on free baseline, reliable JSON.

DeepSeek-V3.1 was EXCLUDED: 50% evidence fidelity = fabricates quotes, the worst
failure mode for a compliance system.

LiteLLM Router gives us retries, fallback chaining, and per-deployment cooldowns
for free. On a 429/timeout/error the call transparently drops to the next tier.

Public API mirrors llm_client.chat_json so extract.py can swap to routing with a
one-line change:  chat_json_routed(system, user) -> json string
"""
import os, logging
from env_util import load_env
load_env()

import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

from litellm import Router
from llm_client import JSON_ENFORCE_SUFFIX, _extract_json


# ── Routing policy (priority order) ───────────────────────────────────────────
def _model_list():
    return [
        {"model_name": "qa-primary",
         "litellm_params": {"model": "mistral/mistral-small-latest",
                            "api_key": os.environ.get("MISTRAL_API_KEY")}},
        {"model_name": "qa-fallback",
         "litellm_params": {"model": "sambanova/Meta-Llama-3.3-70B-Instruct",
                            "api_key": os.environ.get("SAMBANOVA_API_KEY")}},
        {"model_name": "qa-safety",
         "litellm_params": {"model": "github/gpt-4o-mini",
                            "api_key": os.environ.get("GITHUB_TOKEN")}},
    ]

# strict priority: primary -> fallback -> safety
FALLBACKS = [{"qa-primary": ["qa-fallback", "qa-safety"]}]

_router = None


def get_router():
    global _router
    if _router is None:
        _router = Router(
            model_list=_model_list(),
            fallbacks=FALLBACKS,
            num_retries=2,          # retries on the same tier before falling back
            timeout=90,             # per-request timeout (s)
            cooldown_time=60,       # park a failing deployment for 60s
            retry_after=2,
        )
    return _router


def chat_json_routed(system, user, temperature=0.1, max_tokens=4000,
                     return_meta=False):
    """
    Routed JSON completion: tries primary, transparently falls back on failure.
    Returns the JSON string (extracted from any markdown fences). With
    return_meta=True, returns (json_str, served_model_string).
    """
    router = get_router()
    messages = [
        {"role": "system", "content": system + JSON_ENFORCE_SUFFIX},
        {"role": "user",   "content": user},
    ]
    resp = router.completion(
        model="qa-primary",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content or ""
    out = _extract_json(raw)
    if return_meta:
        return out, getattr(resp, "model", "?")
    return out


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, time, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-fallback", action="store_true",
                    help="break the primary to prove fallback fires")
    args = ap.parse_args()

    SYS = "You are a helpful assistant."
    USR = 'Return JSON: {"status": "ok", "n": 42}'

    if not args.test_fallback:
        print("\n[1] Normal routed call (should serve from PRIMARY = mistral):")
        t0 = time.time()
        out, served = chat_json_routed(SYS, USR, return_meta=True)
        print(f"    served by : {served}")
        print(f"    latency   : {int((time.time()-t0)*1000)}ms")
        print(f"    parsed    : {json.loads(out)}")
    else:
        print("\n[2] Fallback test: primary deliberately broken ->")
        broken = Router(
            model_list=[
                {"model_name": "qa-primary",
                 "litellm_params": {"model": "mistral/mistral-small-latest",
                                    "api_key": "sk-DELIBERATELY-BROKEN"}},
                {"model_name": "qa-fallback",
                 "litellm_params": {"model": "sambanova/Meta-Llama-3.3-70B-Instruct",
                                    "api_key": os.environ.get("SAMBANOVA_API_KEY")}},
                {"model_name": "qa-safety",
                 "litellm_params": {"model": "github/gpt-4o-mini",
                                    "api_key": os.environ.get("GITHUB_TOKEN")}},
            ],
            fallbacks=[{"qa-primary": ["qa-fallback", "qa-safety"]}],
            num_retries=1, timeout=90,
        )
        t0 = time.time()
        resp = broken.completion(
            model="qa-primary",
            messages=[{"role": "system", "content": SYS + JSON_ENFORCE_SUFFIX},
                      {"role": "user", "content": USR}],
            max_tokens=30)
        served = getattr(resp, "model", "?")
        print(f"    primary failed -> served by FALLBACK: {served}")
        print(f"    latency   : {int((time.time()-t0)*1000)}ms")
        print(f"    parsed    : {json.loads(_extract_json(resp.choices[0].message.content))}")
