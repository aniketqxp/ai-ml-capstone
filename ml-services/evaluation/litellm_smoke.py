"""Smoke-test the 3 finalist models THROUGH litellm.completion()."""
import os, time, logging
from env_util import load_env
load_env()

import litellm
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from litellm import completion

# (label, model string, extra kwargs)
TESTS = [
    ("mistral",   "mistral/mistral-small-latest",
        {"api_key": os.environ["MISTRAL_API_KEY"]}),
    ("sambanova", "sambanova/Meta-Llama-3.3-70B-Instruct",
        {"api_key": os.environ["SAMBANOVA_API_KEY"]}),
    # GitHub Models -- try native 'github/' provider first
    ("github(native)", "github/gpt-4o-mini",
        {"api_key": os.environ["GITHUB_TOKEN"]}),
    # GitHub Models -- openai-compatible fallback path
    ("github(openai)", "openai/gpt-4o-mini",
        {"api_key": os.environ["GITHUB_TOKEN"],
         "api_base": "https://models.github.ai/inference"}),
]

for label, model, kwargs in TESTS:
    t0 = time.time()
    try:
        resp = completion(
            model=model,
            messages=[{"role": "user",
                       "content": "Reply with JSON: {\"ok\": true}"}],
            max_tokens=30,
            **kwargs,
        )
        content = resp.choices[0].message.content
        ms = int((time.time() - t0) * 1000)
        print(f"  [OK]  {label:<16} {model:<42} {ms:>6}ms  -> {repr(content)[:50]}")
    except Exception as e:
        print(f"  [XX]  {label:<16} {model:<42}  {str(e)[:90]}")
