from env_util import load_env; load_env()
import os, time
from openai import OpenAI

c = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"])

for model in [
    "meta/llama-3.1-8b-instruct",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "mistralai/mistral-7b-instruct-v0.3",
    "microsoft/phi-3-mini-4k-instruct",
]:
    try:
        t0 = time.time()
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=30
        )
        content = (r.choices[0].message.content or "").strip()
        lat = int((time.time() - t0) * 1000)
        print(f"OK   {model:<45} {lat}ms  ->  {content[:80]}")
        break
    except Exception as e:
        print(f"FAIL {model}: {str(e)[:100]}")
