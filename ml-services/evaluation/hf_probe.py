from env_util import load_env; load_env()
import os, urllib.request, json, time
from openai import OpenAI

tok = os.environ["HUGGINGFACE_API_TOKEN"]

# Find which providers support this model
url = "https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct?expand[]=inferenceProviderMapping"
req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
with urllib.request.urlopen(req, timeout=10) as r:
    d = json.loads(r.read())

mapping = d.get("inferenceProviderMapping", {})
print("Supported providers:")
for p, info in mapping.items():
    status = info.get("status", "?")
    pid = info.get("providerId", "?")
    print(f"  {p:<20} status={status}  providerId={pid}")

# Try each live provider
print("\nTrying each provider...")
for provider, info in mapping.items():
    if info.get("status") != "live":
        continue
    provider_model = info.get("providerId", "meta-llama/Llama-3.1-8B-Instruct")
    base = f"https://router.huggingface.co/{provider}/v1"
    try:
        c = OpenAI(base_url=base, api_key=tok)
        t0 = time.time()
        r = c.chat.completions.create(
            model=provider_model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=30
        )
        reply = r.choices[0].message.content.strip()
        print(f"  OK  provider={provider}  {int((time.time()-t0)*1000)}ms  -> {reply[:80]}")
        break
    except Exception as e:
        print(f"  FAIL provider={provider}: {str(e)[:100]}")
