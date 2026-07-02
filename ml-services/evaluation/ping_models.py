"""Ping the lightest model from each provider with 'Hi' and report results."""
import os, time, urllib.request, json
from env_util import load_env
load_env()

from openai import OpenAI

PROVIDERS = [
    ("OpenAI",      "gpt-4o-mini",                          lambda: OpenAI(api_key=os.environ["OPENAI_API_KEY"])),
    ("GitHub",      "openai/gpt-4o-mini",                   lambda: OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])),
    ("Gemini",      "gemini-2.0-flash",                     None),   # handled separately
    ("OpenRouter",  "meta-llama/llama-3.2-3b-instruct:free",lambda: OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])),
]

results = []

for name, model, client_fn in PROVIDERS:
    if name == "Gemini":
        try:
            t0 = time.time()
            key = os.environ["GEMINI_API_KEY"]
            url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                   "gemini-2.0-flash:generateContent?key=" + key)
            body = json.dumps({"contents":[{"parts":[{"text":"Hi"}]}],
                               "generationConfig":{"maxOutputTokens":30}}).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.loads(r.read())
            reply = d["candidates"][0]["content"]["parts"][0]["text"].strip()
            results.append((name, model, f"{(time.time()-t0)*1000:.0f}ms", reply[:80], "OK"))
        except Exception as e:
            results.append((name, model, "-", str(e)[:60], "FAIL"))
    else:
        try:
            t0 = time.time()
            c = client_fn()
            r = c.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":"Hi"}],
                max_tokens=30, temperature=0.7)
            reply = r.choices[0].message.content.strip()
            results.append((name, model, f"{(time.time()-t0)*1000:.0f}ms", reply[:80], "OK"))
        except Exception as e:
            results.append((name, model, "-", str(e)[:60], "FAIL"))

# Print table
print(f"\n  {'Provider':<12} {'Model':<42} {'Latency':>8}  {'Response'}")
print("  " + "-"*100)
for name, model, lat, reply, status in results:
    icon = "v" if status == "OK" else "X"
    print(f"  [{icon}] {name:<10} {model:<42} {lat:>8}  {reply}")
