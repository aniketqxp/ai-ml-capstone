"""Test all API keys in .env and report status."""
import os, urllib.request, json
from env_util import load_env

load_env()
results = {}

# 1. OpenAI
try:
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = c.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role":"user","content":"Say OK"}], max_tokens=5)
    results["OPENAI_API_KEY"] = ("OK", r.choices[0].message.content.strip())
except Exception as e:
    results["OPENAI_API_KEY"] = ("FAIL", str(e)[:120])

# 2. GitHub Models
try:
    from openai import OpenAI
    c = OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
    r = c.chat.completions.create(model="openai/gpt-4o-mini",
        messages=[{"role":"user","content":"Say OK"}], max_tokens=5)
    results["GITHUB_TOKEN"] = ("OK", r.choices[0].message.content.strip())
except Exception as e:
    results["GITHUB_TOKEN"] = ("FAIL", str(e)[:120])

# 3. Gemini
try:
    key = os.environ["GEMINI_API_KEY"]
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "gemini-2.0-flash:generateContent?key=" + key)
    body = json.dumps({"contents":[{"parts":[{"text":"Say OK"}]}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        d = json.loads(resp.read())
    reply = d["candidates"][0]["content"]["parts"][0]["text"].strip()[:20]
    results["GEMINI_API_KEY"] = ("OK", reply)
except Exception as e:
    results["GEMINI_API_KEY"] = ("FAIL", str(e)[:120])

# 4. HuggingFace (HUGGINGFACE_API_TOKEN and HF_TOKEN are the same value -- test once)
try:
    tok = os.environ["HUGGINGFACE_API_TOKEN"]
    req = urllib.request.Request("https://huggingface.co/api/whoami",
        headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=10) as resp:
        d = json.loads(resp.read())
    info = f"user={d.get('name','?')}"
    results["HUGGINGFACE_API_TOKEN"] = ("OK", info)
    # HF_TOKEN is identical value -- just confirm and don't double-test
    same = os.environ.get("HF_TOKEN") == tok
    results["HF_TOKEN"] = ("OK (same value)" if same else "OK", "same as HUGGINGFACE_API_TOKEN" if same else "")
except Exception as e:
    results["HUGGINGFACE_API_TOKEN"] = ("FAIL", str(e)[:120])
    results["HF_TOKEN"] = ("FAIL", "same key, same result")

# 5. OpenRouter
try:
    from openai import OpenAI
    c = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    r = c.chat.completions.create(model="meta-llama/llama-3.2-3b-instruct:free",
        messages=[{"role":"user","content":"Say OK"}], max_tokens=5)
    results["OPENROUTER_API_KEY"] = ("OK", r.choices[0].message.content.strip())
except Exception as e:
    results["OPENROUTER_API_KEY"] = ("FAIL", str(e)[:120])

print()
for k, (status, detail) in results.items():
    icon = "v" if "OK" in status else "X"
    print(f"  [{icon}] {k:<30} {status:<20} {detail}")
