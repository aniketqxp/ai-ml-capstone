from env_util import load_env; load_env()
import os, time, urllib.request, json
from openai import OpenAI

tok = os.environ["CLOUDFLARE_API_TOKEN"]

# Step 1: get account ID from the token
try:
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts",
        headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read())
    accounts = d.get("result", [])
    if not accounts:
        print("No accounts found for this token")
        exit(1)
    account_id = accounts[0]["id"]
    account_name = accounts[0]["name"]
    print(f"Account: {account_name} ({account_id})")
except Exception as e:
    print(f"Failed to get account ID: {e}")
    exit(1)

# Step 2: OpenAI-compatible endpoint
base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
c = OpenAI(base_url=base_url, api_key=tok)

for model in [
    "@cf/meta/llama-3.1-8b-instruct",
    "@cf/mistral/mistral-7b-instruct-v0.1",
    "@cf/google/gemma-7b-it",
]:
    try:
        t0 = time.time()
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=50
        )
        content = (r.choices[0].message.content or "").strip()
        print(f"OK   {model}  {int((time.time()-t0)*1000)}ms  ->  {content[:80]}")
        break
    except Exception as e:
        print(f"FAIL {model}: {str(e)[:100]}")
