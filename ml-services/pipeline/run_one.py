"""
CLI: run the single-call orchestrator for one call_id from the manifest.

  python run_one.py --call_id en_CA_Aviation_1586888            # env ENABLE_ACOUSTIC decides
  python run_one.py --call_id en_CA_Aviation_1586888 --no-acoustic
  python run_one.py --call_id en_CA_Aviation_1586888 --acoustic

Builds the CallSpec from the manifest (base + runtime) and streams stage
timings. The heavy lifting lives in orchestrator.process_call.
"""
import sys
import json
import time
import argparse
from pathlib import Path

_ML = Path(__file__).resolve().parents[1]
for _p in (Path(__file__).resolve().parent, _ML / "evaluation", _ML / "scripts", _ML):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                       # noqa: E402
from assemble import load_manifest  # noqa: E402
from orchestrator import process_call  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call_id", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--acoustic", dest="acoustic", action="store_true",
                   help="force the acoustic stage on")
    g.add_argument("--no-acoustic", dest="acoustic", action="store_false",
                   help="force the acoustic stage off")
    ap.set_defaults(acoustic=None)   # None -> ENABLE_ACOUSTIC env decides
    args = ap.parse_args()

    manifest = load_manifest()
    if args.call_id not in manifest:
        raise SystemExit(f"call_id '{args.call_id}' not in manifest")
    m = manifest[args.call_id]
    spec = {
        "call_id": args.call_id,
        "domain": m["domain"],
        "accent": m["accent"],
        "agent_wav": str(paths.NA_TESTSET / m["agent_wav"].replace("\\", "/")),
        "customer_wav": str(paths.NA_TESTSET / m["customer_wav"].replace("\\", "/")),
    }

    t0 = time.time()
    print(f"Processing {args.call_id} "
          f"(domain={m['domain']}, accent={m['accent']}, acoustic={args.acoustic})")
    print("-" * 68)

    def progress(stage):
        print(f"  [{time.time() - t0:6.1f}s] {stage}")

    artifacts = process_call(spec, progress=progress, enable_acoustic=args.acoustic)

    print("-" * 68)
    print(f"Done in {time.time() - t0:.1f}s")
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
