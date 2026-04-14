from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

from llm_client import OllamaClient
from parse_output import parse_with_retry


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Small real-inference smoke test for 03_LLM")
    parser.add_argument("--prompts", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "prompts_en_v2.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--num-predict", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "cache")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    rows = _read_jsonl(args.prompts)
    if not rows:
        raise SystemExit(f"No records found in {args.prompts}")
    rows = rows[: max(1, args.max_rows)]

    client = OllamaClient(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        num_predict=args.num_predict,
        cache_dir=args.cache_dir,
    )

    started = time.time()
    out = []
    ok = 0
    for i, rec in enumerate(rows, start=1):
        prompt = str(rec["prompt"])

        def _infer_once(prompt_text: str):
            r = client.generate(prompt_text)
            return r.text, r.latency_seconds, r.from_cache

        parsed = parse_with_retry(prompt, _infer_once, max_retries=args.max_retries)
        parse_ok = bool(parsed["parse_ok"])
        if parse_ok:
            ok += 1
        out.append(
            {
                "row_index": int(rec["row_index"]),
                "prompt_version": str(rec.get("prompt_version", "")),
                "bRate": float(rec["bRate"]),
                "pred_bRate": float(parsed["pred_brate"]) if parse_ok else None,
                "parse_ok": parse_ok,
                "retry_count": int(parsed["retry_count"]),
                "raw_completion": str(parsed["raw_completion"]),
                "latency_seconds": float(parsed["latency_seconds"]),
            }
        )
        if i == 1 or i % max(1, args.log_every) == 0 or i == len(rows):
            elapsed = time.time() - started
            print(f"[smoke] {i}/{len(rows)} | parse_ok={ok}/{i} | elapsed={elapsed:.1f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("/", "_").replace(":", "_")
    out_path = args.output_dir / f"smoke_real_infer_{model_tag}_{ts}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - started
    print(f"[smoke] done | rows={len(rows)} parse_success_rate={ok/len(rows):.3f} elapsed={elapsed:.1f}s")
    print(f"[smoke] output -> {out_path}")


if __name__ == "__main__":
    main()
