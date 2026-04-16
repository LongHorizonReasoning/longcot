#!/usr/bin/env python3
"""
Run LLM inference on benchmark questions in parallel and save results.

Usage (run from repo root):
    python run_inference.py --config oai_gpt52
    python run_inference.py --domain math --config oai_gpt52
    python run_inference.py --domain logic --difficulty hard --config or_deepseek
    python run_inference.py --difficulty longcot --config anthropic_sonnet --max-questions 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

# Support running both with the installed `longcot` package (via `uv sync`)
# and directly from the repo root without installation.
try:
    from longcot._llm import LLMProvider, call_with_retry, create_provider
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from _llm import LLMProvider, call_with_retry, create_provider  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "src" / "data"
CONFIGS_DIR = REPO_ROOT / "src" / "configs"

DOMAINS = ("logic", "cs", "chemistry", "chess", "math")
DIFFICULTIES = ("easy", "medium", "hard")
# Benchmark aliases: "longcot-mini" == easy, "longcot" == medium + hard.
DIFFICULTY_CHOICES = (*DIFFICULTIES, "longcot-mini", "longcot")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_questions(
    domain: str | None = None,
    difficulty: str | None = None,
    max_questions: int | None = None,
) -> List[Dict[str, Any]]:
    """Load questions, optionally filtered by domain and/or difficulty."""
    domains = [domain] if domain else list(DOMAINS)
    if difficulty == "longcot":
        diffs = ["medium", "hard"]
    elif difficulty == "longcot-mini":
        diffs = ["easy"]
    elif difficulty:
        diffs = [difficulty]
    else:
        diffs = list(DIFFICULTIES)
    questions: List[Dict[str, Any]] = []
    for dom in domains:
        for diff in diffs:
            path = DATA_DIR / dom / f"{diff}.json"
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for q in data.get("questions", []):
                if "prompt" not in q:
                    continue
                questions.append(q)
    if not questions:
        raise ValueError(f"No questions found for domain={domain!r}, difficulty={difficulty!r}")
    if max_questions is not None:
        questions = questions[:max_questions]
    return questions


def find_config(name: str) -> Path:
    """Resolve a config name to a path in src/configs/."""
    candidates = [CONFIGS_DIR / name, CONFIGS_DIR / f"{name}.yaml"]
    for p in candidates:
        if p.exists():
            return p
    available = [f.stem for f in CONFIGS_DIR.glob("*.yaml")]
    raise FileNotFoundError(
        f"Config {name!r} not found in {CONFIGS_DIR}. Available: {', '.join(sorted(available))}"
    )


def load_config(path: Path, *, resolve_env: bool = True) -> Dict[str, Any]:
    """Load a YAML config file, optionally resolving ${ENV_VAR} placeholders."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _resolve_env(cfg) if resolve_env else cfg


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        v = value.strip()
        if v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1].strip()
            resolved = os.environ.get(env_var)
            if not resolved:
                raise ValueError(f"Environment variable {env_var} is not set")
            return resolved
        return value
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Core parallel runner
# ---------------------------------------------------------------------------

def call_one(
    provider: LLMProvider,
    question: Dict[str, Any],
    *,
    max_retries: int,
    retry_timeouts: bool,
    llm_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Call the provider for a single question and return a result dict."""
    qid = question.get("question_id", "unknown")
    prompt = question["prompt"]

    resp, errors, attempts = call_with_retry(
        provider,
        prompt,
        max_retries=max_retries,
        retry_timeouts=retry_timeouts,
        **llm_kwargs,
    )

    if resp is None:
        return {
            "question_id": qid,
            "successful": False,
            "attempts": attempts,
            "errors": errors,
        }

    result = {
        "question_id": qid,
        "successful": True,
        "attempts": attempts,
        "response_text": resp.content,
        "model": resp.model,
        "usage": resp.usage,
    }
    if resp.reasoning:
        result["reasoning"] = resp.reasoning
    return result


def run_parallel(
    provider: LLMProvider,
    questions: List[Dict[str, Any]],
    *,
    num_workers: int = 8,
    max_retries: int = 2,
    retry_timeouts: bool = False,
    llm_kwargs: Dict[str, Any],
    output_path: str,
    stop_on_codes: tuple[int, ...] = (401, 402, 403),
    stop_threshold: int = 3,
) -> str:
    """Run inference in parallel, writing results to JSONL as they complete."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = len(questions)
    idx = 0
    completed = 0
    consecutive_fatal = 0
    stop_submitting = False

    in_flight: Set[Any] = set()
    future_to_q: Dict[Any, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor, open(out, "w", encoding="utf-8") as f:

        def submit(q: Dict[str, Any]):
            fut = executor.submit(
                call_one, provider, q,
                max_retries=max_retries,
                retry_timeouts=retry_timeouts,
                llm_kwargs=llm_kwargs,
            )
            in_flight.add(fut)
            future_to_q[fut] = q
            return fut

        # Seed initial batch
        while idx < total and len(in_flight) < num_workers:
            submit(questions[idx])
            idx += 1

        while in_flight:
            done_futures, _ = wait(in_flight, timeout=0.5, return_when=FIRST_COMPLETED)
            if not done_futures:
                continue

            for fut in done_futures:
                in_flight.remove(fut)
                q = future_to_q.pop(fut, {})
                result = fut.result()

                f.write(json.dumps(result) + "\n")
                f.flush()
                completed += 1

                qid = result.get("question_id", "?")
                ok = result.get("successful", False)
                status = "ok" if ok else "FAIL"
                print(f"  [{completed}/{total}] {qid}: {status}")

                # Check for fatal status codes
                errors = result.get("errors", [])
                fatal = any(
                    e.get("status_code") in stop_on_codes
                    for e in errors
                    if isinstance(e.get("status_code"), int)
                )
                if fatal:
                    consecutive_fatal += 1
                    if consecutive_fatal >= stop_threshold:
                        print(f"Stopping: {consecutive_fatal} consecutive fatal errors (codes {stop_on_codes})")
                        stop_submitting = True
                elif ok:
                    consecutive_fatal = 0

                # Replenish
                while not stop_submitting and idx < total and len(in_flight) < num_workers:
                    submit(questions[idx])
                    idx += 1

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    available_configs = [f.stem for f in sorted(CONFIGS_DIR.glob("*.yaml"))]

    parser = argparse.ArgumentParser(
        description="Run LLM inference on benchmark questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available configs: {', '.join(available_configs)}
Available domains: {', '.join(DOMAINS)}

Examples:
  python run_inference.py --config oai_gpt52                               # all domains, all difficulties
  python run_inference.py --domain math --config oai_gpt52                 # one domain, all difficulties
  python run_inference.py --difficulty longcot --config or_deepseek        # full LongCoT (medium + hard)
  python run_inference.py --difficulty longcot-mini --config or_deepseek   # LongCoT-Mini (easy)
  python run_inference.py --domain logic --difficulty hard --config or_deepseek
""",
    )
    parser.add_argument("--domain", choices=DOMAINS, help="Question domain (default: all)")
    parser.add_argument("--difficulty", choices=DIFFICULTY_CHOICES, help="Difficulty filter (default: all)")
    parser.add_argument("--config", required=True, help="Config name (from src/configs/)")
    parser.add_argument("--max-questions", type=int, help="Only use the first N questions")
    parser.add_argument("--output", help="Output JSONL path (default: responses/<domain>_<difficulty>_<model>_<ts>.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be submitted without calling the API")

    args = parser.parse_args()

    # --- Load config ---
    config_path = find_config(args.config)
    cfg = load_config(config_path, resolve_env=not args.dry_run)

    provider_name: str = cfg["provider"]
    model: str = cfg["model"]
    api_key = cfg.get("api_key")
    timeout = cfg.get("timeout", 900.0)
    num_workers = cfg.get("num_workers", 8)
    max_retries = cfg.get("max_retries", 2)
    retry_timeouts = cfg.get("retry_timeouts", False)
    llm_kwargs: Dict[str, Any] = dict(cfg.get("llm_kwargs") or {})
    headers = cfg.get("headers")

    # --- Load questions ---
    questions = load_questions(args.domain, args.difficulty, args.max_questions)
    domain_label = args.domain or "all"
    diff_label = args.difficulty or "all"
    print(f"Loaded {len(questions)} questions ({domain_label}/{diff_label})")

    # --- Output path ---
    if args.output:
        output_path = args.output
    else:
        model_slug = model.replace("/", "_").replace(":", "_")
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"responses/{domain_label}_{diff_label}_{model_slug}_{ts}.jsonl"

    # --- Dry run ---
    if args.dry_run:
        print(f"\nDry run — would submit {len(questions)} questions")
        print(f"  Config:      {config_path.stem}")
        print(f"  Provider:    {provider_name}")
        print(f"  Model:       {model}")
        print(f"  Workers:     {num_workers}")
        print(f"  Max retries: {max_retries}")
        print(f"  Output:      {output_path}")
        print(f"  LLM kwargs:  {json.dumps(llm_kwargs, indent=2)}")
        print(f"\nFirst question ID: {questions[0].get('question_id', '?')}")
        print(f"Last question ID:  {questions[-1].get('question_id', '?')}")
        return

    # --- Create provider ---
    provider = create_provider(
        provider_name,
        model=model,
        api_key=api_key,
        timeout=timeout,
        headers=headers,
    )

    # --- Run ---
    print(f"\nConfig:      {config_path.stem}")
    print(f"Provider:    {provider_name}")
    print(f"Model:       {model}")
    print(f"Workers:     {num_workers}")
    print(f"Max retries: {max_retries}")
    print(f"Output:      {output_path}")
    print()

    start = time.time()

    run_parallel(
        provider,
        questions,
        num_workers=num_workers,
        max_retries=max_retries,
        retry_timeouts=retry_timeouts,
        llm_kwargs=llm_kwargs,
        output_path=output_path,
    )

    elapsed = time.time() - start

    # --- Summary ---
    results = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    total = len(results)
    succeeded = sum(1 for r in results if r.get("successful"))
    failed = total - succeeded

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total:     {total}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print(f"  Output:    {output_path}")


if __name__ == "__main__":
    main()
