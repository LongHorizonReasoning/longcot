#!/usr/bin/env python3
"""
Evaluate LLM responses against benchmark answers.

Usage (run from repo root):
    python run_eval.py responses/math_hard_gpt-5.2_20260226.jsonl
    python run_eval.py responses/all_all_model_ts.jsonl --no-fallback
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "src" / "data"

# Import verification pipeline from longcot package.
# If the package isn't installed, register src/ as the longcot package
# so that relative imports within the verification modules work.
try:
    from longcot._verifier import verify
    from longcot._parsing import extract_solution
    from longcot._types import (
        ChemistryVerifyOptions,
        MathVerifyOptions,
        Question,
        VerifyOptions,
    )
except ImportError:
    _src = REPO_ROOT / "src"
    if not _src.is_dir():
        print("Error: longcot package not found. Run: uv sync", file=sys.stderr)
        sys.exit(1)
    import types as _types_mod
    _pkg = _types_mod.ModuleType("longcot")
    _pkg.__path__ = [str(_src)]
    sys.modules["longcot"] = _pkg
    from longcot._verifier import verify  # type: ignore[no-redef]
    from longcot._parsing import extract_solution  # type: ignore[no-redef]
    from longcot._types import (  # type: ignore[no-redef]
        ChemistryVerifyOptions,
        MathVerifyOptions,
        Question,
        VerifyOptions,
    )

DOMAINS = ("logic", "cs", "chemistry", "chess", "math")
DIFFICULTIES = ("easy", "medium", "hard")


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------

def load_all_questions() -> Dict[str, Question]:
    """Load every question from src/data/ into a {question_id: Question} lookup."""
    questions: Dict[str, Question] = {}
    for domain in DOMAINS:
        domain_dir = DATA_DIR / domain
        if not domain_dir.is_dir():
            continue
        for diff in DIFFICULTIES:
            path = domain_dir / f"{diff}.json"
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for q in data.get("questions", []):
                qid = str(q["question_id"])
                questions[qid] = Question(
                    question_id=qid,
                    domain=domain,
                    difficulty=diff,
                    prompt=q["prompt"],
                    problem=q.get("problem"),
                    answer=q.get("answer"),
                )
    return questions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    responses_path: str,
    *,
    verify_options: VerifyOptions | None = None,
) -> Dict[str, Any]:
    """Evaluate a responses JSONL file and return a results dict."""
    questions = load_all_questions()

    responses: list[Dict[str, Any]] = []
    with open(responses_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))

    correct = 0
    incorrect = 0
    failed = 0
    wrong_formatting = 0
    details: list[Dict[str, Any]] = []

    for resp in responses:
        qid = resp.get("question_id", "unknown")
        successful = resp.get("successful", False)

        # Category 1: API / submission failure
        if not successful:
            failed += 1
            details.append({"question_id": qid, "status": "failed"})
            continue

        response_text = resp.get("response_text", "")

        # Look up the question for verification
        question = questions.get(qid)
        if question is None:
            print(f"  Warning: question_id {qid!r} not found in data, counting as failed")
            failed += 1
            details.append({"question_id": qid, "status": "failed", "reason": "missing_question"})
            continue

        # Formatting check: missing `solution =` is counted, but verification still runs.
        has_wrong_formatting = extract_solution(response_text) is None
        if has_wrong_formatting:
            wrong_formatting += 1

        # Verify correctness
        try:
            is_correct = verify(question, response_text, options=verify_options)
        except Exception as e:
            incorrect += 1
            detail = {"question_id": qid, "status": "incorrect", "error": str(e)[:200]}
            if has_wrong_formatting:
                detail["wrong_formatting"] = True
            details.append(detail)
            continue

        if is_correct:
            correct += 1
            detail = {"question_id": qid, "status": "correct"}
            if has_wrong_formatting:
                detail["wrong_formatting"] = True
            details.append(detail)
        else:
            incorrect += 1
            detail = {"question_id": qid, "status": "incorrect"}
            if has_wrong_formatting:
                detail["wrong_formatting"] = True
            details.append(detail)

    total = len(responses)
    verified = correct + incorrect

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "failed": failed,
        "wrong_formatting": wrong_formatting,
        "accuracy": correct / verified if verified > 0 else 0.0,
        "overall_accuracy": correct / total if total > 0 else 0.0,
        "details": details,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM responses against benchmark answers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_eval.py responses/math_hard_gpt-5.2_20260226.jsonl
  python run_eval.py responses/all_all_model_ts.jsonl --no-fallback
""",
    )
    parser.add_argument("responses", help="Path to responses JSONL file")
    parser.add_argument("--output", help="Output JSON path (default: results/<name>.json)")
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="Disable math/chemistry fallback judges (avoids needing GEMINI_API_KEY)",
    )

    args = parser.parse_args()

    verify_options = None
    if args.no_fallback:
        verify_options = VerifyOptions(
            math=MathVerifyOptions(enable_fallback=False),
            chemistry=ChemistryVerifyOptions(enable_fallback=False),
        )

    # --- Warn about math fallback ---
    if not args.no_fallback:
        has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        if not has_key:
            print(
                "Warning: GEMINI_API_KEY / GOOGLE_API_KEY not set. "
                "Math/chemistry fallback judges are unavailable without an API key.\n"
                "  Set the env var, or pass --no-fallback to silence this warning.\n"
            )

    # --- Evaluate ---
    print(f"Evaluating: {args.responses}")
    results = evaluate(args.responses, verify_options=verify_options)

    # --- Output path ---
    if args.output:
        output_path = args.output
    else:
        name = Path(args.responses).stem
        output_path = f"results/{name}.json"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # --- Summary ---
    print(f"  Total:            {results['total']}")
    print(f"  Correct:          {results['correct']}")
    print(f"  Incorrect:        {results['incorrect']}")
    print(f"  Failed:           {results['failed']}")
    print(f"  Wrong formatting: {results['wrong_formatting']}")
    print(f"  Accuracy:         {results['accuracy']:.1%}")
    print(f"  Overall accuracy: {results['overall_accuracy']:.1%}")
    print(f"  Saved to:         {output_path}")


if __name__ == "__main__":
    main()
