"""Question loading from bundled data files."""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

from ._types import Question

DOMAINS = ("logic", "cs", "chemistry", "chess", "math")
DIFFICULTIES = ("easy", "medium", "hard")


def list_domains() -> list[str]:
    """Return the list of available domains."""
    return list(DOMAINS)


def load_questions(
    *,
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> list[Question]:
    """Load questions, optionally filtered by domain and/or difficulty.

    Args:
        domain: One of 'logic', 'cs', 'chemistry', 'chess', 'math'. If None, loads all.
        difficulty: One of 'easy', 'medium', 'hard'. If None, loads all.

    Returns:
        List of Question objects.
    """
    domains = [domain] if domain else list(DOMAINS)
    diffs = [difficulty] if difficulty else list(DIFFICULTIES)

    questions: list[Question] = []
    for d in domains:
        for diff in diffs:
            questions.extend(_load_file(d, diff))
    return questions


def _load_file(domain: str, difficulty: str) -> list[Question]:
    """Load questions from a single data/{domain}/{difficulty}.json file."""
    try:
        data_pkg = resources.files("longcot") / "data" / domain / f"{difficulty}.json"
        text = data_pkg.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError):
        return []

    data = json.loads(text)
    questions = []
    for q in data.get("questions", []):
        questions.append(
            Question(
                question_id=str(q["question_id"]),
                domain=domain,
                difficulty=difficulty,
                prompt=q["prompt"],
                problem=q.get("problem"),
                answer=q.get("answer"),
            )
        )
    return questions
