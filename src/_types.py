from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class Question:
    """A single evaluation question."""

    question_id: str
    domain: str
    difficulty: str
    prompt: str
    problem: Optional[dict[str, Any]] = None
    answer: Optional[Any] = None


@dataclass(frozen=True)
class MathFallbackResult:
    """Structured result returned by a math fallback judge."""

    equivalent: bool
    reason: str


class MathFallbackJudge(Protocol):
    """Callable protocol for math fallback adjudication."""

    def __call__(
        self,
        *,
        question: Question,
        expected: str,
        predicted: str,
        model: str,
        timeout_s: float,
    ) -> MathFallbackResult:
        ...


@dataclass(frozen=True)
class MathVerifyOptions:
    """Math verifier configuration."""

    enable_fallback: bool = True
    fallback_model: str = "gemini-3-flash-preview"
    fallback_timeout_s: float = 20.0
    fallback_max_retries: int = 2
    fallback_judge: Optional[MathFallbackJudge] = None


@dataclass(frozen=True)
class ChemistryFallbackResult:
    """Structured result returned by a chemistry fallback extractor."""

    smiles: Optional[str]
    reason: str


class ChemistryFallbackJudge(Protocol):
    """Callable protocol for chemistry fallback extraction."""

    def __call__(
        self,
        *,
        question: Question,
        response: str,
        model: str,
        timeout_s: float,
    ) -> ChemistryFallbackResult:
        ...


@dataclass(frozen=True)
class ChemistryVerifyOptions:
    """Chemistry verifier configuration."""

    enable_fallback: bool = True
    fallback_model: str = "gemini-3.1-flash"
    fallback_timeout_s: float = 20.0
    fallback_max_retries: int = 2
    fallback_judge: Optional[ChemistryFallbackJudge] = None


@dataclass(frozen=True)
class VerifyOptions:
    """Optional configuration for verification."""

    math: MathVerifyOptions = field(default_factory=MathVerifyOptions)
    chemistry: ChemistryVerifyOptions = field(default_factory=ChemistryVerifyOptions)
