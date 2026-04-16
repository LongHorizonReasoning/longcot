"""Unified verification dispatch."""

from __future__ import annotations

from typing import Callable, Sequence

from ._types import Question, VerifyOptions
from ._verify.logic import (
    _blocksworld, _dungeon, _hanoi, _packaging,
    _sokoban, _sudoku, _trapezoid, _wizards,
)
from ._verify import _chemistry, _chess, _cs, _math

_CHEMISTRY_SMILES_TEMPLATES = {"easy1", "easy2", "med3", "hard3"}
_MATH_TEMPLATES = {"linear", "dag", "dag_first", "conditional", "backtracking"}

# Template name -> verifier(response, question)
_VERIFIERS: dict[str, Callable[[str, Question], bool]] = {
    # Logic
    "BlocksWorld": _blocksworld.verify_output,
    "Dungeon": _dungeon.verify_output,
    "PackagingMinWaste": _packaging.verify_output,
    "RandomHanoi": _hanoi.verify_output,
    "Sokoban": _sokoban.verify_output,
    "Sudoku": _sudoku.verify_output,
    "TrapezoidCounting": _trapezoid.verify_output,
    "WizardsTotalStrength": _wizards.verify_output,
    # CS
    "HM": _cs.verify_json,
    "MFMC": _cs.verify_json,
    "Scheduling": _cs.verify_json,
    "TM": _cs.verify_json,
    "MCM": _cs.verify_json,
    "LLVM": _cs.verify_json,
    "Backprop": _cs.verify_cs_int_list,
    "DistMem": _cs.verify_cs_int_list,
    "VLIW": _cs.verify_cs_integer,
    "CodeTrace": _cs.verify_cs_integer,
    # Chemistry
    "med1": _chemistry.verify_int_list,
    "med2": _chemistry.verify_int_list,
    "med4": _chemistry.verify_string_list,
    "hard1": _chemistry.verify_int_list,
    "hard2": _chemistry.verify_mixed_list,
    "hard4": _chemistry.verify_string_list,
    # Chess
    "uci_to_fen": _chess.verify_fen,
    "piece_combinations": _chess.verify_chess_integer,
    "reconstruct_moves": _chess.verify_reconstruct_moves,
    "best_3_moves": _chess.verify_chess_moves,
    "best_move": _chess.verify_chess_moves,
    "knight_path": _chess.verify_chess_integer,
    "knight_path_enemy": _chess.verify_chess_integer,
    "knight_game": _chess.verify_chess_integer,
    "max_rooks": _chess.verify_chess_integer,
    "forced_checkmate": _chess.verify_forced_checkmate,
}


def verify(question: Question, response: str, options: VerifyOptions | None = None) -> bool:
    """Verify an LLM response against a question.

    Dispatches to the appropriate verifier based on the ``template`` field
    in ``question.problem``.
    """
    template = (question.problem or {}).get("template")
    if template is None:
        raise ValueError(f"Question {question.question_id} missing problem.template")

    if template in _MATH_TEMPLATES:
        verify_options = options if options is not None else VerifyOptions()
        return _math.verify_math(response, question, verify_options.math)

    if template in _CHEMISTRY_SMILES_TEMPLATES:
        verify_options = options if options is not None else VerifyOptions()
        return _chemistry.verify_smiles(response, question, verify_options.chemistry)

    fn = _VERIFIERS.get(template)
    if fn is None:
        raise ValueError(f"No verifier for template: {template}")

    return fn(response, question)


def verify_batch(
    questions: Sequence[Question],
    responses: Sequence[str],
    options: VerifyOptions | None = None,
) -> list[bool]:
    """Verify a batch of responses. Returns list of booleans."""
    if len(questions) != len(responses):
        raise ValueError(f"Length mismatch: {len(questions)} questions vs {len(responses)} responses")
    return [verify(q, r, options=options) for q, r in zip(questions, responses)]
