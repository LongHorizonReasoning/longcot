"""Verification for Random Hanoi puzzles."""

from __future__ import annotations

from ..._parsing import parse_list_solution
from ..._types import Question
from .._regexes import MOVE_LIST_PATTERN


def verify_output(lm_text: str, question: Question) -> bool:
    moves = _parse_moves(lm_text)
    if not moves:
        return False

    instance = question.problem["instance"]
    initial_state = instance["initial_state"]
    goal_state = instance["goal_state"]
    return _simulate(initial_state, goal_state, moves)


def _parse_moves(lm_text: str) -> list[list[int]]:
    moves = parse_list_solution(lm_text, fallback_pattern=MOVE_LIST_PATTERN)
    if moves is None:
        return []
    for move in moves:
        if not isinstance(move, list) or len(move) != 3:
            return []
        if not all(isinstance(x, int) for x in move):
            return []
    return moves


def _simulate(initial: list[list[int]], goal: list[list[int]], moves: list[list[int]]) -> bool:
    state = [stack.copy() for stack in initial]
    for block, from_stack, to_stack in moves:
        if not _valid_move(state, block, from_stack, to_stack):
            return False
        state[to_stack].append(state[from_stack].pop())
    return state == goal


def _valid_move(state: list[list[int]], block: int, from_s: int, to_s: int) -> bool:
    if from_s < 0 or from_s >= len(state) or to_s < 0 or to_s >= len(state):
        return False
    if not state[from_s] or state[from_s][-1] != block:
        return False
    # Hanoi constraint: can't place larger disc on smaller one
    if state[to_s] and state[to_s][-1] < block:
        return False
    if from_s == to_s:
        return False
    return True
