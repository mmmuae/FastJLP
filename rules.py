#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rules engine for next-valid-number construction under digit-pattern constraints.

API:
    next_valid_ge(start: int, lo: int, hi: int) -> int | None

Constraints (overlapping windows):
1) No 5 identical digits in a row anywhere.
2) Each 3-digit pattern (e.g., '456') appears at most 3 times.
3) Each strictly sequential 5-digit run (increasing like 12345 or decreasing like 98765)
   appears at most 2 times.
4) Each 5-digit palindrome (e.g., 12321) appears at most 2 times.

Design:
- Tight-bound digit-by-digit DFS constructs the lexicographically-smallest valid number
  >= start within [lo..hi]. Fixed-size integer arrays for counters avoid hash overhead.

Assumptions:
- Integers are non-negative; lo <= hi. Bounds are inclusive.
- Width = len(str(hi)); we left-pad numbers internally to that width for DP only.
"""

from __future__ import annotations
from typing import List, Optional

__all__ = ["next_valid_ge"]

def _digits_of(n: int, width: int) -> List[int]:
    s = str(n)
    if len(s) > width:
        raise ValueError("number does not fit in the specified width")
    out = [0] * width
    k = width - len(s)
    for i, ch in enumerate(s):
        out[k + i] = ord(ch) - 48  # '0'..'9' -> 0..9
    return out

def next_valid_ge(start: int, lo: int, hi: int) -> Optional[int]:
    """
    Return the smallest integer N such that lo <= N <= hi and N >= start
    that satisfies all rules, or None if no such number exists.
    """
    if lo > hi:
        raise ValueError("invalid bounds: lo > hi")
    if start < lo:
        start = lo
    if start > hi:
        return None

    width = len(str(hi))
    lo_digits = _digits_of(start, width)
    hi_digits = _digits_of(hi, width)

    # Pattern counters (overlapping windows)
    cnt3 = [0] * 1000          # index abc (0..999)
    cnt5_seq = [0] * 100000    # index abcde (0..99999)
    cnt5_pal = [0] * 100000    # index abcde (0..99999)

    res = [-1] * width  # output buffer

    def dfs(pos: int, tight_lo: bool, tight_hi: bool, last_digit: int, run_len: int) -> bool:
        if pos == width:
            return True

        low_d = lo_digits[pos] if tight_lo else 0
        high_d = hi_digits[pos] if tight_hi else 9

        for d in range(low_d, high_d + 1):
            # Rule 1: no 5 identical digits in a row
            new_run = run_len + 1 if d == last_digit else 1
            if new_run >= 5:
                continue

            inc3 = inc5s = inc5p = False
            id3 = id5 = None

            # Rule 2: 3-digit pattern quota
            if pos >= 2:
                a, b = res[pos - 2], res[pos - 1]
                id3 = a * 100 + b * 10 + d
                if cnt3[id3] >= 3:  # would be 4th occurrence
                    continue

            inc = dec = pal = False
            # Rules 3 & 4 on 5-digit window
            if pos >= 4:
                a, b, c, e = res[pos - 4], res[pos - 3], res[pos - 2], res[pos - 1]
                inc = (b == a + 1) and (c == b + 1) and (e == c + 1) and (d == e + 1)
                dec = (b == a - 1) and (c == b - 1) and (e == c - 1) and (d == e - 1)
                pal = (a == d) and (b == e)
                id5 = (((a * 10 + b) * 10 + c) * 10 + e) * 10 + d
                if (inc or dec) and cnt5_seq[id5] >= 2:
                    continue
                if pal and cnt5_pal[id5] >= 2:
                    continue

            # Apply counter updates
            if id3 is not None:
                cnt3[id3] += 1; inc3 = True
            if id5 is not None:
                if inc or dec:
                    cnt5_seq[id5] += 1; inc5s = True
                if pal:
                    cnt5_pal[id5] += 1; inc5p = True

            res[pos] = d
            if dfs(pos + 1,
                   tight_lo and (d == lo_digits[pos]),
                   tight_hi and (d == hi_digits[pos]),
                   d, new_run):
                return True

            # Roll back
            res[pos] = -1
            if inc3:  cnt3[id3] -= 1
            if inc5s: cnt5_seq[id5] -= 1
            if inc5p: cnt5_pal[id5] -= 1

        return False

    if not dfs(0, True, True, -1, 0):
        return None

    # Convert digits to integer
    out = 0
    for x in res:
        out = out * 10 + x
    return out
