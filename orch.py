#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kangaroo CPU orchestrator with entropy picker (coverage-entropy maximizer).

- No .work files; SQLite DB is the source of truth.
- Big integers stored as TEXT to avoid SQLite 64-bit limits.
- Entropy picker (--picker entropy) takes the midpoint of the largest unclaimed
  chunk-interval to smooth coverage and avoid clumping.
- Random and sequential pickers preserved.
- Graceful Ctrl-C: finish the current run, persist results, then exit.

Examples:
  python orch.py --db ranges.db --range-name B27 \
    --min-dec 27000000000000000000000000000000000000000 \
    --max-dec 28000000000000000000000000000000000000000 \
    --pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
    --threads 8 --chunk-bits 48 --picker entropy --stop

  python orch.py --db ranges.db --summary
"""


from __future__ import annotations

import argparse
import atexit
import contextlib
import datetime as _dt
import math
import os
import random
import signal
import sqlite3
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Iterable

from rules import next_valid_ge


# ----------------- Global terminal/signal state -----------------

_ACTIVE_PGID: Optional[int] = None
_CURSOR_HIDDEN: bool = False
_STOP_REQUESTED: bool = False

def _hide_cursor() -> None:
    global _CURSOR_HIDDEN
    if not _CURSOR_HIDDEN:
        try:
            sys.stdout.write("\x1b[?25l"); sys.stdout.flush()
        finally:
            _CURSOR_HIDDEN = True

def _show_cursor() -> None:
    global _CURSOR_HIDDEN
    if _CURSOR_HIDDEN:
        try:
            sys.stdout.write("\x1b[?25h"); sys.stdout.flush()
        finally:
            _CURSOR_HIDDEN = False

atexit.register(_show_cursor)


def _kill_group(sig: int) -> None:
    global _ACTIVE_PGID
    try:
        if _ACTIVE_PGID is not None and os.name != "nt":
            os.killpg(_ACTIVE_PGID, sig)
    except Exception:
        pass

def _on_sigint(signum, frame):
    # Forward ^C to child group; mark stop; restore cursor; and raise KeyboardInterrupt.
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    _show_cursor()
    _kill_group(signal.SIGINT)
    raise KeyboardInterrupt

def _on_sigterm(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    _show_cursor()
    _kill_group(signal.SIGTERM)
    raise KeyboardInterrupt

def _on_sigtstp(signum, frame):
    _show_cursor()
    signal.signal(signal.SIGTSTP, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTSTP)

def _on_sigcont(signum, frame):
    _show_cursor()

def _install_signal_handlers() -> None:
    signal.signal(signal.SIGINT,  _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigterm)
    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, _on_sigtstp)
    if hasattr(signal, "SIGCONT"):
        signal.signal(signal.SIGCONT, _on_sigcont)

# ----------------- Utility helpers -----------------

def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()

def seconds_to_hms(s: int) -> str:
    h = s // 3600; s -= h*3600
    m = s // 60;   s -= m*60
    return f"{h}h{m}m{s}s"

def _fmt_secs(total: int | None) -> str:
    if total is None or total < 0:
        return "?"
    h, r = divmod(int(total), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def _pow2_to_float(expr: str | None) -> Optional[float]:
    if not expr:
        return None
    t = expr.strip()
    try:
        if t.startswith("2^"):
            return float(2.0 ** float(t[2:]))
        return float(2.0 ** float(t))
    except Exception:
        return None

def chunk_size_for_bits(bits: int) -> int:
    return 1 << bits

def span_count(min_dec: int, max_dec: int) -> int:
    return (max_dec - min_dec + 1)

def total_chunks(min_dec: int, max_dec: int, chunk_bits: int) -> int:
    cs = chunk_size_for_bits(chunk_bits)
    return (span_count(min_dec, max_dec) + cs - 1) // cs

def chunk_bounds(min_dec: int, max_dec: int, chunk_bits: int, idx: int) -> Tuple[int, int]:
    cs = chunk_size_for_bits(chunk_bits)
    start = min_dec + idx * cs
    stop = min(start + cs, max_dec + 1) - 1
    return start, stop

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

import hashlib

def auto_rangeset_name(pubkey: str, min_dec: int, max_dec: int, chunk_bits: int) -> str:
    s = f"{pubkey}:{min_dec}:{max_dec}:{int(chunk_bits)}"
    h = hashlib.sha1(s.encode("ascii")).hexdigest()[:10]
    return f"band_{h}"

# ----------------- SQLite schema & migrations -----------------

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS rangesets(
  id           INTEGER PRIMARY KEY,
  name         TEXT UNIQUE NOT NULL,
  min_dec      TEXT NOT NULL,
  max_dec      TEXT NOT NULL,
  chunk_bits   INTEGER NOT NULL,
  next_index   TEXT DEFAULT '0',
  created_ts   TEXT NOT NULL,
  cfg_fingerprint TEXT,
  notes        TEXT
);
CREATE TABLE IF NOT EXISTS chunks(
  id           INTEGER PRIMARY KEY,
  rangeset_id  INTEGER NOT NULL,
  chunk_index  TEXT NOT NULL,
  start_dec    TEXT NOT NULL,
  end_dec      TEXT NOT NULL,
  status       TEXT NOT NULL CHECK(status IN ('queued','running','done','found','aborted','stalled')),
  claimed_ts   TEXT NOT NULL,
  started_ts   TEXT,
  finished_ts  TEXT,
  mk_s_now     REAL,
  mk_s_avg     REAL,
  dead         INTEGER,
  dp           INTEGER,
  expected_ops TEXT,
  nthreads     INTEGER,
  pubkey       TEXT,
  m_factor     REAL,
  dp_forced    INTEGER,
  band_min_dec TEXT,
  band_max_dec TEXT,
  output       TEXT,
  UNIQUE(rangeset_id, chunk_index)
);
-- tiles: will be (re)created/migrated to include pubkey + unique(pubkey,level,start_hex)
"""

DEFAULT_TILE_LEVELS = [52, 48, 44, 40]

def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    for stmt in SCHEMA.strip().split(";\n"):
        if stmt.strip():
            conn.execute(stmt)
    conn.commit()
    _ensure_schema_migrations(conn)
    return conn

def _allowed_statuses(conn: sqlite3.Connection) -> set[str]:
    sql_row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone()
    sql = (sql_row["sql"] or "") if sql_row else ""
    import re
    m = re.search(r"CHECK\s*\(\s*status\s*IN\s*\(([^)]*)\)\s*\)", sql, re.IGNORECASE)
    if not m:
        return {"queued","running","done","found","aborted","stalled","error"}
    raw = m.group(1)
    vals = [v.strip().strip("'\"") for v in raw.split(",")]
    return set(v for v in vals if v)

def _map_status_for_schema(conn: sqlite3.Connection, status: str) -> str:
    allowed = _allowed_statuses(conn)
    if status in allowed:
        return status
    if status == "aborted" and "aborted" not in allowed and "error" in allowed:
        return "error"
    return next(iter(allowed))

def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    # rangesets: ensure next_index / created_ts / cfg_fingerprint
    rs_cols = {r["name"] for r in conn.execute("PRAGMA table_info(rangesets)")}
    if "next_index" not in rs_cols:
        conn.execute("ALTER TABLE rangesets ADD COLUMN next_index TEXT DEFAULT '0'")
    if "created_ts" not in rs_cols:
        conn.execute("ALTER TABLE rangesets ADD COLUMN created_ts TEXT")
        conn.execute("UPDATE rangesets SET created_ts=? WHERE created_ts IS NULL", (utc_now_iso(),))
    if "cfg_fingerprint" not in rs_cols:
        conn.execute("ALTER TABLE rangesets ADD COLUMN cfg_fingerprint TEXT")
        conn.execute("""UPDATE rangesets
                           SET cfg_fingerprint='min:'||min_dec||'|max:'||max_dec||'|bits:'||chunk_bits
                         WHERE cfg_fingerprint IS NULL""")

    # chunks: ensure extended columns exist
    ch_cols = {r["name"] for r in conn.execute("PRAGMA table_info(chunks)")}
    needed = [
        ("claimed_ts",  "TEXT"),
        ("started_ts",  "TEXT"),
        ("finished_ts", "TEXT"),
        ("mk_s_now",    "REAL"),
        ("mk_s_avg",    "REAL"),
        ("dead",        "INTEGER"),
        ("dp",          "INTEGER"),
        ("expected_ops","TEXT"),
        ("nthreads",    "INTEGER"),
        ("pubkey",      "TEXT"),
        ("m_factor",    "REAL"),
        ("dp_forced",   "INTEGER"),
        ("band_min_dec","TEXT"),
        ("band_max_dec","TEXT"),
        ("output",      "TEXT"),
    ]
    for name, typ in needed:
        if name not in ch_cols:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {name} {typ}")

    conn.commit()

    # tiles: ensure per-pubkey uniqueness; migrate if needed
    _migrate_tiles_add_pubkey_if_needed(conn)

def _table_sql(conn: sqlite3.Connection, name: str) -> str:
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return (row["sql"] or "") if row else ""

def _migrate_tiles_add_pubkey_if_needed(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tiles'").fetchone()
    if not row:
        # fresh create in new shape
        for stmt in """
        CREATE TABLE IF NOT EXISTS tiles(
          id           INTEGER PRIMARY KEY,
          pubkey       TEXT,
          level        INTEGER NOT NULL,
          start_hex    TEXT NOT NULL,
          status       TEXT NOT NULL CHECK(status IN ('running','done','found')),
          lease_ts     TEXT NOT NULL,
          rangeset_id  INTEGER,
          chunk_id     INTEGER,
          UNIQUE(pubkey, level, start_hex)
        );
        CREATE INDEX IF NOT EXISTS tiles_by_chunk ON tiles(chunk_id);
        CREATE INDEX IF NOT EXISTS tiles_by_status ON tiles(pubkey, status, level);
        """.strip().split(";\n"):
            if stmt.strip():
                conn.execute(stmt)
        conn.commit()
        return

    # table exists: check columns & unique
    cols = {r["name"]: r for r in conn.execute("PRAGMA table_info(tiles)")}
    need_pubkey = ("pubkey" not in cols)
    sql = _table_sql(conn, "tiles")
    has_unique_per_pubkey = ("UNIQUE(pubkey, level, start_hex)" in (sql or "").replace("\n"," ").replace("  "," "))

    if not need_pubkey and has_unique_per_pubkey:
        # nothing to do
        return

    # Migration: rebuild tiles with pubkey + new unique
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DROP TABLE IF EXISTS tiles_new")
        for stmt in """
        CREATE TABLE tiles_new(
          id           INTEGER PRIMARY KEY,
          pubkey       TEXT,
          level        INTEGER NOT NULL,
          start_hex    TEXT NOT NULL,
          status       TEXT NOT NULL CHECK(status IN ('running','done','found')),
          lease_ts     TEXT NOT NULL,
          rangeset_id  INTEGER,
          chunk_id     INTEGER,
          UNIQUE(pubkey, level, start_hex)
        );
        """.strip().split(";\n"):
            if stmt.strip():
                conn.execute(stmt)

        # copy rows; if chunk_id present, pull pubkey from chunks; else NULL
        conn.execute("""
            INSERT OR IGNORE INTO tiles_new(id, pubkey, level, start_hex, status, lease_ts, rangeset_id, chunk_id)
            SELECT t.id,
                   (SELECT c.pubkey FROM chunks c WHERE c.id=t.chunk_id) AS pubkey,
                   t.level, t.start_hex, t.status, t.lease_ts, t.rangeset_id, t.chunk_id
            FROM tiles t
        """)

        conn.execute("DROP TABLE tiles")
        conn.execute("ALTER TABLE tiles_new RENAME TO tiles")
        conn.execute("CREATE INDEX IF NOT EXISTS tiles_by_chunk ON tiles(chunk_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS tiles_by_status ON tiles(pubkey, status, level)")
        conn.commit()
    except Exception:
        conn.rollback()
        raise

# ----------------- Tile helpers (hierarchical canonical tiling, per pubkey) -----------------

def _hex64_upper(n: int) -> str:
    if n < 0:
        raise ValueError("n must be non-negative")
    h = format(n, "064x").upper()
    if len(h) > 64:
        h = h[-64:]
    return h

def _align_down(n: int, bits: int) -> int:
    mask = (1 << bits) - 1
    return n & ~mask

def _parent_hex(level: int, start_hex: str, parent_level: int) -> str:
    """Parent shares leftmost head_len nibbles; tail_len = parent_level//4 zeros on the right."""
    assert parent_level > level
    head_len = 64 - (parent_level // 4)
    return start_hex[:head_len] + ("0" * (parent_level // 4))

def _now_minus_iso(seconds: int) -> str:
    return (_dt.datetime.now(_dt.UTC) - _dt.timedelta(seconds=seconds)).replace(microsecond=0).isoformat()

def tiles_reap_expired(conn: sqlite3.Connection, lease_ttl_s: int) -> int:
    cutoff = _now_minus_iso(lease_ttl_s)
    cur = conn.execute("DELETE FROM tiles WHERE status='running' AND lease_ts < ?", (cutoff,))
    conn.commit()
    return cur.rowcount

def _tile_row(conn: sqlite3.Connection, level: int, start_hex: str, pubkey: Optional[str]) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM tiles WHERE pubkey IS ? AND level=? AND start_hex=? LIMIT 1",
                        (pubkey, int(level), start_hex)).fetchone()

def _any_child_exists(conn: sqlite3.Connection, level: int, start_hex: str,
                      lease_ttl_s: int, pubkey: Optional[str]) -> Optional[str]:
    """
    Return child status for the SAME pubkey if any immediate child exists; else None.
    Ignores tiles from other pubkeys.
    """
    if level < 4:
        return None
    child_level = level - 4
    head_len    = 64 - (level // 4)
    zeros_tail  = "0" * (child_level // 4)
    pat = start_hex[:head_len] + "[0-9A-F]" + zeros_tail

    row = conn.execute(
        "SELECT status, lease_ts FROM tiles WHERE pubkey IS ? AND level=? AND start_hex GLOB ? LIMIT 1",
        (pubkey, child_level, pat)
    ).fetchone()
    if not row:
        return None
    st, lt = row["status"], row["lease_ts"]
    if st == "running" and lt and lt < _now_minus_iso(lease_ttl_s):
        return None  # expired running—treat as no active child
    return st


def _tile_conflicts_due_to_ancestor(conn: sqlite3.Connection, level: int, start_hex: str,
                                    lease_ttl_s: int, levels: List[int],
                                    pubkey: Optional[str]) -> bool:
    for L in levels:
        if L <= level:
            continue
        anc_hex = _parent_hex(level, start_hex, L)
        row = _tile_row(conn, L, anc_hex, pubkey)
        if not row:
            continue
        st = row["status"]; lt = row["lease_ts"]
        if st == "running":
            if lt and lt >= _now_minus_iso(lease_ttl_s):
                return True
            else:
                continue
        return True
    return False

def _tile_has_children(conn: sqlite3.Connection, level: int, start_hex: str,
                       lease_ttl_s: int, pubkey: Optional[str]) -> bool:
    if level < 4:
        return False
    child_level = level - 4
    head_len    = 64 - (level // 4)
    zeros_tail  = "0" * (child_level // 4)
    pat = start_hex[:head_len] + "[0-9A-F]" + zeros_tail
    rows = conn.execute("""SELECT status, lease_ts FROM tiles
                           WHERE pubkey IS ? AND level=? AND start_hex GLOB ? LIMIT 1""",
                        (pubkey, child_level, pat)).fetchone()
    if not rows:
        return False
    st = rows["status"]; lt = rows["lease_ts"]
    if st == "running":
        return bool(lt and lt >= _now_minus_iso(lease_ttl_s))
    return True

def _insert_tile(conn: sqlite3.Connection, level: int, start_hex: str,
                 rangeset_id: int, chunk_id: int, pubkey: Optional[str]) -> bool:
    try:
        conn.execute("""INSERT INTO tiles(pubkey,level,start_hex,status,lease_ts,rangeset_id,chunk_id)
                        VALUES(?, ?, ?, 'running', ?, ?, ?)""",
                     (pubkey, int(level), start_hex, utc_now_iso(), int(rangeset_id), int(chunk_id)))
        return True
    except sqlite3.IntegrityError:
        return False
    
def _insert_running_tile(conn: sqlite3.Connection, level: int, start_hex: str,
                         rangeset_id: int, chunk_id: int, lease_ttl_s: int,
                         pubkey: Optional[str]) -> str:
    """
    Try to claim this tile as 'running' for the given pubkey.
    Returns: 'inserted' | 'stolen' (expired lease) | 'busy' | 'covered'
    """
    try:
        conn.execute(
            """INSERT INTO tiles(pubkey,level,start_hex,status,lease_ts,rangeset_id,chunk_id)
               VALUES(?, ?, ?, 'running', ?, ?, ?)""",
            (pubkey, int(level), start_hex, utc_now_iso(), int(rangeset_id), int(chunk_id))
        )
        return "inserted"
    except sqlite3.IntegrityError:
        row = _tile_row(conn, level, start_hex, pubkey)
        if not row:
            return "busy"  # race or different row/idx name
        st, lt = row["status"], row["lease_ts"]
        if st in ("done","found"):
            return "covered"
        # running: check lease freshness
        if lt and lt >= _now_minus_iso(lease_ttl_s):
            return "busy"
        # steal expired lease
        conn.execute(
            """UPDATE tiles
                 SET status='running', lease_ts=?, rangeset_id=?, chunk_id=?
               WHERE pubkey IS ? AND level=? AND start_hex=?""",
            (utc_now_iso(), int(rangeset_id), int(chunk_id), pubkey, int(level), start_hex)
        )
        return "stolen"

def _claim_tile_recursive(conn: sqlite3.Connection, level_idx: int, levels: List[int],
                          start_dec: int, end_dec: int, rangeset_id: int, chunk_id: int,
                          lease_ttl_s: int, pubkey: Optional[str]) -> int:
    """Return number of tiles claimed for this chunk."""
    if level_idx >= len(levels):
        return 0
    L = int(levels[level_idx]); size = 1 << L
    claimed = 0

    cur = start_dec
    head_align = ((cur + size - 1) // size) * size
    if cur < head_align and cur <= end_dec:
        claimed += _claim_tile_recursive(conn, level_idx + 1, levels, cur, min(end_dec, head_align - 1),
                                         rangeset_id, chunk_id, lease_ttl_s, pubkey)
        cur = head_align

    while cur + size - 1 <= end_dec:
        t_start = cur
        t_hex = _hex64_upper(_align_down(t_start, L))

        # If an ancestor is already permanent for THIS pubkey, skip; if running&fresh, split.
        anc = _ancestor_active_status(conn, levels, L, t_hex, lease_ttl_s, pubkey)
        if anc in ("done","found"):
            cur += size
            continue
        if anc == "running":
            claimed += _claim_tile_recursive(conn, level_idx + 1, levels, t_start, t_start + size - 1,
                                             rangeset_id, chunk_id, lease_ttl_s, pubkey)
            cur += size
            continue

        # If a child exists for THIS pubkey, split.
        chst = _any_child_exists(conn, L, t_hex, lease_ttl_s, pubkey)
        if chst is not None:
            claimed += _claim_tile_recursive(conn, level_idx + 1, levels, t_start, t_start + size - 1,
                                             rangeset_id, chunk_id, lease_ttl_s, pubkey)
            cur += size
            continue

        res = _insert_running_tile(conn, L, t_hex, rangeset_id, chunk_id, lease_ttl_s, pubkey)
        if res in ("inserted","stolen"):
            claimed += 1
            cur += size
            continue
        if res in ("busy","covered"):
            claimed += _claim_tile_recursive(conn, level_idx + 1, levels, t_start, t_start + size - 1,
                                             rangeset_id, chunk_id, lease_ttl_s, pubkey)
            cur += size
            continue
        # unexpected
        cur += size

    if cur <= end_dec:
        claimed += _claim_tile_recursive(conn, level_idx + 1, levels, cur, end_dec,
                                         rangeset_id, chunk_id, lease_ttl_s, pubkey)
    return claimed


def claim_tiles_for_chunk(conn: sqlite3.Connection, ch: sqlite3.Row, rs: RangeSet,
                          tile_levels: List[int], lease_ttl_s: int,
                          pubkey: Optional[str]) -> bool:
    levels = sorted(set(int(x) for x in tile_levels), reverse=True)
    if not levels:
        levels = DEFAULT_TILE_LEVELS[:]
    for L in levels:
        if L % 4 != 0:
            raise ValueError("--tile-levels must be multiples of 4")

    start_dec = int(ch["start_dec"]); end_dec = int(ch["end_dec"])
    conn.execute("BEGIN IMMEDIATE")
    try:
        n = _claim_tile_recursive(conn, 0, levels, start_dec, end_dec,
                                  int(rs.id), int(ch["id"]), int(lease_ttl_s), pubkey)
        conn.commit()
        return n > 0
    except Exception:
        with contextlib.suppress(Exception):
            conn.rollback()
        return False


def refresh_tile_leases(conn: sqlite3.Connection, chunk_id: int) -> None:
    conn.execute("UPDATE tiles SET lease_ts=? WHERE chunk_id=? AND status='running'", (utc_now_iso(), int(chunk_id)))
    conn.commit()

def finalize_tiles(conn: sqlite3.Connection, chunk_id: int, new_status: str) -> None:
    if new_status not in ("done","found"):
        return
    conn.execute("UPDATE tiles SET status=?, lease_ts=? WHERE chunk_id=?",
                 (new_status, utc_now_iso(), int(chunk_id)))
    conn.commit()

# ----------------- Data classes -----------------

class RangeSet:
    __slots__ = ("id","name","min_dec","max_dec","chunk_bits")
    def __init__(self, id: int, name: str, min_dec: int | str, max_dec: int | str, chunk_bits: int | str):
        self.id = int(id)
        self.name = str(name)
        self.min_dec = int(min_dec)
        self.max_dec = int(max_dec)
        self.chunk_bits = int(chunk_bits)

def upsert_rangeset(conn: sqlite3.Connection, name: str, min_dec: int, max_dec: int, chunk_bits: int,
                    *, force_reinit: bool = False) -> RangeSet:
    row = conn.execute("SELECT * FROM rangesets WHERE name=?", (name,)).fetchone()
    fp_new = f"min:{min_dec}|max:{max_dec}|bits:{int(chunk_bits)}"
    if not row:
        cur = conn.execute(
            "INSERT INTO rangesets(name,min_dec,max_dec,chunk_bits,next_index,created_ts,cfg_fingerprint) "
            "VALUES(?,?,?,?, '0', ?, ?)",
            (name, str(min_dec), str(max_dec), int(chunk_bits), utc_now_iso(), fp_new))
        conn.commit()
        return RangeSet(cur.lastrowid, name, min_dec, max_dec, chunk_bits)
    rs = RangeSet(row["id"], row["name"], row["min_dec"], row["max_dec"], row["chunk_bits"])
    fp_old = row["cfg_fingerprint"] or f"min:{rs.min_dec}|max:{rs.max_dec}|bits:{rs.chunk_bits}"
    if fp_old == fp_new:
        return rs
    cnt = conn.execute("SELECT COUNT(1) AS n FROM chunks WHERE rangeset_id=?", (rs.id,)).fetchone()["n"] or 0
    if cnt and not force_reinit:
        raise ValueError(
            f"Range-set '{name}' already exists with different bounds/bits and has {cnt} chunk(s). "
            f"Use a NEW --range-name, or pass --force-reinit-range to purge."
        )
    if cnt and force_reinit:
        conn.execute("DELETE FROM tiles WHERE rangeset_id=? OR chunk_id IN (SELECT id FROM chunks WHERE rangeset_id=?)",
                     (rs.id, rs.id))
        conn.execute("DELETE FROM chunks WHERE rangeset_id=?", (rs.id,))
        conn.execute("UPDATE rangesets SET next_index='0' WHERE id=?", (rs.id,))
        conn.commit()
    conn.execute("UPDATE rangesets SET min_dec=?, max_dec=?, chunk_bits=?, cfg_fingerprint=? WHERE id=?",
                 (str(min_dec), str(max_dec), int(chunk_bits), fp_new, rs.id))
    conn.commit()
    return RangeSet(rs.id, name, min_dec, max_dec, chunk_bits)

def rangeset_by_name(conn: sqlite3.Connection, name: str) -> Optional[RangeSet]:
    row = conn.execute("SELECT * FROM rangesets WHERE name=?", (name,)).fetchone()
    return RangeSet(row["id"], row["name"], row["min_dec"], row["max_dec"], row["chunk_bits"]) if row else None

# ----------------- Chunk picking -----------------

def try_insert_chunk(conn: sqlite3.Connection, rs: RangeSet, idx: int) -> Optional[sqlite3.Row]:
    s, e = chunk_bounds(rs.min_dec, rs.max_dec, rs.chunk_bits, idx)
    try:
        conn.execute(
            """INSERT INTO chunks(rangeset_id,chunk_index,start_dec,end_dec,status,claimed_ts,band_min_dec,band_max_dec)
               VALUES(?,?,?,?, 'running', ?, ?, ?)""",
            (rs.id, str(idx), str(s), str(e), utc_now_iso(), str(rs.min_dec), str(rs.max_dec)))
        conn.commit()
        return conn.execute("SELECT * FROM chunks WHERE rangeset_id=? AND chunk_index=?",
                            (rs.id, str(idx))).fetchone()
    except sqlite3.IntegrityError:
        return None

def pick_random(conn: sqlite3.Connection, rs: RangeSet) -> Optional[sqlite3.Row]:
    r = conn.execute("""SELECT * FROM chunks
                          WHERE rangeset_id=? AND status='running'
                       ORDER BY claimed_ts ASC LIMIT 1""", (rs.id,)).fetchone()
    if r: return r
    N = total_chunks(rs.min_dec, rs.max_dec, rs.chunk_bits)
    for _ in range(64):
        idx = random.randrange(0, N)
        r = try_insert_chunk(conn, rs, idx)
        if r: return r
    for idx in range(N):
        r = try_insert_chunk(conn, rs, idx)
        if r: return r
    return None

def build_intervals_from_claimed(N: int, claimed_idx_sorted: List[int]) -> List[Tuple[int,int,int]]:
    if N <= 0: return []
    if not claimed_idx_sorted: return [(N, 0, N-1)]
    spans: List[Tuple[int,int,int]] = []
    if claimed_idx_sorted[0] > 0:
        spans.append((claimed_idx_sorted[0], 0, claimed_idx_sorted[0]-1))
    for i in range(len(claimed_idx_sorted)-1):
        a, b = claimed_idx_sorted[i], claimed_idx_sorted[i+1]
        if b > a + 1:
            spans.append((b - a - 1, a+1, b-1))
    if claimed_idx_sorted[-1] < N-1:
        spans.append((N-1 - claimed_idx_sorted[-1], claimed_idx_sorted[-1]+1, N-1))
    return spans

def pick_entropy(conn: sqlite3.Connection, rs: RangeSet) -> Optional[sqlite3.Row]:
    r = conn.execute("""SELECT * FROM chunks
                          WHERE rangeset_id=? AND status='running'
                       ORDER BY claimed_ts ASC LIMIT 1""", (rs.id,)).fetchone()
    if r: return r

    N = total_chunks(rs.min_dec, rs.max_dec, rs.chunk_bits)
    claimed = [int(row["chunk_index"]) for row in conn.execute(
        "SELECT chunk_index FROM chunks WHERE rangeset_id=?", (rs.id,))]
    if not claimed:
        idx = (N - 1) // 2
        r = try_insert_chunk(conn, rs, idx)
        if r: return r
        return pick_random(conn, rs)

    claimed.sort()
    intervals = build_intervals_from_claimed(N, claimed)
    if not intervals: return None
    length, L, R = max(intervals, key=lambda t: (t[0], -t[1]))
    mid = (L + R) // 2
    for k in (mid, clamp(mid-1, L, R), clamp(mid+1, L, R)):
        r = try_insert_chunk(conn, rs, k)
        if r: return r
    # second attempt after reloading claimed
    claimed2 = [int(row["chunk_index"]) for row in conn.execute(
        "SELECT chunk_index FROM chunks WHERE rangeset_id=?", (rs.id,))]
    claimed2.sort()
    intervals2 = build_intervals_from_claimed(N, claimed2)
    if not intervals2: return None
    length, L, R = max(intervals2, key=lambda t: (t[0], -t[1]))
    mid = (L + R) // 2
    for k in (mid, clamp(mid-1, L, R), clamp(mid+1, L, R)):
        r = try_insert_chunk(conn, rs, k)
        if r: return r
    return pick_random(conn, rs)

def pick_sequential(conn: sqlite3.Connection, rs: RangeSet, *,
                    rules_enabled: bool = False,
                    rules_max_tries: int = 1024) -> Optional[sqlite3.Row]:
    """
    Sequential picker with optional rules-based validate-and-jump.

    When rules_enabled is False:
        - Behavior identical to original implementation.
    When rules_enabled is True:
        - Compute the next rules-valid decimal V >= start of the "next_index" chunk.
        - If V is within that chunk, claim that chunk (as usual).
        - If V lies in a later chunk, jump to that chunk index and try to claim it.
        - On contention, linearly probe forward up to rules_max_tries attempts,
          then fall back to the original probing strategy.
    This preserves robustness and existing locking/retry semantics.
    """
    # If any running chunk exists, resume it first (unchanged behavior).
    r = conn.execute("""SELECT * FROM chunks
                          WHERE rangeset_id=? AND status='running'
                       ORDER BY claimed_ts ASC LIMIT 1""", (rs.id,)).fetchone()
    if r:
        return r

    row = conn.execute("SELECT next_index FROM rangesets WHERE id=?", (rs.id,)).fetchone()
    next_idx = int(row["next_index"]) if row else 0
    N = total_chunks(rs.min_dec, rs.max_dec, rs.chunk_bits)
    if next_idx >= N:
        return None

    # Fast path: no rules → original logic.
    if not rules_enabled:
        r = try_insert_chunk(conn, rs, next_idx)
        if r:
            conn.execute("UPDATE rangesets SET next_index=? WHERE id=?", (str(next_idx+1), rs.id))
            conn.commit()
            return r
        for idx in range(next_idx+1, min(next_idx+1024, N)):
            r = try_insert_chunk(conn, rs, idx)
            if r:
                conn.execute("UPDATE rangesets SET next_index=? WHERE id=?", (str(idx+1), rs.id))
                conn.commit()
                return r
        for idx in range(N):
            r = try_insert_chunk(conn, rs, idx)
            if r:
                return r
        return None

    # Rules-enabled path: compute next valid number >= start of next_idx chunk.
    start_dec, _end_dec = chunk_bounds(rs.min_dec, rs.max_dec, rs.chunk_bits, next_idx)
    try:
        v = next_valid_ge(max(start_dec, rs.min_dec), rs.min_dec, rs.max_dec)
    except Exception:
        # Defensive: if rules computation fails for any unexpected reason,
        # fallback to original behavior rather than break orchestration.
        v = None

    if v is None:
        # No rules-valid numbers remain at or beyond this point; give up.
        return None

    # Determine the chunk index containing v.
    cs = chunk_size_for_bits(rs.chunk_bits)
    idx_v = (v - rs.min_dec) // cs
    if idx_v < 0:
        idx_v = 0
    elif idx_v >= N:
        return None  # Out of range (should not happen if next_valid_ge respected bounds)

    # Try to claim the target chunk first.
    r = try_insert_chunk(conn, rs, idx_v)
    if r:
        conn.execute("UPDATE rangesets SET next_index=? WHERE id=?", (str(idx_v+1), rs.id))
        conn.commit()
        return r

    # Linear probe forward for a bounded number of attempts (contention handling).
    # We do not wrap-around here to preserve forward progress guarantee.
    limit = min(N, idx_v + max(1, int(rules_max_tries)))
    for idx in range(idx_v + 1, limit):
        r = try_insert_chunk(conn, rs, idx)
        if r:
            conn.execute("UPDATE rangesets SET next_index=? WHERE id=?", (str(idx+1), rs.id))
            conn.commit()
            return r

    # Fallback to original probing strategy to avoid starvation if contention is high.
    # First, try near future window from next_idx+1.
    for idx in range(next_idx+1, min(next_idx+1024, N)):
        r = try_insert_chunk(conn, rs, idx)
        if r:
            conn.execute("UPDATE rangesets SET next_index=? WHERE id=?", (str(idx+1), rs.id))
            conn.commit()
            return r
    # Then try the entire space.
    for idx in range(N):
        r = try_insert_chunk(conn, rs, idx)
        if r:
            return r

    return None

def claim_or_resume_chunk(conn: sqlite3.Connection, rs: RangeSet, picker: str,
                          *, seq_rules_enabled: bool = False,
                          seq_rules_max_tries: int = 1024) -> Optional[sqlite3.Row]:
    if picker == "sequential":
        return pick_sequential(conn, rs, rules_enabled=seq_rules_enabled, rules_max_tries=seq_rules_max_tries)
    if picker == "entropy":
        return pick_entropy(conn, rs)
    return pick_random(conn, rs)

def parse_header(lines: List[str]) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("Number of CPU thread:"):
            d["threads"] = ln.split(":")[1].strip()
        elif ln.startswith("Suggested DP:"):
            d["suggested_dp"] = ln.split(":")[1].strip()
        elif ln.startswith("Expected operations:"):
            d["ops"] = ln.split(":", 1)[1].strip()
        elif ln.startswith("Range width:"):
            d["width"] = ln.split(":", 1)[1].strip()
        elif ln.startswith("Expected RAM:"):
            d["ram"] = ln.split(":", 1)[1].strip()
        elif ln.startswith("DP size:"):
            d["dp_size"] = ln.split(":", 1)[1].strip()
    return d

def parse_progress_line(ln: str) -> Dict[str, object]:
    d: Dict[str, object] = {}
    try:
        parts = ln.strip().split("]")
        for p in parts:
            p = p.strip("[] ")
            if not p:
                continue
            if p.endswith("MK/s") and "GPU" not in p:
                d["now_mks"] = float(p.split()[0])
            elif p.startswith("Count "):
                d["count"] = p.split("Count", 1)[1].strip()
            elif p.startswith("Dead "):
                d["dead"] = int(p.split()[1])
            elif "(" in p and "Avg" in p and "s" in p:
                pre = p.split("s", 1)[0]
                try: d["elapsed_s_hint"] = int(pre)
                except Exception: pass
    except Exception:
        pass
    return d

def print_banner(rs: RangeSet, ch: sqlite3.Row, header: Dict[str, str],
                 prog: Dict[str, object], pubkey_hex: str, set_idx_text: str,
                 stop_on_found: bool, dp_forced: Optional[int], m_factor: Optional[int],
                 found_priv_hex: Optional[str], freeze: bool) -> None:
    """
    Persistent, in-place banner with DP (running/suggested), elapsed & ETA.
    """
    start_dec = ch["start_dec"]; end_dec = ch["end_dec"]
    start_hex = f"{int(start_dec):064X}"
    end_hex   = f"{int(end_dec):064X}"

    suggested_dp = header.get("suggested_dp", None)
    dp_size_str  = header.get("dp_size", None)
    if dp_forced is not None:
        dp_line = f"DP: running={int(dp_forced)}  suggested={suggested_dp if suggested_dp is not None else '?'}"
    else:
        dp_line = f"DP: running=auto  suggested={suggested_dp if suggested_dp is not None else '?'}"

    lines: List[str] = []
    if found_priv_hex:
        lines += ["\x1b[1;32m" + "="*72,
                  f"FOUND PRIVATE KEY: {found_priv_hex}",
                  "="*72 + "\x1b[0m"]

    lines.append(f"PubKey: {pubkey_hex}")
    lines.append(f"set={rs.name}  idx={set_idx_text}")
    lines.append(f"Threads: {header.get('threads', '?')}")
    lines.append(f"Set Min (DEC): {rs.min_dec}")
    lines.append(f"Set Max (DEC): {rs.max_dec}")
    lines.append(f"Chunk Bits: {rs.chunk_bits}")
    lines.append(f"Chunk Start (DEC): {start_dec}")
    lines.append(f"Chunk End   (DEC): {end_dec}")
    lines.append(f"Chunk Start (HEX): 0x{start_hex}")
    lines.append(f"Chunk End   (HEX): 0x{end_hex}")
    lines.append(dp_line)
    if dp_size_str:
        lines.append(f"DP size: {dp_size_str}")
    if m_factor is not None:
        lines.append(f"MaxStep (m): {m_factor}")
    if header.get("ops"):
        lines.append(f"Expected ops: {header.get('ops')}")
    if header.get("ram"):
        lines.append(f"Expected RAM: {header.get('ram')}")

    if "count" in prog or "dead" in prog:
        lines.append(f"Progress: {prog.get('count', '?')}  Dead: {prog.get('dead', '?')}")
    if "now_mks" in prog or "avg_mks" in prog:
        now_mks = float(prog.get("now_mks", 0.0))
        avg_mks = float(prog.get("avg_mks", 0.0))
        lines.append(f"Speed: {now_mks:.2f} MK/s (Avg {avg_mks:.2f} MK/s)")

    elapsed_s = prog.get("elapsed_s")
    eta_s     = prog.get("eta_s")
    if isinstance(elapsed_s, int):
        if isinstance(eta_s, int):
            lines.append(f"Time: {_fmt_secs(elapsed_s)}  ETA≈ {_fmt_secs(eta_s)}")
        else:
            lines.append(f"Time: {_fmt_secs(elapsed_s)}")

    text = "\n".join(lines)

    if not hasattr(print_banner, "_initialized"):
        print_banner._initialized = True  # type: ignore[attr-defined]
        sys.stdout.write("\x1b[2J\x1b[H")  # full clear
    else:
        sys.stdout.write("\x1b[H\x1b[J")  # repaint in place

    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    if freeze:
        return
# ----------------- Run kangaroo -----------------

class RunResult:
    __slots__ = ("status","found_priv_hex","header","stats","raw")
    def __init__(self, status: str, found_priv_hex: Optional[str], header: Dict[str,str|int],
                 stats: Dict[str, object], raw: str) -> None:
        self.status = status
        self.found_priv_hex = found_priv_hex
        self.header = header
        self.stats = stats
        self.raw = raw
def run_kangaroo(ch: sqlite3.Row, rs: RangeSet, pubkey_hex: str,
                 threads: int, dp: Optional[int], m_factor: Optional[int],
                 kangaroo_path: str, set_idx_text: str,
                 stop_on_found: bool,
                 banner_refresh_s: float,
                 lease_refresh_s: int,
                 conn: sqlite3.Connection) -> RunResult:
    """
    Spawn Kangaroo, stream/parse its output, maintain leases, render a persistent banner,
    forward signals to the child process group, and return a summarized RunResult.
    """
    # ---------- build command ----------
    args = [kangaroo_path, "-t", str(threads),
            "--start-dec", ch["start_dec"],
            "--end-dec",   ch["end_dec"],
            "--pubkey",    pubkey_hex]
    if dp is not None:
        args += ["-d", str(int(dp))]
    if m_factor is not None:
        args += ["-m", str(int(m_factor))]

    # ---------- start process in its own group (for clean signal handling) ----------
    creationflags = 0
    preexec = None
    if os.name == "nt":
        creationflags = 0x00000200  # CREATE_NEW_PROCESS_GROUP
    else:
        preexec = os.setsid

    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            preexec_fn=preexec,
                            creationflags=creationflags)

    # Track process group so signal handlers can forward INT/TERM/KILL
    global _ACTIVE_PGID
    try:
        pgid = os.getpgid(proc.pid) if os.name != "nt" else proc.pid
    except Exception:
        pgid = None
    _ACTIVE_PGID = pgid

    # ---------- state for parsing & banner ----------
    header_lines: List[str] = []
    header: Dict[str, str | int] = {}
    now_mks_hist: List[float] = []
    dead_last = 0
    elapsed_start = time.time()
    expected_total_ops: Optional[float] = None
    last_count_ops: Optional[float] = None
    found_priv: Optional[str] = None
    out_accum: List[str] = []
    aborted_by_user = False
    last_banner_refresh = time.monotonic()

    _hide_cursor()

    def _safe_kill_group() -> None:
        """INT → TERM → KILL escalator, respecting process group when available."""
        if proc.poll() is not None:
            return
        try:
            if _ACTIVE_PGID is not None and os.name != "nt":
                os.killpg(_ACTIVE_PGID, signal.SIGINT)
            else:
                proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        for _ in range(20):
            if proc.poll() is not None: break
            time.sleep(0.1)
        if proc.poll() is not None: return
        try:
            if _ACTIVE_PGID is not None and os.name != "nt":
                os.killpg(_ACTIVE_PGID, signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            pass
        for _ in range(30):
            if proc.poll() is not None: break
            time.sleep(0.1)
        if proc.poll() is not None: return
        try:
            if _ACTIVE_PGID is not None and os.name != "nt":
                os.killpg(_ACTIVE_PGID, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass

    # ---------- main read loop ----------
    try:
        assert proc.stdout is not None
        last_lease_refresh = time.time()

        try:
            for ln in proc.stdout:
                out_accum.append(ln)
                s = ln.strip()

                # lease heartbeat
                if (time.time() - last_lease_refresh) >= max(1, int(lease_refresh_s)):
                    with contextlib.suppress(Exception):
                        refresh_tile_leases(conn, int(ch["id"]))
                    last_lease_refresh = time.time()

                # header fields
                if s.startswith(("Kangaroo v", "Start:", "Stop :", "Keys :", "Number of CPU thread:",
                                 "Range width:", "Jump Avg distance:", "Number of kangaroos:",
                                 "Suggested DP:", "Expected operations:", "Expected RAM:", "DP size:")):
                    header_lines.append(s)
                    header = parse_header(header_lines)
                    if expected_total_ops is None and header.get("ops"):
                        base = _pow2_to_float(str(header.get("ops")))
                        if base:
                            expected_total_ops = base * (float(m_factor) if m_factor else 1.0)

                    # fast banner refresh on header
                    now = time.monotonic()
                    if (now - last_banner_refresh) >= 0.1:
                        print_banner(
                            rs, ch, header,
                            {"elapsed_s": int(time.time() - elapsed_start)},
                            pubkey_hex, set_idx_text, stop_on_found,
                            dp, m_factor, found_priv, False
                        )
                        last_banner_refresh = now
                    continue

                # progress line
                if s.startswith("[") and "MK/s" in s:
                    prog = parse_progress_line(s)
                    if "now_mks" in prog:
                        now_mks_hist.append(float(prog["now_mks"]))
                    dead_last = prog.get("dead", dead_last)
                    if isinstance(prog.get("count"), str):
                        last_count_ops = _pow2_to_float(str(prog["count"]))

                    # periodic banner refresh
                    now = time.monotonic()
                    if (now - last_banner_refresh) >= max(0.1, float(banner_refresh_s)):
                        elapsed_s = int(time.time() - elapsed_start)
                        avg_mks = (sum(now_mks_hist)/len(now_mks_hist)) if now_mks_hist else 0.0
                        eta_s = None
                        if expected_total_ops and avg_mks > 0.0 and last_count_ops is not None:
                            remaining = max(0.0, expected_total_ops - last_count_ops)
                            eta_s = int(remaining / (avg_mks * 1_000_000.0))
                        print_banner(
                            rs, ch, header,
                            {
                                "now_mks": now_mks_hist[-1] if now_mks_hist else 0.0,
                                "avg_mks": avg_mks,
                                "dead":    dead_last,
                                "count":   prog.get("count", None),
                                "elapsed_s": elapsed_s,
                                "eta_s":     eta_s if eta_s is not None else None,
                            },
                            pubkey_hex, set_idx_text, stop_on_found,
                            dp, m_factor, found_priv, False
                        )
                        last_banner_refresh = now
                    continue

                # found key line
                if "Priv:" in s:
                    tail = s.split("Priv:", 1)[1].strip()
                    if tail:
                        found_priv = tail.split()[0].strip()
                    else:
                        with contextlib.suppress(StopIteration):
                            nxt = next(proc.stdout)
                            out_accum.append(nxt)
                            found_priv = nxt.strip().split()[0]

                    # final frozen banner with FOUND
                    elapsed_s = int(time.time() - elapsed_start)
                    avg_mks = (sum(now_mks_hist)/len(now_mks_hist)) if now_mks_hist else 0.0
                    print_banner(
                        rs, ch, header,
                        {"now_mks": now_mks_hist[-1] if now_mks_hist else 0.0,
                         "avg_mks": avg_mks,
                         "dead":    dead_last,
                         "elapsed_s": elapsed_s},
                        pubkey_hex, set_idx_text, stop_on_found,
                        dp, m_factor, found_priv, True
                    )
                    _safe_kill_group()
                    break

        except KeyboardInterrupt:
            aborted_by_user = True
            _safe_kill_group()

    finally:
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()
        _show_cursor()
        if _ACTIVE_PGID == pgid:
            _ACTIVE_PGID = None

    # ---------- summarize ----------
    rc = proc.returncode if (proc and proc.poll() is not None) else 0
    status = "found" if found_priv else ("done" if rc == 0 else "aborted")
    avg_mks = (sum(now_mks_hist) / len(now_mks_hist)) if now_mks_hist else 0.0
    elapsed_s = int(time.time() - elapsed_start)
    stats = {
        "now_mks": now_mks_hist[-1] if now_mks_hist else 0.0,
        "avg_mks": avg_mks,
        "dead": dead_last,
        "elapsed_s": elapsed_s,
        "elapsed_text": seconds_to_hms(elapsed_s),
    }
    if aborted_by_user:
        header["aborted_by_user"] = 1
    return RunResult(status, found_priv, header, stats, "".join(out_accum))


# ----------------- Backfill (pubkey-scoped) -----------------

def _tile_row(conn: sqlite3.Connection, level: int, start_hex: str, pubkey: Optional[str]) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM tiles WHERE pubkey IS ? AND level=? AND start_hex=? LIMIT 1",
        (pubkey, int(level), start_hex)
    ).fetchone()

def _insert_tile_status(conn: sqlite3.Connection, level: int, start_hex: str,
                        status: str, rangeset_id: Optional[int], chunk_id: Optional[int],
                        pubkey: Optional[str]) -> str:
    assert status in ("done","found")
    try:
        conn.execute("""INSERT INTO tiles(pubkey,level,start_hex,status,lease_ts,rangeset_id,chunk_id)
                        VALUES(?, ?, ?, ?, ?, ?, ?)""",
                     (pubkey, int(level), start_hex, status, utc_now_iso(),
                      (int(rangeset_id) if rangeset_id is not None else None),
                      (int(chunk_id) if chunk_id is not None else None)))
        return "inserted"
    except sqlite3.IntegrityError:
        row = _tile_row(conn, level, start_hex, pubkey)
        if not row:
            return "race"
        if row["status"] in ("done","found"):
            return "covered"
        return "conflict-running"

def _ancestor_active_status(conn: sqlite3.Connection, levels: List[int],
                            level: int, start_hex: str, lease_ttl_s: int,
                            pubkey: Optional[str]) -> Optional[str]:
    for L in levels:
        if L <= level:
            continue
        anc_hex = _parent_hex(level, start_hex, L)
        row = _tile_row(conn, L, anc_hex, pubkey)
        if not row:
            continue
        st = row["status"]; lt = row["lease_ts"]
        if st in ("done","found"):
            return st
        if st == "running" and lt and lt >= _now_minus_iso(lease_ttl_s):
            return "running"
    return None

def _seal_tile_recursive(conn: sqlite3.Connection, level_idx: int, levels: List[int],
                         start_dec: int, end_dec: int,
                         final_status: str, rangeset_id: int, chunk_id: int,
                         lease_ttl_s: int, pubkey: Optional[str]) -> bool:
    if level_idx >= len(levels):
        return True
    L = int(levels[level_idx]); size = 1 << L

    cur = start_dec
    head_align = ((cur + size - 1) // size) * size
    if cur < head_align and cur <= end_dec:
        if not _seal_tile_recursive(conn, level_idx + 1, levels, cur, min(end_dec, head_align - 1),
                                    final_status, rangeset_id, chunk_id, lease_ttl_s, pubkey):
            return False
        cur = head_align

    while cur + size - 1 <= end_dec:
        t_start = cur
        t_hex = _hex64_upper(_align_down(t_start, L))

        anc = _ancestor_active_status(conn, levels, L, t_hex, lease_ttl_s, pubkey)
        if anc in ("done","found"):
            cur += size
            continue
        if anc == "running":
            if not _seal_tile_recursive(conn, level_idx + 1, levels, t_start, t_start + size - 1,
                                        final_status, rangeset_id, chunk_id, lease_ttl_s, pubkey):
                return False
            cur += size
            continue

        chst = _any_child_exists(conn, L, t_hex, lease_ttl_s, pubkey)
        if chst is not None:
            if not _seal_tile_recursive(conn, level_idx + 1, levels, t_start, t_start + size - 1,
                                        final_status, rangeset_id, chunk_id, lease_ttl_s, pubkey):
                return False
            cur += size
            continue

        res = _insert_tile_status(conn, L, t_hex, final_status, rangeset_id, chunk_id, pubkey)
        if res in ("inserted","covered"):
            cur += size
            continue
        if res == "conflict-running":
            if not _seal_tile_recursive(conn, level_idx + 1, levels, t_start, t_start + size - 1,
                                        final_status, rangeset_id, chunk_id, lease_ttl_s, pubkey):
                return False
            cur += size
            continue
        return False

    if cur <= end_dec:
        return _seal_tile_recursive(conn, level_idx + 1, levels, cur, end_dec,
                                    final_status, rangeset_id, chunk_id, lease_ttl_s, pubkey)
    return True

def backfill_tiles_for_chunk(conn: sqlite3.Connection, ch: sqlite3.Row, rs: RangeSet,
                             tile_levels: List[int], lease_ttl_s: int) -> int:
    levels = sorted(set(int(x) for x in tile_levels), reverse=True)
    if not levels:
        levels = DEFAULT_TILE_LEVELS[:]
    for L in levels:
        if L % 4 != 0:
            raise ValueError("--tile-levels must be multiples of 4")

    start_dec = int(ch["start_dec"]); end_dec = int(ch["end_dec"])
    final_status = "found" if ch["status"] == "found" else "done"
    pubkey = ch["pubkey"]

    inserted0 = conn.execute("SELECT COUNT(*) AS c FROM tiles").fetchone()["c"] or 0
    try:
        conn.execute("BEGIN IMMEDIATE")
        ok = _seal_tile_recursive(conn, 0, levels, start_dec, end_dec,
                                  final_status, int(rs.id), int(ch["id"]), lease_ttl_s, pubkey)
        conn.commit()
    except sqlite3.Error:
        with contextlib.suppress(Exception):
            conn.rollback()
        return 0

    inserted1 = conn.execute("SELECT COUNT(*) AS c FROM tiles").fetchone()["c"] or 0
    return int(inserted1) - int(inserted0)

def backfill_driver(conn: sqlite3.Connection, range_name: Optional[str],
                    tile_levels: List[int], lease_ttl_s: int) -> None:
    if range_name:
        rs = rangeset_by_name(conn, range_name)
        if not rs:
            print(f"Range-set '{range_name}' not found.")
            return
        rs_list = [rs]
    else:
        rs_list = [RangeSet(r["id"], r["name"], r["min_dec"], r["max_dec"], r["chunk_bits"])
                   for r in conn.execute("SELECT * FROM rangesets").fetchall()]

    for rs in rs_list:
        print(f"\nBackfilling: {rs.name}")
        rows = conn.execute("""SELECT * FROM chunks
                               WHERE rangeset_id=? AND status IN ('done','found')
                               ORDER BY claimed_ts ASC""", (rs.id,)).fetchall()
        total_ins = 0
        for ch in rows:
            total_ins += backfill_tiles_for_chunk(conn, ch, rs, tile_levels, lease_ttl_s)
        print(f"  inserted_tiles={total_ins}")


# ----------------- Compaction (pubkey-scoped) -----------------

def _compact_once(conn: sqlite3.Connection, parent_level: int) -> int:
    if parent_level < 4:
        return 0
    child_level = parent_level - 4
    parent_pref_len = 64 - (parent_level // 4)
    zeros_tail = "0" * (parent_level // 4)

    # Group by (pubkey, head)
    candidates = conn.execute("""
        SELECT pubkey,
               substr(start_hex,1, ?) AS head,
               COUNT(*) AS cnt,
               SUM(CASE WHEN status IN ('done','found') THEN 1 ELSE 0 END) AS good,
               SUM(CASE WHEN status='found' THEN 1 ELSE 0 END) AS found_cnt
          FROM tiles
         WHERE level=?
      GROUP BY pubkey, substr(start_hex,1, ?)
        HAVING cnt=16 AND good=16
    """, (parent_pref_len, child_level, parent_pref_len)).fetchall()

    merged = 0
    for row in candidates:
        parent_hex = row["head"] + zeros_tail
        pubkey = row["pubkey"]
        existing = _tile_row(conn, parent_level, parent_hex, pubkey)
        if existing:
            if existing["status"] == "running":
                continue
        else:
            status = "found" if (row["found_cnt"] or 0) > 0 else "done"
            try:
                conn.execute("""INSERT INTO tiles(pubkey,level,start_hex,status,lease_ts,rangeset_id,chunk_id)
                                VALUES(?, ?, ?, ?, ?, NULL, NULL)""",
                             (pubkey, parent_level, parent_hex, status, utc_now_iso()))
            except sqlite3.IntegrityError:
                ex2 = _tile_row(conn, parent_level, parent_hex, pubkey)
                if not ex2 or ex2["status"] == "running":
                    continue
        conn.execute("DELETE FROM tiles WHERE pubkey IS ? AND level=? AND substr(start_hex,1,?)=?",
                     (pubkey, child_level, parent_pref_len, row["head"]))
        merged += 1

    conn.commit()
    return merged

def compact_tiles(conn: sqlite3.Connection, tile_levels: List[int]) -> int:
    levels = sorted(set(int(x) for x in tile_levels))
    total = 0
    for parent_level in sorted(levels, reverse=True):
        changed = 1
        while changed:
            changed = _compact_once(conn, parent_level)
            total += changed
    return total


# ----------------- Summary view -----------------

def show_summary(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
      SELECT rs.name, rs.chunk_bits, COUNT(c.id) AS total,
             SUM(CASE WHEN c.status='done'   THEN 1 ELSE 0 END) as done,
             SUM(CASE WHEN c.status='found'  THEN 1 ELSE 0 END) as found,
             SUM(CASE WHEN c.status='running'THEN 1 ELSE 0 END) as running,
             MIN(c.claimed_ts) as since
        FROM rangesets rs
   LEFT JOIN chunks c ON c.rangeset_id=rs.id
    GROUP BY rs.id
    ORDER BY rs.name
    """).fetchall()
    if not rows:
        print("No rangesets yet.")
        return
    print("\n=== Summary ===")
    for r in rows:
        total = int(r["total"] or 0)
        running = int(r["running"] or 0)
        done = int(r["done"] or 0)
        found = int(r["found"] or 0)
        since = r["since"] or "N/A"
        dstr = f"{done}+{found}/{total}" if total else "0/0"
        print(f"{r['name']:<12} bits={r['chunk_bits']:<2}  chunks={dstr:<12}  running={running:<4}  since={since}")


# ----------------- Main -----------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Kangaroo CPU orchestrator (global tiles, no overlaps).")
    ap.add_argument("--db", required=True, help="SQLite database path")
    ap.add_argument("--range-name", help="Logical set name (e.g., B27)")
    ap.add_argument("--min-dec", type=int, help="Minimum decimal (inclusive)")
    ap.add_argument("--max-dec", type=int, help="Maximum decimal (inclusive)")
    ap.add_argument("--chunk-bits", type=int, default=48, help="Chunk width in bits (default: 48)")
    ap.add_argument("--pubkey", help="Target compressed pubkey hex (66 hex chars)")
    ap.add_argument("--threads", type=int, default=8, help="Kangaroo CPU threads (-t)")
    ap.add_argument("--dp", type=int, help="Pass -d to kangaroo (optional)")
    ap.add_argument("--max-step", type=int, help="Pass -m to kangaroo (optional, multiplier on expected ops)")
    ap.add_argument("--rest", type=int, default=0, help="Rest between chunks in milliseconds (default: 0)")
    ap.add_argument("--stop", action="store_true", help="Exit when rangeset is exhausted (default)")
    ap.add_argument("--stop-on-found", action="store_true", help="Exit immediately after first match")
    ap.add_argument("--kangaroo", default="./kangaroo", help="Path to kangaroo binary")
    ap.add_argument("--summary", action="store_true", help="Print summary and exit")
    ap.add_argument("--picker", choices=["random","sequential","entropy"], default="random",
                    help="Chunk selection strategy (default: random)")
    ap.add_argument("--sequential", action="store_true",
                    help="(Deprecated) same as --picker sequential")
    ap.add_argument("--sequential-rules", action="store_true",
                    help="Enable rules-based validate-and-jump for the sequential picker.")
    ap.add_argument("--sequential-rules-max-tries", type=int, default=1024,
                    help="Max additional forward claim attempts after the jump target (contention bound).")
    ap.add_argument("--tile-levels", default="52,48,44,40",
                    help="Comma-separated tile levels L (multiples of 4). Tile size=2^L. Default: 52,48,44,40")
    ap.add_argument("--lease-ttl-s", type=int, default=900,
                    help="Lease TTL in seconds for running tiles (default: 900)")
    ap.add_argument("--lease-refresh-s", type=int, default=60,
                    help="How often to refresh tile leases while running (default: 60)")
    ap.add_argument("--banner-refresh-s", type=float, default=2.0,
                    help="Seconds between banner refreshes (persistent, in-place; default: 2.0)")
    # Maintenance modes
    ap.add_argument("--backfill-tiles", action="store_true",
                    help="Seal historical done/found chunks into tiles and exit")
    ap.add_argument("--compact-tiles", action="store_true",
                    help="Coalesce fully covered fine tiles into coarser parents and exit")
    ap.add_argument("--all", action="store_true",
                    help="With --backfill-tiles/--compact-tiles, process all rangesets (ignore --range-name)")

    args = ap.parse_args()

    if args.summary:
        with contextlib.closing(open_db(args.db)) as conn:
            show_summary(conn)
        return 0

    if args.backfill_tiles or args.compact_tiles:
        with contextlib.closing(open_db(args.db)) as conn:
            tile_levels = [int(x) for x in str(args.tile_levels).split(",") if x.strip()]
            if args.backfill_tiles:
                rng = None if args.all else (args.range_name or None)
                backfill_driver(conn, rng, tile_levels, int(args.lease_ttl_s))
            if args.compact_tiles:
                merged = compact_tiles(conn, tile_levels)
                print(f"\nCompaction: merged {merged} parent groups.")
        return 0

    if args.sequential:
        args.picker = "sequential"
    if not args.range_name:
        if args.min_dec is None or args.max_dec is None or not args.pubkey:
            ap.error("--min-dec, --max-dec and --pubkey are required when --range-name is omitted")
        args.range_name = auto_rangeset_name(args.pubkey, int(args.min_dec), int(args.max_dec), int(args.chunk_bits))
    # optional: print(f"[info] using auto range-name: {args.range_name}")

    if args.min_dec is None or args.max_dec is None:
        ap.error("--min-dec and --max-dec are required")
    if not args.pubkey:
        ap.error("--pubkey is required")

    conn = open_db(args.db)
    _install_signal_handlers()
    try:
        rs = upsert_rangeset(conn, args.range_name, int(args.min_dec), int(args.max_dec), int(args.chunk_bits))
    except Exception as e:
        print(f"DB error: {e}", file=sys.stderr)
        return 2

    N = total_chunks(rs.min_dec, rs.max_dec, rs.chunk_bits)

    try:
        while True:
            if _STOP_REQUESTED:
                print("\nStop requested. Exiting loop.")
                break

            with contextlib.suppress(Exception):
                _ = tiles_reap_expired(conn, int(args.lease_ttl_s))

            ch = claim_or_resume_chunk(conn, rs, picker=args.picker, seq_rules_enabled=bool(args.sequential_rules), seq_rules_max_tries=int(args.sequential_rules_max_tries))
            if ch is None:
                print("\nRange-set exhausted (no chunks left).")
                total_rows = conn.execute("SELECT COUNT(*) AS c FROM chunks WHERE rangeset_id=?", (rs.id,)).fetchone()["c"] or 0
                done_rows  = conn.execute("SELECT COUNT(*) AS c FROM chunks WHERE rangeset_id=? AND status IN ('done','found')", (rs.id,)).fetchone()["c"] or 0
                print(f"Done: {done_rows}/{N} (claimed total rows: {total_rows})")
                break

            conn.execute("""UPDATE chunks
                               SET started_ts=?, nthreads=?, pubkey=?, dp=?, m_factor=?, dp_forced=?
                             WHERE id=?""",
                         (utc_now_iso(), int(args.threads), args.pubkey,
                          (args.dp if args.dp is not None else None),
                          (args.max_step if args.max_step is not None else None),
                          (1 if args.dp is not None else 0),
                          ch["id"]))
            conn.commit()

            tile_levels = [int(x) for x in str(args.tile_levels).split(",") if x.strip()]
            # FIX: pass pubkey into tile claim
            if not claim_tiles_for_chunk(conn, ch, rs, tile_levels, int(args.lease_ttl_s), args.pubkey):
                conn.execute("DELETE FROM chunks WHERE id=?", (ch["id"],))
                conn.commit()
                continue

            idx_int = int(ch["chunk_index"])
            set_idx_text = f"{idx_int}/{N}"

            try:
                rr = run_kangaroo(ch, rs, args.pubkey, args.threads,
                                  dp=args.dp, m_factor=args.max_step,
                                  kangaroo_path=args.kangaroo, set_idx_text=set_idx_text,
                                  stop_on_found=args.stop_on_found,
                                  banner_refresh_s=float(args.banner_refresh_s),
                                  lease_refresh_s=int(args.lease_refresh_s),
                                  conn=conn)
            except KeyboardInterrupt:
                conn.execute("""UPDATE chunks
                                   SET status=?, finished_ts=?
                                 WHERE id=?""",
                             (_map_status_for_schema(conn, "aborted"), utc_now_iso(), ch["id"]))
                conn.commit()
                print("\nInterrupted. Chunk marked as aborted.")
                break

            planned_stop = (args.max_step is not None)
            if rr.found_priv_hex:
                chunk_final = "found"
            elif rr.status == "done":
                chunk_final = "done"
            elif rr.status == "aborted" and planned_stop and not rr.header.get("aborted_by_user"):
                chunk_final = "done"
            else:
                chunk_final = "aborted"

            new_status = _map_status_for_schema(conn, chunk_final)
            conn.execute("""UPDATE chunks
                               SET status=?,
                                   finished_ts=?,
                                   mk_s_now=?,
                                   mk_s_avg=?,
                                   dead=?,
                                   expected_ops=?
                             WHERE id=?""",
                         (new_status, utc_now_iso(), float(rr.stats.get("now_mks",0.0)),
                          float(rr.stats.get("avg_mks",0.0)), int(rr.stats.get("dead",0)),
                          (str(rr.header.get("ops")) if rr.header.get("ops") is not None else None),
                          ch["id"]))
            conn.commit()

            if chunk_final in ("done","found"):
                finalize_tiles(conn, int(ch["id"]), chunk_final)

            if _STOP_REQUESTED:
                print("\nStop requested. Exiting loop.")
                break

            if rr.found_priv_hex and args.stop_on_found:
                print("Stopping (found).")
                break

            if args.rest and args.rest > 0:
                time.sleep(args.rest / 1000.0)
    except KeyboardInterrupt:
        _show_cursor()
        print("\nInterrupted. Exiting gracefully...")

    conn.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        _show_cursor()
        print(f"Fatal: {e}", file=sys.stderr)
        sys.exit(2)
