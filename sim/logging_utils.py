from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def json_dumps_canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_json(obj: Any) -> str:
    return sha256_hex(json_dumps_canonical(obj).encode("utf-8"))


def now_utc_compact() -> str:
    # YYYYMMDDTHHMMSSZ
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass(frozen=True)
class RunPaths:
    run_dir: str
    events_path: str
    metrics_path: str
    manifest_path: str
    checkpoints_dir: str

    @staticmethod
    def create(base_runs_dir: str, run_id: str) -> "RunPaths":
        run_dir = os.path.join(base_runs_dir, run_id)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        ensure_dir(checkpoints_dir)
        return RunPaths(
            run_dir=run_dir,
            events_path=os.path.join(run_dir, "events.jsonl"),
            metrics_path=os.path.join(run_dir, "metrics.jsonl"),
            manifest_path=os.path.join(run_dir, "manifest.json"),
            checkpoints_dir=checkpoints_dir,
        )


class JsonlWriter:
    def __init__(self, path: str):
        self._path = path
        parent = os.path.dirname(path)
        if parent:
            ensure_dir(parent)
        self._fp = open(path, "a", encoding="utf-8")

    @property
    def path(self) -> str:
        return self._path

    def write(self, obj: Dict[str, Any]) -> None:
        self._fp.write(json_dumps_canonical(obj))
        self._fp.write("\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


def make_run_id(prefix: str = "run") -> str:
    # Non-deterministic; only for filesystem organization.
    suffix = sha256_hex(f"{time.time_ns()}".encode("utf-8"))[:8]
    return f"{prefix}_{now_utc_compact()}_{suffix}"


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json_dumps_canonical(manifest))
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def file_exists(path: str) -> bool:
    return os.path.exists(path)


def maybe_truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def stable_env_id(kind: str, tick: int, phase: str, turn_index: int) -> str:
    # Deterministic within a run.
    payload = f"{kind}:{tick}:{phase}:{turn_index}".encode("utf-8")
    return sha256_hex(payload)[:16]


def observation_digest(obs_obj: Any) -> str:
    # Digest for audit trails; do not treat as cryptographic proof.
    return hash_json(obs_obj)[:16]
