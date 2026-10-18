"""Shared helpers: project paths, .env loading, JSON I/O, hashing."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    """Read ROOT/.env into os.environ (never overrides variables already set)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass


def require_env(name, hint):
    load_env()
    v = os.environ.get(name, "").strip()
    if not v or v.startswith("paste_"):
        raise SystemExit(f"{name} is not set. {hint}")
    return v


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()[:16]
