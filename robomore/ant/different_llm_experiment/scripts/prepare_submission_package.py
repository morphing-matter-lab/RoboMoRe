#!/usr/bin/env python3
"""Export a sanitized submission package from a local RoboMoRe workspace.

This script copies selected files into a clean output directory and masks
common private information patterns:
  - API keys and bearer tokens
  - absolute local paths (/home/<user>/..., C:\\Users\\<user>\\...)
  - emails and localhost/private custom endpoints

Usage:
  python scripts/prepare_submission_package.py \
    --source /path/to/RoboMoRe \
    --output /path/to/submission_package
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".sh",
    ".xml",
    ".ipynb",
    ".rst",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".conda",
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
}

EXCLUDED_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".tmp",
    ".log",
    ".lnk",
}

EXCLUDED_FILE_NAMES = {
    "MUJOCO_LOG.TXT",
}

EXCLUDED_NAME_PATTERNS = [
    re.compile(r"^SAC_.*steps$", re.IGNORECASE),
]

MAX_COPY_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB


def sanitize_text(content: str) -> str:
    """Mask common sensitive patterns."""
    patterns = [
        # OpenAI-like keys.
        (r"sk-[A-Za-z0-9_-]{12,}", "sk-<MASKED>"),
        # Generic API key assignments.
        (
            r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)\s*=\s*['\"][^'\"]+['\"]",
            r"\1=\"<MASKED>\"",
        ),
        # Bearer tokens.
        (r"(?i)\bBearer\s+[A-Za-z0-9._\-+/=]{16,}", "Bearer <MASKED>"),
        # Emails.
        (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<MASKED_EMAIL>"),
        # Linux/macOS absolute home paths.
        (r"/home/[^/\s]+/", "/home/<user>/"),
        (r"/Users/[^/\s]+/", "/Users/<user>/"),
        # Windows absolute user paths.
        (r"[A-Za-z]:\\Users\\[^\\/\s]+\\", r"C:\\Users\\<user>\\"),
        # Localhost style URLs.
        (r"http://localhost(?::\d+)?", "<LOCALHOST_URL>"),
        (r"http://127\.0\.0\.1(?::\d+)?", "<LOCALHOST_URL>"),
        # Generic private/proxy style URLs (non-public IP or localhost-like hosts).
        (r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0)(?::\d+)?[^\s\"']*", "<LOCALHOST_URL>"),
    ]

    out = content
    for pattern, replacement in patterns:
        out = re.sub(pattern, replacement, out)
    return out


def should_skip(path: Path) -> bool:
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    if any(path.name.endswith(suf) for suf in EXCLUDED_FILE_SUFFIXES):
        return True
    if any(pat.match(path.name) for pat in EXCLUDED_NAME_PATTERNS):
        return True
    try:
        if path.stat().st_size > MAX_COPY_FILE_SIZE_BYTES:
            return True
    except OSError:
        return True
    return False


def should_descend(path: Path) -> bool:
    return path.name not in EXCLUDED_DIR_NAMES


def copy_and_sanitize(source: Path, destination: Path) -> tuple[int, int]:
    copied = 0
    sanitized = 0

    for entry in source.rglob("*"):
        rel = entry.relative_to(source)
        out_path = destination / rel

        if entry.is_dir():
            if should_descend(entry):
                out_path.mkdir(parents=True, exist_ok=True)
            continue

        if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
            continue
        if should_skip(entry):
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if entry.suffix.lower() in TEXT_EXTENSIONS:
            try:
                raw = entry.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                shutil.copy2(entry, out_path)
                copied += 1
                continue
            cleaned = sanitize_text(raw)
            out_path.write_text(cleaned, encoding="utf-8")
            copied += 1
            if cleaned != raw:
                sanitized += 1
        else:
            shutil.copy2(entry, out_path)
            copied += 1

    return copied, sanitized


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sanitized submission package.")
    parser.add_argument("--source", type=Path, required=True, help="Path to repository root.")
    parser.add_argument("--output", type=Path, required=True, help="Output folder path.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output directory if it already exists.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()

    if not source.exists() or not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")

    if output.exists():
        if not args.force:
            raise SystemExit(
                f"Output already exists: {output}. Re-run with --force to overwrite."
            )
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)

    copied, sanitized = copy_and_sanitize(source, output)
    print(f"Done. Copied files: {copied}, sanitized files: {sanitized}")
    print(f"Sanitized package path: {output}")


if __name__ == "__main__":
    main()
