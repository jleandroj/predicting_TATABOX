"""Minimal, dependency-free experiment tracking.

The point of this module is to capture *what makes a run reproducible* without
reaching for a hosted service (no account, no API key, no secrets): the
parameters used, the metrics produced, and the provenance (timestamp + git
commit). Each run is written as a self-contained JSON file under a runs
directory, so results are diffable, versionable, and easy to inspect.

Vendored from ``biomllab.tracking`` (bio-ml-lab) — kept dependency-free so this
repo stands alone.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_RUNS_DIR = Path("runs")


def _git_sha() -> str | None:
    """Return the short git commit SHA, or ``None`` if not in a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None


@dataclass
class Run:
    """A single tracked experiment run.

    Attributes:
        name: Human-readable label for the experiment.
        params: Inputs that define the run (config, hyperparameters, dataset id).
        metrics: Outputs measured from the run.
        run_id: Unique id; generated if not supplied.
        timestamp: UTC ISO-8601 time the run was recorded.
        git_sha: Short commit SHA for provenance, or ``None`` outside a repo.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_sha: str | None = field(default_factory=_git_sha)

    def to_json(self) -> str:
        """Serialize the run to a stable, human-readable JSON string."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def log_run(
    name: str,
    *,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> Path:
    """Record a run as a JSON file and return its path.

    The file is named ``<timestamp-safe>-<run_id>.json`` so runs sort
    chronologically. The runs directory is created if it does not exist.

    >>> import tempfile, pathlib
    >>> d = pathlib.Path(tempfile.mkdtemp())
    >>> p = log_run("smoke", params={"k": 1}, metrics={"acc": 0.5}, runs_dir=d)
    >>> p.exists()
    True
    """
    run = Run(name=name, params=params or {}, metrics=metrics or {})
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = run.timestamp.replace(":", "").replace("-", "").replace(".", "")
    path = runs_dir / f"{stamp}-{run.run_id}.json"
    path.write_text(run.to_json() + "\n", encoding="utf-8")
    return path


def load_run(path: Path) -> Run:
    """Load a previously logged run from its JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Run(**data)
