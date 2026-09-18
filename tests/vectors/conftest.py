from __future__ import annotations

from pathlib import Path

import trajectory_editor


vectors_package = Path(__file__).resolve().parents[2] / "vector" / "src" / "trajectory_editor"
if str(vectors_package) not in trajectory_editor.__path__:
    trajectory_editor.__path__.append(str(vectors_package))
