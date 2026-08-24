from __future__ import annotations

import json
import platform
from pathlib import Path

from mpi4py import MPI
import basix
import dolfinx
import ffcx
import numpy
import petsc4py
import ufl


def write_metadata(path: Path) -> None:
    if MPI.COMM_WORLD.rank != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": platform.python_version(),
        "dolfinx": dolfinx.__version__,
        "basix": basix.__version__,
        "ufl": ufl.__version__,
        "ffcx": ffcx.__version__,
        "petsc4py": petsc4py.__version__,
        "numpy": numpy.__version__,
        "mpi_size": MPI.COMM_WORLD.size,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
