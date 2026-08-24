from pathlib import Path
import subprocess
import sys
from mpi4py import MPI

from src.manufactured import run
from src.metadata import write_metadata


def main() -> None:
    root = Path(__file__).resolve().parent
    write_metadata(root / "results" / "software_versions.json")
    run(root / "config.json", root / "results")
    if MPI.COMM_WORLD.rank == 0:
        subprocess.run(
            [sys.executable, "-m", "plotting.make_figures", "--results", str(root / "results"), "--figures", str(root / "results" / "figures")],
            check=True,
            cwd=root,
        )


if __name__ == "__main__":
    main()
