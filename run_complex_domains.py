from pathlib import Path
import subprocess
import sys

from mpi4py import MPI

from src.complex_domains import run


def main() -> None:
    root = Path(__file__).resolve().parent
    run(root / "config_complex.json", root / "results" / "complex_domains")
    if MPI.COMM_WORLD.rank == 0:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "plotting.complex_domain_figures",
                "--results",
                str(root / "results" / "complex_domains"),
                "--figures",
                str(root / "results" / "complex_domains" / "figures"),
            ],
            check=True,
            cwd=root,
        )


if __name__ == "__main__":
    main()

