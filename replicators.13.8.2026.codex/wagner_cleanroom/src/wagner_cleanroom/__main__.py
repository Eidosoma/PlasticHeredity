import os


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from .cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
