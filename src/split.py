"""Create the one shared split and persist its indices.   Owner: Parthraj Singh Bhati.

Split by site or by time if those columns exist, otherwise a seeded random split.
The seed comes from src.config.SEED. Train/validation/test indices are written to
data/splits/ so every member trains on byte-identical data.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
