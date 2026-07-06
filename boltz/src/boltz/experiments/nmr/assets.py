from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NMRAssets:
    """Locations of required pickled hydrogen libraries."""

    assets_dir: Path

    @property
    def fragments_pickle(self) -> Path:
        return self.assets_dir / "fragments.pickle"

    @property
    def names_pickle(self) -> Path:
        return self.assets_dir / "names.pickle"

    def validate(self) -> None:
        if not self.assets_dir.exists():
            raise FileNotFoundError(f"NMR assets dir does not exist: {self.assets_dir}")
        if not self.fragments_pickle.exists():
            raise FileNotFoundError(
                f"Missing `fragments.pickle` in {self.assets_dir}. "
                "Provide it via --nmr_assets_dir or BOLTZ_NMR_ASSETS_DIR."
            )
        if not self.names_pickle.exists():
            raise FileNotFoundError(
                f"Missing `names.pickle` in {self.assets_dir}. "
                "Provide it via --nmr_assets_dir or BOLTZ_NMR_ASSETS_DIR."
            )


def resolve_assets_dir(explicit: str | Path | None = None) -> Path:
    """Resolve assets directory from argument or environment variable."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("BOLTZ_NMR_ASSETS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # Default to adjacent folder (user can drop pickles here).
    return (Path(__file__).resolve().parent / "assets").resolve()

