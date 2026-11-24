"""Content-addressed storage for source grids and solved maps."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile

import numpy as np


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


class ArrayCache:
    def __init__(self, directory, refresh):
        self.directory = directory
        self.refresh = refresh

    def load(self, name, *fields):
        if self.directory is None or self.refresh:
            return None
        try:
            with np.load(self.directory / f"{name}.npz", allow_pickle=False) as archive:
                return [archive[field] for field in fields]
        except (OSError, EOFError, ValueError, KeyError, BadZipFile):
            return None

    def save(self, name, **arrays):
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="writing-", dir=self.directory) as directory:
            path = self.directory / f"{name}.npz"
            # Write a complete archive on the same filesystem before publishing it.
            temporary = Path(directory) / path.name
            np.savez_compressed(temporary, allow_pickle=False, **arrays)
            temporary.replace(path)
