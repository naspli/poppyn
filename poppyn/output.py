"""Publish complete catalogues and protect unrelated files on reruns."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
from stat import S_ISREG
from tempfile import TemporaryDirectory

from PIL.PngImagePlugin import PngImageFile

from .render import MAP_FILENAME


def validate_destination(output, report_path, *, raw):
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError(f"output must be a directory: {output}")
    if not output.exists() or not any(output.iterdir()):
        return
    map_path = output / MAP_FILENAME
    try:
        if map_path.is_file() and not map_path.is_symlink():
            # Poppyn writes the export tag before the PNG pixel data.
            with PngImageFile(map_path) as image:
                kind = image.info.get("poppyn-kind")
            if kind not in ("binary", "raw"):
                raise ValueError
            directory_pattern = r"tiles_[1-9][0-9]*(?:/[0-9]+){0,2}"
            file_pattern = rf"{re.escape(MAP_FILENAME)}|tiles_[1-9][0-9]*/(?:manifest\.json|[0-9]+/[0-9]+/[0-9]+\.png)"
        else:
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            if manifest["version"] not in (1, 2) or not manifest["levels"]:
                raise ValueError
            kind = manifest.get("kind", "binary")
            directory_pattern = r"[0-9]+(?:/[0-9]+)?"
            file_pattern = r"manifest\.json|[0-9]+/(?:map(?:-annotated)?\.png|[0-9]+/[0-9]+\.png)"
    except (OSError, SyntaxError, ValueError, KeyError, TypeError):
        raise ValueError(f"output must be empty or an existing Poppyn export: {output}") from None
    existing_raw = kind == "raw"
    if existing_raw != raw:
        kind = "raw heatmaps" if existing_raw else "binary maps"
        raise ValueError(f"choose another output directory. {output} already contains {kind}")
    for item in output.rglob("*"):
        if report_path is not None and item == report_path and item.is_file() and not item.is_symlink():
            continue
        relative = item.relative_to(output).as_posix()
        pattern = directory_pattern if item.is_dir() else file_pattern
        if item.is_symlink() or not re.fullmatch(pattern, relative):
            raise ValueError(f"output contains a file or directory outside the catalogue layout: {item}")


@contextmanager
def output_lock(path):
    while True:
        lock = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o666)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            locked_file = os.fstat(lock)
            # A waiting run can acquire a lock that the previous run already unlinked.
            try:
                current_file = path.lstat()
            except FileNotFoundError:
                continue
            if not os.path.samestat(locked_file, current_file):
                continue
            if not S_ISREG(locked_file.st_mode) or locked_file.st_size:
                raise ValueError(f"output lock must be an empty regular file: {path}")
            try:
                yield
            finally:
                path.unlink(missing_ok=True)
            return
        finally:
            os.close(lock)


@contextmanager
def catalogue_stage(output, report_path, *, raw):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(output.parent / f".{output.name}.lock"):
        validate_destination(output, report_path, raw=raw)
        with TemporaryDirectory(prefix="poppyn-output-", dir=output.parent) as directory:
            staged = Path(directory) / "next"
            staged.mkdir()
            yield staged


def publish_catalogue(staged, output, report, report_path):
    if report_path is not None and report_path.is_relative_to(output):
        inside = staged / report_path.relative_to(output)
        if inside.exists():
            raise ValueError(f"report path conflicts with a generated catalogue entry: {report_path}")
        write_json(inside, report)
        report_path = None

    previous = staged.parent / "previous"
    if output.exists():
        output.rename(previous)
    try:
        staged.rename(output)
        if report_path is not None:
            write_json(report_path, report)
    except BaseException:
        if not staged.exists():
            output.rename(staged)
        if previous.exists():
            previous.rename(output)
        raise


def write_json(path, data):
    with TemporaryDirectory(prefix="poppyn-json-", dir=path.parent) as directory:
        staged = Path(directory) / path.name
        staged.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        staged.replace(path)
