"""Publish complete catalogues and protect unrelated files on reruns."""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory


def validate_destination(output, report_path, *, raw):
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError(f"output must be a directory: {output}")
    if not output.exists() or not any(output.iterdir()):
        return
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["version"] not in (1, 2) or not manifest["levels"]:
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError(f"output must be empty or a Poppyn catalogue: {output}") from None
    existing_raw = manifest.get("kind") == "raw"
    if existing_raw != raw:
        kind = "raw heatmaps" if existing_raw else "binary maps"
        raise ValueError(f"choose another output directory. {output} already contains {kind}")
    for item in output.rglob("*"):
        if report_path is not None and item == report_path and item.is_file() and not item.is_symlink():
            continue
        relative = item.relative_to(output).as_posix()
        pattern = r"[0-9]+(?:/[0-9]+)?" if item.is_dir() else (
            r"manifest\.json|[0-9]+/(?:map(?:-annotated)?\.png|[0-9]+/[0-9]+\.png)"
        )
        if item.is_symlink() or not re.fullmatch(pattern, relative):
            raise ValueError(f"output contains a file or directory outside the catalogue layout: {item}")


@contextmanager
def catalogue_stage(output, report_path, *, raw):
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
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
