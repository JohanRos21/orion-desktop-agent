"""Validate that the ORION Colab wake word notebook is JSON and Python-valid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_NOTEBOOK = Path("training/wakeword/train_orion_colab.ipynb")


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def _cell_title(source: str) -> str:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "<celda vacia>"


def _python_for_compile(source: str) -> str:
    lines = source.splitlines(keepends=True)
    first_code_index = next(
        (index for index, line in enumerate(lines) if line.strip()),
        None,
    )
    if first_code_index is None:
        return "pass\n"

    first = lines[first_code_index].lstrip()
    if first.startswith("%%bash"):
        return "pass  # Colab %%bash cell omitted by validator\n"

    transformed: list[str] = []
    skipping_magic_continuation = False
    for line in lines:
        if skipping_magic_continuation:
            transformed.append("pass  # Colab magic continuation omitted by validator\n")
            skipping_magic_continuation = line.rstrip().endswith("\\")
            continue

        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("!") or (
            stripped.startswith("%") and not stripped.startswith("%%")
        ):
            transformed.append(f"{indent}pass  # Colab magic omitted by validator\n")
            skipping_magic_continuation = line.rstrip().endswith("\\")
        else:
            transformed.append(line)
    return "".join(transformed)


def validate_notebook(path: Path) -> int:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Notebook JSON ERROR: {path}: {exc}")
        return 1

    print("Notebook JSON OK")

    errors = 0
    code_cells = 0
    for cell_number, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        code_cells += 1
        source = _cell_source(cell)
        title = _cell_title(source)
        compiled_source = _python_for_compile(source)
        try:
            compile(compiled_source, f"{path}:cell-{cell_number}", "exec")
        except SyntaxError as exc:
            errors += 1
            print(f"Celda {cell_number}: {title}")
            print(f"  SyntaxError: {exc.msg} en linea {exc.lineno}, columna {exc.offset}")
            if exc.text:
                print(f"  {exc.text.rstrip()}")

    if errors:
        print(f"{errors} celda(s) Python no compilan")
        return 1

    print("Todas las celdas Python compilan correctamente")
    print(f"Celdas Python compiladas: {code_cells}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate JSON and Python syntax for the ORION Colab notebook.",
    )
    parser.add_argument(
        "notebook",
        nargs="?",
        type=Path,
        default=DEFAULT_NOTEBOOK,
        help=f"Notebook path. Default: {DEFAULT_NOTEBOOK}",
    )
    args = parser.parse_args()
    return validate_notebook(args.notebook)


if __name__ == "__main__":
    sys.exit(main())
