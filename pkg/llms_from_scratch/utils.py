# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Internal utility functions (not intended for public use)

import ast
import re
import types
from pathlib import Path

import nbformat
import requests


def _extract_imports(src: str):
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in tree.body:
        if isinstance(node, ast.Import):
            parts = []
            for n in node.names:
                parts.append(f"{n.name} as {n.asname}" if n.asname else n.name)
            out.append("import " + ", ".join(parts))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            parts = []
            for n in node.names:
                parts.append(f"{n.name} as {n.asname}" if n.asname else n.name)
            level = "." * node.level if getattr(node, "level", 0) else ""
            out.append(f"from {level}{module} import " + ", ".join(parts))
    return out


def _extract_defs_and_classes_from_code(src):
    lines = src.splitlines()
    kept = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("@"):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith(("def ", "class ")):
                kept.append(line)
                i += 1
                continue
        if stripped.startswith("def ") or stripped.startswith("class "):
            kept.append(line)
            base_indent = len(line) - len(stripped)
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() == "":
                    kept.append(nxt)
                    i += 1
                    continue
                indent = len(nxt) - len(nxt.lstrip())
                if indent <= base_indent and not nxt.lstrip().startswith(("#", "@")):
                    break
                kept.append(nxt)
                i += 1
            continue
        i += 1

    code = "\n".join(kept)

    # General rule:
    # replace functions defined like `def load_weights_into_xxx(ClassName, ...`
    # with `def load_weights_into_xxx(model, ...`
    code = re.sub(
        r"(def\s+load_weights_into_\w+\s*\()\s*\w+\s*,",
        r"\1model,",
        code
    )
    return code


def import_definitions_from_notebook(nb_dir_or_path, notebook_name=None, *, extra_globals=None):
    nb_path = Path(nb_dir_or_path)
    if notebook_name is not None:
        nb_file = nb_path / notebook_name if nb_path.is_dir() else nb_path
    else:
        nb_file = nb_path

    if not nb_file.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_file}")

    nb = nbformat.read(nb_file, as_version=4)

    import_lines = []
    seen = set()
    for cell in nb.cells:
        if cell.cell_type == "code":
            for line in _extract_imports(cell.source):
                if line not in seen:
                    import_lines.append(line)
                    seen.add(line)

    for required in ("import torch", "import torch.nn as nn"):
        if required not in seen:
            import_lines.append(required)
            seen.add(required)

    pieces = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            pieces.append(_extract_defs_and_classes_from_code(cell.source))

    src = "\n\n".join(import_lines + pieces)

    mod_name = nb_file.stem.replace("-", "_").replace(" ", "_") or "notebook_defs"
    mod = types.ModuleType(mod_name)

    if extra_globals:
        mod.__dict__.update(extra_globals)

    exec(src, mod.__dict__)
    return mod


def download_file(url, out_dir="."):
    """Simple file download utility for tests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name
    dest = out_dir / filename

    if dest.exists():
        return dest

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return dest
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")
