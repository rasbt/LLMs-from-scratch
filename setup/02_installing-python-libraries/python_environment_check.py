# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from importlib.metadata import PackageNotFoundError, import_module
import importlib.metadata
from os.path import dirname, exists, join, realpath
from packaging.version import parse as version_parse
import platform
import sys

if version_parse(platform.python_version()) < version_parse("3.9"):
    print("[FAIL] We recommend Python 3.9 or newer but"
          " found version %s" % (sys.version))
else:
    print("[OK] Your Python version is %s" % (platform.python_version()))


def get_packages(pkgs):
    versions = []
    for p in pkgs:
        try:
            imported = import_module(p)
            try:
                version = (getattr(imported, "__version__", None) or
                           getattr(imported, "version", None) or
                           getattr(imported, "version_info", None))
                if version is None:
                    # If common attributes don"t exist, use importlib.metadata
                    version = importlib.metadata.version(p)
                versions.append(version)
            except PackageNotFoundError:
                # Handle case where package is not installed
                versions.append("0.0")
        except ImportError:
            # Fallback if importlib.import_module fails for unexpected reasons
            versions.append("0.0")
    return versions


def get_requirements_dict():
    PROJECT_ROOT = dirname(realpath(__file__))
    PROJECT_ROOT_UP_TWO = dirname(dirname(PROJECT_ROOT))
    REQUIREMENTS_FILE = join(PROJECT_ROOT_UP_TWO, "requirements.txt")
    if not exists(REQUIREMENTS_FILE):
        REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

    d = {}
    with open(REQUIREMENTS_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            if "," in line:
                left, right = line.split(",")
                lower = right.split("#")[0].strip()
                package, _, upper = left.split(" ")
                package = package.strip()
                _, lower = lower.split(" ")
                lower = lower.strip()
                upper = upper.strip()
                d[package] = (upper, lower)
            else:
                line = line.split("#")[0].strip()
                line = line.split(" ")
                line = [ln.strip() for ln in line]
                d[line[0]] = line[-1]
    return d


def check_packages(d):
    versions = get_packages(d.keys())

    for (pkg_name, suggested_ver), actual_ver in zip(d.items(), versions):
        if isinstance(suggested_ver, tuple):
            lower, upper = suggested_ver[0], suggested_ver[1]
        else:
            lower = suggested_ver
            upper = None
        if actual_ver == "N/A":
            continue
        actual_ver = version_parse(actual_ver)
        lower = version_parse(lower)
        if upper is not None:
            upper = version_parse(upper)
        if actual_ver < lower and upper is None:
            print(f"[FAIL] {pkg_name} {actual_ver}, please upgrade to >= {lower}")
        elif actual_ver < lower:
            print(f"[FAIL] {pkg_name} {actual_ver}, please upgrade to >= {lower} and < {upper}")
        elif upper is not None and actual_ver >= upper:
            print(f"[FAIL] {pkg_name} {actual_ver}, please downgrade to >= {lower} and < {upper}")
        else:
            print(f"[OK] {pkg_name} {actual_ver}")


def main():
    d = get_requirements_dict()
    check_packages(d)


if __name__ == "__main__":
    main()
