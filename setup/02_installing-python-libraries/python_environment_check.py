# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from importlib.metadata import PackageNotFoundError, import_module, version as get_version
from os.path import dirname, exists, join, realpath
from packaging.version import parse as version_parse
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import platform
import sys

if version_parse(platform.python_version()) < version_parse("3.9"):
    print("[FAIL] We recommend Python 3.9 or newer but found version %s" % sys.version)
else:
    print("[OK] Your Python version is %s" % platform.python_version())


def get_packages(pkgs):
    """
    Returns a dictionary mapping package names (in lowercase) to their installed version.
    """
    PACKAGE_MODULE_OVERRIDES = {
        "tensorflow-cpu": ["tensorflow", "tensorflow_cpu"],
    }
    result = {}
    for p in pkgs:
        # Determine possible module names to try.
        module_names = PACKAGE_MODULE_OVERRIDES.get(p.lower(), [p])
        version_found = None
        for module_name in module_names:
            try:
                imported = import_module(module_name)
                version_found = getattr(imported, "__version__", None)
                if version_found is None:
                    try:
                        version_found = get_version(module_name)
                    except PackageNotFoundError:
                        version_found = None
                if version_found is not None:
                    break  # Stop if we successfully got a version.
            except ImportError:
                # Also try replacing hyphens with underscores as a fallback.
                alt_module = module_name.replace("-", "_")
                if alt_module != module_name:
                    try:
                        imported = import_module(alt_module)
                        version_found = getattr(imported, "__version__", None)
                        if version_found is None:
                            try:
                                version_found = get_version(alt_module)
                            except PackageNotFoundError:
                                version_found = None
                        if version_found is not None:
                            break
                    except ImportError:
                        continue
                continue
        if version_found is None:
            version_found = "0.0"
        result[p.lower()] = version_found
    return result


def get_requirements_dict():
    """
    Parses requirements.txt and returns a dictionary mapping package names (lowercase)
    to a specifier string (e.g. ">=2.18.0,<3.0"). It uses packaging.requirements.Requirement
    to properly handle environment markers.
    """

    PROJECT_ROOT = dirname(realpath(__file__))
    PROJECT_ROOT_UP_TWO = dirname(dirname(PROJECT_ROOT))
    REQUIREMENTS_FILE = join(PROJECT_ROOT_UP_TWO, "requirements.txt")
    if not exists(REQUIREMENTS_FILE):
        REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

    reqs = {}
    with open(REQUIREMENTS_FILE) as f:
        for line in f:
            # Remove inline comments and trailing whitespace.
            # This splits on the first '#' and takes the part before it.
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                req = Requirement(line)
            except Exception as e:
                print(f"Skipping line due to parsing error: {line} ({e})")
                continue
            # Evaluate the marker if present.
            if req.marker is not None and not req.marker.evaluate():
                continue
            # Store the package name and its version specifier.
            spec = str(req.specifier) if req.specifier else ">=0"
            reqs[req.name.lower()] = spec
    return reqs


def check_packages(reqs):
    """
    Checks the installed versions of packages against the requirements.
    """
    installed = get_packages(reqs.keys())
    for pkg_name, spec_str in reqs.items():
        spec_set = SpecifierSet(spec_str)
        actual_ver = installed.get(pkg_name, "0.0")
        if actual_ver == "N/A":
            continue
        actual_ver_parsed = version_parse(actual_ver)
        # If the installed version is a pre-release, allow pre-releases in the specifier.
        if actual_ver_parsed.is_prerelease:
            spec_set.prereleases = True
        if actual_ver_parsed not in spec_set:
            print(f"[FAIL] {pkg_name} {actual_ver_parsed}, please install a version matching {spec_set}")
        else:
            print(f"[OK] {pkg_name} {actual_ver_parsed}")


def main():
    reqs = get_requirements_dict()
    check_packages(reqs)


if __name__ == "__main__":
    main()
