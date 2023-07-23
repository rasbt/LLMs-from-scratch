# Sebastian Raschka, 2023

from os.path import dirname, join, realpath
from packaging.version import parse as version_parse
import platform
import sys

if version_parse(platform.python_version()) < version_parse('3.9'):
    print('[FAIL] We recommend Python 3.9 or newer but'
          ' found version %s' % (sys.version))
else:
    print('[OK] Your Python version is %s' % (platform.python_version()))


def get_packages(pkgs):
    versions = []
    for p in pkgs:
        try:
            imported = __import__(p)
            try:
                versions.append(imported.__version__)
            except AttributeError:
                try:
                    versions.append(imported.version)
                except AttributeError:
                    try:
                        versions.append(imported.version_info)
                    except AttributeError:
                        versions.append('0.0')
        except ImportError:
            print(f'[FAIL]: {p} is not installed and/or cannot be imported.')
            versions.append('N/A')
    return versions


def get_requirements_dict():
    PROJECT_ROOT = dirname(realpath(__file__))
    REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")
    d = {}
    with open(REQUIREMENTS_FILE) as f:
        for line in f:
            line = line.split(" ")
            d[line[0]] = line[-1]
    return d


def check_packages(d):
    versions = get_packages(d.keys())

    for (pkg_name, suggested_ver), actual_ver in zip(d.items(), versions):
        if actual_ver == 'N/A':
            continue
        actual_ver, suggested_ver = version_parse(actual_ver), version_parse(suggested_ver)
        if actual_ver < suggested_ver:
            print(f'[FAIL] {pkg_name} {actual_ver}, please upgrade to >= {suggested_ver}')
        else:
            print(f'[OK] {pkg_name} {actual_ver}')


if __name__ == '__main__':
    d = get_requirements_dict()
    check_packages(d)
