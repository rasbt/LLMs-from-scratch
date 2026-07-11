#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "html5lib>=1.1",
#   "nbconvert>=7.16",
#   "nbformat>=5.10",
#   "pytest>=9.1.1,<10",
#   "requests>=2.32",
# ]
# ///
"""Check Markdown and notebook links with modern pytest collection APIs."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

import html5lib
import nbformat
import pytest
import requests
from nbconvert.filters import markdown2html


SUPPORTED_EXTENSIONS = {".ipynb", ".md"}
_SESSION_KEY = pytest.StashKey[requests.Session]()


@dataclass(frozen=True)
class Link:
    """A link and the label used in pytest output."""

    label: str
    target: str


class BrokenLinkError(Exception):
    """Describe why a link check failed."""

    def __init__(self, target: str, reason: str) -> None:
        super().__init__(reason)
        self.target = target
        self.reason = reason


def links_from_html(label: str, html: str) -> Iterator[Link]:
    """Extract HTTP and relative links from rendered HTML."""
    parsed = html5lib.parse(html, namespaceHTMLElements=False)
    attributes = {"a": "href", "iframe": "src", "img": "src"}

    for element in parsed.iter():
        attribute = attributes.get(element.tag)
        if attribute is None:
            continue

        target = element.get(attribute)
        if not target or target.startswith("#"):
            continue

        scheme = urlsplit(target).scheme.lower()
        if scheme and scheme not in {"http", "https"}:
            continue

        yield Link(
            label=f"{label} <{element.tag} {attribute}={target}>",
            target=target,
        )


def links_from_file(path: Path) -> Iterator[Link]:
    """Extract links from a Markdown file or Markdown notebook cells."""
    if path.suffix.lower() == ".md":
        yield from links_from_html(str(path), markdown2html(path.read_text(encoding="utf-8")))
        return

    notebook = nbformat.read(path, as_version=4)
    for cell_number, cell in enumerate(notebook.cells):
        if cell.cell_type != "markdown":
            continue
        yield from links_from_html(f"Cell {cell_number}", markdown2html(cell.source))


def is_ignored(target: str, patterns: list[str]) -> bool:
    """Return whether any configured regular expression matches a target."""
    return any(re.match(pattern, target) for pattern in patterns)


def local_link_error(source: Path, target: str) -> str | None:
    """Return an error for a missing relative target, otherwise ``None``."""
    parsed = urlsplit(target)
    target_path = unquote(parsed.path)

    if not target_path:
        return None
    if target_path.startswith("/"):
        return "absolute path link"

    candidate = source.parent / target_path
    if candidate.exists():
        return None

    if candidate.suffix == ".html":
        for extension in SUPPORTED_EXTENSIONS:
            if candidate.with_suffix(extension).exists():
                return None

    return f"No such file: {candidate}"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register link-checker options."""
    group = parser.getgroup("link checker")
    group.addoption("--check-links", action="store_true", help="Check links for validity")
    group.addoption(
        "--check-links-ignore",
        action="append",
        default=[],
        metavar="REGEX",
        help="Ignore targets matching this regular expression; may be repeated",
    )
    group.addoption(
        "--check-links-timeout",
        type=float,
        default=float(os.environ.get("CHECK_LINKS_TIMEOUT", "10")),
        metavar="SECONDS",
        help="Timeout for each HTTP request",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Create one HTTP session for the test run."""
    if not config.getoption("check_links"):
        return

    session = requests.Session()
    session.headers["User-Agent"] = "LLMs-from-scratch-link-checker"
    config.stash[_SESSION_KEY] = session


def pytest_unconfigure(config: pytest.Config) -> None:
    """Close the shared HTTP session."""
    session = config.stash.get(_SESSION_KEY, None)
    if session is not None:
        session.close()


def pytest_collect_file(file_path: Path, parent: pytest.Collector) -> LinkFile | None:
    """Collect supported documentation files when link checking is enabled."""
    if not parent.config.getoption("check_links"):
        return None
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None

    return LinkFile.from_parent(parent, path=file_path)


class LinkFile(pytest.File):
    """Collect one pytest item per link in a documentation file."""

    def collect(self) -> Iterator[LinkItem]:
        patterns = self.config.getoption("check_links_ignore")
        for link in links_from_file(self.path):
            if is_ignored(link.target, patterns):
                continue
            yield LinkItem.from_parent(self, name=link.label, target=link.target)


class LinkItem(pytest.Item):
    """Validate one HTTP or relative link."""

    def __init__(self, *, target: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.target = target

    def runtest(self) -> None:
        parsed = urlsplit(self.target)
        if parsed.scheme in {"http", "https"}:
            self._check_http_link(parsed._replace(fragment="").geturl())
            return

        error = local_link_error(self.path, self.target)
        if error is not None:
            raise BrokenLinkError(self.target, error)

    def _check_http_link(self, target: str) -> None:
        session = self.config.stash[_SESSION_KEY]
        timeout = self.config.getoption("check_links_timeout")

        try:
            response = session.get(target, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            raise BrokenLinkError(self.target, str(error)) from error

    def repr_failure(self, excinfo: pytest.ExceptionInfo[BaseException]) -> str:
        """Report concise link failures."""
        if isinstance(excinfo.value, BrokenLinkError):
            return f"{excinfo.value.target}: {excinfo.value.reason}"
        return super().repr_failure(excinfo)

    def reportinfo(self) -> tuple[Path, int | None, str]:
        """Associate failures with the source documentation file."""
        return self.path, None, self.target


def main(arguments: list[str] | None = None) -> int:
    """Run pytest with only this documentation collector enabled."""
    args = sys.argv[1:] if arguments is None else arguments
    return pytest.main(
        ["--check-links", "-p", "no:python", *args],
        plugins=[sys.modules[__name__]],
    )


if __name__ == "__main__":
    raise SystemExit(main())
