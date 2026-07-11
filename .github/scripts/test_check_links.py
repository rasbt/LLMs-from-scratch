"""Tests for the repository-local hyperlink checker."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from check_links import is_ignored, links_from_file, local_link_error


class LinkExtractionTests(unittest.TestCase):
    """Ensure the checker covers the repository's documentation formats."""

    def test_markdown_links_and_images_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "README.md"
            path.write_text(
                "[local](guide.md) ![cover](https://example.com/cover.png) "
                "[mail](mailto:test@example.com) [section](#section)",
                encoding="utf-8",
            )

            targets = [link.target for link in links_from_file(path)]

        self.assertEqual(targets, ["guide.md", "https://example.com/cover.png"])

    def test_notebook_checks_markdown_cells_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.ipynb"
            path.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "markdown",
                                "id": "markdown-cell",
                                "metadata": {},
                                "source": ["[docs](https://example.com/docs)"],
                            },
                            {
                                "cell_type": "code",
                                "execution_count": None,
                                "id": "code-cell",
                                "metadata": {},
                                "outputs": [],
                                "source": ["url = 'https://example.com/code'"],
                            },
                        ],
                        "metadata": {},
                        "nbformat": 4,
                        "nbformat_minor": 5,
                    }
                ),
                encoding="utf-8",
            )

            targets = [link.target for link in links_from_file(path)]

        self.assertEqual(targets, ["https://example.com/docs"])


class LinkValidationTests(unittest.TestCase):
    """Ensure local validation and configured exclusions remain reliable."""

    def test_local_link_accepts_query_and_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "README.md"
            target = Path(directory) / "guide.md"
            source.touch()
            target.touch()

            error = local_link_error(source, "guide.md?view=1#section")

        self.assertIsNone(error)

    def test_missing_local_link_reports_resolved_path(self) -> None:
        source = Path("docs/README.md")

        error = local_link_error(source, "missing.md")

        self.assertEqual(error, "No such file: docs/missing.md")

    def test_ignore_patterns_match_from_the_start(self) -> None:
        self.assertTrue(is_ignored("https://github.com/example", [r"https://github[.]com/.*"]))
        self.assertFalse(is_ignored("https://example.com", [r"https://github[.]com/.*"]))


if __name__ == "__main__":
    unittest.main()
