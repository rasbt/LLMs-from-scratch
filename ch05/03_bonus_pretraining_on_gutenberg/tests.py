# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

from pathlib import Path
import os
import subprocess


def test_pretraining():

    sequence = "a b c d"
    repetitions = 1000
    content = sequence * repetitions

    folder_path = Path("gutenberg") / "data"
    file_name = "repeated_sequence.txt"

    os.makedirs(folder_path, exist_ok=True)

    with open(folder_path/file_name, "w") as file:
        file.write(content)

    result = subprocess.run(
        ["python", "pretraining_simple.py", "--debug", "true"],
        capture_output=True, text=True
    )
    print(result.stdout)
    assert "Maximum GPU memory allocated" in result.stdout
