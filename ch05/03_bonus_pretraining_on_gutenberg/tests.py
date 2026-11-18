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


def test_checkpoints_follow_save_ckpt_freq(tmp_path):

    folder_path = Path("gutenberg") / "data"
    os.makedirs(folder_path, exist_ok=True)

    with open(folder_path/"repeated_sequence.txt", "w") as file:
        file.write("a b c d" * 1000)

    save_ckpt_freq = 20
    result = subprocess.run(
        ["python", "pretraining_simple.py", "--debug", "true",
         "--output_dir", str(tmp_path), "--save_ckpt_freq", str(save_ckpt_freq)],
        capture_output=True, text=True
    )
    assert "Maximum GPU memory allocated" in result.stdout

    saved_steps = [
        int(path.stem.split("_")[-1]) for path in tmp_path.glob("model_pg_*.pth")
        if path.stem != "model_pg_final"
    ]
    assert saved_steps
    assert all(step > 0 and step % save_ckpt_freq == 0 for step in saved_steps)
