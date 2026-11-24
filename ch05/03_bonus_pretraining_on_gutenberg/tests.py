# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

from pathlib import Path
import os
import subprocess
import sys


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

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # In debug mode this input takes exactly 60 updates, ending on a checkpoint.
    (data_dir / "repeated_sequence.txt").write_text("a b c d" * 900, encoding="utf-8")

    script = Path(__file__).with_name("pretraining_simple.py")
    subprocess.run(
        [sys.executable, str(script), "--debug", "true", "--data_dir", str(data_dir),
         "--output_dir", str(tmp_path), "--save_ckpt_freq", "20"],
        cwd=tmp_path, check=True
    )

    saved_steps = [
        int(path.stem.split("_")[-1]) for path in tmp_path.glob("model_pg_*.pth")
        if path.stem != "model_pg_final"
    ]
    assert sorted(saved_steps) == [20, 40, 60]
