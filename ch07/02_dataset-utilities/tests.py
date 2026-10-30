# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

import importlib.util
from pathlib import Path


def load_find_near_duplicates_module():
    script_path = Path(__file__).resolve().parent / "find-near-duplicates.py"
    spec = importlib.util.spec_from_file_location("find_near_duplicates", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_near_duplicates_with_empty_entries():
    module = load_find_near_duplicates_module()

    json_data = [
        {"instruction": "Name a prime number.", "input": "", "output": "7"},
        {"instruction": "Translate 'cat' to German.", "input": "", "output": "Katze"},
        {"instruction": "Sort the numbers.", "input": "3, 1, 2", "output": "1, 2, 3"},
        {"instruction": "Classify the animal.", "input": "Penguin", "output": "Bird"},
        {"instruction": "Summarize the text.",
         "input": "The quick brown fox jumps over the lazy dog.", "output": "A fox jumps over a dog."},
        {"instruction": "Paraphrase the text.",
         "input": "The quick brown fox jumps over the lazy dog!", "output": "A dog is jumped over by a fox."},
    ]

    filtered_json_data, near_duplicates = module.find_near_duplicates(json_data, threshold=0.9, key="input")

    assert len(near_duplicates) == 1
    first, second, _ = near_duplicates[0]
    assert first is json_data[4]
    assert second is json_data[5]
    assert filtered_json_data == json_data[:5]


def test_near_duplicates_with_all_entries_empty():
    module = load_find_near_duplicates_module()

    json_data = [
        {"instruction": "Name a prime number.", "input": "", "output": "7"},
        {"instruction": "Name a color.", "input": "", "output": "Red"},
    ]

    filtered_json_data, near_duplicates = module.find_near_duplicates(json_data, key="input")

    assert near_duplicates == []
    assert filtered_json_data == json_data
