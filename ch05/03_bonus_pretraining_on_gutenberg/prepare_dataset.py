# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Script that processes the Project Gutenberg files into fewer larger files.
"""

import argparse
import os
import re


def combine_files(file_paths, target_dir, max_size_mb=500, separator="<|endoftext|>", fallback_encoding="latin1"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    current_content = []
    current_size = 0
    file_counter = 1

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            # Attempt to read the file with a fallback encoding
            print(f"Warning: UnicodeDecodeError encountered. Trying fallback encoding for {file_path}")
            with open(file_path, "r", encoding=fallback_encoding) as file:
                content = file.read()

        # Regular expression to replace multiple blank lines with a single blank line
        content = re.sub(r'\n\s*\n', '\n\n', content)
        estimated_size = len(content.encode("utf-8"))

        if current_size + estimated_size > max_size_mb * 1024 * 1024:
            target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
            with open(target_file_path, "w", encoding="utf-8") as target_file:
                target_file.write(separator.join(current_content))
            file_counter += 1
            current_content = [content]
            current_size = estimated_size
        else:
            current_content.append(content)
            current_size += estimated_size

    if current_content:
        target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
        with open(target_file_path, "w", encoding="utf-8") as target_file:
            target_file.write(separator.join(current_content))
    return file_counter


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess and combine text files for pretraining")

    parser.add_argument("--data_dir", type=str, default="gutenberg/data",
                        help="Directory containing the downloaded raw training data")
    parser.add_argument("--max_size_mb", type=int, default=500,
                        help="The maximum file size for each concatenated file in megabytes")
    parser.add_argument("--output_dir", type=str, default="gutenberg_preprocessed",
                        help="Directory where the preprocessed data will be saved")

    args = parser.parse_args()

    all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(args.data_dir)
                 for name in files if name.endswith((".txt", ".txt.utf8")) and "raw" not in path]

    print(f"{len(all_files)} file(s) to process.")
    file_counter = combine_files(all_files, args.output_dir, max_size_mb=args.max_size_mb)
    print(f"{file_counter} file(s) saved in {os.path.abspath(args.output_dir)}")
