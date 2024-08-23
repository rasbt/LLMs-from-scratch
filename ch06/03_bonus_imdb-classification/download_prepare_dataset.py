# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import os
import sys
import tarfile
import time
import urllib.request
import pandas as pd


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
    else:
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        percent = count * block_size * 100 / total_size

        speed = int(progress_size / (1024 * duration)) if duration else 0
        sys.stdout.write(
            f"\r{int(percent)}% | {progress_size / (1024**2):.2f} MB "
            f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
        )
        sys.stdout.flush()


def download_and_extract_dataset(dataset_url, target_file, directory):
    if not os.path.exists(directory):
        if os.path.exists(target_file):
            os.remove(target_file)
        urllib.request.urlretrieve(dataset_url, target_file, reporthook)
        print("\nExtracting dataset ...")
        with tarfile.open(target_file, "r:gz") as tar:
            tar.extractall()
    else:
        print(f"Directory `{directory}` already exists. Skipping download.")


def load_dataset_to_dataframe(basepath="aclImdb", labels={"pos": 1, "neg": 0}):
    data_frames = []  # List to store each chunk of DataFrame
    for subset in ("test", "train"):
        for label in ("pos", "neg"):
            path = os.path.join(basepath, subset, label)
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                    # Create a DataFrame for each file and add it to the list
                    data_frames.append(pd.DataFrame({"text": [infile.read()], "label": [labels[label]]}))
    # Concatenate all DataFrame chunks together
    df = pd.concat(data_frames, ignore_index=True)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # Shuffle the DataFrame
    return df


def partition_and_save(df, sizes=(35000, 5000, 10000)):
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Get indices for where to split the data
    train_end = sizes[0]
    val_end = sizes[0] + sizes[1]

    # Split the DataFrame
    train = df_shuffled.iloc[:train_end]
    val = df_shuffled.iloc[train_end:val_end]
    test = df_shuffled.iloc[val_end:]

    # Save to CSV files
    train.to_csv("train.csv", index=False)
    val.to_csv("validation.csv", index=False)
    test.to_csv("test.csv", index=False)


if __name__ == "__main__":
    dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    print("Downloading dataset ...")
    download_and_extract_dataset(dataset_url, "aclImdb_v1.tar.gz", "aclImdb")
    print("Creating data frames ...")
    df = load_dataset_to_dataframe()
    print("Partitioning and saving data frames ...")
    partition_and_save(df)
