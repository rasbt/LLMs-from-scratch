# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os
from tempfile import TemporaryDirectory

import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir, source=None):
    """Try OpenAI, Hugging Face, then Backblaze, or pass a source to use only that host."""
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    sources = {
        "openai": "https://openaipublic.blob.core.windows.net/gpt-2/models",
        "hf": "https://huggingface.co/rasbt/gpt2-tensorflow/resolve/main",
        "backblaze": "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2",
    }
    if source is not None and source not in sources:
        raise ValueError(f"Unknown source {source!r}. Choose from {tuple(sources)}.")
    base_urls = sources.values() if source is None else [sources[source]]
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        urls = [f"{base_url}/{model_size}/{filename}" for base_url in base_urls]
        file_path = os.path.join(model_dir, filename)
        download_file(urls[0], file_path, *urls[1:])

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None, second_backup_url=None):
    urls = [u for u in (url, backup_url, second_backup_url) if u is not None]
    file_size = None
    for i, download_url in enumerate(urls):
        try:
            with requests.get(download_url, stream=True, timeout=60, headers={"Accept-Encoding": "identity"}) as response:
                response.raise_for_status()
                file_size = int(response.headers.get("Content-Length", 0))
                if file_size and os.path.isfile(destination) and os.path.getsize(destination) == file_size:
                    print(f"File already exists and is up-to-date: {destination}")
                    return

                # Replace the checkpoint file only after the download finishes.
                with TemporaryDirectory(dir=os.path.dirname(os.path.abspath(destination))) as tmp_dir:
                    tmp_path = os.path.join(tmp_dir, "download")
                    desc = os.path.basename(destination)
                    with open(tmp_path, "wb") as file, tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as bar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            file.write(chunk)
                            bar.update(len(chunk))
                        if file_size and file.tell() != file_size:
                            raise requests.exceptions.ChunkedEncodingError(f"Incomplete download from {download_url}")
                    os.replace(tmp_path, destination)
                return
        except requests.exceptions.RequestException:
            if i + 1 == len(urls):
                if file_size is None and os.path.isfile(destination):
                    print(f"Could not verify the download online. Using local file: {destination}")
                    return
                raise
            print(f"Download from {download_url} failed. Trying {urls[i + 1]}.")


# Alternative way using `requests`
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
