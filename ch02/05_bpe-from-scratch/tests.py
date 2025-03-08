import os
import sys
import io
import nbformat
import types
import pytest

import tiktoken


def import_definitions_from_notebook(fullname, names):
    current_dir = os.getcwd()
    path = os.path.join(current_dir, fullname + ".ipynb")
    path = os.path.normpath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Notebook file not found at: {path}")

    with io.open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    mod = types.ModuleType(fullname)
    sys.modules[fullname] = mod

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell_code = cell.source
            for name in names:
                if f"def {name}" in cell_code or f"class {name}" in cell_code:
                    exec(cell_code, mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def imported_module():
    fullname = "bpe-from-scratch"
    names = ["BPETokenizerSimple", "download_file_if_absent"]
    return import_definitions_from_notebook(fullname, names)


@pytest.fixture(scope="module")
def gpt2_files(imported_module):
    """Fixture to handle downloading GPT-2 files."""
    download_file_if_absent = getattr(imported_module, "download_file_if_absent", None)

    search_directories = [".", "../02_bonus_bytepair-encoder/gpt2_model/"]
    files_to_download = {
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe": "vocab.bpe",
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json": "encoder.json"
    }
    paths = {filename: download_file_if_absent(url, filename, search_directories)
             for url, filename in files_to_download.items()}

    return paths


def test_tokenizer_training(imported_module, gpt2_files):
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)
    download_file_if_absent = getattr(imported_module, "download_file_if_absent", None)

    tokenizer = BPETokenizerSimple()

    filename = "the-verdict.txt"

    download_file_if_absent(
        url=(
             "https://raw.githubusercontent.com/rasbt/"
             "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
             "the-verdict.txt"
        ),
        filename=filename,
        search_dirs="."
    )

    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer.train(text, vocab_size=1000, allowed_special={"<|endoftext|>"})
    assert len(tokenizer.vocab) == 1000, "Tokenizer vocabulary size mismatch."
    assert len(tokenizer.bpe_merges) == 742, "Tokenizer BPE merges count mismatch."

    input_text = "Jack embraced beauty through art and life."
    token_ids = tokenizer.encode(input_text)
    assert token_ids == [424, 256, 654, 531, 302, 311, 256, 296, 97, 465, 121, 595, 841, 116, 287, 466, 256, 326, 972, 46], "Token IDs do not match expected output."

    assert tokenizer.decode(token_ids) == input_text, "Decoded text does not match the original input."

    tokenizer.save_vocab_and_merges(vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt")
    tokenizer2 = BPETokenizerSimple()
    tokenizer2.load_vocab_and_merges(vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt")
    assert tokenizer2.decode(token_ids) == input_text, "Decoded text mismatch after reloading tokenizer."


def test_gpt2_tokenizer_openai_simple(imported_module, gpt2_files):
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)

    tokenizer_gpt2 = BPETokenizerSimple()
    tokenizer_gpt2.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )

    assert len(tokenizer_gpt2.vocab) == 50257, "GPT-2 tokenizer vocabulary size mismatch."

    input_text = "This is some text"
    token_ids = tokenizer_gpt2.encode(input_text)
    assert token_ids == [1212, 318, 617, 2420], "Tokenized output does not match expected GPT-2 encoding."


def test_gpt2_tokenizer_openai_edgecases(imported_module, gpt2_files):
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)

    tokenizer_gpt2 = BPETokenizerSimple()
    tokenizer_gpt2.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )
    tik_tokenizer = tiktoken.get_encoding("gpt2")

    test_cases = [
        ("Hello,", [15496, 11]),
        ("Implementations", [3546, 26908, 602])
    ]

    errors = []

    for input_text, expected_tokens in test_cases:
        tik_tokens = tik_tokenizer.encode(input_text)
        gpt2_tokens = tokenizer_gpt2.encode(input_text)

        print(f"Text: {input_text}")
        print(f"Expected Tokens: {expected_tokens}")
        print(f"tiktoken Output: {tik_tokens}")
        print(f"BPETokenizerSimple Output: {gpt2_tokens}")
        print("-" * 40)

        if tik_tokens != expected_tokens:
            errors.append(f"Tiktokenized output does not match expected GPT-2 encoding for '{input_text}'.\n"
                          f"Expected: {expected_tokens}, Got: {tik_tokens}")

        if gpt2_tokens != expected_tokens:
            errors.append(f"Tokenized output does not match expected GPT-2 encoding for '{input_text}'.\n"
                          f"Expected: {expected_tokens}, Got: {gpt2_tokens}")

    if errors:
        pytest.fail("\n".join(errors))
