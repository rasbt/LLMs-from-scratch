import os
import sys
import io
import nbformat
import types
import pytest

import tiktoken


def import_definitions_from_notebook(fullname, names):
    """Loads function definitions from a Jupyter notebook file into a module."""
    path = os.path.join(os.path.dirname(__file__), fullname + ".ipynb")
    path = os.path.normpath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Notebook file not found at: {path}")

    with io.open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    mod = types.ModuleType(fullname)
    sys.modules[fullname] = mod

    # Execute all code cells to capture dependencies
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, mod.__dict__)

    # Ensure required names are in module
    missing_names = [name for name in names if name not in mod.__dict__]
    if missing_names:
        raise ImportError(f"Missing definitions in notebook: {missing_names}")

    return mod


@pytest.fixture(scope="module")
def imported_module():
    fullname = "bpe-from-scratch"
    names = ["BPETokenizerSimple", "download_file_if_absent"]
    return import_definitions_from_notebook(fullname, names)


@pytest.fixture(scope="module")
def verdict_file(imported_module):
    """Fixture to handle downloading The Verdict file."""
    download_file_if_absent = getattr(imported_module, "download_file_if_absent", None)

    verdict_path = download_file_if_absent(
        url=(
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        ),
        filename="the-verdict.txt",
        search_dirs=["ch02/01_main-chapter-code/", "../01_main-chapter-code/", "."]
    )

    return verdict_path


@pytest.fixture(scope="module")
def gpt2_files(imported_module):
    """Fixture to handle downloading GPT-2 files."""
    download_file_if_absent = getattr(imported_module, "download_file_if_absent", None)

    search_directories = ["ch02/02_bonus_bytepair-encoder/gpt2_model/", "../02_bonus_bytepair-encoder/gpt2_model/", "."]
    files_to_download = {
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe": "vocab.bpe",
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json": "encoder.json"
    }
    paths = {filename: download_file_if_absent(url, filename, search_directories)
             for url, filename in files_to_download.items()}

    return paths


def test_tokenizer_training(imported_module, verdict_file):
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)
    tokenizer = BPETokenizerSimple()

    with open(verdict_file, "r", encoding="utf-8") as f:  # added ../01_main-chapter-code/
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
        ("Implementations", [3546, 26908, 602]),
        ("asdf asdfasdf a!!, @aba 9asdf90asdfk", [292, 7568, 355, 7568, 292, 7568, 257, 3228, 11, 2488, 15498, 860, 292, 7568, 3829, 292, 7568, 74]),
        ("Hello, world. Is this-- a test?", [15496, 11, 995, 13, 1148, 428, 438, 257, 1332, 30])
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


def test_gpt2_newline_and_eot_ids(imported_module, gpt2_files):
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)

    tok = BPETokenizerSimple()
    tok.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )

    assert "Ċ" in tok.inverse_vocab, "Missing GPT-2 newline glyph 'Ċ' in inverse_vocab"
    assert "<|endoftext|>" in tok.inverse_vocab, "Missing EOT in inverse_vocab"

    assert tok.inverse_vocab["Ċ"] == 198, "Ċ must map to id 198"
    assert tok.inverse_vocab["<|endoftext|>"] == 50256, "EOT must be 50256"

    if "\n" not in tok.inverse_vocab:
        tok.inverse_vocab["\n"] = tok.inverse_vocab["Ċ"]
    assert tok.inverse_vocab["\n"] == 198, r"'\n' must map to 198 via Ċ"

    assert tok.vocab[198] == "Ċ", "Don't overwrite vocab[198]; keep it 'Ċ'"
    assert tok.vocab[50256] == "<|endoftext|>", "Don't map <|endoftext|> to anything else"


def test_no_eot_aliasing_and_disallowed_logic(imported_module, gpt2_files):
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)
    tok = BPETokenizerSimple()
    tok.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )
    tik = tiktoken.get_encoding("gpt2")

    text = "Hello<|endoftext|>\nworld"
    # When not allowed, our encode should raise ValueError like tiktoken
    with pytest.raises(ValueError):
        tok.encode(text)

    # When allowed, both tokenizers should match
    ids_ours = tok.encode(text, allowed_special={"<|endoftext|>"})
    ids_tik = tik.encode(text, allowed_special={"<|endoftext|>"})
    assert ids_ours == ids_tik, "Mismatch vs tiktoken with EOT allowed"


@pytest.mark.parametrize(
    "text",
    [
        "a\nb",
        "a\n\nb",
        "\nHello",
        "Hello\n",
        "a\r\nb",
    ],
)
def test_newline_roundtrip_and_equivalence(imported_module, gpt2_files, text):
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)
    tok = BPETokenizerSimple()
    tok.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )
    tik = tiktoken.get_encoding("gpt2")

    ids_ours = tok.encode(text)
    ids_tik = tik.encode(text)

    assert ids_ours == ids_tik, f"Mismatch vs tiktoken for: {repr(text)}"
    # Each "\n" should correspond to id 198
    expected_lf_count = text.count("\n")
    assert ids_ours.count(198) == expected_lf_count

    dec = tok.decode(ids_ours)
    assert dec == text


def test_space_newline_space_patterns(imported_module, gpt2_files):
    BPETokenizerSimple = getattr(imported_module, "BPETokenizerSimple", None)
    tok = BPETokenizerSimple()
    tok.load_vocab_and_merges_from_openai(
        vocab_path=gpt2_files["encoder.json"], bpe_merges_path=gpt2_files["vocab.bpe"]
    )
    tik = tiktoken.get_encoding("gpt2")

    samples = [
        "Hello \nworld",
        "Hello\n world",
    ]
    for s in samples:
        assert tok.encode(s) == tik.encode(s), f"Mismatch vs tiktoken: {repr(s)}"