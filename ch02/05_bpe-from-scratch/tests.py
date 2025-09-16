from bpe import BpeTokenizer


# Test case 1: Test with the vocabulary and merges from the chapter
def test_bpe_tokenizer_from_chapter():
    # Setup with vocabulary and merges from the chapter
    vocab = {"<|endoftext|>": 10257, "a": 97, "b": 98, "c": 99}
    merges = {("a", "b"): 10258}
    tokenizer = BpeTokenizer(vocab, merges)

    # Test encoding
    text = "abc"
    expected_ids = [10258, 99]
    assert tokenizer.encode(text) == expected_ids, "Encoding failed for 'abc'"

    # Test decoding
    ids = [10258, 99]
    expected_text = "abc"
    assert tokenizer.decode(ids) == expected_text, "Decoding failed for [10258, 99]"


# Test case 2: Test with a different vocabulary and merges
def test_bpe_tokenizer_different_vocab_merges():
    # Setup with a different vocabulary and merges
    vocab = {"a": 1, "b": 2, "c": 3, "d": 4}
    merges = {("a", "b"): 5, ("c", "d"): 6}
    tokenizer = BpeTokenizer(vocab, merges)

    # Test encoding
    text = "abcd"
    expected_ids = [5, 6]
    assert tokenizer.encode(text) == expected_ids, "Encoding failed for 'abcd'"

    # Test decoding
    ids = [5, 6]
    expected_text = "abcd"
    assert tokenizer.decode(ids) == expected_text, "Decoding failed for [5, 6]"
