# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).

import ast
import copy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATHS = [
    "ch05/11_qwen3/qwen3-chat-interface/qwen3-chat-interface-multiturn.py",
]


class Tokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))


@pytest.fixture(params=SCRIPT_PATHS)
def helpers(request):
    path = REPO_ROOT / request.param
    tree = ast.parse(path.read_text())
    # Load only the pure helpers, without downloading a model or starting a chat app.
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"build_prompt_from_history", "encode_chat_history"}]
    namespace = {}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["build_prompt_from_history"], namespace["encode_chat_history"]


@pytest.fixture
def history():
    return [
        {"role": "system", "content": "Answer in German."},
        {"role": "user", "content": "Old question " * 20},
        {"role": "assistant", "content": "Old answer " * 20},
        {"role": "user", "content": "Recent question"},
        {"role": "assistant", "content": "Recent answer"},
        {"role": "user", "content": "Newest question 🚀"},
    ]


def test_exact_fit_keeps_the_original_prompt_and_assistant_header(helpers, history):
    build, encode = helpers
    expected = Tokenizer().encode(build(history))
    actual, removed = encode(history, Tokenizer(), len(expected) + 16, 16)
    assert actual == expected
    assert removed == 0
    assert bytes(actual).decode().endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("old_turns", [1, 2])
def test_trimming_keeps_system_latest_message_and_complete_recent_turns(helpers, history, old_turns):
    build, encode = helpers
    original = copy.deepcopy(history)
    expected_history = [history[0], *history[1 + 2 * old_turns:]]
    expected = Tokenizer().encode(build(expected_history))
    actual, removed = encode(history, Tokenizer(), len(expected) + 16, 16)
    assert actual == expected
    assert removed == old_turns
    assert history == original


def test_one_token_over_budget_removes_a_whole_turn(helpers, history):
    build, encode = helpers
    full_length = len(Tokenizer().encode(build(history)))
    actual, removed = encode(history, Tokenizer(), full_length + 15, 16)
    assert actual == Tokenizer().encode(build([history[0], *history[3:]]))
    assert removed == 1
    assert len(actual) <= full_length - 1


@pytest.mark.parametrize("role", ["system", "user"])
def test_oversized_required_messages_raise_without_changing_history(helpers, history, role):
    _, encode = helpers
    index = 0 if role == "system" else -1
    history[index]["content"] = "Required content " * 100
    original = copy.deepcopy(history)
    with pytest.raises(ValueError, match="system message and latest user message.*available for input"):
        encode(history, Tokenizer(), 256, 16)
    assert history == original


def test_all_leading_system_messages_are_preserved(helpers, history):
    build, encode = helpers
    history.insert(1, {"role": "system", "content": "Keep the response concise."})
    expected = Tokenizer().encode(build([*history[:2], history[-1]]))
    actual, removed = encode(history, Tokenizer(), len(expected) + 16, 16)
    assert actual == expected
    assert removed == 2


@pytest.mark.parametrize("budget", [-1, 256, 257])
def test_invalid_output_budget_is_reported_explicitly(helpers, history, budget):
    _, encode = helpers
    with pytest.raises(ValueError, match="output token limit"):
        encode(history, Tokenizer(), 256, budget)
