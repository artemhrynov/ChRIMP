from pathlib import Path

import pytest
from tokenizers import Tokenizer


TOKENIZER_PATHS = [
    Path("src/chrimp/agent/tokenizer/mechsmiles_tokenizer.json"),
    Path("src/chrimp/agent/tokenizer/mechsmiles_tokenizer_folder/tokenizer.json"),
]

STEREO_MODE_EXAMPLES = [
    ("TH(1,'invert',((5,6),))", "'invert'"),
    ("TH(1,'retain',((5,6),))", "'retain'"),
    ("TH(1,'clear',())", "'clear'"),
    ("TH(1,'unknown',())", "'unknown'"),
]

FULL_MECHSMILES_EXAMPLES = [
    "[C@:1]([F:2])([Cl:3])([Br:4])[I:5]|(6,1)|TH(1,'invert',((5,6),))",
    "[OH-:1].[C@:2]([H:3])([Cl:4])[Br:5]|(1,2)|TH(2,'retain',((5,1),))",
    "[C@:1]([F:2])([Cl:3])([Br:4])[I:5]|((2,3),1)|TH(1,'invert',((4,6),))",
    "[OH-:1].[C@:2]([H:3])([Cl:4])[Br:5]|(1,2);((2,5),5)|TH(2,'invert',((5,1),))",
    "[C@:1]([F:2])([Cl:3])([Br:4])[I:5]|(6,1)|TH(1,'clear',())",
    "[C@:1]([F:2])([Cl:3])([Br:4])[I:5]|(6,1)|TH(1,'unknown',())",
    "F[C@:1]([Cl:2])([Br:3])[I:4]|(5,1)|TH(1,'retain',((4,5),))",
    "F[C@@:1]([Cl:2])([Br:3])[I:4]|(5,1)|TH(1,'invert',((4,5),))",
]


@pytest.fixture(params=TOKENIZER_PATHS, ids=lambda path: path.name)
def tokenizer(request):
    path = request.param
    assert path.exists(), f"Tokenizer file does not exist: {path}"
    return Tokenizer.from_file(str(path))


def test_tokenizer_vocab_contains_tetrahedral_stereo_tokens(tokenizer):
    vocab = tokenizer.get_vocab()

    expected_tokens = [
        "TH",
        "'invert'",
        "'retain'",
        "'clear'",
        "'unknown'",
    ]

    for token in expected_tokens:
        assert token in vocab, f"Missing token from vocabulary: {token}"

    ids = [vocab[token] for token in expected_tokens]
    assert len(ids) == len(set(ids)), "Stereo tokens should have unique token IDs"


def test_tokenizer_splits_invert_stereo_update_exactly(tokenizer):
    text = "TH(1,'invert',((5,6),))"

    tokens = tokenizer.encode(text).tokens

    assert tokens == [
        "TH",
        "(",
        "1",
        ",",
        "'invert'",
        ",",
        "(",
        "(",
        "5",
        ",",
        "6",
        ")",
        ",",
        ")",
        ")",
    ]


@pytest.mark.parametrize("text, mode_token", STEREO_MODE_EXAMPLES)
def test_tokenizer_recognizes_all_tetrahedral_stereo_modes(tokenizer, text, mode_token):
    tokens = tokenizer.encode(text).tokens

    assert "[unk]" not in tokens
    assert "TH" in tokens
    assert mode_token in tokens


@pytest.mark.parametrize("text", FULL_MECHSMILES_EXAMPLES)
def test_tokenizer_handles_full_stereo_mechsmiles_without_unknown_tokens(tokenizer, text):
    tokens = tokenizer.encode(text).tokens

    assert "[unk]" not in tokens


def test_tokenizer_keeps_existing_mechsmiles_syntax_working(tokenizer):
    text = "[OH-:1].[C:2][Br:3]|(1,2);((2,3),3)"

    tokens = tokenizer.encode(text).tokens

    assert "[unk]" not in tokens
    assert "|" in tokens
    assert ";" in tokens
    assert "(" in tokens
    assert ")" in tokens
    assert "," in tokens