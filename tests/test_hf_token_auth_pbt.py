from __future__ import annotations

import os
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st


_env_keys = st.text(
    alphabet=st.characters(blacklist_characters="\x00", blacklist_categories=("Cs",)),
    min_size=1,
    max_size=64,
).filter(lambda k: k != "HF_TOKEN")

_env_values = st.text(
    alphabet=st.characters(blacklist_characters="\x00", blacklist_categories=("Cs",)),
    max_size=128,
)

_base_env_without_token = st.dictionaries(
    keys=_env_keys,
    values=_env_values,
    max_size=20,
)

_base_env_any = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(
            blacklist_characters="\x00", blacklist_categories=("Cs",)
        ),
        min_size=1,
        max_size=64,
    ),
    values=_env_values,
    max_size=20,
)


@given(
    token=st.text(
        alphabet=st.characters(
            blacklist_characters="\x00", blacklist_categories=("Cs",)
        ),
        min_size=1,
    ),
    base_env=_base_env_without_token,
)
@settings(max_examples=200)
def test_hf_token_always_injected_when_present(token: str, base_env: dict):
    from run_modal_training import _hf_env

    with patch.dict(os.environ, {"HF_TOKEN": token}, clear=False):
        result = _hf_env(base_env)

    assert result["HF_TOKEN"] == token


@given(base_env=_base_env_without_token)
@settings(max_examples=200)
def test_hf_env_unchanged_when_no_token(base_env: dict):
    from run_modal_training import _hf_env

    clean_env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
    with patch.dict(os.environ, clean_env, clear=True):
        result = _hf_env(base_env)

    assert result == base_env


@given(
    base_env=_base_env_any,
    token=st.one_of(
        st.none(),
        st.text(
            alphabet=st.characters(
                blacklist_characters="\x00", blacklist_categories=("Cs",)
            ),
            min_size=0,
            max_size=128,
        ),
    ),
)
@settings(max_examples=300)
def test_hf_env_is_superset_of_base_env(base_env: dict, token):
    from run_modal_training import _hf_env

    if token is not None:
        env_patch = {"HF_TOKEN": token}
        clean = False
    else:
        env_patch = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        clean = True

    with patch.dict(os.environ, env_patch, clear=clean):
        result = _hf_env(base_env)

    for key, value in base_env.items():
        if key == "HF_TOKEN":
            continue  # intentionally overwritten by _hf_env
        assert key in result, f"Key {key!r} was removed by _hf_env"
        assert result[key] == value, (
            f"Key {key!r} was modified: expected {value!r}, got {result[key]!r}"
        )
