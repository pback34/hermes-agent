from __future__ import annotations

from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    _filter_compacted_messages,
    _is_inter_session_injection,
)


QUEUE_DELIVERY = (
    "\n[INTER-SESSION MESSAGE from coder:worker (id: 42, to: coder:main)]\n"
    "Priority: normal\nPlease inspect the patch.\n"
    "[To reply, use send_to_session with to_profile='coder', to_label='worker']\n"
)
INTERRUPT_DELIVERY = (
    "\n============================================================\n"
    "[INTERRUPTING INTER-SESSION MESSAGE from coder:worker (id: 43)]\n"
    "Stop and inspect this immediately.\n"
)


def test_compacted_tail_drops_rendered_inter_session_injections():
    assert _is_inter_session_injection({"role": "user", "content": QUEUE_DELIVERY})
    assert _is_inter_session_injection({"role": "user", "content": INTERRUPT_DELIVERY})
    result = _filter_compacted_messages(
        [
            {"role": "user", "content": QUEUE_DELIVERY},
            {"role": "user", "content": INTERRUPT_DELIVERY},
            {"role": "assistant", "content": QUEUE_DELIVERY},
        ]
    )
    assert result == [{"role": "assistant", "content": QUEUE_DELIVERY}]


def test_similar_user_text_is_not_treated_as_injection():
    message = {
        "role": "user",
        "content": "I saw [INTER-SESSION MESSAGE from someone (id: 42)] in a log.",
    }
    assert not _is_inter_session_injection(message)
    assert _filter_compacted_messages([message]) == [message]


def test_exact_duplicate_user_content_collapses_but_other_roles_do_not():
    duplicate = {"role": "user", "content": "same request"}
    result = _filter_compacted_messages(
        [duplicate.copy(), {"role": "assistant", "content": "same request"}, duplicate.copy()]
    )
    assert result == [duplicate, {"role": "assistant", "content": "same request"}]


def test_two_compactions_do_not_reintroduce_injected_tail_row():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.5,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    compressor.tail_token_budget = 10
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial request"},
        {"role": "assistant", "content": "initial reply"},
        {"role": "user", "content": "older request"},
        {"role": "assistant", "content": "older reply"},
        {"role": "user", "content": QUEUE_DELIVERY},
        {"role": "assistant", "content": "working on it"},
    ]
    with (
        patch.object(compressor, "_protect_head_size", return_value=2),
        patch.object(compressor, "_find_tail_cut_by_tokens", return_value=5),
        patch.object(compressor, "_generate_summary", return_value="summary"),
    ):
        first = compressor.compress(messages, current_tokens=90_000)
        second = compressor.compress(first, current_tokens=90_000)

    assert not any(message.get("content") == QUEUE_DELIVERY for message in first)
    assert not any(message.get("content") == QUEUE_DELIVERY for message in second)
    assert sum(message.get("content") == "summary" for message in second) <= 1
