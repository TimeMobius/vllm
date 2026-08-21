import pytest

from tools.rwkv7_logprob_regression import compare_traces


def _row(name, text, tokens, selected, top):
    return {
        "name": name,
        "trace": {
            "text": text,
            "tokens": tokens,
            "selected_logprobs": selected,
            "top_logprobs": top,
        },
    }


def test_compare_traces_reports_distribution_error_and_top1():
    reference = [
        _row("p", "ab", ["a", "b"], [-0.1, -0.2], [{"a": -0.1, "x": -1.0}, {"b": -0.2}])
    ]
    candidate = [
        _row(
            "p",
            "ab",
            ["a", "b"],
            [-0.11, -0.25],
            [{"a": -0.11, "x": -1.2}, {"b": -0.25}],
        )
    ]

    result = compare_traces(reference, candidate)

    assert result["text_mismatch_count"] == 0
    assert result["top1_disagreement_count"] == 0
    assert result["max_common_topk_abs_error"] == pytest.approx(0.2)
    assert result["max_selected_logprob_abs_error"] == pytest.approx(0.05)


def test_compare_traces_detects_text_and_top1_mismatch():
    reference = [_row("p", "ab", ["a"], [-0.1], [{"a": -0.1, "x": -1.0}])]
    candidate = [_row("p", "ac", ["a"], [-0.1], [{"x": -0.01, "a": -0.1}])]

    result = compare_traces(reference, candidate)

    assert result["text_mismatch_count"] == 1
    assert result["top1_disagreement_count"] == 1
    assert result["top1_disagreement_rate"] == 1.0
