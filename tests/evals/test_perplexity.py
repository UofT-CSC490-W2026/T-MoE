import math

import torch

from evals.perplexity import (
    compute_document_nll,
    evaluate_text_documents,
    summarize_language_model_metrics,
)


class _PerfectNextTokenModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.vocab_size = vocab_size

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape

        logits = torch.zeros(batch, seq_len, self.vocab_size, dtype=torch.float32)

        for pos in range(seq_len - 1):
            next_tokens = input_ids[:, pos + 1]

            logits[:, pos, :] = -20.0

            logits[:, pos, next_tokens] = 20.0

        return logits, None, None


class _UniformModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.vocab_size = vocab_size

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape

        logits = torch.zeros(batch, seq_len, self.vocab_size, dtype=torch.float32)

        return logits, None, None


class _WhitespaceTokenizer:
    def __init__(self, token_map):
        self.token_map = token_map

        self.model_max_length = 128

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        tokens = [self.token_map[token] for token in text.split()]

        return {"input_ids": torch.tensor([tokens], dtype=torch.long)}


def test_compute_document_nll_counts_each_target_once_with_overlap():
    model = _PerfectNextTokenModel(vocab_size=16)

    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], dtype=torch.long)

    total_nll, total_tokens = compute_document_nll(
        model,
        input_ids,
        stride=2,
        max_length=4,
        device="cpu",
    )

    assert total_tokens == 6

    assert total_nll < 1e-4


def test_summarize_language_model_metrics_computes_ppl_and_bpb():
    total_nll = 6 * math.log(10)

    summary = summarize_language_model_metrics(
        total_nll=total_nll,
        total_tokens=6,
        total_bytes=12,
    )

    assert math.isclose(summary["ppl"], 10.0, rel_tol=1e-6)

    expected_bpb = total_nll / (math.log(2) * 12)

    assert math.isclose(summary["bpb"], expected_bpb, rel_tol=1e-6)


def test_evaluate_text_documents_aggregates_metrics():
    tokenizer = _WhitespaceTokenizer(
        {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3, "epsilon": 4}
    )

    model = _UniformModel(vocab_size=5)

    summary = evaluate_text_documents(
        model,
        tokenizer,
        ["alpha beta gamma", "delta epsilon"],
        stride=2,
        max_length=4,
        device="cpu",
        include_bpb=True,
    )

    assert summary["documents_scored"] == 2

    assert summary["tokens_scored"] == 3

    assert math.isclose(summary["ppl"], 5.0, rel_tol=1e-6)

    assert "bpb" in summary
