"""Extra coverage for evals/results_schema.py — lines 354 and 372."""
from unittest.mock import MagicMock


def test_log_results_to_wandb_no_checkpoint_step_with_mmlu():
    """Covers line 354: run.log(log_payload) when checkpoint_step is None + mmlu table."""
    from evals import results_schema

    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run
    mock_wandb.Table = MagicMock(return_value=MagicMock())

    payload = {
        "task": "lm_harness",
        "results": {"acc": 0.5},
        "metadata": {"mmlu_subjects": {"math": 0.7, "science": 0.8}},
        "checkpoint_step": None,
        "experiment_name": "test_exp",
        "git_commit": "abc123",
    }

    orig = results_schema.WANDB_AVAILABLE
    orig_wandb = results_schema.wandb
    try:
        results_schema.WANDB_AVAILABLE = True
        results_schema.wandb = mock_wandb
        result = results_schema.log_results_to_wandb(payload, config={})
        assert result is True
        # Verify run.log was called (no step arg since checkpoint_step is None)
        mock_run.log.assert_called()
    finally:
        results_schema.WANDB_AVAILABLE = orig
        results_schema.wandb = orig_wandb


def test_log_results_to_wandb_with_checkpoint_step_and_mmlu():
    """Covers line 372: run.log(..., step=int(checkpoint_step)) with mmlu table."""
    from evals import results_schema

    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run
    mock_wandb.Table = MagicMock(return_value=MagicMock())

    payload = {
        "task": "lm_harness",
        "results": {"acc": 0.5},
        "metadata": {"mmlu_subjects": {"math": 0.7}},
        "checkpoint_step": 1000,
        "experiment_name": "test_exp",
        "git_commit": "abc123",
    }

    orig = results_schema.WANDB_AVAILABLE
    orig_wandb = results_schema.wandb
    try:
        results_schema.WANDB_AVAILABLE = True
        results_schema.wandb = mock_wandb
        result = results_schema.log_results_to_wandb(payload, config={})
        assert result is True
        # Verify run.log was called with step kwarg
        calls = mock_run.log.call_args_list
        step_calls = [c for c in calls if c.kwargs.get("step") == 1000 or (len(c.args) > 1 and c.args[1] == 1000)]
        assert len(step_calls) > 0
    finally:
        results_schema.WANDB_AVAILABLE = orig
        results_schema.wandb = orig_wandb


def test_log_results_to_wandb_run_finish_not_callable():
    """Covers the else branch: wandb.finish() when run.finish is not callable."""
    from evals import results_schema

    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_run.finish = "not_callable"  # not callable → triggers wandb.finish()
    mock_wandb.init.return_value = mock_run

    payload = {
        "task": "perplexity",
        "results": {"ppl": 20.0},
        "metadata": {},
        "checkpoint_step": None,
        "experiment_name": "test",
    }

    orig = results_schema.WANDB_AVAILABLE
    orig_wandb = results_schema.wandb
    try:
        results_schema.WANDB_AVAILABLE = True
        results_schema.wandb = mock_wandb
        result = results_schema.log_results_to_wandb(payload, config={})
        assert result is True
        mock_wandb.finish.assert_called_once()
    finally:
        results_schema.WANDB_AVAILABLE = orig
        results_schema.wandb = orig_wandb
