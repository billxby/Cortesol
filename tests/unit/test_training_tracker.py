from __future__ import annotations

from cortesol.train.tracker import parse_worker_metrics


def test_worker_tracker_parses_sft_curve_gpu_and_eta() -> None:
    text = "\n".join(
        (
            'HEARTBEAT {"stage":"sft_step","ts":100.0,"step":10,"loss":0.7,'
            '"grad_norm":4.0,"learning_rate":0.0001,'
            '"gpu":{"device_name":"RTX 5090","gpu_util_pct":99}}',
            "{'loss': '0.7', 'grad_norm': '4', 'learning_rate': '0.0001', "
            "'entropy': '0.6', 'mean_token_accuracy': '0.8', "
            "'num_tokens': '1000', 'epoch': '1'}",
            'HEARTBEAT {"stage":"sft_step","ts":160.0,"step":20,"loss":0.3,'
            '"grad_norm":2.0,"learning_rate":0.00005,'
            '"gpu":{"device_name":"RTX 5090","gpu_util_pct":100,'
            '"memory_used_gb":18.5}}',
            "{'loss': '0.3', 'grad_norm': '2', 'learning_rate': '0.00005', "
            "'entropy': '0.2', 'mean_token_accuracy': '0.94', "
            "'num_tokens': '2000', 'epoch': '2'}",
        )
    )
    result = parse_worker_metrics(text, max_steps=50)
    assert result["step"] == 20
    assert result["progress"] == 0.4
    assert result["seconds_per_step"] == 6.0
    assert result["eta_seconds"] == 180.0
    assert result["latest"]["loss"] == 0.3
    assert result["curve"][-1]["mean_token_accuracy"] == 0.94
    assert result["gpu"]["gpu_util_pct"] == 100


def test_worker_tracker_handles_setup_without_training_samples() -> None:
    text = 'HEARTBEAT {"stage":"sft_model_load","ts":100.0,"gpu":{"memory_used_gb":0.6}}'
    result = parse_worker_metrics(text, max_steps=32)
    assert result["stage"] == "sft_model_load"
    assert result["step"] == 0
    assert result["curve"] == []
    assert result["eta_seconds"] is None
