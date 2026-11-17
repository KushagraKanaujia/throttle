"""`throttle fit`: offline memory estimate of the fewest GPUs for a model."""

from __future__ import annotations

import json

import pytest

import throttle.fit as fit
from throttle.cli import main


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(["fit", *argv])
    out, err = capsys.readouterr()
    return code, out, err


def test_kv_math_matches_hand_calculation():
    spec = fit.MODELS["llama-3.3-70b"]
    # 2 (K and V) x 80 layers x 8 heads x 128 dims x 2 bytes = 327,680 bytes/token
    assert fit.kv_gb_per_token(spec, "auto") * fit.GIB == pytest.approx(327_680)
    assert fit.kv_gb_per_token(spec, "auto") * 32768 == pytest.approx(10.0)


def test_weights_scale_with_quant():
    spec = fit.MODELS["qwen2.5-72b"]
    assert fit.weights_gb(spec, "fp8") == pytest.approx(fit.weights_gb(spec, "bf16") / 2)


def test_fewest_h100s_for_70b_fp8(capsys):
    code, out, _ = run(capsys, "--model", "llama-3.3-70b", "--quant", "fp8",
                       "--context", "32k", "--users", "16", "--gpu", "h100", "--json")
    assert code == 0
    data = json.loads(out)
    assert data["estimate"] is True
    (h100,) = data["options"]
    # 65.8 GiB weights + 16 x 10 GiB KV needs more than 2 cards, fits on 4
    assert h100["gpus"] == 4
    assert h100["max_users"] >= 16


def test_current_flags_spare_cards_with_saving(capsys):
    code, out, _ = run(capsys, "--model", "llama-3.3-70b", "--quant", "fp8",
                       "--context", "8k", "--users", "16", "--gpu", "h100",
                       "--price", "h100=2.00", "--current", "h100=4", "--json")
    assert code == 0
    data = json.loads(out)
    assert data["current"]["status"] == "maybe_fewer"
    assert data["current"]["fewest_estimate"] == 2
    assert data["current"]["monthly_saving_if_confirmed"] == pytest.approx(2 * 2.00 * 730)
    assert "--space tp=2,4" in data["measure_command"]
    assert "meta-llama/Llama-3.3-70B-Instruct" in data["measure_command"]


def test_current_on_other_gpu_is_not_mixed_into_space(capsys):
    _, out, _ = run(capsys, "--model", "llama-3.3-70b", "--quant", "fp8", "--context", "32k",
                    "--users", "16", "--gpu", "l40s", "--price", "l40s=1",
                    "--current", "h100=4", "--json")
    data = json.loads(out)
    assert data["pick"] == "l40s"
    assert "--space tp=8" in data["measure_command"]


def test_vllm_counts_respect_attention_heads():
    # Qwen2.5-7B has 28 attention heads: 8-way tensor parallel is not possible
    assert fit.gpu_counts("vllm", fit.MODELS["qwen2.5-7b"]) == [1, 2, 4]
    assert fit.gpu_counts("llamacpp", fit.MODELS["qwen2.5-7b"]) == list(range(1, 9))


def test_kv_copied_when_more_cards_than_kv_heads():
    one = fit.max_users(8, 70.0, 10.0, 1.0, kv_heads=4)
    none = fit.max_users(8, 70.0, 10.0, 1.0, kv_heads=0)
    assert one == none // 2


def test_does_not_fit_exits_1_with_advice(capsys):
    code, out, _ = run(capsys, "--model", "qwen2.5-72b", "--context", "128k",
                       "--users", "64", "--gpu", "l4")
    assert code == 1
    assert "Nothing here fits" in out
    assert "--quant" in out


def test_table_says_estimate_and_points_to_measurement(capsys):
    code, out, _ = run(capsys, "--model", "qwen2.5-7b", "--users", "4")
    assert code == 0
    assert "ESTIMATE" in out
    assert "throttle autotune" in out
    assert "Qwen/Qwen2.5-7B-Instruct" in out


def test_custom_gpu_and_llamacpp(capsys):
    code, out, _ = run(capsys, "--model", "qwen2.5-32b", "--quant", "q4_k_m",
                       "--engine", "llamacpp", "--gpu", "rtx3090", "--gpu", "1060=6",
                       "--context", "32k", "--users", "2", "--json")
    assert code == 0
    data = json.loads(out)
    by_name = {o["gpu"]: o for o in data["options"]}
    assert by_name["1060"]["vram_gb"] == 6
    assert by_name["rtx3090"]["gpus"] == 2
    assert data["measure_command"].startswith("throttle check")


def test_config_json(tmp_path, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"num_hidden_layers": 32, "num_attention_heads": 32,
                               "num_key_value_heads": 8, "hidden_size": 4096}))
    code, out, _ = run(capsys, "--config", str(cfg), "--params", "8", "--json")
    assert code == 0
    data = json.loads(out)
    assert data["model"]["head_dim"] == 128
    assert "YOUR/MODEL" in data["measure_command"]


@pytest.mark.parametrize("argv, message", [
    ((), "give --model"),
    (("--model", "nope"), "unknown model"),
    (("--model", "qwen2.5-7b", "--gpu", "b9000"), "unknown GPU"),
    (("--model", "qwen2.5-7b", "--price", "h100=abc"), "--price wants"),
    (("--model", "qwen2.5-7b", "--gpu", "h100", "--price", "l4=1"), "not in the list"),
    (("--model", "qwen2.5-7b", "--current", "h100=1.5"), "whole card count"),
    (("--config", "missing.json", "--params", "7"), "could not read"),
])
def test_usage_errors(capsys, argv, message):
    code, _, err = run(capsys, *argv)
    assert code == 2
    assert message in err


def test_bad_context_is_rejected_by_argparse(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["fit", "--model", "qwen2.5-7b", "--context", "lots"])
    assert exc.value.code == 2


def test_lists(capsys):
    assert main(["fit", "--list-models"]) == 0
    assert "llama-3.3-70b" in capsys.readouterr().out
    assert main(["fit", "--list-gpus"]) == 0
    assert "h100" in capsys.readouterr().out
