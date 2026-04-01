from __future__ import annotations

from sim.inference import InferenceEngine


def test_inference_engine_defaults_to_ollama() -> None:
    engine = InferenceEngine(cfg={}, prompt_budgets={})
    assert engine._mode == "ollama"


def test_inference_engine_normalizes_base_ollama_url() -> None:
    engine = InferenceEngine(
        cfg={"ollama_url": "http://localhost:11434", "model": "llama3.1"},
        prompt_budgets={},
    )
    assert engine._ollama_url == "http://localhost:11434/api/generate"


def test_inference_engine_keeps_full_generate_url() -> None:
    engine = InferenceEngine(
        cfg={
            "ollama_url": "http://localhost:11434/api/generate",
            "model": "llama3.1",
        },
        prompt_budgets={},
    )
    assert engine._ollama_url == "http://localhost:11434/api/generate"
