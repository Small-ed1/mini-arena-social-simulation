from __future__ import annotations

from types import SimpleNamespace

from sim.inference import InferenceEngine, _normalize_guest_obj, _normalize_host_obj


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


def test_normalize_guest_offer_alias() -> None:
    obj = _normalize_guest_obj(
        {
            "type": "offer",
            "reason_short": "Offer help",
            "actor_id": "guest_1",
            "prop_id": "prop_flashlight",
            "target_guest_id": "guest_2",
        }
    )
    assert obj["type"] == "interact"
    assert obj["verb"] == "offer"


def test_normalize_host_progression_alias() -> None:
    obs = SimpleNamespace(guests=[SimpleNamespace(guest_id="guest_2")])
    obj = _normalize_host_obj(
        {
            "type": "progression",
            "reason_short": "Move thread forward",
            "actor_id": "host",
        },
        obs,
    )
    assert obj["type"] == "allocate_spotlight"
    assert obj["target_guest_id"] == "guest_2"
