import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import wavetag.audio.pipeline as audio_pipeline_module
from wavetag.audio.optout_code import AudioOptOutConfig, detect_optout, embed_optout
from wavetag.audio.pipeline import ProtectResult, WaveTagAudioPipeline
from wavetag.audio.volto_plus import apply_volto_plus, detect_volto_plus
from wavetag.audio.volto_removal import RemovalConfig, lateral_crop_aco, longitudinal_crop
from wavetag.common.payload import pack_optout
from wavetag.robustness.audio_attacks import apply_attack, get_attack_suite, to_pcm16

SR = 16000


def test_optout_roundtrip_and_wrong_key(speech_like):
    cfg, pl = AudioOptOutConfig(), pack_optout(4242)
    y = to_pcm16(embed_optout(speech_like, pl, cfg))
    assert detect_optout(y, cfg, 5, pl)["ok"]
    assert not detect_optout(y, AudioOptOutConfig(key=1), 5, pl)["ok"]      # 키 불일치
    assert not detect_optout(speech_like, cfg, 5, pl)["ok"]                  # 미삽입 원본 -> 오검출 없음


def test_optout_survives_shift_and_noise(speech_like):
    cfg, pl, rng = AudioOptOutConfig(), pack_optout(4242), np.random.default_rng(0)
    y = to_pcm16(embed_optout(speech_like, pl, cfg))
    suite = get_attack_suite()
    for name in ("shift_50ms", "noise_20dB", "quant_8bit"):
        assert detect_optout(apply_attack(y, suite[name], SR, rng), cfg, 5, pl)["ok"], name


def test_optout_dies_under_8k_resample(speech_like):
    """문서의 '압축·변환 후 유지' 주장에 대한 한계 확인: 8kHz 다운샘플은 고주파 코드를 지운다(예상된 실패)."""
    cfg, pl, rng = AudioOptOutConfig(), pack_optout(4242), np.random.default_rng(0)
    y = to_pcm16(embed_optout(speech_like, pl, cfg))
    assert not detect_optout(apply_attack(y, get_attack_suite()["resample_8k"], SR, rng), cfg, 5, pl)["ok"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 필요")
def test_optout_survives_mp3(speech_like):
    cfg, pl, rng = AudioOptOutConfig(), pack_optout(4242), np.random.default_rng(0)
    y = to_pcm16(embed_optout(speech_like, pl, cfg))
    assert detect_optout(apply_attack(y, get_attack_suite()["mp3_128k"], SR, rng), cfg, 5, pl)["ok"]


def test_volto_plus_detect(speech_like):
    y = apply_volto_plus(speech_like, 7, SR, 0.05)
    good = detect_volto_plus(y, 7, SR)
    assert good["peak_ratio"] > 15 and good["loc_error"] == 0
    assert detect_volto_plus(speech_like, 7, SR)["peak_ratio"] < 8            # 무삽입
    assert detect_volto_plus(y, 8, SR)["peak_ratio"] < 8                      # 다른 uid


def _dummy_embed(w):
    S = np.abs(np.fft.rfft(np.atleast_2d(w), axis=1))
    return np.stack([np.log1p(S[:, b].mean(1) * 50) for b in np.array_split(np.arange(S.shape[1]), 64)], 1)


def test_removal_reduces_similarity(speech_like):
    from wavetag.audio.metrics import cosine
    x = speech_like[: SR * 2]
    cfg = RemovalConfig(aco_max_candidates=64, aco_iters=2, aco_max_steps=30, target_sim=0.5)
    ref = _dummy_embed(x)[0]
    x1, t, s1 = longitudinal_crop(x, _dummy_embed, ref, cfg)
    x2, info = lateral_crop_aco(x1, SR, _dummy_embed, ref, t, cfg, pesq_ref=x)
    assert len(x2) == len(x)
    assert float(cosine(_dummy_embed(x2)[0], ref)[0]) <= 1.0
    assert info["removed_units"] >= 1


def test_pipeline_preserves_original_when_protection_fails(speech_like, monkeypatch):
    pipeline = object.__new__(WaveTagAudioPipeline)
    pipeline.cfg = SimpleNamespace(strength_levels=["strong"])
    failed = ProtectResult(
        audio=np.zeros_like(speech_like),
        passed=False,
        strength="strong",
        metrics={"sim_aux": 0.8, "quality_ok": False},
    )
    monkeypatch.setattr(pipeline, "protect_once", lambda *_: failed)

    result = pipeline.protect(speech_like, uid=1)

    np.testing.assert_array_equal(result.audio, speech_like)
    assert result.audio is not speech_like
    assert not result.passed
    assert result.info["fallback"] == "original_audio"


def _validation_pipeline(monkeypatch, pesq_value, trigger_ratio=10.0, trigger_loc_error=0,
                         optout_ok=True):
    cfg = SimpleNamespace(
        sr=SR,
        seed=0,
        strength=SimpleNamespace(strong=SimpleNamespace(target_sim=0.3, wm_amp_ratio=0.03)),
        audio=SimpleNamespace(
            removal=SimpleNamespace(),
            wm_mode="plus",
            volto_plus=SimpleNamespace(slot_ms=100.0, seed_base=1234),
            trigger_detect=SimpleNamespace(min_peak_ratio=8.0, max_loc_error_samples=16),
            volto_star=SimpleNamespace(seg_sec=3.0, min_correlation=0.1),
        ),
        pass_criteria=SimpleNamespace(
            pesq_mode="wb", pesq_min=1.99, snr_min_db=6.0, sim_max=1.1,
        ),
    )
    pipeline = object.__new__(WaveTagAudioPipeline)
    pipeline.cfg = cfg
    pipeline.sr = SR
    pipeline.embed_fn = lambda batch: np.ones((len(batch), 2), dtype=np.float32)
    pipeline.star_model = None
    pipeline.device = None
    pipeline.optout_cfg = AudioOptOutConfig()

    monkeypatch.setattr(audio_pipeline_module, "longitudinal_crop",
                        lambda x, *_: (x, 0.0, 0.5))
    monkeypatch.setattr(audio_pipeline_module, "lateral_crop_aco",
                        lambda x, *_args, **_kwargs: (x, {}))
    monkeypatch.setattr(audio_pipeline_module, "apply_volto_plus", lambda x, *_: x)
    monkeypatch.setattr(audio_pipeline_module, "embed_optout", lambda x, *_: x)
    monkeypatch.setattr(
        audio_pipeline_module,
        "detect_optout",
        lambda _x, _cfg, _n, payload: {
            "ok": optout_ok,
            "payload": payload if optout_ok else None,
        },
    )
    monkeypatch.setattr(
        audio_pipeline_module,
        "detect_volto_plus",
        lambda *_: {"peak_ratio": trigger_ratio, "loc_error": trigger_loc_error},
    )
    monkeypatch.setattr(audio_pipeline_module, "pesq_score", lambda *_: pesq_value)
    monkeypatch.setattr(audio_pipeline_module, "snr_db", lambda *_: 10.0)
    return pipeline


def test_pipeline_requires_watermark_detection(speech_like, monkeypatch):
    pipeline = _validation_pipeline(monkeypatch, 3.0, trigger_ratio=7.0)

    result = pipeline.protect_once(speech_like, uid=42, strength="strong")

    assert not result.passed
    assert not result.metrics["trigger_detected"]
    assert not result.metrics["watermark_ok"]


def test_pipeline_requires_optout_payload_detection(speech_like, monkeypatch):
    pipeline = _validation_pipeline(monkeypatch, 3.0, optout_ok=False)

    result = pipeline.protect_once(speech_like, uid=42, strength="strong")

    assert not result.passed
    assert not result.metrics["optout_ok"]
    assert not result.metrics["watermark_ok"]


def test_pipeline_rejects_nan_pesq(speech_like, monkeypatch):
    pipeline = _validation_pipeline(monkeypatch, float("nan"))

    result = pipeline.protect_once(speech_like, uid=42, strength="strong")

    assert not result.passed
    assert not result.metrics["quality_ok"]
    assert result.metrics["watermark_ok"]


def test_pipeline_passes_when_pesq_and_watermarks_pass(speech_like, monkeypatch):
    pipeline = _validation_pipeline(monkeypatch, 3.0)

    result = pipeline.protect_once(speech_like, uid=42, strength="strong")

    assert result.passed
    assert result.metrics["quality_ok"]
    assert result.metrics["watermark_ok"]
