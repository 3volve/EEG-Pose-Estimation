from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from config import (
    EEG_BANDSTOP_HZ,
    EEG_BANDSTOP_ORDER,
    EEG_PREPROCESSING_VERSION,
    EEG_SAMPLE_RATE,
    EEG_SOURCE_CHANNEL_INDICES,
)
from eeg_encoding.permanent_holdout import PERMANENT_HOLDOUT_CATEGORIES


def _load_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "__main__.py"
    spec = importlib.util.spec_from_file_location(
        f"eeg_pose_cli_{uuid4().hex}",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _output_dir(name: str) -> Path:
    path = Path("test_outputs") / f"bootstrap_atomic_{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def _write_complete_bootstrap_archive(path: Path) -> None:
    test_block_names = [
        "rest_between_blocks" if category == "rest" else category
        for category in PERMANENT_HOLDOUT_CATEGORIES
    ]
    roles = np.asarray(
        ["support", "query", "validation"] + ["test"] * len(test_block_names)
    )
    block_names = np.asarray(
        ["neutral_rest", "left_arm_raise", "right_arm_raise", *test_block_names]
    )
    sample_count = len(roles)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        eeg=np.zeros((sample_count, 4, 200), dtype=np.float32),
        pose_latent=np.zeros((sample_count, 4), dtype=np.float32),
        pose_confidence=np.full(sample_count, 0.99, dtype=np.float32),
        interpolation_confidence=np.full(sample_count, 0.99, dtype=np.float32),
        pose_reconstruction_error=np.zeros(sample_count, dtype=np.float32),
        profile_session_id=np.asarray("bootstrap-test"),
        profile_round_role=roles,
        profile_block_id=np.arange(sample_count, dtype=np.int64),
        profile_block_name=block_names,
        profile_block_repeat_index=np.asarray(
            [0, 1, 2] + [3] * len(test_block_names),
            dtype=np.int64,
        ),
        profile_block_accepted=np.ones(sample_count, dtype=np.bool_),
        profile_block_is_rest=np.asarray(
            ["rest" in name for name in block_names],
            dtype=np.bool_,
        ),
        profile_frame_trusted=np.ones(sample_count, dtype=np.bool_),
        profile_frame_split_eligible=np.ones(sample_count, dtype=np.bool_),
        profile_block_summary_json=np.asarray(json.dumps([])),
        pipeline_version=np.asarray(EEG_PREPROCESSING_VERSION),
        source_channel_indices=np.asarray(EEG_SOURCE_CHANNEL_INDICES),
        sample_rate_hz=np.asarray(EEG_SAMPLE_RATE),
        bandstop_low_hz=np.asarray(EEG_BANDSTOP_HZ[0]),
        bandstop_high_hz=np.asarray(EEG_BANDSTOP_HZ[1]),
        bandstop_order=np.asarray(EEG_BANDSTOP_ORDER),
    )


def _bootstrap_args(out_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        user_id="bootstrap-user",
        out=str(out_path),
        epochs=1,
        inner_epochs=1,
        batch_size=2,
        lr=1e-3,
        pose_checkpoint=None,
        device="cpu",
    )


def test_bootstrap_validation_requires_all_ten_holdout_categories() -> None:
    cli = _load_cli_module()
    output_dir = _output_dir("validation")
    archive_path = output_dir / "complete.npz"
    _write_complete_bootstrap_archive(archive_path)

    session = cli._validate_bootstrap_session(archive_path)

    assert len(session.test_indices) == 10

    with np.load(archive_path) as archive:
        values = {key: archive[key] for key in archive.files}
    keep = np.asarray(values["profile_block_name"]) != "torso_side_lean"
    for key, value in list(values.items()):
        if np.asarray(value).shape[:1] == (len(keep),):
            values[key] = np.asarray(value)[keep]
    incomplete_path = output_dir / "incomplete.npz"
    np.savez_compressed(incomplete_path, **values)

    with pytest.raises(RuntimeError, match="complete permanent holdout.*torso_side_lean"):
        cli._validate_bootstrap_session(incomplete_path)


def test_bootstrap_commits_checkpoint_only_after_evaluation_and_holdout_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    output_dir = _output_dir("success")
    requested_out = output_dir / "models" / "base.pt"
    paths = SimpleNamespace(
        holdout_manifest=output_dir / "profile" / "permanent_holdout" / "manifest.json",
        sessions_root=output_dir / "profile" / "sessions",
    )
    calls: list[str] = []

    monkeypatch.setattr(cli.eeg_encoding, "profile_paths", lambda _user_id: paths)

    def collect(_args, session_data):
        calls.append("collect")
        Path(session_data).parent.mkdir(parents=True, exist_ok=True)
        Path(session_data).write_bytes(b"capture")

    monkeypatch.setattr(cli, "collect_profile_build_session", collect)
    monkeypatch.setattr(
        cli,
        "_validate_bootstrap_session",
        lambda _path: calls.append("validate") or object(),
    )

    def train(_data, out_path, **_kwargs):
        calls.append("train")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"temporary-checkpoint")
        return SimpleNamespace(training_report=object())

    monkeypatch.setattr(cli.eeg_encoding, "train_model", train)
    monkeypatch.setattr(
        cli.eeg_encoding,
        "evaluate_profile_post_adaptation",
        lambda *_args, **_kwargs: calls.append("test") or [],
    )

    def seed(**_kwargs):
        calls.append("seed")
        paths.holdout_manifest.parent.mkdir(parents=True, exist_ok=True)
        paths.holdout_manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli.eeg_encoding, "seed_profile_holdout", seed)
    monkeypatch.setattr(
        cli.eeg_encoding,
        "format_training_report",
        lambda _report: "training report",
    )

    cli.bootstrap_base(_bootstrap_args(requested_out))

    assert calls == ["collect", "validate", "train", "test", "seed"]
    assert requested_out.read_bytes() == b"temporary-checkpoint"
    assert not list(requested_out.parent.glob(".*.bootstrap-*.tmp"))


def test_bootstrap_failure_removes_only_temporary_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    output_dir = _output_dir("failure")
    requested_out = output_dir / "models" / "base.pt"
    paths = SimpleNamespace(
        holdout_manifest=output_dir / "profile" / "permanent_holdout" / "manifest.json",
        sessions_root=output_dir / "profile" / "sessions",
    )
    captured_paths: list[Path] = []

    monkeypatch.setattr(cli.eeg_encoding, "profile_paths", lambda _user_id: paths)

    def collect(_args, session_data):
        session_path = Path(session_data)
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_bytes(b"capture-to-keep")
        captured_paths.append(session_path)

    monkeypatch.setattr(cli, "collect_profile_build_session", collect)
    monkeypatch.setattr(cli, "_validate_bootstrap_session", lambda _path: object())

    def train(_data, out_path, **_kwargs):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"temporary-checkpoint")
        return SimpleNamespace(training_report=object())

    monkeypatch.setattr(cli.eeg_encoding, "train_model", train)

    def fail_test(*_args, **_kwargs):
        raise RuntimeError("test evaluation failed")

    monkeypatch.setattr(
        cli.eeg_encoding,
        "evaluate_profile_post_adaptation",
        fail_test,
    )

    with pytest.raises(RuntimeError, match="test evaluation failed"):
        cli.bootstrap_base(_bootstrap_args(requested_out))

    assert not requested_out.exists()
    assert not list(requested_out.parent.glob(".*.bootstrap-*.tmp"))
    assert captured_paths[0].read_bytes() == b"capture-to-keep"


def test_bootstrap_finalization_failure_rolls_back_seed_and_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    output_dir = _output_dir("finalize_failure")
    requested_out = output_dir / "models" / "base.pt"
    paths = SimpleNamespace(
        holdout_manifest=output_dir / "profile" / "permanent_holdout" / "manifest.json",
        sessions_root=output_dir / "profile" / "sessions",
    )

    monkeypatch.setattr(cli.eeg_encoding, "profile_paths", lambda _user_id: paths)

    def collect(_args, session_data):
        Path(session_data).parent.mkdir(parents=True, exist_ok=True)
        Path(session_data).write_bytes(b"capture-to-keep")

    monkeypatch.setattr(cli, "collect_profile_build_session", collect)
    monkeypatch.setattr(cli, "_validate_bootstrap_session", lambda _path: object())

    def train(_data, out_path, **_kwargs):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"temporary-checkpoint")
        return SimpleNamespace(training_report=object())

    monkeypatch.setattr(cli.eeg_encoding, "train_model", train)
    monkeypatch.setattr(
        cli.eeg_encoding,
        "evaluate_profile_post_adaptation",
        lambda *_args, **_kwargs: [],
    )

    def seed(**_kwargs):
        paths.holdout_manifest.parent.mkdir(parents=True, exist_ok=True)
        paths.holdout_manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli.eeg_encoding, "seed_profile_holdout", seed)
    monkeypatch.setattr(
        cli.eeg_encoding,
        "format_training_report",
        lambda _report: "training report",
    )
    real_replace = Path.replace

    def fail_final_replace(path: Path, target: Path):
        if Path(target) == requested_out:
            raise OSError("simulated final checkpoint failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_final_replace)

    with pytest.raises(OSError, match="simulated final checkpoint failure"):
        cli.bootstrap_base(_bootstrap_args(requested_out))

    assert not requested_out.exists()
    assert not paths.holdout_manifest.exists()
    assert not list(paths.sessions_root.glob("bootstrap_*/bootstrap_report.json"))
