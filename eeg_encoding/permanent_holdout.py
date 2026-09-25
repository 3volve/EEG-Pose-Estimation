from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Collection, Mapping, Sequence

import numpy as np


MANIFEST_VERSION = 1
PERMANENT_HOLDOUT_CATEGORIES: tuple[str, ...] = (
    "left_arm_raise",
    "right_arm_raise",
    "left_elbow_bend",
    "right_elbow_bend",
    "left_forward_reach",
    "right_forward_reach",
    "both_arms_raise",
    "arms_open_close",
    "torso_side_lean",
    "rest",
)
REST_BLOCK_NAMES = frozenset(("neutral_rest", "rest_between_blocks", "rest"))


@dataclass(frozen=True, slots=True)
class HoldoutCandidate:
    category: str
    source_session_id: str
    source_path: str
    source_block_id: int
    source_block_name: str
    source_repeat_index: int
    sample_indices: tuple[int, ...]
    sample_count: int
    source_metadata: dict[str, object]
    preprocessing_signature: str
    preprocessing_checksum: str
    data_checksum: str
    eligible: bool = True
    ineligible_reason: str = ""


@dataclass(frozen=True, slots=True)
class HoldoutSlot:
    category: str
    source_session_id: str
    source_path: str
    source_block_id: int
    source_block_name: str
    source_repeat_index: int
    sample_indices: tuple[int, ...]
    sample_count: int
    source_metadata: dict[str, object]
    preprocessing_signature: str
    preprocessing_checksum: str
    data_checksum: str
    installed_after_decision_id: str | None = None
    installed_after_committed: bool | None = None


@dataclass(frozen=True, slots=True)
class PermanentHoldoutManifest:
    version: int
    categories: tuple[str, ...]
    cursor: int
    preprocessing_signature: str
    preprocessing_checksum: str
    slots: dict[str, HoldoutSlot | None]

    @property
    def next_category(self) -> str:
        return self.categories[self.cursor]

    @property
    def missing_categories(self) -> tuple[str, ...]:
        return tuple(
            category
            for category in self.categories
            if self.slots[category] is None
        )

    @property
    def complete(self) -> bool:
        return not self.missing_categories


@dataclass(frozen=True, slots=True)
class HoldoutUpdate:
    updated: bool
    reason: str
    category: str
    cursor_before: int
    cursor_after: int
    manifest: PermanentHoldoutManifest


def canonical_preprocessing_signature(settings: Mapping[str, object] | str) -> str:
    """Return the stable, human-readable identity stored with every slot."""
    if isinstance(settings, str):
        if not settings:
            raise ValueError("preprocessing signature must not be empty")
        return settings
    return json.dumps(
        settings,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def checksum_preprocessing_signature(signature: str) -> str:
    if not signature:
        raise ValueError("preprocessing signature must not be empty")
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def normalize_holdout_category(block_name: str) -> str | None:
    if block_name in REST_BLOCK_NAMES:
        return "rest"
    if block_name in PERMANENT_HOLDOUT_CATEGORIES:
        return block_name
    return None


def candidates_from_test_blocks(
    archive_path: str | Path,
    test_indices: Sequence[int] | np.ndarray,
    *,
    preprocessing: Mapping[str, object] | str,
    eligible_block_ids: Collection[int] | None = None,
) -> tuple[HoldoutCandidate, ...]:
    """Describe whole test blocks without copying their samples.

    A block is eligible only when all of its samples are in ``test_indices``, its
    saved acceptance mask is true, and (when supplied) its id is present in
    ``eligible_block_ids``. Unknown categories, including the retired
    ``hip_shift`` block, are ignored.
    """
    path = Path(archive_path).resolve()
    _reject_ambiguous_merged_archive(path)
    signature = canonical_preprocessing_signature(preprocessing)
    preprocessing_checksum = checksum_preprocessing_signature(signature)
    requested_ids = (
        {int(block_id) for block_id in eligible_block_ids}
        if eligible_block_ids is not None
        else None
    )

    with np.load(path) as archive:
        required = ("eeg", "profile_block_id", "profile_block_name")
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(
                f"holdout source {path} is missing required arrays: {missing}"
            )

        sample_total = len(archive["eeg"])
        indices = np.asarray(test_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("test_indices must be one-dimensional")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("test_indices must not contain duplicates")
        if np.any(indices < 0) or np.any(indices >= sample_total):
            raise ValueError(
                f"test_indices are outside holdout source sample range [0, {sample_total})"
            )

        block_ids = _per_sample_array(
            archive,
            "profile_block_id",
            sample_total,
            dtype=np.int64,
        )
        block_names = _per_sample_array(
            archive,
            "profile_block_name",
            sample_total,
        ).astype(str)
        repeat_indices = (
            _per_sample_array(
                archive,
                "profile_block_repeat_index",
                sample_total,
                dtype=np.int64,
            )
            if "profile_block_repeat_index" in archive.files
            else np.zeros(sample_total, dtype=np.int64)
        )
        accepted = (
            _per_sample_array(
                archive,
                "profile_block_accepted",
                sample_total,
                dtype=bool,
            )
            if "profile_block_accepted" in archive.files
            else np.ones(sample_total, dtype=bool)
        )
        eligible_test_mask = np.ones(sample_total, dtype=bool)
        if "profile_round_role" in archive.files:
            roles = _per_sample_array(
                archive,
                "profile_round_role",
                sample_total,
            ).astype(str)
            eligible_test_mask &= roles == "test"
        session_id = _optional_scalar_text(
            archive,
            "profile_session_id",
            default=path.parent.name,
        )
        posture = _optional_scalar_text(archive, "profile_posture", default="")
        summaries = _block_summaries(archive)

        candidates = []
        for block_id in _unique_in_order(block_ids[indices]):
            all_block_indices = np.flatnonzero(
                (block_ids == block_id) & eligible_test_mask
            )
            selected_indices = np.sort(indices[block_ids[indices] == block_id])
            names = set(block_names[selected_indices])
            repeats = set(int(value) for value in repeat_indices[selected_indices])
            block_name = next(iter(names)) if len(names) == 1 else ""
            category = normalize_holdout_category(block_name)
            if category is None:
                continue

            reasons = []
            if not np.array_equal(selected_indices, all_block_indices):
                reasons.append("partial_test_block")
            if not np.all(accepted[selected_indices]):
                reasons.append("block_not_accepted")
            if requested_ids is not None and block_id not in requested_ids:
                reasons.append("block_not_eligible")
            if len(names) != 1:
                reasons.append("inconsistent_block_name")
            if len(repeats) != 1:
                reasons.append("inconsistent_repeat_index")

            summary = dict(summaries.get(block_id, {}))
            source_metadata: dict[str, object] = {
                "profile_posture": posture,
                "block_summary": summary,
            }
            candidates.append(
                HoldoutCandidate(
                    category=category,
                    source_session_id=session_id,
                    source_path=str(path),
                    source_block_id=block_id,
                    source_block_name=block_name,
                    source_repeat_index=next(iter(repeats)) if len(repeats) == 1 else -1,
                    sample_indices=tuple(int(index) for index in selected_indices),
                    sample_count=len(selected_indices),
                    source_metadata=source_metadata,
                    preprocessing_signature=signature,
                    preprocessing_checksum=preprocessing_checksum,
                    data_checksum=checksum_archive_rows(
                        archive,
                        selected_indices,
                        preprocessing_checksum=preprocessing_checksum,
                    ),
                    eligible=not reasons,
                    ineligible_reason=",".join(reasons),
                )
            )
    return tuple(candidates)


def create_manifest_from_first_session(
    candidates: Sequence[HoldoutCandidate],
    *,
    categories: Sequence[str] = PERMANENT_HOLDOUT_CATEGORIES,
) -> PermanentHoldoutManifest:
    """Seed one permanent slot per category from the first session's test blocks."""
    category_tuple = _validated_categories(categories)
    candidate_tuple = tuple(candidates)
    signature, checksum = _shared_preprocessing_identity(candidate_tuple)
    slots: dict[str, HoldoutSlot | None] = {
        category: None for category in category_tuple
    }
    for candidate in candidate_tuple:
        _validate_candidate(candidate, category_tuple)
        if (
            candidate.eligible
            and candidate.preprocessing_checksum == checksum
            and slots[candidate.category] is None
        ):
            slots[candidate.category] = _slot_from_candidate(candidate)
    missing = tuple(
        category for category, slot in slots.items() if slot is None
    )
    if missing:
        raise ValueError(
            "The first-session permanent holdout requires one eligible test "
            f"block for every category; missing: {list(missing)}"
        )
    return PermanentHoldoutManifest(
        version=MANIFEST_VERSION,
        categories=category_tuple,
        cursor=0,
        preprocessing_signature=signature,
        preprocessing_checksum=checksum,
        slots=slots,
    )


def seed_manifest_from_first_session(
    manifest_path: str | Path,
    candidates: Sequence[HoldoutCandidate],
    *,
    categories: Sequence[str] = PERMANENT_HOLDOUT_CATEGORIES,
) -> PermanentHoldoutManifest:
    path = Path(manifest_path)
    if path.exists():
        raise FileExistsError(f"permanent holdout manifest already exists: {path}")
    manifest = create_manifest_from_first_session(candidates, categories=categories)
    save_manifest(path, manifest)
    return manifest


def replace_next_slot_after_decision(
    manifest_path: str | Path,
    candidates: Sequence[HoldoutCandidate],
    *,
    decision_id: str,
    committed: bool,
) -> HoldoutUpdate:
    """Replace at most the cursor's slot, then persist the advanced cursor.

    The manifest is left byte-for-byte untouched when the requested category has
    no eligible candidate with matching preprocessing.
    """
    if not decision_id:
        raise ValueError("decision_id must not be empty")
    path = Path(manifest_path)
    manifest = load_manifest(path)
    require_complete_manifest(manifest, source=path)
    cursor_before = manifest.cursor
    category = manifest.next_category

    matching = []
    for candidate in candidates:
        _validate_candidate(candidate, manifest.categories)
        if candidate.category == category:
            matching.append(candidate)
    if not matching:
        return _unchanged_update(manifest, category, "candidate_absent")

    eligible = [candidate for candidate in matching if candidate.eligible]
    if not eligible:
        return _unchanged_update(manifest, category, "candidate_ineligible")

    compatible = [
        candidate
        for candidate in eligible
        if candidate.preprocessing_signature == manifest.preprocessing_signature
        and candidate.preprocessing_checksum == manifest.preprocessing_checksum
    ]
    if not compatible:
        return _unchanged_update(manifest, category, "preprocessing_mismatch")

    sessions_in_other_slots = {
        slot.source_session_id
        for other_category, slot in manifest.slots.items()
        if other_category != category and slot is not None
    }
    diverse = [
        candidate
        for candidate in compatible
        if candidate.source_session_id not in sessions_in_other_slots
    ]
    if not diverse:
        return _unchanged_update(
            manifest,
            category,
            "source_session_already_used",
        )

    candidate = diverse[0]
    slots = dict(manifest.slots)
    slots[category] = _slot_from_candidate(
        candidate,
        decision_id=decision_id,
        committed=committed,
    )
    updated_manifest = replace(
        manifest,
        cursor=(cursor_before + 1) % len(manifest.categories),
        slots=slots,
    )
    save_manifest(path, updated_manifest)
    return HoldoutUpdate(
        updated=True,
        reason="replaced",
        category=category,
        cursor_before=cursor_before,
        cursor_after=updated_manifest.cursor,
        manifest=updated_manifest,
    )


def save_manifest(
    manifest_path: str | Path,
    manifest: PermanentHoldoutManifest,
) -> None:
    _validate_manifest(manifest)
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": manifest.version,
        "categories": list(manifest.categories),
        "cursor": manifest.cursor,
        "preprocessing_signature": manifest.preprocessing_signature,
        "preprocessing_checksum": manifest.preprocessing_checksum,
        "slots": {
            category: (
                _slot_to_json(slot)
                if slot is not None
                else None
            )
            for category, slot in manifest.slots.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(manifest_path: str | Path) -> PermanentHoldoutManifest:
    path = Path(manifest_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"permanent holdout manifest must contain an object: {path}")
    categories = tuple(str(value) for value in raw.get("categories", ()))
    raw_slots = raw.get("slots")
    if not isinstance(raw_slots, dict):
        raise ValueError(f"permanent holdout manifest slots must be an object: {path}")
    slots = {
        category: (
            _slot_from_json(raw_slots.get(category), category=category, source=path)
            if raw_slots.get(category) is not None
            else None
        )
        for category in categories
    }
    manifest = PermanentHoldoutManifest(
        version=int(raw.get("version", -1)),
        categories=categories,
        cursor=int(raw.get("cursor", -1)),
        preprocessing_signature=str(raw.get("preprocessing_signature", "")),
        preprocessing_checksum=str(raw.get("preprocessing_checksum", "")),
        slots=slots,
    )
    _validate_manifest(manifest, source=path)
    if set(raw_slots) != set(categories):
        raise ValueError(
            f"permanent holdout manifest slot keys do not match categories: {path}"
        )
    return manifest


def require_complete_manifest(
    manifest: PermanentHoldoutManifest,
    *,
    source: str | Path | None = None,
) -> None:
    if manifest.complete:
        return
    location = f" in {source}" if source is not None else ""
    raise ValueError(
        "Permanent holdout manifest is incomplete"
        f"{location}; missing categories: {list(manifest.missing_categories)}"
    )


def _validated_categories(categories: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(category) for category in categories)
    if len(values) != 10 or len(set(values)) != 10:
        raise ValueError("permanent holdout requires 10 unique categories")
    if "rest" not in values:
        raise ValueError("permanent holdout categories must contain normalized rest")
    return values


def _shared_preprocessing_identity(
    candidates: Sequence[HoldoutCandidate],
) -> tuple[str, str]:
    if not candidates:
        raise ValueError("first-session holdout candidates must not be empty")
    identities = {
        (candidate.preprocessing_signature, candidate.preprocessing_checksum)
        for candidate in candidates
    }
    if len(identities) != 1:
        raise ValueError("first-session holdout candidates use mixed preprocessing")
    signature, checksum = next(iter(identities))
    if checksum_preprocessing_signature(signature) != checksum:
        raise ValueError("candidate preprocessing checksum does not match its signature")
    return signature, checksum


def _validate_candidate(
    candidate: HoldoutCandidate,
    categories: Sequence[str],
) -> None:
    if candidate.category not in categories:
        raise ValueError(f"unknown permanent holdout category: {candidate.category}")
    if candidate.sample_count != len(candidate.sample_indices):
        raise ValueError(
            f"candidate {candidate.category} sample_count does not match sample_indices"
        )
    if candidate.sample_count <= 0:
        raise ValueError(f"candidate {candidate.category} has no samples")
    expected = checksum_preprocessing_signature(candidate.preprocessing_signature)
    if candidate.preprocessing_checksum != expected:
        raise ValueError(
            f"candidate {candidate.category} preprocessing checksum is invalid"
        )
    if candidate.eligible and candidate.ineligible_reason:
        raise ValueError(
            f"eligible candidate {candidate.category} has an ineligible reason"
        )


def _validate_manifest(
    manifest: PermanentHoldoutManifest,
    *,
    source: Path | None = None,
) -> None:
    location = f": {source}" if source is not None else ""
    if manifest.version != MANIFEST_VERSION:
        raise ValueError(
            f"unsupported permanent holdout manifest version "
            f"{manifest.version}{location}"
        )
    categories = _validated_categories(manifest.categories)
    if not 0 <= manifest.cursor < len(categories):
        raise ValueError(f"permanent holdout cursor is out of range{location}")
    if set(manifest.slots) != set(categories):
        raise ValueError(
            f"permanent holdout slot keys do not match categories{location}"
        )
    expected = checksum_preprocessing_signature(manifest.preprocessing_signature)
    if manifest.preprocessing_checksum != expected:
        raise ValueError(
            f"permanent holdout preprocessing checksum is invalid{location}"
        )
    for category, slot in manifest.slots.items():
        if slot is None:
            continue
        if slot.category != category:
            raise ValueError(
                f"permanent holdout slot {category} contains {slot.category}{location}"
            )
        if slot.sample_count != len(slot.sample_indices) or slot.sample_count <= 0:
            raise ValueError(
                f"permanent holdout slot {category} has invalid sample metadata{location}"
            )
        if (
            slot.preprocessing_signature != manifest.preprocessing_signature
            or slot.preprocessing_checksum != manifest.preprocessing_checksum
        ):
            raise ValueError(
                f"permanent holdout slot {category} uses different preprocessing"
                f"{location}"
            )


def _slot_from_candidate(
    candidate: HoldoutCandidate,
    *,
    decision_id: str | None = None,
    committed: bool | None = None,
) -> HoldoutSlot:
    return HoldoutSlot(
        category=candidate.category,
        source_session_id=candidate.source_session_id,
        source_path=candidate.source_path,
        source_block_id=candidate.source_block_id,
        source_block_name=candidate.source_block_name,
        source_repeat_index=candidate.source_repeat_index,
        sample_indices=candidate.sample_indices,
        sample_count=candidate.sample_count,
        source_metadata=dict(candidate.source_metadata),
        preprocessing_signature=candidate.preprocessing_signature,
        preprocessing_checksum=candidate.preprocessing_checksum,
        data_checksum=candidate.data_checksum,
        installed_after_decision_id=decision_id,
        installed_after_committed=committed,
    )


def _slot_to_json(slot: HoldoutSlot) -> dict[str, object]:
    payload = asdict(slot)
    payload["sample_indices"] = list(slot.sample_indices)
    return payload


def _slot_from_json(
    raw: object,
    *,
    category: str,
    source: Path,
) -> HoldoutSlot:
    if not isinstance(raw, dict):
        raise ValueError(f"permanent holdout slot {category} must be an object: {source}")
    try:
        metadata = raw["source_metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(
                f"permanent holdout slot {category} source_metadata must be an object: "
                f"{source}"
            )
        decision_committed = raw.get("installed_after_committed")
        if decision_committed is not None and not isinstance(decision_committed, bool):
            raise ValueError(
                f"permanent holdout slot {category} decision must be boolean: {source}"
            )
        return HoldoutSlot(
            category=str(raw["category"]),
            source_session_id=str(raw["source_session_id"]),
            source_path=str(raw["source_path"]),
            source_block_id=int(raw["source_block_id"]),
            source_block_name=str(raw["source_block_name"]),
            source_repeat_index=int(raw["source_repeat_index"]),
            sample_indices=tuple(int(value) for value in raw["sample_indices"]),
            sample_count=int(raw["sample_count"]),
            source_metadata=dict(metadata),
            preprocessing_signature=str(raw["preprocessing_signature"]),
            preprocessing_checksum=str(raw["preprocessing_checksum"]),
            data_checksum=str(raw["data_checksum"]),
            installed_after_decision_id=(
                str(raw["installed_after_decision_id"])
                if raw.get("installed_after_decision_id") is not None
                else None
            ),
            installed_after_committed=decision_committed,
        )
    except KeyError as error:
        raise ValueError(
            f"permanent holdout slot {category} is missing {error.args[0]}: {source}"
        ) from error


def _unchanged_update(
    manifest: PermanentHoldoutManifest,
    category: str,
    reason: str,
) -> HoldoutUpdate:
    return HoldoutUpdate(
        updated=False,
        reason=reason,
        category=category,
        cursor_before=manifest.cursor,
        cursor_after=manifest.cursor,
        manifest=manifest,
    )


def _per_sample_array(
    archive: np.lib.npyio.NpzFile,
    key: str,
    sample_total: int,
    *,
    dtype=None,
) -> np.ndarray:
    value = np.asarray(archive[key], dtype=dtype)
    if value.shape != (sample_total,):
        raise ValueError(
            f"holdout source array {key} has shape {value.shape}; "
            f"expected ({sample_total},)"
        )
    return value


def _optional_scalar_text(
    archive: np.lib.npyio.NpzFile,
    key: str,
    *,
    default: str,
) -> str:
    if key not in archive.files:
        return default
    value = np.asarray(archive[key])
    if value.ndim != 0:
        raise ValueError(f"holdout source metadata {key} must be scalar")
    raw = value.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return str(raw)


def _block_summaries(
    archive: np.lib.npyio.NpzFile,
) -> dict[int, dict[str, object]]:
    if "profile_block_summary_json" not in archive.files:
        return {}
    raw = _optional_scalar_text(
        archive,
        "profile_block_summary_json",
        default="[]",
    )
    summaries = json.loads(raw)
    if not isinstance(summaries, list):
        raise ValueError("profile_block_summary_json must contain a list")
    result = {}
    for summary in summaries:
        if not isinstance(summary, dict) or "block_id" not in summary:
            raise ValueError(
                "profile_block_summary_json entries must be objects with block_id"
            )
        result[int(summary["block_id"])] = dict(summary)
    return result


def checksum_archive_rows(
    archive: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    *,
    preprocessing_checksum: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(preprocessing_checksum.encode("ascii"))
    sample_total = len(archive["eeg"])
    for key in sorted(archive.files):
        value = np.asarray(archive[key])
        if value.ndim == 0 or value.shape[0] != sample_total or value.dtype.hasobject:
            continue
        rows = np.ascontiguousarray(value[indices])
        digest.update(key.encode("utf-8"))
        digest.update(rows.dtype.str.encode("ascii"))
        digest.update(str(rows.shape).encode("ascii"))
        digest.update(rows.tobytes())
    return digest.hexdigest()


def _reject_ambiguous_merged_archive(path: Path) -> None:
    if path.name != "paired_profile_session.npz":
        return
    source_paths = sorted(path.parent.glob("source_*.npz"))
    if len(source_paths) > 1:
        raise ValueError(
            "Permanent holdout provenance requires one source archive per "
            f"profile session, but {path} merges {len(source_paths)} sources."
        )


def _unique_in_order(values: np.ndarray) -> tuple[int, ...]:
    seen: set[int] = set()
    result = []
    for value in values:
        block_id = int(value)
        if block_id not in seen:
            result.append(block_id)
            seen.add(block_id)
    return tuple(result)
