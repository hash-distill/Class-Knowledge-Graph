"""ARIC regression tests; synthetic arrays here are test fixtures, not evaluation data."""
from __future__ import annotations

import importlib
import io
import json
import os
import tarfile

import numpy as np
import pytest


def api():
    return importlib.import_module("learning.aric")


def test_split_keeps_original_images_disjoint_and_is_deterministic():
    m = api()
    ids = [f"{i:05d}" for i in range(100)]
    split = m.split_image_ids(ids, 41)
    assert split == m.split_image_ids(list(reversed(ids)), 41)
    assert [len(split[k]) for k in ("train", "validation", "test")] == [60, 20, 20]
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["train"])
    assert set().union(*map(set, split.values())) == set(ids)
    with pytest.raises(ValueError, match="duplicate"):
        m.split_image_ids(ids + ids[:1], 41)


def test_hash_sample_is_reproducible_and_spread_across_id_range():
    ids = [f"{i:05d}" for i in range(1000)]
    picked = api().select_image_ids(ids, 80, 41)
    assert picked == api().select_image_ids(list(reversed(ids)), 80, 41)
    assert len(picked) == 80
    assert min(map(int, picked)) < 100 and max(map(int, picked)) > 900


def test_targets_ignore_generated_captions_and_duplicate_people():
    annotation = {"caption": "reading teaching", "text": "reading", "instances": [
        {"label": "listening"}, {"label": "listening"}, {"label": "writing"}]}
    assert api().annotation_labels(annotation) == ["listening", "writing"]


def test_pooling_has_no_spatial_token_assumption_and_rejects_invalid():
    a = np.ones((1, 401, 768), dtype=np.float32)
    a[0, :, 0] = 2
    pooled = api().reduce_global_feature(a)
    assert pooled.shape == (768,)
    assert np.linalg.norm(pooled) == pytest.approx(1)
    assert pooled[0] == pytest.approx(2 * pooled[1])
    np.testing.assert_allclose(pooled, api().reduce_global_feature(a[:, ::-1]))
    with pytest.raises(ValueError, match="shape"):
        api().reduce_global_feature(a[:, :400])
    a[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        api().reduce_global_feature(a)


def test_safe_npy_reader_rejects_pickle_arrays():
    payload = io.BytesIO()
    np.save(payload, np.array([{"bad": "pickle"}], dtype=object))
    with pytest.raises(ValueError):
        api().load_global_feature(payload.getvalue())


def test_cache_reads_selected_ids_only_and_reuses_matching_source(tmp_path):
    archive = tmp_path / "features.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for i in range(5):
            payload = io.BytesIO()
            np.save(payload, np.ones((1, 401, 768), dtype=np.float32) * (i + 1))
            data = payload.getvalue()
            info = tarfile.TarInfo(f"features/{i:05d}.npy")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    target = tmp_path / "features.npz"
    ids, features, info = api().cache_features(archive, ["00001", "00003"], target)
    assert ids == ["00001", "00003"]
    assert features.shape == (2, 768)
    assert info["cached_images"] == 2
    re_ids, re_features, _ = api().cache_features(archive, ids, target)
    assert re_ids == ids
    np.testing.assert_array_equal(re_features, features)
    archive.unlink()
    with pytest.raises(ValueError, match="--from-bundle"):
        api().cache_features(archive, ids, target)


def write_tiny_global_archive(path, emphasized_dimension):
    array = np.ones((1, 401, 768), dtype=np.float32)
    array[0, :, emphasized_dimension] = 3
    payload = io.BytesIO()
    np.save(payload, array)
    data = payload.getvalue()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("features/00001.npy")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))


def test_cache_rejects_different_archive_with_same_ids_and_shape(tmp_path):
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    write_tiny_global_archive(first, 0)
    write_tiny_global_archive(second, 1)
    cache = tmp_path / "features.npz"
    api().cache_features(first, ["00001"], cache)
    with pytest.raises(ValueError, match="fresh output"):
        api().cache_features(second, ["00001"], cache)


def test_cache_rejects_modified_archive_at_same_path(tmp_path):
    archive = tmp_path / "source.tar.gz"
    write_tiny_global_archive(archive, 0)
    cache = tmp_path / "features.npz"
    api().cache_features(archive, ["00001"], cache)
    initial = archive.stat()
    write_tiny_global_archive(archive, 1)
    os.utime(archive, ns=(initial.st_atime_ns, initial.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="fresh output"):
        api().cache_features(archive, ["00001"], cache)


@pytest.mark.parametrize("key, invalid_value", [("reduction", "different_reduction"),
                                                ("source_shape", [1, 400, 768])])
def test_cache_rejects_changed_reduction_contract(tmp_path, key, invalid_value):
    archive = tmp_path / "source.tar.gz"
    write_tiny_global_archive(archive, 0)
    cache = tmp_path / "features.npz"
    api().cache_features(archive, ["00001"], cache)
    metadata_path = cache.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata[key] = invalid_value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="fresh output"):
        api().cache_features(archive, ["00001"], cache)


def test_prepare_rejects_stale_cache_before_overwriting_existing_audit(tmp_path):
    original = tmp_path / "original.tar.gz"
    write_tiny_global_archive(original, 0)
    output = tmp_path / "output"
    output.mkdir()
    api().cache_features(original, ["00001"], output / "features.npz")
    audit = output / "data_audit.json"
    audit.write_text('{"existing_run": "preserve"}')
    root = tmp_path / "different_dataset"
    (root / "annotations").mkdir(parents=True)
    write_tiny_global_archive(root / "ARIC_CLIP_global_view_features.tar.gz", 1)
    for i in range(5):
        image_id = f"{i:05d}"
        (root / "annotations" / f"{image_id}.json").write_text(json.dumps({
            "image_name": image_id, "image_width": 1920, "image_height": 1080,
            "instances": [{"label": "reading"}]}))
    with pytest.raises(ValueError):
        api().prepare_dataset(root, output, max_images=5)
    assert json.loads(audit.read_text()) == {"existing_run": "preserve"}


def test_rare_class_selection_uses_training_support_only():
    y = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1]], dtype=np.float32)
    kept, dropped = api().supported_labels(y, ["common", "balanced", "rare"], 2)
    assert kept == [0, 1]
    assert dropped == [{"label": "rare", "positive": 1, "negative": 3}]


def test_threshold_selection_uses_validation_and_handles_unobserved_label():
    y = np.array([[0, 0], [0, 0], [1, 0], [1, 0]], dtype=np.float32)
    p = np.array([[.1, .1], [.2, .1], [.3, .2], [.4, .2]], dtype=np.float32)
    thresholds = api().choose_thresholds(y, p)
    assert thresholds.shape == (2,)
    assert .2 < thresholds[0] <= .3
    assert thresholds[1] == .5


def test_head_learns_separable_fixture_and_reload_matches(tmp_path):
    rng = np.random.default_rng(24)
    x = rng.normal(size=(120, 6)).astype(np.float32)
    y = np.column_stack([x[:, 0] > 0, x[:, 1] < 0]).astype(np.float32)
    model = api().fit_head(x[:80], y[:80], x[80:100], y[80:100],
                           epochs=160, seed=12, regularization=.001)
    probabilities = api().predict_reduced(model, x[100:])
    assert np.mean((probabilities >= .5) == y[100:]) > .85
    np.testing.assert_allclose(model["mean"], x[:80].mean(axis=0), atol=1e-6)
    path = tmp_path / "model.npz"
    api().save_head(model, path, ["a", "b"], np.array([.5, .5]))
    restored = api().load_head(path)
    np.testing.assert_array_equal(probabilities, api().predict_reduced(restored, x[100:]))
    assert restored["labels"].tolist() == ["a", "b"]


def test_metrics_report_multilabel_confusions_and_absent_support():
    y = np.array([[1, 0], [0, 0], [1, 0]], dtype=np.float32)
    p = np.array([[.9, .9], [.9, .1], [.1, .1]], dtype=np.float32)
    result = api().multilabel_metrics(y, p, np.array([.5, .5]), ["reading", "absent"])
    assert result["per_class"]["reading"]["confusion"] == {"tn": 0, "fp": 1, "fn": 1, "tp": 1}
    assert result["per_class"]["absent"]["average_precision"] is None
    assert result["map_eligible_classes"] == 1
    json.dumps(result, allow_nan=False)


def test_constant_score_average_precision_is_prevalence():
    result = api().multilabel_metrics(np.array([[1], [0], [1], [0]]),
                                     np.full((4, 1), .5), np.array([.5]), ["a"])
    assert result["mean_average_precision"] == .5


def test_prepare_and_train_bundle_are_portable_and_export_honest_contract(tmp_path):
    root = tmp_path / "ARIC_supplement"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    with tarfile.open(root / "ARIC_CLIP_global_view_features.tar.gz", "w:gz") as tf:
        for i in range(30):
            image_id, label = f"{i:05d}", "reading" if i % 2 else "listening"
            (annotations / f"{image_id}.json").write_text(json.dumps({
                "image_name": image_id, "image_width": 1920, "image_height": 1080,
                "caption": "target leaking caption must not be read",
                "instances": [{"label": label, "instance_name": f"{label}/{image_id}_{label}-1.pkl"}]}))
            array = np.ones((1, 401, 768), dtype=np.float32)
            array[0, :, i % 2] = 3
            payload = io.BytesIO()
            np.save(payload, array)
            data = payload.getvalue()
            member = tarfile.TarInfo(f"features/{image_id}.npy")
            member.size = len(data)
            tf.addfile(member, io.BytesIO(data))
    output = tmp_path / "output"
    bundle = api().prepare_dataset(root, output, max_images=30, seed=41)
    assert bundle.exists()
    contract = json.loads((output / "contract.json").read_text())
    assert contract["task"] == "frame_multilabel_behavior_presence"
    assert contract["raw_rgb_compatible"] is False
    assert contract["input_modalities"] == ["global_shallow_visual_features"]
    assert contract["uses_generated_captions"] is False
    (root / "ARIC_CLIP_global_view_features.tar.gz").unlink()
    result = api().train_from_bundle(bundle, output, min_support=2, epochs=60, seed=41)
    assert result["dataset"]["sample_count"] == 30
    assert sum(result["dataset"]["split_counts"].values()) == 30
    assert result["reload_max_absolute_error"] == 0
    assert result["feature_contract"]["raw_rgb_compatible"] is False
    assert (output / "model.npz").exists()
    assert (output / "test_predictions.npz").exists()
    assert result["test"]["macro_f1"] > .8
    assert result["baselines"]["train_prevalence"]["test"]["mean_average_precision"] < result["test"]["mean_average_precision"]
    assert result["baselines"]["all_positive"]["test"]["macro_f1"] < result["test"]["macro_f1"]


def test_training_rejects_overlapping_portable_image_splits(tmp_path):
    path = tmp_path / "unsafe.npz"
    np.savez(path, features=np.ones((5, 768), dtype=np.float32),
             targets=np.ones((5, 1), dtype=np.float32), image_ids=np.array(list("abcde")),
             labels=np.array(["reading"]), train_indices=np.array([0, 1, 2]),
             validation_indices=np.array([2]), test_indices=np.array([4]))
    with pytest.raises(ValueError, match="overlap"):
        api().train_from_bundle(path, tmp_path / "bad-output")
