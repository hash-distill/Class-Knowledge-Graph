"""Conservative offline ARIC frame-presence baseline, using NumPy only.

Global shallow features are opaque tokens: no spatial layout or original
individual-instance extractor is assumed. Captions are never model inputs.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import platform
from collections import Counter
import tarfile
import time

import numpy as np


def select_image_ids(image_ids, max_images=2048, seed=41):
    """Hash-rank the complete annotation population, independent of archive order."""
    ids = list(image_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate original image ID")
    if max_images < 5:
        raise ValueError("at least five images are required")
    key = lambda value: hashlib.sha256(f"aric-sample:{seed}:{value}".encode()).digest()
    return sorted(sorted(ids, key=key)[:max_images])


def split_image_ids(image_ids, seed=41):
    """60/20/20 original-image split; lesson identities are unavailable."""
    ids = list(image_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate original image ID")
    if len(ids) < 5:
        raise ValueError("at least five images are required")
    key = lambda value: hashlib.sha256(f"aric-split:{seed}:{value}".encode()).digest()
    ordered = sorted(ids, key=key)
    train_end, val_end = int(len(ids) * .6), int(len(ids) * .8)
    return {"train": sorted(ordered[:train_end]),
            "validation": sorted(ordered[train_end:val_end]),
            "test": sorted(ordered[val_end:])}


def annotation_labels(annotation):
    return sorted({item["label"] for item in annotation.get("instances", [])
                   if isinstance(item.get("label"), str) and item["label"]})


def reduce_global_feature(array):
    """Mean all 401 opaque tokens, then L2 normalize; output is 768 float32 values."""
    array = np.asarray(array)
    if array.shape != (1, 401, 768):
        raise ValueError(f"unsupported global feature shape {array.shape}; expected (1,401,768)")
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("global feature must be finite float32")
    mean = array[0].mean(axis=0, dtype=np.float64)
    norm = np.linalg.norm(mean)
    if norm < 1e-12:
        raise ValueError("global feature has zero pooled norm")
    return (mean / norm).astype(np.float32)


def load_global_feature(payload):
    return reduce_global_feature(np.load(io.BytesIO(payload), allow_pickle=False))


def _cache_source_identity(archive_path):
    archive_path = Path(archive_path)
    try:
        source_stat = archive_path.stat()
    except OSError as error:
        raise ValueError("cannot verify source archive for cache reuse; restore the source "
                         "or train the portable package with --from-bundle") from error
    return {"archive": str(archive_path.resolve()), "archive_bytes": source_stat.st_size,
            "archive_mtime_ns": source_stat.st_mtime_ns,
            "reduction": "mean_all_401_tokens_then_l2", "source_shape": [1, 401, 768]}


def _validate_cache_source(archive_path, cache_path):
    expected = _cache_source_identity(archive_path)
    try:
        metadata = json.loads(Path(cache_path).with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("existing cache has no valid source identity; use a fresh output directory") from error
    changed = [key for key, value in expected.items() if metadata.get(key) != value]
    if changed:
        raise ValueError("existing cache source/reduction identity differs (" + ", ".join(changed)
                         + "); use a fresh output directory")
    return metadata


def cache_features(archive_path, selected_ids, cache_path):
    """Stream tar without extraction; save only requested reduced vectors atomically."""
    archive_path, cache_path = Path(archive_path), Path(cache_path)
    selected_ids = sorted(selected_ids)
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists():
        metadata = _validate_cache_source(archive_path, cache_path)
        with np.load(cache_path, allow_pickle=False) as stored:
            ids = stored["image_ids"].tolist()
            x = stored["features"].copy()
        if ids != selected_ids or x.shape != (len(ids), 768) or not np.isfinite(x).all():
            raise ValueError("existing feature cache does not match requested IDs/feature contract")
        metadata["reused_cache"] = True
        return ids, x, metadata
    started = time.perf_counter()
    wanted, found, provenance = set(selected_ids), {}, {}
    source_identity = _cache_source_identity(archive_path)
    examined = 0
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".npy"):
                continue
            examined += 1
            image_id = Path(member.name).stem
            if image_id in wanted:
                if image_id in found:
                    raise ValueError(f"duplicate feature member for {image_id}")
                if member.size > 2_000_000:
                    raise ValueError(f"unexpected oversized feature member {member.name}")
                with archive.extractfile(member) as stream:
                    payload = stream.read()
                found[image_id] = load_global_feature(payload)
                provenance[image_id] = {"member": member.name, "bytes": member.size,
                                        "sha256": hashlib.sha256(payload).hexdigest()}
                if len(found) % 128 == 0:
                    print(f"ARIC cached {len(found)}/{len(wanted)} selected images; "
                          f"scanned {examined} feature members; {time.perf_counter()-started:.1f}s", flush=True)
                if len(found) == len(wanted):
                    break
    missing = sorted(wanted - found.keys())
    if missing:
        raise ValueError(f"missing {len(missing)} selected feature IDs: {missing[:10]}")
    features = np.stack([found[image_id] for image_id in selected_ids])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(temporary, image_ids=np.array(selected_ids), features=features)
    temporary.replace(cache_path)
    metadata = {"cached_images": len(selected_ids), "scanned_members": examined,
                "seconds": time.perf_counter()-started, "reused_cache": False,
                "members": provenance, **source_identity}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return selected_ids, features, metadata


def supported_labels(y_train, labels, min_support=15):
    positive = y_train.sum(axis=0).astype(int)
    negative = len(y_train)-positive
    kept = [i for i in range(len(labels)) if min(positive[i], negative[i]) >= min_support]
    dropped = [{"label": labels[i], "positive": int(positive[i]), "negative": int(negative[i])}
               for i in range(len(labels)) if i not in kept]
    return kept, dropped


def _sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -35, 35)))


def predict_reduced(model, features):
    """Predict sigmoid scores for saved reduced features, never raw RGB images."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(model["mean"]) or not np.isfinite(x).all():
        raise ValueError("features must be a finite matrix matching the saved reduced feature dimension")
    standardized = (x-model["mean"])/model["scale"]
    return _sigmoid(standardized @ model["weights"] + model["bias"])


def fit_head(x_train, y_train, x_validation, y_validation, epochs=240, seed=41,
             regularization=.01, learning_rate=.01):
    """Full-batch, class-weighted logistic head, Adam; best validation BCE checkpoint."""
    if not len(x_train) or not len(x_validation) or y_train.shape[1] == 0:
        raise ValueError("nonempty training, validation, and supported labels are required")
    rng = np.random.default_rng(seed)
    mean = x_train.mean(axis=0).astype(np.float32)
    scale = np.maximum(x_train.std(axis=0), 1e-4).astype(np.float32)
    x = ((x_train-mean)/scale).astype(np.float32)
    xv = ((x_validation-mean)/scale).astype(np.float32)
    weights = rng.normal(0, .001, (x.shape[1], y_train.shape[1])).astype(np.float32)
    bias = np.zeros(y_train.shape[1], dtype=np.float32)
    positives = y_train.sum(axis=0)
    positive_weight = np.clip((len(y_train)-positives)/np.maximum(positives, 1), .1, 10).astype(np.float32)
    sample_weight = np.where(y_train > 0, positive_weight, 1).astype(np.float32)
    parameters = [weights, bias]
    first = [np.zeros_like(p) for p in parameters]
    second = [np.zeros_like(p) for p in parameters]
    best_loss, best_epoch, best_parameters = float("inf"), 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        probability = _sigmoid(x @ weights + bias)
        residual = (probability-y_train)*sample_weight/len(x)
        gradients = [x.T @ residual + regularization*weights, residual.sum(axis=0)]
        for i, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            first[i] = .9*first[i] + .1*gradient
            second[i] = .999*second[i] + .001*gradient*gradient
            parameter -= learning_rate*(first[i]/(1-.9**epoch))/(np.sqrt(second[i]/(1-.999**epoch))+1e-8)
        pv = np.clip(_sigmoid(xv @ weights+bias), 1e-6, 1-1e-6)
        loss = float(-np.mean(y_validation*np.log(pv)+(1-y_validation)*np.log(1-pv)))
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_parameters = [weights.copy(), bias.copy()]
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "validation_bce": loss})
    return {"weights": best_parameters[0], "bias": best_parameters[1], "mean": mean,
            "scale": scale, "positive_weight": positive_weight,
            "training": {"epochs": epochs, "best_epoch": best_epoch, "validation_bce": best_loss,
                         "seconds": time.perf_counter()-started, "history": history,
                         "regularization": regularization, "learning_rate": learning_rate, "seed": seed}}


def choose_thresholds(y_validation, probabilities):
    thresholds = np.full(y_validation.shape[1], .5, dtype=np.float32)
    for i in range(y_validation.shape[1]):
        truth = y_validation[:, i] > 0
        if not truth.any() or truth.all():
            continue
        best = (-1., -1., .5)
        for threshold in np.round(np.arange(.1, .91, .05), 2):
            pred = probabilities[:, i] >= threshold
            tp = int((truth & pred).sum())
            f1 = 2*tp/max(int(truth.sum()+pred.sum()), 1)
            candidate = (f1, -abs(float(threshold)-.5), float(threshold))
            if candidate > best:
                best = candidate
        thresholds[i] = best[2]
    return thresholds


def save_head(model, path, labels, thresholds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in model.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(path, **arrays, labels=np.array(labels), thresholds=thresholds)


def load_head(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def multilabel_metrics(y, probabilities, thresholds, labels):
    truth, predicted = y > 0, probabilities >= thresholds
    per_class, aps, f1s, f1s_supported = {}, [], [], []
    total_tp = total_fp = total_fn = 0
    for i, label in enumerate(labels):
        yt, yp = truth[:, i], predicted[:, i]
        tp, fp = int((yt & yp).sum()), int((~yt & yp).sum())
        fn, tn = int((yt & ~yp).sum()), int((~yt & ~yp).sum())
        precision, recall = tp/max(tp+fp, 1), tp/max(tp+fn, 1)
        f1 = 2*tp/max(2*tp+fp+fn, 1)
        support = int(yt.sum())
        ap = None
        if 0 < support < len(y):
            order = np.argsort(-probabilities[:, i], kind="stable")
            ranked, scores = yt[order], probabilities[order, i]
            ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores)-1]
            cumulative = np.cumsum(ranked)[ends]
            recall_steps = cumulative/support
            ap = float(np.sum(np.diff(np.r_[0, recall_steps]) * cumulative/(ends+1)))
            aps.append(ap)
        per_class[label] = {"support": support, "negative_support": len(y)-support,
                            "precision": precision, "recall": recall, "f1": f1,
                            "average_precision": ap, "threshold": float(thresholds[i]),
                            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}
        f1s.append(f1)
        if support > 0:
            f1s_supported.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return {"samples": len(y), "classes": len(labels), "macro_f1": float(np.mean(f1s)),
            "macro_f1_positive_support": float(np.mean(f1s_supported)) if f1s_supported else None,
            "micro_f1": 2*total_tp/max(2*total_tp+total_fp+total_fn, 1),
            "mean_average_precision": float(np.mean(aps)) if aps else None,
            "map_eligible_classes": len(aps), "subset_accuracy": float(np.all(truth == predicted, axis=1).mean()),
            "hamming_loss": float(np.mean(truth != predicted)), "per_class": per_class}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(dataset_root, output, max_images=2048, seed=41):
    """Audit annotations, hash-sample original images, and produce a portable bundle."""
    started = time.perf_counter()
    root, output = Path(dataset_root), Path(output)
    supplement = root if (root / "annotations").is_dir() else root / "ARIC_supplement"
    if (output / "features.npz").exists():
        _validate_cache_source(supplement / "ARIC_CLIP_global_view_features.tar.gz", output / "features.npz")
    output.mkdir(parents=True, exist_ok=True)
    files = sorted((supplement / "annotations").glob("*.json"))
    if not files:
        raise ValueError(f"no ARIC JSON annotations in {supplement}")
    chosen = select_image_ids([path.stem for path in files], max_images, seed)
    chosen_set = set(chosen)
    selected, instance_support, frame_support, dimensions = {}, Counter(), Counter(), Counter()
    for path in files:
        payload = path.read_bytes()
        annotation = json.loads(payload)
        image_id = path.stem
        if str(annotation["image_name"]) != image_id:
            raise ValueError(f"annotation identity mismatch for {path}")
        labels = annotation_labels(annotation)
        frame_support.update(labels)
        instance_support.update(item["label"] for item in annotation["instances"])
        dimensions[f"{annotation['image_width']}x{annotation['image_height']}"] += 1
        if image_id in chosen_set:
            selected[image_id] = {"image_id": image_id, "labels": labels,
                                  "annotation_sha256": hashlib.sha256(payload).hexdigest()}
    all_labels = sorted(frame_support)
    split = split_image_ids(chosen, seed)
    assigned = {image_id: name for name, ids in split.items() for image_id in ids}
    manifest = [{**selected[image_id], "split": assigned[image_id]} for image_id in chosen]
    _write_json(output / "split_manifest.json", {"seed": seed, "unit": "original_image_id",
                "selection": "sha256_rank_over_all_annotation_ids", "samples": manifest})
    audit = {"annotation_images": len(files), "annotation_instances": sum(instance_support.values()),
             "selected_images": len(chosen), "annotation_frame_support": dict(frame_support),
             "annotation_instance_support": dict(instance_support), "image_dimensions": dict(dimensions),
             "generated_captions_used": False, "individual_features_present": (supplement.parent / "ARIC_dataset/image_features").exists()}
    index_root = supplement.parent / "ARIC_dataset/index"
    groups, index_counts = {}, {}
    for part in ("train", "test"):
        ids, count = set(), 0
        for path in sorted((index_root / part).glob("*.txt")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ids.add(Path(line.rsplit(" ", 1)[0].replace("\\", "/")).stem.split("_", 1)[0])
                count += 1
        groups[part], index_counts[part] = ids, count
    audit["original_index_instances"] = index_counts
    audit["original_index_image_overlap"] = len(groups["train"] & groups["test"])
    _write_json(output / "data_audit.json", audit)
    contract = {"schema_version": "1.0", "model_id": f"aric-global-frame-presence-seed{seed}",
                "task": "frame_multilabel_behavior_presence", "prediction_unit": "original_frame",
                "input_modalities": ["global_shallow_visual_features"],
                "source_array_shape": [1, 401, 768], "source_dtype": "float32",
                "reduced_feature_dimension": 768, "reduction": "mean_all_401_tokens_then_l2",
                "uses_generated_captions": False, "raw_rgb_compatible": False,
                "individual_instance_classifier": False, "online_status": "offline_research_only",
                "encoder_preprocessing_verified": False,
                "source_readme": str((supplement / "ReadMe.md").resolve()),
                "known_preprocessing": "source README: resize original whole image to 640x640, shallow pretrained CLIP features",
                "unverified_preprocessing": ["exact encoder checkpoint", "shallow layer identity", "normalization", "token spatial ordering"],
                "output": "independent sigmoid scores; class-weighted training, not calibrated probabilities",
                "limitations": ["Frame presence does not localize a person, count people, or estimate engagement.",
                    "Original image IDs are separated; lesson/video identities are unavailable, so cross-lesson generalization is unverified.",
                    "Different image IDs can be adjacent frames or the same classroom; original-image split alone does not remove that dependence.",
                    "Label presence is derived from supplied instance annotations; missing annotations are not an independently audited absence.",
                    "No raw images, exact extractor parity, individual-feature package, or raw-RGB deployment has been validated."]}
    _write_json(output / "contract.json", contract)
    print(f"ARIC annotation audit: {len(files)} original images; selecting {len(chosen)} across full ID population; "
          f"{len(all_labels)} labels; official index overlap={audit['original_index_image_overlap']}", flush=True)
    ids, features, cache = cache_features(supplement / "ARIC_CLIP_global_view_features.tar.gz", chosen, output / "features.npz")
    y = np.array([[label in selected[image_id]["labels"] for label in all_labels] for image_id in ids], dtype=np.float32)
    rows = {image_id: i for i, image_id in enumerate(ids)}
    indices = {f"{name}_indices": np.array([rows[image_id] for image_id in group], dtype=np.int64)
               for name, group in split.items()}
    bundle = output / "training_bundle.npz"
    np.savez_compressed(bundle, features=features, targets=y, labels=np.array(all_labels),
                        image_ids=np.array(ids), seed=np.array(seed), **indices)
    preparation = {"seconds": time.perf_counter()-started, "cache_seconds": cache["seconds"],
                   "bundle_sha256": _file_sha256(bundle), "feature_cache_sha256": _file_sha256(output / "features.npz"),
                   "platform": platform.platform(), "numpy_version": np.__version__, "python": platform.python_version()}
    _write_json(output / "preparation.json", preparation)
    print(f"ARIC portable bundle saved: {bundle} ({bundle.stat().st_size} bytes)", flush=True)
    return bundle


def train_from_bundle(bundle_path, output, min_support=15, epochs=240, seed=41):
    """Train without source dataset access; only validation selects checkpoints/thresholds."""
    started = time.perf_counter()
    bundle_path, output = Path(bundle_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(bundle_path, allow_pickle=False) as bundle:
        x, y = bundle["features"].copy(), bundle["targets"].copy()
        labels, image_ids = bundle["labels"].tolist(), bundle["image_ids"].tolist()
        indices = {name: bundle[f"{name}_indices"].copy() for name in ("train", "validation", "test")}
    flattened = np.concatenate(list(indices.values()))
    if len(flattened) != len(x) or len(np.unique(flattened)) != len(x) or set(flattened) != set(range(len(x))):
        raise ValueError("portable bundle splits overlap, omit, or duplicate original-image rows")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("portable bundle duplicates original image IDs")
    train, validation, test = (indices[key] for key in ("train", "validation", "test"))
    kept, dropped = supported_labels(y[train], labels, min_support)
    if not kept:
        raise ValueError("no labels have enough training positive and negative support")
    selected_labels = [labels[i] for i in kept]
    y = y[:, kept]
    print(f"ARIC fit: {len(x)} images; train/validation/test={len(train)}/{len(validation)}/{len(test)}; "
          f"{len(kept)} supported labels, {len(dropped)} excluded using training support only", flush=True)
    model = fit_head(x[train], y[train], x[validation], y[validation], epochs=epochs, seed=seed)
    validation_p = predict_reduced(model, x[validation])
    thresholds = choose_thresholds(y[validation], validation_p)
    model_path = output / "model.npz"
    save_head(model, model_path, selected_labels, thresholds)
    test_p = predict_reduced(model, x[test])
    reloaded = load_head(model_path)
    reload_error = float(np.max(np.abs(test_p-predict_reduced(reloaded, x[test]))))
    np.savez_compressed(output / "test_predictions.npz", image_ids=np.array(image_ids)[test],
                        targets=y[test], scores=test_p, thresholds=thresholds, labels=np.array(selected_labels))
    prevalence = y[train].mean(axis=0)
    baseline_scores = {"train_prevalence": prevalence, "all_negative": np.zeros_like(prevalence),
                       "all_positive": np.ones_like(prevalence)}
    baselines = {}
    for name, scores in baseline_scores.items():
        baselines[name] = {part: multilabel_metrics(y[rows], np.broadcast_to(scores, (len(rows), len(scores))),
                          np.full(len(scores), .5), selected_labels) for part, rows in indices.items() if part != "train"}
    source_contract = bundle_path.parent / "contract.json"
    if source_contract.exists():
        contract = json.loads(source_contract.read_text(encoding="utf-8"))
    else:
        contract = {"task": "frame_multilabel_behavior_presence", "raw_rgb_compatible": False,
                    "online_status": "offline_research_only", "reduction": "mean_all_401_tokens_then_l2"}
    contract["labels"] = selected_labels
    contract["threshold_selection"] = "per-class F1 on validation only; grid 0.10 to 0.90 step 0.05; absent/all-positive validation class uses 0.5"
    _write_json(output / "contract.json", contract)
    class_support = {part: {label: {"positive": int(y[rows, i].sum()), "negative": int(len(rows)-y[rows, i].sum())}
                    for i, label in enumerate(selected_labels)} for part, rows in indices.items()}
    result = {"schema_version": "1.0", "model_id": contract.get("model_id", f"aric-global-frame-presence-seed{seed}"),
              "status": "trained_evaluated_offline", "feature_contract": contract,
              "dataset": {"sample_count": len(x), "split_counts": {name: len(rows) for name, rows in indices.items()},
                          "split_unit": "original_image_id", "supported_classes": len(kept), "class_support": class_support,
                          "dropped_labels_training_support": dropped, "min_train_positive_and_negative": min_support},
              "training": model["training"], "validation": multilabel_metrics(y[validation], validation_p, thresholds, selected_labels),
              "test": multilabel_metrics(y[test], test_p, thresholds, selected_labels), "baselines": baselines,
              "test_default_0_5_threshold": multilabel_metrics(y[test], test_p, np.full(len(kept), .5), selected_labels),
              "reload_max_absolute_error": reload_error, "runtime_seconds": time.perf_counter()-started,
              "compute": {"device": "CPU", "platform": platform.platform(), "processor": platform.processor(),
                          "python_version": platform.python_version(), "numpy_version": np.__version__},
              "provenance": {"bundle_sha256": _file_sha256(bundle_path), "model_sha256": _file_sha256(model_path),
                             "seed": seed, "test_used_for_tuning": False},
              "export_paths": {"model": str(model_path.resolve()), "contract": str((output / "contract.json").resolve()),
                               "metrics": str((output / "metrics.json").resolve()), "bundle": str(bundle_path.resolve())}}
    _write_json(output / "metrics.json", result)
    print(json.dumps({"samples": len(x), "classes": len(kept), "test_macro_f1": result["test"]["macro_f1"],
                      "test_mAP": result["test"]["mean_average_precision"], "seconds": result["runtime_seconds"]}), flush=True)
    return result
