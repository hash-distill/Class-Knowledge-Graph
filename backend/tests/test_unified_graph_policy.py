"""The demo's graph display policy must not rewrite the preserved evidence."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/build_unified_chemistry_demo.py"
spec = spec_from_file_location("unified_chemistry_demo", MODULE)
demo = module_from_spec(spec)
spec.loader.exec_module(demo)


def sample():
    nodes = [
        {"id": "K1", "label": "氨基酸", "properties": {"source_refs": ["T1", "S1"]}},
        {"id": "K2", "label": "二发氨基酸", "properties": {"source_refs": ["T1"]}},
        {"id": "K3", "label": "谷氨酸", "properties": {"source_refs": ["S1"]}},
    ]
    return {"lesson": {"duration_sec": 90., "source_video": {"sha256": "a" * 64}},
        "transcript": [{"id": "T1", "text": "二发氨基酸"}],
        "transcript_unguided": [{"id": "T0", "text": "二发安吉酸"}],
        "slides": [{"id": "S1", "method": "video_ocr", "text": "认识氨基酸\n谷 氨 酸"}],
        "graph": {"nodes": nodes, "edges": [
            {"id": "E1", "source": "K1", "target": "K2"},
            {"id": "E2", "source": "K1", "target": "K3"}], "provenance": {}},
        "alignment": [{"concept_ids": ["K1", "K2", "K3"]}],
        "report": {"summary": "旧3概念", "limitations": [],
            "statistics": {"concepts": 3, "relations": 2, "activity_sampled_frames": 181},
            "sections": [{"title": n["label"], "evidence_ids": n["properties"]["source_refs"]} for n in nodes] +
                        [{"title": "可见行为与课堂情境", "text": "行为原文", "evidence_ids": ["V1"]}]}}


def test_filter_is_pure_and_keeps_only_source_supported_existing_candidates():
    bundle = sample()
    untouched = deepcopy(bundle)
    filtered, archive = demo.filter_graph_with_observed_courseware(bundle)
    assert bundle == untouched
    assert archive["original_graph"] == untouched["graph"]
    assert [n["id"] for n in filtered["graph"]["nodes"]] == ["K1", "K3"]
    assert [e["id"] for e in filtered["graph"]["edges"]] == ["E2"]
    assert filtered["alignment"][0]["concept_ids"] == ["K1", "K3"]
    assert filtered["transcript"] == untouched["transcript"]
    assert filtered["transcript_unguided"] == untouched["transcript_unguided"]
    assert [r["title"] for r in filtered["report"]["sections"]] == ["氨基酸", "谷氨酸", "可见行为与课堂情境"]
    assert filtered["report"]["statistics"] == {"concepts": 2, "relations": 1,
        "activity_sampled_frames": 181, "graph_candidates_original": 3, "graph_candidates_hidden": 1}
    assert "展示 2 个" in filtered["report"]["summary"]
    assert filtered["graph"]["provenance"]["display_policy"]["teacher_or_chemical_accuracy_validated"] is False


def test_reapplying_with_archive_drops_non_model_overlay_terms_without_losing_archive():
    filtered, archive = demo.filter_graph_with_observed_courseware(sample())
    filtered["graph"]["nodes"].append({"id": "KD4", "label": "认识", "properties": {"source_refs": ["S1"]}})
    again, unchanged_archive = demo.filter_graph_with_observed_courseware(filtered, candidate_archive=archive)
    assert [n["id"] for n in again["graph"]["nodes"]] == ["K1", "K3"]
    assert unchanged_archive == archive
    assert len(unchanged_archive["original_graph"]["nodes"]) == 3


def test_no_character_substitution_or_external_document_fallback():
    bundle = sample()
    bundle["slides"] = [{"id": "S1", "method": "video_ocr", "text": "氨基栈"},
                        {"id": "C1", "method": "external_document", "text": "氨基酸谷氨酸"}]
    filtered, _ = demo.filter_graph_with_observed_courseware(bundle)
    assert filtered["graph"]["nodes"] == []
    assert filtered["graph"]["edges"] == []
    assert filtered["alignment"][0]["concept_ids"] == []
    assert filtered["report"]["statistics"]["concepts"] == 0


def build_fixture(tmp_path, monkeypatch):
    """Synthetic media/LLM fixtures test publication; they are not ASR evidence."""
    from review import builder
    from review.common import sha256
    video = tmp_path / "synthetic.mp4"
    video.write_bytes(b"synthetic test source, not decodable video")
    media = tmp_path / "fresh-output"
    media.mkdir()
    (media / "screen.jpg").write_bytes(b"synthetic image")
    source = {"sha256": sha256(video)}
    transcript = [{"id": "T1", "text": "二发氨基酸", "start_sec": 1., "end_sec": 5.}]
    slides = [{"id": "S1", "method": "video_ocr", "text": "氨基酸 谷氨酸", "title": "课件",
               "start_sec": 0., "end_sec": 10., "image_path": "screen.jpg"}]
    demo.atomic_json(media / "asr.json", {"status": "complete", "segments": transcript,
                                          "provenance": {"source_video": source}})
    demo.atomic_json(media / "asr.unguided.json", {"segments": [{"id": "T0", "text": "二发安吉酸", "start_sec": 1., "end_sec": 5.}],
                                                  "provenance": {"source_video": source}})
    demo.atomic_json(media / "slides.json", {"status": "complete", "slides": slides,
                                             "provenance": {"source_video": source}})
    monkeypatch.setattr(builder, "validate_media_timebase", lambda *args: 90.)
    calls = []
    def concepts(*args):
        calls.append("synthetic_model_call")
        return deepcopy(sample()["graph"])
    monkeypatch.setattr(builder, "extract_concepts", concepts)
    class FixtureProvider:
        def status(self): return {"model": "synthetic-proofreader", "source": "test_only"}
        def complete(self, *args, **kwargs):
            return {"segments": []}, {"model": "synthetic-proofreader", "source": "test_only"}
    monkeypatch.setattr(builder, "LLMProvider", FixtureProvider)
    monkeypatch.setattr(builder, "CorrectionProvider", FixtureProvider)
    monkeypatch.setattr(demo, "transcribe", lambda *a, **k: pytest.fail("build must not rerun ASR"))
    return video, media, calls


def test_fresh_build_publishes_filtered_cache_and_rebuild_preserves_original_archive(tmp_path, monkeypatch):
    from review import builder, courseware_service
    from review.mutation import LessonMutationLock
    from review.provider import ReviewError
    video, media, calls = build_fixture(tmp_path, monkeypatch)
    original_builder_root = builder.ROOT
    asr_bytes = (media / "asr.json").read_bytes()
    raw_bytes = (media / "asr.unguided.json").read_bytes()
    published_under_lease = []
    original_persist = courseware_service.persist_bundle
    def persist(directory, bundle):
        with pytest.raises(ReviewError) as error:
            with LessonMutationLock(directory):
                pass
        assert error.value.code == "lesson_busy"
        published_under_lease.append(True)
        original_persist(directory, bundle)
    monkeypatch.setattr(courseware_service, "persist_bundle", persist)
    first = demo.build(video, media, "synthetic-unified", False)
    archive_bytes = (media / "graph-candidates-original.json").read_bytes()
    assert len(demo.read_json(media / "graph-candidates-original.json")["original_graph"]["nodes"]) == 3
    assert [n["id"] for n in first["graph"]["nodes"]] == ["K1", "K3"]
    assert demo.read_json(media / "graph.json")["graph"] == first["graph"]
    second = demo.build(video, media, "synthetic-unified", False)
    assert [n["id"] for n in second["graph"]["nodes"]] == ["K1", "K3"]
    assert (media / "graph-candidates-original.json").read_bytes() == archive_bytes
    assert (media / "asr.json").read_bytes() == asr_bytes
    assert (media / "asr.unguided.json").read_bytes() == raw_bytes
    assert calls == ["synthetic_model_call"]
    assert published_under_lease == [True, True]
    assert builder.ROOT == original_builder_root


def test_rebuild_with_existing_courseware_filters_base_then_preserves_overlay(tmp_path, monkeypatch):
    import sys
    import types
    from review.courseware_service import overlay_bundle, source_fingerprint
    video, media, calls = build_fixture(tmp_path, monkeypatch)
    base = demo.build(video, media, "synthetic-unified", False)
    document = {"document_id": "Dfixture", "filename": "synthetic.pdf", "sha256": "b" * 64,
        "pages": [{"id": "DP1", "text": "氨基酸 谷氨酸"}],
        "matches": [{"id": "M1", "status": "auto_matched"}],
        "lesson_fingerprint": source_fingerprint(base)}
    def derived(*args):
        return {"courseware_evidence": [{"id": "C1", "page_id": "DP1", "text": "氨基酸 谷氨酸",
                                         "start_sec": 0., "end_sec": 10.}],
                "transcript_revisions": [], "revised_evidence_blocks": []}
    monkeypatch.setitem(sys.modules, "review.courseware_terms", types.SimpleNamespace(revise_transcript=derived))
    demo.atomic_json(media / "bundle.base.json", base)
    demo.atomic_json(media / "bundle.json", overlay_bundle(base, document))
    result = demo.build(video, media, "synthetic-unified", False)
    assert result["courseware"] == document
    assert result["courseware_evidence"][0]["id"] == "C1"
    assert result["transcript"] == base["transcript"]
    assert [n["id"] for n in result["graph"]["nodes"]] == ["K1", "K3"]
    filtered_base = demo.read_json(media / "bundle.base.json")
    assert "courseware" not in filtered_base
    assert [n["id"] for n in filtered_base["graph"]["nodes"]] == ["K1", "K3"]
    assert len(demo.read_json(media / "graph-candidates-original.json")["original_graph"]["nodes"]) == 3
    assert calls == ["synthetic_model_call"]
