"""Build full-source ASR/OCR artifacts; no training or synthetic transcript."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT.parent / "speech-models" / "lesson1.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "lesson1-real")
    parser.add_argument("--mode", choices=["asr", "ocr", "all"], default="all")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--language", default="zh", help="Whisper language code, e.g. zh or en; default preserves existing Chinese pipeline")
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--roi", nargs=4, type=float, default=[0.49, 0.055, 0.91, 0.47])
    parser.add_argument('--ocr-languages', choices=['chi_sim', 'chi_sim+eng', 'eng'], default='chi_sim+eng')
    parser.add_argument('--ocr-scale', type=float, default=3)
    parser.add_argument('--ocr-body', nargs=2, type=float, default=[.13, .90])
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    active_mode = args.mode
    try:
        if args.mode in ("asr", "all"):
            active_mode = "asr"
            from review.asr import run_asr
            run_asr(args.video, args.output, cpu_threads=args.cpu_threads, beam_size=args.beam_size,
                    resume=not args.no_resume, language=args.language)
        if args.mode in ("ocr", "all"):
            active_mode = "ocr"
            from review.media import run_ocr
            run_ocr(args.video, args.output, sample_interval=args.sample_interval, roi=args.roi,
                    resume=not args.no_resume, languages=args.ocr_languages, scale=args.ocr_scale, body_region=args.ocr_body)
    except Exception as error:
        from review.asr import atomic_json, utc_now
        atomic_json(args.output / f"{active_mode}.progress.json",
                    {"status": "failed", "failed_at": utc_now(),
                     "error_type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
