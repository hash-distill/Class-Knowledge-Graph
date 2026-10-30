"""Conservative observed-frame to document-page matching; never invent a clock."""
from __future__ import annotations
import math
import json
import os
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import unicodedata

LEXICAL_THRESHOLD = 0.43
IMAGE_THRESHOLD = 0.62
MIN_MARGIN = 0.075


def _normalize(text):
    return ''.join(char for char in unicodedata.normalize('NFKC', str(text)).lower() if char.isalnum())


@lru_cache(maxsize=2000)
def _grams(text):
    text = _normalize(text)
    return {text[index:index + 2] for index in range(len(text) - 1)}


def lexical_score(observed, page):
    left, right = _grams(observed), _grams(page)
    intersection = len(left & right)
    if min(len(left), len(right)) < 7 or intersection < 5:
        return 0.0
    dice = 2 * intersection / (len(left) + len(right))
    containment = intersection / min(len(left), len(right))
    return 0.65 * dice + 0.35 * containment


def _image_features(path, root):
    try:
        import cv2
        import numpy as np
        # Bound CPU work and avoid native thread-pool collisions when PDF/OCR
        # libraries and OpenCV are loaded in the same Windows worker process.
        cv2.setNumThreads(1)
        # Keep this CPU matcher independent of OpenCL device/driver startup,
        # which can block ORB even when native threading is limited above.
        cv2.ocl.setUseOpenCL(False)
        if not path:
            return None
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            return None
        # imdecode supports Windows paths containing Chinese characters.
        image = cv2.imdecode(np.fromfile(resolved, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        factor = min(1.0, 1100 / max(image.shape))
        if factor < 1:
            image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        # Binary matching cost grows roughly quadratically with keypoint count.
        # 800 spread keypoints retain a bounded geometric check for every page.
        points, descriptors = cv2.ORB_create(nfeatures=800, fastThreshold=12).detectAndCompute(image, None)
        if descriptors is None or len(points) < 20:
            return None
        return points, descriptors, image.shape
    except (ImportError, OSError, ValueError):
        return None


def _visual_score(left, right):
    if left is None or right is None:
        return 0.0, {'inliers': 0}
    import cv2
    import numpy as np
    try:
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left[1], right[1], k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
        # Repeated glyphs may all hit one descriptor; they are not spatial evidence.
        unique = {}
        for match in sorted(matches, key=lambda item: item.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        if len(matches) < 16:
            return 0.0, {'inliers': 0, 'keypoint_matches': len(matches)}
        source = np.float32([left[0][item.queryIdx].pt for item in matches]).reshape(-1, 1, 2)
        target = np.float32([right[0][item.trainIdx].pt for item in matches]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
        if homography is None or mask is None or not np.isfinite(homography).all():
            return 0.0, {'inliers': 0}
        valid = mask.ravel().astype(bool)
        inliers = int(valid.sum())
        ratio = inliers / len(matches)
        extent = np.ptp(target[valid, 0, :], axis=0) if inliers else np.zeros(2)
        coverage = float(extent.prod() / (right[2][0] * right[2][1]))
        height, width = left[2]
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, homography)
        area = abs(float(cv2.contourArea(transformed)))
        page_area = right[2][0] * right[2][1]
        if inliers < 12 or ratio < 0.62 or coverage < 0.12 or not 0.2 < area / page_area < 5:
            return 0.0, {'inliers': inliers, 'inlier_ratio': round(ratio, 3), 'coverage': round(coverage, 3)}
        score = 0.45 + 0.55 * ratio * min(1.0, inliers / 30)
        return score, {'inliers': inliers, 'inlier_ratio': round(ratio, 3), 'coverage': round(coverage, 3)}
    except cv2.error:
        return 0.0, {'inliers': 0}


def _align_in_process(document, bundle, lesson_dir):
    """Return one decision per observed S slide, allowing repeated/out-of-order pages.

    Only auto_matched rows carry a page_id. The source slide's observed time span
    is copied verbatim; ambiguous candidates never become timed page evidence.
    """
    root = Path(lesson_dir).resolve()
    pages = document.get('pages', [])
    features = {}
    decisions = []
    for slide in bundle.get('slides', []):
        start, end = slide.get('start_sec'), slide.get('end_sec')
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not math.isfinite(start + end) or end <= start:
            continue
        lexical_scores = {page['id']: lexical_score(slide.get('text', ''), page.get('text', '')) for page in pages}
        ranked_lexical = sorted(lexical_scores.values(), reverse=True)
        # Clear lexical evidence is an independent accepted matching method.
        # All raw page scores still participate in its margin; image fallback is
        # used only when the lexical result does not confidently distinguish it.
        strong_lexical = bool(ranked_lexical and ranked_lexical[0] >= 0.60 and
                              ranked_lexical[0] - (ranked_lexical[1] if len(ranked_lexical) > 1 else 0) >= 0.15)
        observation = None if strong_lexical else _image_features(slide.get('image_path'), root)
        if observation is not None and not features:
            features = {page['id']: _image_features(page.get('image_path'), root) for page in pages}
        candidates = []
        for page in pages:
            lexical = lexical_scores[page['id']]
            visual, visual_details = _visual_score(observation, features.get(page['id']))
            # Compare raw scores before applying acceptance thresholds. Cutting
            # a near-tied runner-up to zero would fabricate a large margin.
            score = max(visual, lexical)
            candidates.append({'page_id': page['id'], 'page_number': page['page_number'],
                               'score': round(score, 4), 'lexical_score': round(lexical, 4),
                               'visual_score': round(visual, 4), 'visual_evaluated': observation is not None,
                               **visual_details})
        candidates.sort(key=lambda row: (-row['score'], row['page_number']))
        first = candidates[0] if candidates else None
        score = first['score'] if first else 0.0
        margin = score - (candidates[1]['score'] if len(candidates) > 1 else 0)
        qualified = first and (first['lexical_score'] >= LEXICAL_THRESHOLD or first['visual_score'] >= IMAGE_THRESHOLD)
        status = 'unmatched' if not qualified else ('auto_matched' if margin >= MIN_MARGIN else 'ambiguous')
        method = 'observed_ocr_lexical'
        if first and first['visual_score'] >= IMAGE_THRESHOLD and first['visual_score'] >= first['lexical_score']:
            method = 'observed_frame_orb_ransac'
        if not qualified:
            method = 'insufficient_observed_evidence'
        decisions.append({'id': f"M{document['document_id'][1:]}_{slide['id']}",
                          'document_id': document['document_id'], 'slide_id': slide['id'],
                          'page_id': first['page_id'] if status == 'auto_matched' else None,
                          'start_sec': start, 'end_sec': end, 'status': status,
                          'score': score, 'margin': round(margin, 4), 'method': method,
                          'candidates': candidates[:3],
                          'thresholds': {'lexical': LEXICAL_THRESHOLD, 'visual': IMAGE_THRESHOLD, 'margin': MIN_MARGIN},
                          'uncertainty_sec': slide.get('uncertainty_sec', 0)})
    return decisions


def align_document(document, bundle, lesson_dir):
    """Run native image matching in an isolated, time-bounded worker.

    PyMuPDF and OpenCV native runtimes can deadlock when both are loaded in one
    Windows process. The worker imports only OpenCV; plain JSON crosses the
    process boundary and no lesson metadata or source files are changed.
    """
    payload = {'document': document, 'bundle': {'slides': bundle.get('slides', [])},
               'lesson_dir': str(Path(lesson_dir).resolve())}
    try:
        result = subprocess.run([sys.executable, '-m', 'review.courseware_alignment'],
                                input=json.dumps(payload, ensure_ascii=False), capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=300,
                                cwd=Path(__file__).resolve().parents[1],
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('Courseware screen matching timed out. No estimated timestamps were generated. Reduce the page count and try again.') from exc
    if result.returncode:
        raise ValueError('Courseware screen matching failed: ' + result.stderr.strip()[-1000:])
    return json.loads(result.stdout)


if __name__ == '__main__':
    payload = json.load(sys.stdin)
    json.dump(_align_in_process(payload['document'], payload['bundle'], payload['lesson_dir']),
              sys.stdout, ensure_ascii=True, allow_nan=False)
