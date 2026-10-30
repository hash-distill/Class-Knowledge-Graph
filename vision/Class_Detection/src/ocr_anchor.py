"""OCR anchor detection for screen/blackboard regions.

Detects text changes on the projected screen or blackboard to
generate knowledge-point anchor timestamps for VSAM alignment.
"""

from __future__ import annotations

import difflib
import hashlib
from typing import Optional

import numpy as np


class OCRAnchorDetector:
    """Detect text changes on screen/blackboard crops.

    Uses PaddleOCR lazily (imported on first use) so that the
    rest of the pipeline can run without PaddlePaddle installed.
    """

    def __init__(
        self,
        change_threshold: float = 0.3,
        lang: str = "ch",
    ) -> None:
        self.change_threshold = change_threshold
        self.lang = lang

        self._ocr = None  # lazy init
        self._prev_hash: Optional[str] = None
        self._prev_text: str = ""

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        return self._ocr

    def detect_change(self, crop: np.ndarray) -> Optional[str]:
        """Run OCR on a screen/blackboard crop.

        Returns the detected text if a significant change is found,
        or ``None`` if the content is unchanged.
        """
        ocr = self._get_ocr()
        results = ocr.ocr(crop, cls=True)

        # Flatten OCR results to a single text string
        lines: list[str] = []
        if results and results[0]:
            for entry in results[0]:
                if entry and len(entry) >= 2:
                    text = entry[1][0] if isinstance(entry[1], (list, tuple)) else str(entry[1])
                    lines.append(text)
        current_text = " ".join(lines)
        current_hash = hashlib.md5(current_text.encode("utf-8")).hexdigest()

        if self._prev_hash is None:
            # First observation — store and trigger if there is text
            self._prev_hash = current_hash
            self._prev_text = current_text
            if current_text.strip():
                return current_text
            return None

        if current_hash == self._prev_hash:
            return None

        # Compute simple change ratio (character-level Jaccard)
        ratio = self._change_ratio(self._prev_text, current_text)
        self._prev_hash = current_hash
        self._prev_text = current_text

        if ratio >= self.change_threshold:
            return current_text
        return None

    @staticmethod
    def _change_ratio(old: str, new: str) -> float:
        """Sequence-based similarity ratio (0 to 1).
        
        Returns 1.0 - ratio to represent 'change distance'.
        """
        if not old and not new:
            return 0.0
        if not old or not new:
            return 1.0
            
        ratio = difflib.SequenceMatcher(None, old, new).ratio()
        return 1.0 - ratio
