"""Source-grounded homophone suggestions; original ASR is immutable.

No wrong-to-right dictionary and no model training. A replacement must occur in
an auto-matched page and have exactly the same no-tone pronunciation. Numeric,
negation and formula-like spans are excluded. Suggestions are not confirmation.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import re
import unicodedata

METHOD = 'aligned_page_exact_homophone_suggestion_v1'
CHINESE = re.compile(r'^[\u3400-\u4dbf\u4e00-\u9fff]{2,10}$')
RUNS = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]+')
PROTECTED = set('不非无無未没沒莫勿别別否零〇一二三四五六七八九十百千万萬亿億两兩壹贰貳叁參肆伍陆陸柒捌玖拾')
MATH_SUFFIX = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]{1,6}(?:定理|公理|公式|法则|法則|函数|函數|方程|不等式|恒等式|恆等式|向量|矩阵|矩陣|导数|導數|积分|積分)')
CLAUSE_BOUNDARY = re.compile(r'[。！？!?；;\n，]')


def _formula_clause(text, start, end):
    """Protect the containing mathematical clause, including nested brackets.

    NFKC is used only for detection; the original characters are never rewritten.
    A formula elsewhere in the same clause is sufficient to abstain, while a
    separate natural-language clause can still receive a term suggestion.
    """
    left = max((match.end() for match in CLAUSE_BOUNDARY.finditer(text, 0, start)), default=0)
    next_boundary = CLAUSE_BOUNDARY.search(text, end)
    right = next_boundary.start() if next_boundary else len(text)
    clause = unicodedata.normalize('NFKC', text[left:right])
    if re.search(r'[=<>≤≥≠≈∑∫√×÷^_]', clause):
        return True
    # Operators may be separated from a Chinese term by arbitrary brackets and
    # spaces: (相量)-x, x + (相量), and their full-width variants all abstain.
    if re.search(r'[\w\)\]\}]\s*[-+*/−]\s*[\w\(\[\{]', clause):
        return True
    before = unicodedata.normalize('NFKC', text[left:start]).rstrip()
    after = unicodedata.normalize('NFKC', text[end:right]).lstrip()
    return bool(re.search(r'[A-Za-z0-9α-ωΑ-Ω]', before[-1:] + after[:1]))


@lru_cache(maxsize=20000)
def _pronunciation(text):
    from pypinyin import Style, pinyin
    # Phrase-level pronunciation handles documented polyphonic words; we do not
    # broaden to every possible heteronym, which would introduce false matches.
    return tuple(item[0] for item in pinyin(text, style=Style.NORMAL, heteronym=False, strict=True))


@lru_cache(maxsize=500)
def _terms(text):
    import jieba.posseg
    result = set()
    for line in text.splitlines():
        words = list(jieba.posseg.cut(line[:4000]))
        for index, item in enumerate(words):
            candidate = item.word
            if CHINESE.fullmatch(candidate) and item.flag.startswith(('n', 'v')):
                result.add(candidate)
            # Combine adjacent dictionary tokens only when they end in a math
            # suffix and stay short; every candidate remains a literal page span.
            for count in (2, 3):
                if index + count > len(words):
                    continue
                phrase = ''.join(word.word for word in words[index:index + count])
                if len(phrase) <= 10 and MATH_SUFFIX.fullmatch(phrase) and phrase in line:
                    result.add(phrase)
        for match in MATH_SUFFIX.finditer(line):
            # Long prose prefixes are avoided by requiring jieba noun boundaries.
            candidate = match.group()
            if len(candidate) <= 6 and any(candidate == word.word for word in words):
                result.add(candidate)
    return {term for term in result if CHINESE.fullmatch(term) and not (set(term) & PROTECTED)}


def _documents_and_matches(courseware):
    if isinstance(courseware, list):
        documents = courseware
        matches = [match for document in documents for match in document.get('matches', [])]
    elif 'documents' in courseware:
        documents = courseware['documents']
        matches = courseware.get('matches', [match for document in documents for match in document.get('matches', [])])
    else:
        documents = [courseware]
        matches = courseware.get('matches', [])
    return documents, matches


def revise_transcript(bundle, courseware_with_matches):
    """Produce reversible suggestions and additional evidence, never edit input."""
    documents, matches = _documents_and_matches(courseware_with_matches)
    pages = {page['id']: page for document in documents for page in document.get('pages', [])}
    observed = {row['id']: row for row in bundle.get('slides', [])}
    active = []
    for match in matches:
        slide = observed.get(match.get('slide_id'))
        if match.get('status') not in {'auto_matched', 'confirmed'} or match.get('page_id') not in pages or not slide:
            continue
        # Metadata supplied by an API must still agree with an observed S span.
        if match.get('start_sec') != slide.get('start_sec') or match.get('end_sec') != slide.get('end_sec'):
            continue
        active.append(match)
    evidence = []
    for match in active:
        page = pages[match['page_id']]
        evidence.append({'id': 'C' + match['id'][1:], 'kind': 'courseware',
                         'text': page.get('text', ''), 'page_id': page['id'],
                         'document_id': page.get('document_id', match.get('document_id')),
                         'page_number': page['page_number'], 'image_url': page.get('image_url'),
                         'start_sec': match['start_sec'], 'end_sec': match['end_sec'],
                         'source_refs': [match['slide_id'], page['id']], 'match_ids': [match['id']],
                         'match_id': match['id'], 'method': match['method'],
                         'status': match['status'], 'score': match['score']})
    terms = {page_id: _terms(page.get('text', '')) for page_id, page in pages.items()}
    revisions, blocks = [], []
    for segment in bundle.get('transcript', []):
        aligned = [match for match in active if match['end_sec'] > segment['start_sec'] and match['start_sec'] < segment['end_sec']]
        if not aligned:
            continue
        phonetic = defaultdict(lambda: defaultdict(lambda: {'page_ids': set(), 'match_ids': set()}))
        for match in aligned:
            for term in terms[match['page_id']]:
                slot = phonetic[_pronunciation(term)][term]
                slot['page_ids'].add(match['page_id'])
                slot['match_ids'].add(match['id'])
        original = segment['text']
        proposals = []
        for run in RUNS.finditer(original):
            for start in range(run.start(), run.end()):
                for length in sorted({len(key) for key in phonetic}, reverse=True):
                    end = start + length
                    if end > run.end():
                        continue
                    source = original[start:end]
                    if set(source) & PROTECTED:
                        continue
                    if _formula_clause(original, start, end):
                        continue
                    alternatives = phonetic.get(_pronunciation(source), {})
                    if len(alternatives) != 1:
                        continue
                    replacement, refs = next(iter(alternatives.items()))
                    if replacement == source or len(source) != len(replacement):
                        continue
                    # Do not normalize script-only differences in otherwise
                    # correctly recognized words (e.g. 定義 -> 定义).
                    try:
                        from .retrieval import normalize
                        if normalize(source) == normalize(replacement):
                            continue
                    except ImportError:
                        pass
                    # At most two differing characters keeps suggestions local.
                    if sum(a != b for a, b in zip(source, replacement)) > 2:
                        continue
                    proposals.append((start, end, source, replacement, refs))
        selected = []
        for item in sorted(proposals, key=lambda row: (-(row[1] - row[0]), row[0])):
            if all(item[1] <= old[0] or item[0] >= old[1] for old in selected):
                selected.append(item)
        if not selected:
            continue
        selected.sort(key=lambda row: row[0])
        corrected, cursor, edits = [], 0, []
        for start, end, source, replacement, refs in selected:
            corrected.extend([original[cursor:start], replacement])
            cursor = end
            edits.append({'original': source, 'replacement': replacement,
                          'start_char': start, 'end_char': end,
                          'page_ids': sorted(refs['page_ids']), 'match_ids': sorted(refs['match_ids']), 'method': METHOD})
        corrected.append(original[cursor:])
        corrected_text = ''.join(corrected)
        fingerprint = hashlib.sha256((segment['id'] + '\0' + original + '\0' + corrected_text + '\0' +
                                     ','.join(sorted(match['id'] for match in aligned))).encode()).hexdigest()[:12]
        revision_id = 'R' + segment['id'] + '_' + fingerprint
        revision = {'id': revision_id, 'transcript_id': segment['id'], 'original_text': original,
                    'corrected_text': corrected_text, 'start_sec': segment['start_sec'],
                    'end_sec': segment['end_sec'], 'edits': edits, 'status': 'suggested', 'method': METHOD}
        revisions.append(revision)
    revision_lookup = {revision['transcript_id']: revision for revision in revisions}
    transcript_lookup = {segment['id']: segment for segment in bundle.get('transcript', [])}
    base_blocks = bundle.get('evidence_blocks') or [
        {**segment, 'kind': 'transcript', 'source_refs': [segment['id']]}
        for segment in bundle.get('transcript', [])]
    for block in base_blocks:
        refs = block.get('source_refs', [block['id']])
        changed = [revision_lookup[ref] for ref in refs if ref in revision_lookup]
        if not changed:
            blocks.append(deepcopy(block))
            continue
        # Rebuild the entire existing speech block so a suggested term never
        # loses the original neighboring explanation used in retrieval.
        if not all(ref in transcript_lookup for ref in refs):
            blocks.append(deepcopy(block))
            continue
        text = ' '.join(revision_lookup[ref]['corrected_text'] if ref in revision_lookup
                        else transcript_lookup[ref]['text'] for ref in refs)
        fingerprint = hashlib.sha256((block['id'] + '\0' + text).encode()).hexdigest()[:12]
        edits = [edit for revision in changed for edit in revision['edits']]
        blocks.append({**deepcopy(block), 'id': 'BR' + block['id'] + '_' + fingerprint,
                       'kind': 'transcript_revision', 'text': text,
                       'source_refs': list(refs), 'revision_ids': [revision['id'] for revision in changed],
                       'page_ids': sorted({page for edit in edits for page in edit['page_ids']}),
                       'match_ids': sorted({match for edit in edits for match in edit['match_ids']}),
                       'status': 'suggested', 'method': METHOD})
    return {'transcript_revisions': revisions, 'revised_evidence_blocks': blocks, 'courseware_evidence': evidence}
