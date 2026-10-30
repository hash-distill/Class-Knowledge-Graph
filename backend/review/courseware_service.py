"""Transactional courseware overlays; original lesson evidence is never rewritten."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import threading
import uuid
from .common import atomic_json, read_json, sha256
from .provider import ReviewError
from .mutation import LessonMutationLock

EXTENSIONS = {'.pdf', '.pptx', '.ppt'}
MAX_UPLOAD = 64 * 1024 * 1024
OVERLAY_KEYS = ('courseware', 'courseware_evidence', 'transcript_revisions', 'revised_evidence_blocks')


def now():
    return datetime.now(timezone.utc).isoformat()


def source_fingerprint(bundle):
    import hashlib
    fields = {'source': bundle['lesson']['source_video']['sha256'],
              'transcript': bundle['transcript'], 'slides': bundle['slides']}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def image_fingerprint(directory, records):
    """Hash the actual pixels consumed by matching, not just stable path strings."""
    root = Path(directory).resolve()
    result = {}
    for row in records:
        relative = row.get('image_path')
        if not relative or relative in result:
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ReviewError('invalid_image_path', '匹配图像必须位于该课堂的产物目录。', 409)
        result[relative] = sha256(path) if path.is_file() else None
    return result


def overlay_bundle(base, document):
    from .retrieval import normalize
    from .courseware_terms import revise_transcript
    bundle = deepcopy(base)
    if document.get('lesson_fingerprint') != source_fingerprint(base):
        raise ReviewError('stale_courseware', '原始课堂证据已变化，请重新导入课件以重新对齐。', 409)
    bundle['courseware'] = deepcopy(document)
    derived = revise_transcript(base, document)
    for key in OVERLAY_KEYS[1:]:
        bundle[key] = derived.get(key, [])
    timed = bundle['courseware_evidence']
    # Attach only actual document occurrences to existing concepts; derived new
    # concepts are confined to terms which were used in traceable revisions.
    nodes = bundle['graph']['nodes']
    restricted_graph = bundle['graph'].get('provenance', {}).get('display_policy', {}).get('name') in (
        'model_candidates_with_observed_video_courseware_literal_support_v1',
        'model_candidates_with_proofread_or_observed_courseware_support_v2')
    for node in nodes:
        refs = [r['id'] for r in timed if normalize(node['label']) in normalize(r['text'])]
        node['properties']['source_refs'] = list(dict.fromkeys(node['properties']['source_refs'] + refs))
    labels = {normalize(n['label']) for n in nodes}
    for revision in bundle['transcript_revisions']:
        for edit in revision['edits']:
            term = edit['replacement']
            if normalize(term) in labels or restricted_graph:
                continue
            refs = [r['id'] for r in timed if r['page_id'] in edit['page_ids'] and normalize(term) in normalize(r['text'])]
            if refs:
                nodes.append({'id': f"KD{len(nodes)+1:04d}", 'label': term,
                              'properties': {'kind': 'courseware_term', 'source_refs': refs,
                                             'status': 'courseware_literal_unreviewed'}})
                labels.add(normalize(term))
    matched = sum(m['status'] in ('auto_matched', 'confirmed') for m in document['matches'])
    bundle['lesson']['provenance']['courseware'] = {
        'document_id': document['document_id'], 'filename': document['filename'],
        'sha256': document['sha256'], 'alignment': 'observed_screen_evidence',
        'transcript_policy': 'raw_preserved_with_separate_suggestions'}
    bundle['report']['sections'].append({
        'title': '独立课件与语音术语修订',
        'text': f"导入 {document['filename']}，{len(document['pages'])} 页；{matched}/{len(document['matches'])} 个实测屏幕段已匹配或人工确认。产生 {len(bundle['transcript_revisions'])} 条课件辅助转写建议，原始 ASR 不变。",
        'evidence_ids': [r['id'] for r in timed]})
    bundle['report']['limitations'].append('独立课件自动匹配与同音术语修订尚未人工核准；未匹配或有歧义的页面不生成课堂时间。修订建议不替代原始转写。')
    bundle['report'].setdefault('statistics', {}).update({
        'courseware_pages': len(document['pages']), 'courseware_matches': matched,
        'transcript_revisions': len(bundle['transcript_revisions']), 'concepts': len(nodes)})
    return bundle


def persist_bundle(directory, bundle):
    """Publish bundle last, restoring reports if a write fails before commit."""
    directory = Path(directory)
    report = bundle['report']
    markdown = [f"# {report['title']}", report['summary']]
    for section in report['sections']:
        markdown.extend([f"## {section['title']}", section['text'], '证据：' + ', '.join(section['evidence_ids'])])
    markdown.extend(['## 数据边界', *['- ' + s for s in report['limitations']]])
    previous = {directory / name: (directory / name).read_bytes() if (directory / name).exists() else None
                for name in ('report.json', 'report.md', 'bundle.json')}
    try:
        atomic_json(directory / 'report.json', report)
        temporary = directory / 'report.md.partial'
        temporary.write_text('\n\n'.join(markdown), encoding='utf-8')
        temporary.replace(directory / 'report.md')
        atomic_json(directory / 'bundle.json', bundle)
    except Exception:
        for path, content in previous.items():
            if content is None:
                path.unlink(missing_ok=True)
            elif not path.exists() or path.read_bytes() != content:
                rollback = path.with_name(path.name + '.rollback')
                rollback.write_bytes(content)
                rollback.replace(path)
        raise


def run_courseware_import(path, directory, base, expected_version, *, check=lambda: None,
                          progress=lambda value: None, publish=None):
    """Shared import implementation; heavy work runs in the caller's process."""
    from .courseware_documents import import_document
    from .courseware_alignment import align_document
    import hashlib
    path, directory = Path(path), Path(directory)
    check()
    document = deepcopy(import_document(path, directory, base['lesson']['id']))
    document['filename'] = path.name
    progress(35)
    check()
    cache_key = {'lesson': source_fingerprint(base), 'document': document['sha256'],
                 'pages': hashlib.sha256(json.dumps(document['pages'], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                 'images': image_fingerprint(directory, [*base['slides'], *document['pages']]),
                 'aligner': sha256(Path(__file__).with_name('courseware_alignment.py'))}
    cache_path = directory / 'courseware' / document['document_id'] / 'automatic-match-cache.json'
    cached = read_json(cache_path) if cache_path.exists() else {}
    reused = cached.get('fingerprint') == cache_key and isinstance(cached.get('matches'), list)
    document['matches'] = deepcopy(cached['matches']) if reused else align_document(document, base, directory)
    progress(70)
    check()
    document['alignment_cache_reused'] = reused
    document['lesson_fingerprint'] = source_fingerprint(base)
    document['imported_at'] = now()
    bundle = overlay_bundle(base, document)
    progress(90)

    def commit():
        if sha256(directory / 'bundle.json') != expected_version:
            raise ReviewError('stale_lesson', '导入期间课堂证据已变化，已放弃旧任务结果。请刷新后重新导入。', 409)
        if image_fingerprint(directory, [*base['slides'], *document['pages']]) != cache_key['images']:
            raise ReviewError('stale_lesson_images', '导入期间课堂或课件图像已变化，已放弃旧匹配结果。请重新导入。', 409)
        if not reused:
            atomic_json(cache_path, {'fingerprint': cache_key, 'matches': document['matches']})
        atomic_json(directory / 'bundle.base.json', base)
        atomic_json(directory / 'courseware' / document['document_id'] / 'alignment.json', document)
        persist_bundle(directory, bundle)
        return document['document_id']

    check()
    serialized=json.dumps(bundle,ensure_ascii=False,allow_nan=False,indent=2).replace('\n',os.linesep).encode('utf-8')
    publication={'directory':str(directory.resolve()),'expected_hash':hashlib.sha256(serialized).hexdigest(),
                 'document_id':document['document_id']}
    return publish(commit,publication) if publish is not None else commit()


class CoursewareService:
    def __init__(self, store, job_queue=None):
        self.store = store
        self.job_queue = job_queue
        self.lock = threading.Lock()
        self.jobs_lock = threading.Lock()
        self.jobs = {}

    def acquire(self, lesson_id=None):
        if lesson_id and self.job_queue is not None and self.job_queue.is_busy(lesson_id):
            raise ReviewError('courseware_busy', '这节课已有待处理课件，请稍后重试。', 409)
        if not self.lock.acquire(blocking=False):
            raise ReviewError('courseware_busy', '另一个课件操作正在处理，请稍后重试。', 409)
        try:
            return LessonMutationLock(self.store.entry(lesson_id)['directory']).acquire() if lesson_id else None
        except Exception:
            self.lock.release()
            raise

    def base(self, lesson_id):
        directory = Path(self.store.entry(lesson_id)['directory'])
        bundle = self.store.load(lesson_id)
        if bundle.get('courseware'):
            path = directory / 'bundle.base.json'
            if not path.is_file():
                raise ReviewError('missing_original_bundle', '缺少原始证据包，无法安全重新对齐。', 409)
            base=read_json(path)
            if bundle.get('activity_analysis'):
                from .activity import overlay_activity
                base=overlay_activity(base,bundle['activity_analysis'],directory)
            return directory, base
        return directory, deepcopy(bundle)

    def status(self, job_id):
        if self.job_queue is not None:
            return self.job_queue.status(job_id)
        with self.jobs_lock:
            result = self.jobs.get(job_id)
            if result is None:
                raise ReviewError('job_not_found', '任务不存在或服务已重启，请刷新课堂确认结果。', 404)
            return deepcopy(result)

    def submit(self, lesson_id, filename, body):
        """Called with bounded bytes already read; only one import may run at once."""
        if (not isinstance(filename, str) or not filename or len(filename) > 180 or
                any(c in filename for c in '/\\:\x00') or Path(filename).suffix.lower() not in EXTENSIONS):
            raise ReviewError('invalid_courseware', '请选择文件名有效的 PDF、PPTX 或 PPT 课件。', 400)
        if not body or len(body) > MAX_UPLOAD:
            raise ReviewError('courseware_too_large', '课件文件需大于零且不超过 64 MiB。', 413)
        lease = self.acquire(lesson_id)
        try:
            directory, base = self.base(lesson_id)
            expected_version = sha256(directory / 'bundle.json')
            job_id = uuid.uuid4().hex
            upload_dir = directory / 'courseware' / '_uploads' / job_id
            upload_dir.mkdir(parents=True)
            path = upload_dir / filename
            path.write_bytes(body)
            if self.job_queue is not None:
                payload = {'path': str(path.resolve()), 'directory': str(directory.resolve()),
                           'base': base, 'expected_version': expected_version,
                           'upload_sha256': sha256(path),
                           'source_video':str(Path(self.store.entry(lesson_id)['source_video']).resolve()),
                           'source_images': image_fingerprint(directory, base['slides']),
                           'source_media':{str(media.resolve()):sha256(media) if media.is_file() else None
                               for media in (Path(self.store.entry(lesson_id)['source_video']),directory/'audio.wav')}}
                job = self.job_queue.enqueue_courseware(lesson_id, filename, payload, job_id=job_id)
                if lease: lease.release()
                self.lock.release()
                return job
            job = {'job_id': job_id, 'lesson_id': lesson_id, 'status': 'running',
                   'filename': filename, 'started_at': now()}
            with self.jobs_lock:
                # Retain bounded status history, never evict an active operation.
                for key in list(self.jobs)[:-99]:
                    if self.jobs[key]['status'] != 'running': self.jobs.pop(key)
                self.jobs[job_id] = job
            threading.Thread(target=self._import, args=(job_id, path, directory, base, lease, expected_version), daemon=True).start()
            return deepcopy(job)
        except Exception:
            if lease: lease.release()
            self.lock.release()
            raise

    def _import(self, job_id, path, directory, base, lease, expected_version):
        completion = {}
        try:
            document_id = run_courseware_import(path, directory, base, expected_version)
            completion = dict(status='complete', completed_at=now(), document_id=document_id)
        except Exception as exc:
            completion = dict(status='failed', completed_at=now(), error={
                'code': getattr(exc, 'code', 'courseware_import_failed'),
                'message': str(exc)[:700] or '课件导入失败，原课堂数据保持不变。'})
        finally:
            # A completed status authorizes the next action. Release both mutation
            # locks before publishing it; status readers observe this atomically.
            with self.jobs_lock:
                if lease: lease.release()
                self.lock.release()
                self.jobs[job_id].update(completion)

    def confirm(self, lesson_id, payload):
        lease = self.acquire(lesson_id)
        try:
            current = self.store.load(lesson_id)
            document = deepcopy(current.get('courseware'))
            if not document or payload.get('document_id') != document['document_id']:
                raise ReviewError('stale_courseware', '课件已变化，请刷新后再确认。', 409)
            match = next((m for m in document['matches'] if m['id'] == payload.get('match_id')), None)
            if not match:
                raise ReviewError('invalid_match', '没有该观测时间段。', 400)
            page_id = payload.get('page_id')
            if page_id is not None and page_id not in {p['id'] for p in document['pages']}:
                raise ReviewError('invalid_page', '页面不属于当前课件。', 400)
            match.update(page_id=page_id, status='confirmed' if page_id else 'rejected',
                         method='manual_observed_interval', confirmed_at=now())
            directory, base = self.base(lesson_id)
            bundle = overlay_bundle(base, document)
            persist_bundle(directory, bundle)
            return bundle
        finally:
            if lease: lease.release()
            self.lock.release()

    def remove(self, lesson_id):
        lease = self.acquire(lesson_id)
        try:
            directory, base = self.base(lesson_id)
            persist_bundle(directory, base)
            return base
        finally:
            if lease: lease.release()
            self.lock.release()
