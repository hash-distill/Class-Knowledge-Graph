"""Source-bound human annotations, kept apart from immutable model observations."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile
from urllib.parse import unquote
import zipfile

from PIL import Image
from .provider import ReviewError

LABELS = ('write', 'read', 'lookup', 'turn_head', 'raise_hand', 'stand', 'discuss', 'unknown')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def invalid(message):
    raise ReviewError('invalid_vision_review', message, 400)


def frame_context(store, bundle, frame_id):
    lesson_id = bundle['lesson']['id']
    analysis = bundle.get('activity_analysis') or {}
    frame = next((row for row in analysis.get('frames', []) if row['id'] == frame_id), None)
    if frame is None:
        raise ReviewError('vision_frame_not_found', '当前课堂没有这一采样帧。', 404)
    source = bundle['lesson'].get('source_video', {}).get('sha256', '')
    if not re.fullmatch('[0-9a-f]{64}', source) or analysis.get('source_sha256', source) != source:
        raise ReviewError('vision_source_missing', '当前帧缺少一致的录像来源摘要，暂不能复核。', 409)
    prefix = f'/api/review/media/{lesson_id}/'
    url = frame.get('image_url', '')
    if not url.startswith(prefix) or '?' in url or '#' in url:
        raise ReviewError('vision_image_invalid', '复核图片必须来自本课注册的原始帧。', 409)
    path = store.media(lesson_id, unquote(url[len(prefix):]))
    if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp') or path.stat().st_size > 16*1024*1024:
        raise ReviewError('vision_image_invalid', '复核图片格式或大小不受支持。', 409)
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
    if width * height > 24_000_000 or width != frame.get('width', width) or height != frame.get('height', height):
        raise ReviewError('vision_image_changed', '帧图片尺寸与检测坐标不一致，请重新处理。', 409)
    image_sha = hashlib.sha256(data).hexdigest()
    return {'frame': frame, 'image': data, 'image_sha256': image_sha, 'width': width,
            'height': height, 'source_video_sha256': source,
            'basis_fingerprint': digest({'source': source, 'frame': frame, 'image': image_sha})}


def validate_annotations(rows, context):
    if not isinstance(rows, list) or len(rows) > 200:
        invalid('每帧最多保存 200 个已复核对象。')
    model = {p['track_id'] for p in context['frame'].get('participants', [])}
    seen = set(); cleaned = []
    for row in rows:
        if not isinstance(row, dict): invalid('复核对象格式无效。')
        identifier = row.get('id')
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', identifier) or identifier in seen:
            invalid('复核对象编号无效或重复。')
        seen.add(identifier)
        origin = row.get('origin')
        if origin not in ('model', 'manual') or (origin == 'model' and identifier not in model) or (
                origin == 'manual' and (not identifier.startswith('manual-') or identifier in model)):
            invalid('复核对象来源与当前检测结果不一致。')
        label, role, status = row.get('label'), row.get('role'), row.get('status')
        if label not in LABELS or role not in ('unknown', 'student', 'teacher') or status not in ('confirmed', 'ignored'):
            invalid('请选择有效的状态、对象角色和复核结论。')
        reason = row.get('reason', '')
        if not isinstance(reason, str) or len(reason) > 500 or status == 'ignored' and not reason.strip():
            invalid('排除误检需填写原因，说明最多 500 字。')
        box = row.get('bbox')
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= context['width'] or not 0 <= box[1] < box[3] <= context['height']
                or box[2]-box[0] < 2 or box[3]-box[1] < 2):
            invalid('边框需位于原图内，宽高至少为 2 像素。')
        cleaned.append({'id': identifier, 'origin': origin, 'label': label, 'role': role,
                        'status': status, 'bbox': box, 'reason': reason.strip()})
    return cleaned


class VisionReviewStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS vision_review_versions (
                lesson_id TEXT NOT NULL, frame_id TEXT NOT NULL, version INTEGER NOT NULL,
                basis TEXT NOT NULL, annotations TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(lesson_id,frame_id,version))''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10); db.row_factory = sqlite3.Row
        try:
            with db: yield db
        finally: db.close()

    def latest(self, db, lesson_id, frame_id):
        return db.execute('SELECT * FROM vision_review_versions WHERE lesson_id=? AND frame_id=? ORDER BY version DESC LIMIT 1',
                          (lesson_id, frame_id)).fetchone()

    def view(self, row, lesson_id, context):
        changed = bool(row and row['basis'] != context['basis_fingerprint'])
        return {'lesson_id': lesson_id, 'frame_id': context['frame']['id'], 'frame': context['frame'],
                'version': row['version'] if row else 0, 'basis_fingerprint': context['basis_fingerprint'],
                'source_changed': changed, 'updated_at': row['updated_at'] if row else None,
                'annotations': json.loads(row['annotations']) if row and not changed else [],
                'width': context['width'], 'height': context['height'], 'image_sha256': context['image_sha256']}

    def get(self, lesson_id, context):
        with self.connect() as db:
            return self.view(self.latest(db, lesson_id, context['frame']['id']), lesson_id, context)

    def save(self, lesson_id, context, version, basis, rows, restore_version=None):
        if basis != context['basis_fingerprint']:
            raise ReviewError('vision_source_changed', '帧证据已变化，请重新加载并复核。', 409)
        frame_id = context['frame']['id']
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = self.latest(db, lesson_id, frame_id)
            current = previous['version'] if previous else 0
            if type(version) is not int or current != version:
                raise ReviewError('vision_review_changed', '该帧已被另一窗口更新。请保留草稿并重新加载后核对。', 409)
            if restore_version is not None:
                old = db.execute('SELECT * FROM vision_review_versions WHERE lesson_id=? AND frame_id=? AND version=?',
                                 (lesson_id, frame_id, restore_version)).fetchone()
                if not old or old['basis'] != basis:
                    raise ReviewError('vision_history_unavailable', '该历史版本不属于当前帧证据，不能恢复。', 409)
                rows = json.loads(old['annotations'])
            rows = validate_annotations(rows, context)
            db.execute('INSERT INTO vision_review_versions VALUES (?,?,?,?,?,?)',
                       (lesson_id, frame_id, current+1, basis, json.dumps(rows, ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat()))
            return self.view(self.latest(db, lesson_id, frame_id), lesson_id, context)

    def history(self, lesson_id, frame_id):
        with self.connect() as db:
            return [{'version': r['version'], 'basis_fingerprint': r['basis'], 'annotations': json.loads(r['annotations']),
                     'updated_at': r['updated_at']} for r in db.execute(
                'SELECT * FROM vision_review_versions WHERE lesson_id=? AND frame_id=? ORDER BY version DESC LIMIT 100',
                (lesson_id, frame_id))]

    def export(self, store, bundle):
        lesson_id = bundle['lesson']['id']
        with self.connect() as db:
            rows = db.execute('''SELECT r.* FROM vision_review_versions r JOIN
                (SELECT frame_id,MAX(version) version FROM vision_review_versions WHERE lesson_id=? GROUP BY frame_id) latest
                ON r.frame_id=latest.frame_id AND r.version=latest.version WHERE r.lesson_id=? ORDER BY r.frame_id''',
                (lesson_id, lesson_id)).fetchall()
        if not rows or not any(json.loads(r['annotations']) for r in rows):
            raise ReviewError('vision_review_empty', '尚无已保存的人工复核对象。请先逐对象确认并保存。', 409)
        archive = tempfile.SpooledTemporaryFile(max_size=8*1024*1024)
        manifest = {'schema_version': '1.0', 'lesson_id': lesson_id,
                    'source_video_sha256': bundle['lesson']['source_video']['sha256'],
                    'partial_annotations': True, 'human_annotation_not_accuracy_evaluation': True,
                    'classes': list(LABELS[:-1]), 'classification_sample_count': 0, 'frames': [], 'archived_missing_frames': []}
        available_frames = {frame['id'] for frame in bundle.get('activity_analysis', {}).get('frames', [])}
        input_bytes = 0
        try:
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
                for index, row in enumerate(rows):
                    annotations = json.loads(row['annotations'])
                    if not annotations: continue
                    if row['frame_id'] not in available_frames:
                        manifest['archived_missing_frames'].append({'frame_id': row['frame_id'], 'review_version': row['version'],
                                                                  'reason': 'frame_removed_from_current_analysis'})
                        continue
                    context = frame_context(store, bundle, row['frame_id'])
                    if context['basis_fingerprint'] != row['basis']:
                        raise ReviewError('vision_source_changed', '有复核记录对应的图片或模型结果已变化，请重新复核后导出。', 409)
                    input_bytes += len(context['image'])
                    if input_bytes > 128*1024*1024 or manifest['classification_sample_count'] > 2000:
                        raise ReviewError('vision_export_limit', '本次样本过多，请分课堂导出。', 413)
                    filename = f'frames/{index:04d}.jpg'
                    with Image.open(io.BytesIO(context['image'])) as original:
                        image = original.convert('RGB')
                        buffer = io.BytesIO(); image.save(buffer, format='JPEG', quality=95)
                        output.writestr(filename, buffer.getvalue())
                        for number, item in enumerate(annotations):
                            if item['status'] != 'confirmed' or item['role'] != 'student' or item['label'] == 'unknown': continue
                            crop_path = f'crops/{item["label"]}/{index:04d}-{number:03d}.jpg'
                            crop = image.crop(tuple(item['bbox']))
                            data = io.BytesIO(); crop.save(data, format='JPEG', quality=95)
                            output.writestr(crop_path, data.getvalue())
                            item['classification_crop'] = crop_path
                            manifest['classification_sample_count'] += 1
                    manifest['frames'].append({'frame_id': row['frame_id'], 'timestamp_sec': context['frame']['timestamp_sec'],
                        'scene_id': context['frame'].get('scene_id'), 'image': filename,
                        'original_image_sha256': context['image_sha256'], 'basis_fingerprint': row['basis'],
                        'width': context['width'], 'height': context['height'], 'review_version': row['version'],
                        'reviewed_at': row['updated_at'], 'annotations': annotations})
                if not manifest['frames']:
                    raise ReviewError('vision_review_empty', '原复核帧已不在当前分析中，请复核当前画面后再导出。', 409)
                output.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
                output.writestr('README.txt', '人工视觉复核样本\n\n仅 crops 内显式确认学生且非未知状态的对象可作为七类状态分类标注。\n'
                    'manifest 保留未知、教师、排除对象与来源。未复核对象不是负样本。整帧为部分标注，不可直接作为完整人物检测训练标签。\n'
                    '图片是原始帧的 JPEG 副本，original_image_sha256 指原始文件；裁剪使用标注框。\n'
                    '没有自动划分训练和测试集；应按课堂录像分组，防止相邻帧泄漏。人工标注不等于独立准确率验收。\n')
                if manifest['archived_missing_frames']:
                    output.writestr('ARCHIVED_FRAMES.txt', '以下旧帧已不在当前分析中，历史保留但不导出为当前训练样本。\n' +
                                    json.dumps(manifest['archived_missing_frames'], ensure_ascii=False, indent=2))
            archive.seek(0)
            return archive
        except BaseException:
            archive.close(); raise
