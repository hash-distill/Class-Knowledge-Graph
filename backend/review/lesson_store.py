"""Read-only access to original lesson bundles and registered local media."""
from pathlib import Path
from .common import ROOT, read_json, valid_id
from .provider import ReviewError


class LessonStore:
    def __init__(self,registry_path=None):
        self.registry_path=Path(registry_path or ROOT/'configs/lessons.json').expanduser().resolve()
        self.cache={}

    def registry(self):
        try:
            result=read_json(self.registry_path) if self.registry_path.exists() else {}
            if not isinstance(result,dict):raise ValueError('Registry must be an object')
            return result
        except (OSError,ValueError) as exc:
            raise ReviewError('invalid_lesson_registry','Invalid lesson registry. Check the JSON object and file access permissions.',409) from exc

    def entry(self,lesson_id):
        try:valid_id(lesson_id)
        except ValueError as exc:raise ReviewError('invalid_lesson','Invalid lesson ID.',400) from exc
        registry=self.registry()
        if lesson_id not in registry:raise ReviewError('lesson_not_found','The lesson does not exist or has not finished processing.',404)
        entry=registry[lesson_id]
        if not isinstance(entry,dict) or not isinstance(entry.get('directory'),str) or not entry['directory'].strip():
            raise ReviewError('invalid_lesson_registry','The lesson registry must include a valid directory path.',409)
        if 'source_video' in entry and (not isinstance(entry['source_video'],str) or not entry['source_video'].strip()):
            raise ReviewError('invalid_lesson_registry','The source_video path in the lesson registry is invalid.',409)
        resolved=dict(entry)
        for key in ('directory','source_video'):
            if key not in entry:continue
            path=Path(entry[key]).expanduser()
            resolved[key]=str((path if path.is_absolute() else self.registry_path.parent/path).resolve())
        return resolved

    def load(self,lesson_id):
        path=Path(self.entry(lesson_id)['directory'])/'bundle.json'
        if not path.is_file():raise ReviewError('lesson_not_ready','Lesson analysis is not complete.',409)
        dependencies = [path, *[path.parent/name for name in ('model.upgrade.json', 'asr.json',
                        'asr.baseline.json', 'audio.wav', 'transcript.proofreading.json')]]
        stamp=tuple((p.stat().st_mtime_ns,p.stat().st_size) if p.is_file() else None for p in dependencies)
        cached=self.cache.get(lesson_id)
        if cached and cached[0]==stamp:return cached[1]
        bundle=read_json(path)
        if bundle.get('data_source')!='lesson_review' or bundle.get('schema_version')!='3.0' or bundle['lesson'].get('status')!='complete':
            raise ReviewError('invalid_lesson_bundle','The lesson artifacts failed the integrity check.',409)
        if bundle['lesson']['id']!=lesson_id:raise ReviewError('lesson_id_mismatch','The lesson source IDs do not match.',409)
        if (path.parent/'model.upgrade.json').is_file() or 'model_upgrade_experiment' in bundle:
            from .model_upgrade import overlay_model_upgrade
            bundle=overlay_model_upgrade(bundle,path.parent)
        self.cache[lesson_id]=(stamp,bundle)
        return bundle

    def list_lessons(self):
        results=[]
        for lesson_id, entry in self.registry().items():
            if not isinstance(entry,dict):continue
            if entry.get('demo_visible') is False:continue
            try:
                lesson=self.load(lesson_id)['lesson']
                results.append({key:lesson[key] for key in ['id','title','duration_sec','status']})
            except (ReviewError,OSError,ValueError,KeyError):continue
        return results

    def media(self,lesson_id,relative):
        entry=self.entry(lesson_id)
        if relative=='source.mp4':
            if not entry.get('source_video'):raise ReviewError('media_not_found','No original video path is registered for this lesson.',404)
            return Path(entry['source_video'])
        root=Path(entry['directory']).resolve()
        if '\\' in relative or ':' in relative:raise ReviewError('invalid_path','Invalid media path.',400)
        candidate=(root/relative).resolve()
        if not candidate.is_relative_to(root) or candidate.suffix.lower() not in ('.jpg','.jpeg','.png','.webp','.mp4','.wav','.mp3','.pdf','.pptx','.ppt'):
            raise ReviewError('invalid_path','This media path is not allowed.',403)
        if not candidate.is_file():raise ReviewError('media_not_found','The media file does not exist.',404)
        return candidate
