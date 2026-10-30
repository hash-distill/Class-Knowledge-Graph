"""Versioned teacher records; machine observations remain immutable."""
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import uuid

from .common import valid_id
from .provider import ReviewError


def now():
    return datetime.now(timezone.utc).isoformat()


def source_revision(bundle):
    source={'video':bundle['lesson'].get('source_video',{}).get('sha256'),
            'transcript':bundle.get('transcript',[])}
    return hashlib.sha256(json.dumps(source,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def text(value,label,maximum=4000,*,required=True):
    if not isinstance(value,str) or len(value)>maximum or required and not value.strip():
        raise ReviewError('invalid_workspace',f'{label}需为有效文字，最多 {maximum} 字。',400)
    return value.strip()


def blank():
    return {'goals':'','correction_history':[],'actions':[],'assessments':[],'attempts':[]}


class WorkspaceStore:
    def __init__(self,path):
        self.path=Path(path).resolve()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS workspaces (
                    lesson_id TEXT PRIMARY KEY, source_revision TEXT NOT NULL,
                    version INTEGER NOT NULL, document TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS workspace_history (
                    id INTEGER PRIMARY KEY, lesson_id TEXT NOT NULL, source_revision TEXT NOT NULL,
                    archived_at TEXT NOT NULL, snapshot TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS workspace_history_lesson ON workspace_history(lesson_id,id);
            ''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.row_factory=sqlite3.Row
        try:
            with db:yield db
        finally:db.close()

    def _row(self,db,bundle):
        lesson_id=valid_id(bundle['lesson']['id'])
        db.execute('INSERT OR IGNORE INTO workspaces VALUES (?,?,?,?,?)',
                   (lesson_id,source_revision(bundle),0,json.dumps(blank()),now()))
        return db.execute('SELECT * FROM workspaces WHERE lesson_id=?',(lesson_id,)).fetchone()

    def _view(self,row,bundle,db):
        document=json.loads(row['document'])
        return {**document,'lesson_id':row['lesson_id'],'version':row['version'],
                'source_revision':source_revision(bundle),'stored_source_revision':row['source_revision'],
                'source_changed':row['source_revision']!=source_revision(bundle),
                'updated_at':row['updated_at'],
                'corrections':[r for r in document['correction_history'] if not r.get('revoked_at')],
                'history':[dict(r) for r in db.execute(
                    'SELECT source_revision,archived_at FROM workspace_history WHERE lesson_id=? ORDER BY id DESC',
                    (row['lesson_id'],))]}

    def get(self,bundle):
        with self.connect() as db:return self._view(self._row(db,bundle),bundle,db)

    def _update(self,bundle,expected_version,edit):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=self._row(db,bundle)
            if row['source_revision']!=source_revision(bundle):
                raise ReviewError('workspace_source_changed','课堂原始证据已变化，请先确认新版本。',409)
            if type(expected_version) is not int or row['version']!=expected_version:
                raise ReviewError('workspace_changed','教学记录已更新，请重新加载后保存。',409)
            document=json.loads(row['document'])
            edit(document)
            db.execute('UPDATE workspaces SET version=version+1,document=?,updated_at=? WHERE lesson_id=?',
                       (json.dumps(document,ensure_ascii=False,allow_nan=False),now(),row['lesson_id']))
            return self._view(self._row(db,bundle),bundle,db)

    def set_goals(self,bundle,value,expected_version):
        value=text(value,'教学目标',required=False)
        return self._update(bundle,expected_version,lambda d:d.update(goals=value))

    def correct(self,bundle,transcript_id,replacement,reason,expected_version):
        original=next((r for r in bundle.get('transcript',[]) if r['id']==transcript_id),None)
        if original is None:raise ReviewError('invalid_transcript','转写片段不存在。',400)
        replacement=text(replacement,'核对稿');reason=text(reason,'核对说明',1000,required=False)
        def edit(document):
            for row in document['correction_history']:
                if row['transcript_id']==transcript_id and not row.get('revoked_at'):row['revoked_at']=now()
            document['correction_history'].append({'id':uuid.uuid4().hex,'transcript_id':transcript_id,
                'original_text':original['text'],'replacement':replacement,'reason':reason,
                'created_at':now(),'revoked_at':None,'human_verified':True})
        return self._update(bundle,expected_version,edit)

    def revert(self,bundle,transcript_id,expected_version):
        def edit(document):
            active=[r for r in document['correction_history'] if r['transcript_id']==transcript_id and not r.get('revoked_at')]
            if not active:raise ReviewError('correction_not_found','该片段没有生效的教师核对稿。',404)
            for row in active:row['revoked_at']=now()
        return self._update(bundle,expected_version,edit)

    def _references(self,bundle,identifiers,*,required=False):
        if (not isinstance(identifiers,list) or len(identifiers)>20 or
                any(not isinstance(i,str) for i in identifiers) or required and not identifiers):
            raise ReviewError('invalid_evidence','请选择有效的课堂证据。',400)
        effective=self.overlay(bundle)
        lookup={r['id']:r for key in ('transcript','slides','evidence_blocks','proofread_evidence_blocks',
                'revised_evidence_blocks','courseware_evidence','teacher_transcript','teacher_evidence_blocks')
                for r in effective.get(key,[])}
        lookup.update({r['id']:r for r in bundle.get('activity_analysis',{}).get('evidence',[])})
        if any(i not in lookup for i in identifiers):
            raise ReviewError('invalid_evidence','引用不属于当前课堂证据。',400)
        return [{key:deepcopy(row[key]) for key in ('id','text','start_sec','end_sec','kind','source_refs',
                 'quality','uncertainties','method','workspace_version','proofreading_ids') if key in row}
                for i in dict.fromkeys(identifiers) for row in [lookup[i]]]

    def add_action(self,bundle,title,evidence_ids,expected_version):
        title=text(title,'改进任务',500);refs=self._references(bundle,evidence_ids)
        def edit(document):
            document['actions'].append({'id':uuid.uuid4().hex,'title':title,'status':'planned','outcome':'',
                'evidence_ids':list(dict.fromkeys(evidence_ids)),'evidence_snapshot':refs,'created_at':now(),'updated_at':now()})
        return self._update(bundle,expected_version,edit)

    def update_action(self,bundle,action_id,status,outcome,expected_version):
        if status not in ('planned','done'):raise ReviewError('invalid_workspace','任务状态无效。',400)
        outcome=text(outcome,'实际执行结果',required=status=='done')
        def edit(document):
            row=next((r for r in document['actions'] if r['id']==action_id),None)
            if row is None:raise ReviewError('action_not_found','改进任务不存在。',404)
            row.update(status=status,outcome=outcome,updated_at=now())
        return self._update(bundle,expected_version,edit)

    def add_assessment(self,bundle,question,reference_answer,evidence_ids,expected_version):
        question=text(question,'练习题');reference_answer=text(reference_answer,'参考答案')
        refs=self._references(bundle,evidence_ids,required=True)
        def edit(document):
            document['assessments'].append({'id':uuid.uuid4().hex,'question':question,
                'reference_answer':reference_answer,'evidence_ids':list(dict.fromkeys(evidence_ids)),
                'evidence_snapshot':refs,'teacher_confirmed':True,'created_at':now()})
        return self._update(bundle,expected_version,edit)

    def add_attempt(self,bundle,assessment_id,answer,feedback,score,expected_version):
        answer=text(answer,'实际作答');feedback=text(feedback,'教师反馈',required=False)
        if score is not None and (isinstance(score,bool) or not isinstance(score,(int,float)) or
                                  not math.isfinite(score) or not 0<=score<=1):
            raise ReviewError('invalid_workspace','得分需为 0–1 的数字，未评分时留空。',400)
        def edit(document):
            if not any(r['id']==assessment_id for r in document['assessments']):
                raise ReviewError('assessment_not_found','练习题不存在。',404)
            document['attempts'].append({'id':uuid.uuid4().hex,'assessment_id':assessment_id,
                'answer':answer,'feedback':feedback,'score':score,'created_at':now(),'source':'teacher_recorded'})
        return self._update(bundle,expected_version,edit)

    def history(self,lesson_id):
        with self.connect() as db:
            return [{**dict(row),'snapshot':json.loads(row['snapshot'])} for row in db.execute(
                'SELECT * FROM workspace_history WHERE lesson_id=? ORDER BY id DESC',(lesson_id,))]

    def reconcile(self,bundle,expected_version):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');row=self._row(db,bundle)
            if type(expected_version) is not int or row['version']!=expected_version:
                raise ReviewError('workspace_changed','教学记录已更新，请重新加载后保存。',409)
            if row['source_revision']==source_revision(bundle):return self._view(row,bundle,db)
            snapshot=self._view(row,bundle,db)
            db.execute('INSERT INTO workspace_history(lesson_id,source_revision,archived_at,snapshot) VALUES(?,?,?,?)',
                       (row['lesson_id'],row['source_revision'],now(),json.dumps(snapshot,ensure_ascii=False)))
            db.execute('UPDATE workspaces SET source_revision=?,version=version+1,document=?,updated_at=? WHERE lesson_id=?',
                       (source_revision(bundle),json.dumps(blank()),now(),row['lesson_id']))
            return self._view(self._row(db,bundle),bundle,db)

    def overlay(self,bundle):
        # Derived views may be passed back by embedding callers. Rebuild them
        # from current records so revoked or previous-source edits never survive.
        bundle={k:v for k,v in bundle.items() if k not in ('teacher_transcript','teacher_evidence_blocks','teacher_review')}
        workspace=self.get(bundle)
        if workspace['source_changed'] or not workspace['corrections']:return bundle
        from .blocks import speech_blocks
        result=deepcopy(bundle)
        changes={r['transcript_id']:r for r in workspace['corrections']}
        machine={r['transcript_id']:r for r in bundle.get('transcript_proofreading',{}).get('segments',[])}
        rows=[]
        for i,raw in enumerate(bundle['transcript']):
            correction=changes.get(raw['id']);baseline=machine.get(raw['id'],raw)
            rows.append({**deepcopy(raw),'id':f'HT{i+1:04d}','transcript_id':raw['id'],
                'text':correction['replacement'] if correction else baseline['text'],
                'source_refs':[raw['id']],'source':'teacher_revision' if correction else 'machine_context',
                'human_verified':bool(correction),'correction_id':correction['id'] if correction else None,
                'uncertainties':[] if correction else deepcopy(baseline.get('uncertainties',[])),
                'uncertain_spans':[] if correction else deepcopy(baseline.get('uncertain_spans',[])),
                'proofreading_ids':[baseline['id']] if baseline is not raw else []})
        lookup={r['id']:r for r in rows};blocks=[]
        for i,block in enumerate(speech_blocks(rows)):
            members=[lookup[r] for r in block['source_refs']]
            blocks.append({**block,'id':f'HB{i+1:04d}','kind':'transcript_teacher',
                'source_refs':[r['transcript_id'] for r in members],'method':'teacher_revision_with_machine_context',
                'quality':{'human_verified':all(r['human_verified'] for r in members)},
                'uncertainties':list(dict.fromkeys(v for r in members for v in r['uncertainties'])),
                'proofreading_ids':list(dict.fromkeys(v for r in members for v in r['proofreading_ids'])),
                'workspace_version':workspace['version']})
        result.update(teacher_transcript=rows,teacher_evidence_blocks=blocks,
                      teacher_review={'version':workspace['version'],'source_revision':workspace['source_revision'],
                                      'corrected_segment_count':len(changes)})
        return result
