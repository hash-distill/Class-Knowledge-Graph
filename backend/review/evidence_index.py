"""Persistent, source-fingerprinted Chinese lexical evidence search."""
import hashlib
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3

from .retrieval import evidence_catalog,normalize


def terms(value):
    normalized=normalize(value)
    chinese=re.findall(r'[\u3400-\u9fff]+',normalized)
    tokens=[part[i:i+2] for part in chinese for i in range(max(0,len(part)-1))]
    tokens.extend(re.findall(r'[a-z0-9_]+',normalized))
    return list(dict.fromkeys(tokens))[:256]


class EvidenceIndex:
    def __init__(self,path):
        self.path=Path(path).resolve();self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS evidence_revisions(lesson_id TEXT PRIMARY KEY,revision TEXT NOT NULL)')
            db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(lesson_id UNINDEXED,evidence_id UNINDEXED,document UNINDEXED,tokens)')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        try:
            with db:yield db
        finally:db.close()

    def update(self,bundle):
        rows=evidence_catalog(bundle);lesson_id=bundle['lesson']['id']
        encoded=json.dumps(rows,sort_keys=True,ensure_ascii=False,allow_nan=False)
        revision=hashlib.sha256(encoded.encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT revision FROM evidence_revisions WHERE lesson_id=?',(lesson_id,)).fetchone()
            if old and old[0]==revision:return False
            db.execute('DELETE FROM evidence_fts WHERE lesson_id=?',(lesson_id,))
            db.executemany('INSERT INTO evidence_fts(lesson_id,evidence_id,document,tokens) VALUES(?,?,?,?)',
                [(lesson_id,row['id'],json.dumps(row,ensure_ascii=False),' '.join(terms(row['text']))) for row in rows])
            db.execute('INSERT OR REPLACE INTO evidence_revisions VALUES(?,?)',(lesson_id,revision))
        return True

    def search(self,bundle,query,limit=10):
        self.update(bundle)
        query_terms=terms(query)
        if not query_terms:return []
        expression=' OR '.join('"'+token.replace('"','""')+'"' for token in query_terms)
        with self.connect() as db:
            rows=db.execute('SELECT document FROM evidence_fts WHERE evidence_fts MATCH ? AND lesson_id=? ORDER BY bm25(evidence_fts),evidence_id LIMIT ?',
                            (expression,bundle['lesson']['id'],max(1,min(int(limit),30)))).fetchall()
        return [json.loads(row[0]) for row in rows]
