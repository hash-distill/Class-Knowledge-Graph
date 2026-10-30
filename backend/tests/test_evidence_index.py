import importlib.util
from copy import deepcopy
import pytest


def index(tmp_path):
    assert importlib.util.find_spec('review.evidence_index') is not None, 'SQLite evidence index is missing.'
    from review.evidence_index import EvidenceIndex
    return EvidenceIndex(tmp_path/'evidence.sqlite3')


def bundle(lesson='one'):
    return {'lesson':{'id':lesson},'transcript':[
        {'id':'T0001','start_sec':0,'end_sec':5,'text':'氨基酸分子具有氨基和羧基'},
        {'id':'T0002','start_sec':5,'end_sec':10,'text':'三角形中的正弦定理'}],'slides':[]}


def test_chinese_search_is_persisted_and_scoped_to_lesson(tmp_path):
    db=index(tmp_path);one=bundle();two=bundle('two');two['transcript'][0]['text']='其他课堂内容'
    db.update(one);db.update(two)
    found=index(tmp_path).search(one,'氨基酸')
    assert found[0]['id']=='T0001' and found[0]['start_sec']==0
    assert db.search(two,'氨基酸')==[]


def test_reindex_removes_old_text_and_quotes_are_data(tmp_path):
    db=index(tmp_path);original=bundle();db.update(original)
    changed=deepcopy(original);changed['transcript'][0]['text']='经过教师核对的羧基'
    db.update(changed)
    assert db.search(changed,'氨基酸')==[]
    assert db.search(changed,'羧基')[0]['id']=='T0001'
    assert db.search(changed,'" OR * NOT "')==[]
