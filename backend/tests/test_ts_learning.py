"""Data leakage and learned-head checks; synthetic arrays are code tests only."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.ts_context import parse_targets, recording_group, assign_group_splits, train_head, load_head, predict_features, classification_metrics, validate_cache, source_role, TARGET_MAPPING_VERSION


def test_model_integrity_and_source_partition_are_explicit(tmp_path):
    import hashlib, json
    np.savez_compressed(tmp_path/'model.npz', mean=np.zeros(2))
    model_hash=hashlib.sha256((tmp_path/'model.npz').read_bytes()).hexdigest()
    (tmp_path/'model.json').write_text(json.dumps({'weights_sha256':model_hash}))
    assert load_head(tmp_path)['mean'].shape == (2,)
    (tmp_path/'model.json').write_text(json.dumps({'weights_sha256':'0'*64}))
    with pytest.raises(ValueError, match='checksum'):
        load_head(tmp_path)
    video=tmp_path/'sample.mp4';video.write_bytes(b'real-source-identity-test')
    digest=hashlib.sha256(video.read_bytes()).hexdigest()
    manifest={'records':[{'video_sha256':digest,'split':'train'}]}
    assert source_role(video,manifest) == 'train'
    assert source_role(video,{'records':[]}) == 'external'


def test_targets_only_read_label_header_not_descriptive_answers():
    raw = "1 Raise a question Students' direct answers\n2\n3 Group discussion and Writing appear only in the answer description."
    result = parse_targets(raw)
    assert result['guiding_questions'] == 1
    assert result['oral_expression'] == 1
    assert result['cooperation_exchange'] == 0
    assert result['practical_operation'] == 0
    with pytest.raises(ValueError, match='label'):
        parse_targets('1 Unknown target label\n2')


def test_source_recording_groups_never_cross_splits():
    rows = [{'id': f'{g}_part{i}', 'group': g} for g in ['Art_001','Chinese_001','Math_006','Physics_001'] for i in range(2)]
    result = assign_group_splits(rows, test_groups=['Math_006'], val_groups=['Art_001'])
    assert recording_group('Mathematics_006_part32.mp4') == 'Mathematics_006'
    assert {r['split'] for r in result if r['group'] == 'Math_006'} == {'test'}
    with pytest.raises(ValueError, match='overlap'):
        assign_group_splits(rows, test_groups=['Math_006'], val_groups=['Math_006'])
    with pytest.raises(ValueError, match='unknown'):
        assign_group_splits(rows, test_groups=['MISSING'], val_groups=['Art_001'])


def test_actual_header_aliases_are_not_silently_dropped():
    assert parse_targets('1 Following up with a counter-question Text display')['guiding_questions'] == 1
    assert parse_targets('1 Students read aloud Raise a question')['oral_expression'] == 1
    assert parse_targets('1 Reporting Conversation Text display')['cooperation_exchange'] == 1


def test_average_precision_ties_do_not_depend_on_record_order():
    p = np.array([[.5],[.5]],dtype=np.float32)
    a = classification_metrics(np.array([[1],[0]]),p,.5)
    b = classification_metrics(np.array([[0],[1]]),p,.5)
    assert a['mean_average_precision'] == b['mean_average_precision'] == .5


def test_cached_targets_and_groups_must_match_the_manifest():
    manifest={'target_mapping_version':TARGET_MAPPING_VERSION,'records':[{'id':'a','group':'g','split':'train','targets':[1]}]}
    values={'ids':np.array(['a']),'splits':np.array(['train']),'targets':np.array([[1]],dtype=np.float32)}
    validate_cache(values,manifest)
    with pytest.raises(ValueError,match='target mismatch'):
        validate_cache({**values,'targets':np.array([[0]])},manifest)
    with pytest.raises(ValueError,match='split mismatch'):
        validate_cache({**values,'splits':np.array(['test'])},manifest)


def test_training_scaler_uses_train_only_and_model_reload_is_identical(tmp_path):
    pytest.importorskip('torch')
    x = np.array([[-2.,0],[-1.,0],[1.,0],[2.,0],[-1.5,0],[1.5,0],[-100.,0],[100.,0]],dtype=np.float32)
    y = (x[:,0]>0).astype(np.float32)[:,None]
    splits = ['train']*4+['val']*2+['test']*2
    report = train_head(x,y,splits,tmp_path,labels=['test_class'],epochs=80,seed=7,device='cpu')
    head = load_head(tmp_path)
    assert np.allclose(head['mean'],x[:4].mean(0))
    assert report['training_performed'] is True
    assert report['split_counts'] == {'train':4,'val':2,'test':2}
    p = predict_features(x,head)
    assert p.shape == (8,1)
    assert p[0,0] < p[3,0]
    assert np.array_equal(p,predict_features(x,load_head(tmp_path)))
    assert (tmp_path/'metrics.json').is_file()
    with pytest.raises(ValueError, match='finite'):
        predict_features(np.array([[np.nan,0]],dtype=np.float32),head)
