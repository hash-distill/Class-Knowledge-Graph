"""Small, grouped TS-Interact context baseline; target descriptions are never inputs."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import time
import numpy as np

CLASSES = {
    'knowledge_presentation': ('Knowledge presentation', ['Explain and state','Summarize the lecture','Evaluation summary','Demonstrate and explain']),
    'guiding_questions': ('Questions and task guidance', ['Raise a question','Assign tasks','Follow up with a counter-question','Following up with a counter-question']),
    'resource_display': ('Teaching resource display', ['Physical display','Experimental demonstration','Text display','Picture display','Data display','Blackboard writing display']),
    'process_management': ('Classroom management and feedback', ['Activity guidance','Communication guidance','Order management','Emotional motivation','Performance evaluation']),
    'independent_exploration': ('Independent thinking and exploration', ['Independent thinking','Information query']),
    'oral_expression': ('Student oral expression', ['Student read aloud','Students read aloud','Speak and answer together',"Student's direct answers","Students' direct answers"]),
    'practical_operation': ('Writing and practical activity', ['Start writing by hand','Hands-on operation']),
    'cooperation_exchange': ('Cooperation and discussion', ['Group discussion','Report conversation','Reporting conversation','Work display']),
}
TARGET_MAPPING_VERSION = 'ts_header_aliases_v2'


def _normalized(text):
    return re.sub(r'[^a-z0-9]+',' ',text.lower()).strip()


def parse_targets(text):
    first = text.lstrip('\ufeff').splitlines()[0] if text.strip() else ''
    if not re.match(r'^\s*1(?:\s|[.:])', first):
        raise ValueError('TS label header must start with field 1')
    header = ' '+_normalized(re.sub(r'^\s*1[\s.:]*','',first))+' '
    targets = {key:int(any(' '+_normalized(p)+' ' in header for p in phrases)) for key,(_,phrases) in CLASSES.items()}
    if not any(targets.values()):
        raise ValueError('No supported label in TS target header')
    return targets


def recording_group(filename):
    value = re.sub(r'_part\d+$','',Path(filename).stem)
    if value == Path(filename).stem:
        raise ValueError('TS recording filename must include _part<number>')
    return value


def assign_group_splits(rows, *, test_groups, val_groups):
    test, val = set(test_groups),set(val_groups)
    available = {r['group'] for r in rows}
    if test & val: raise ValueError('Validation and test groups overlap')
    if not test or not val or (test|val)-available: raise ValueError('Empty or unknown held-out group')
    result = [{**r,'split':'test' if r['group'] in test else 'val' if r['group'] in val else 'train'} for r in rows]
    if not any(r['split']=='train' for r in result): raise ValueError('No training groups remain')
    return result


def dataset_rows(root):
    root=Path(root).resolve(); rows=[]
    for video in sorted(root.rglob('*.mp4')):
        label=video.parent.parent/'label'/f'{video.stem}.txt'
        if not label.is_file(): raise ValueError(f'Missing paired TS label: {video.stem}')
        targets=parse_targets(label.read_text(encoding='utf-8-sig'))
        rows.append({'id':video.stem,'group':recording_group(video.name),
                     'video':video.relative_to(root).as_posix(),'video_sha256':hashlib.file_digest(video.open('rb'),'sha256').hexdigest(),
                     'targets':[targets[k] for k in CLASSES]})
    if not rows:raise ValueError('No TS MP4 videos found')
    return rows


def classification_metrics(y, probabilities, thresholds):
    prediction=probabilities>=thresholds
    truth=y.astype(bool)
    tp=(prediction&truth).sum(0);fp=(prediction&~truth).sum(0);fn=(~prediction&truth).sum(0)
    f1=2*tp/np.maximum(2*tp+fp+fn,1)
    support=truth.sum(0)
    ap=[]
    for i in range(y.shape[1]):
        order=np.argsort(-probabilities[:,i],kind='stable'); ranked=truth[order,i]
        if support[i]:
            scores=probabilities[order,i]
            ends=np.r_[np.flatnonzero(np.diff(scores)),len(scores)-1]
            cumulative=np.cumsum(ranked)[ends]
            increments=np.diff(np.r_[0,cumulative])
            ap.append(float(np.sum(increments*cumulative/(ends+1))/support[i]))
        else:ap.append(None)
    return {'macro_f1':float(f1.mean()),'macro_f1_supported':float(f1[support>0].mean()) if (support>0).any() else None,
            'mean_average_precision':float(np.mean([v for v in ap if v is not None])) if any(v is not None for v in ap) else None,
            'sample_count':len(y),'per_label':[{'support':int(support[i]),'f1':float(f1[i]),'average_precision':ap[i],
                        'true_positive':int(tp[i]),'false_positive':int(fp[i]),'false_negative':int(fn[i])} for i in range(y.shape[1])]}


def _sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))


def train_head(features, targets, splits, output, *, labels=None, seed=42, epochs=300, device='cpu', learning_rate=.005):
    import torch
    from copy import deepcopy
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    x=np.asarray(features,dtype=np.float32);y=np.asarray(targets,dtype=np.float32)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)!=len(splits) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Training inputs must be finite matched matrices')
    if not np.isin(y,[0,1]).all():raise ValueError('Targets must be binary multilabel values')
    masks={s:np.array([v==s for v in splits]) for s in ['train','val','test']}
    if any(not m.any() for m in masks.values()) or sum(int(m.sum()) for m in masks.values())!=len(x):raise ValueError('All three split partitions are required')
    labels=list(labels or CLASSES)
    if len(labels)!=y.shape[1]:raise ValueError('Label dimension mismatch')
    torch.manual_seed(seed);np.random.seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    if device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('Requested CUDA is not available')
    mean=x[masks['train']].mean(0);std=np.maximum(x[masks['train']].std(0),.01)
    normalized=np.clip((x-mean)/std,-20,20).astype(np.float32)
    xt=torch.from_numpy(normalized).to(device);yt=torch.from_numpy(y).to(device)
    positive=y[masks['train']].sum(0);negative=masks['train'].sum()-positive
    pos_weight=torch.from_numpy(np.clip(negative/np.maximum(positive,1),.3,5).astype(np.float32)).to(device)
    model=torch.nn.Linear(x.shape[1],y.shape[1]).to(device)
    with torch.no_grad():
        model.weight.zero_();model.bias.copy_(torch.tensor(np.log((positive+.5)/(negative+.5)),device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.1)
    train_index=torch.from_numpy(np.flatnonzero(masks['train'])).to(device)
    val_index=torch.from_numpy(np.flatnonzero(masks['val'])).to(device)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best=math.inf;state=None;best_epoch=0;history=[];started=time.perf_counter()
    if device.startswith('cuda'):torch.cuda.synchronize()
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad()
        loss=loss_fn(model(xt[train_index]),yt[train_index]);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():val_loss=float(torch.nn.functional.binary_cross_entropy_with_logits(model(xt[val_index]),yt[val_index]).cpu())
        history.append({'epoch':epoch,'train_loss':float(loss.detach().cpu()),'val_loss':val_loss})
        if val_loss<best-1e-6:best=val_loss;best_epoch=epoch;state=deepcopy(model.state_dict())
        if epoch-best_epoch>=35:break
    if device.startswith('cuda'):torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    model.load_state_dict(state)
    weight=model.weight.detach().cpu().numpy();bias=model.bias.detach().cpu().numpy()
    probs=_sigmoid(normalized@weight.T+bias)
    candidates=np.arange(.2,.81,.05)
    threshold=max(candidates,key=lambda t:classification_metrics(y[masks['val']],probs[masks['val']],t)['macro_f1'])
    thresholds=np.full(y.shape[1],threshold,dtype=np.float32)
    np.savez_compressed(output/'model.npz',weight=weight,bias=bias,mean=mean,std=std,thresholds=thresholds,labels=np.array(labels))
    prior=(positive+.5)/(masks['train'].sum()+1)
    report={'schema_version':'1.0','task':'TS clip-level multilabel instructional context','training_performed':True,
            'seed':seed,'device':device,'gpu_name':torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else None,
            'epochs_run':len(history),'best_epoch':best_epoch,'training_elapsed_sec':round(elapsed,4),
            'split_counts':{s:int(m.sum()) for s,m in masks.items()},'feature_dimension':x.shape[1],
            'threshold_selection':'global threshold on validation Macro-F1 only','threshold':float(threshold),
            'labels':labels,'validation':classification_metrics(y[masks['val']],probs[masks['val']],thresholds),
            'test':classification_metrics(y[masks['test']],probs[masks['test']],thresholds),
            'baseline_test':classification_metrics(y[masks['test']],np.tile(prior,(int(masks['test'].sum()),1)),.5),
            'history':history,'limitations':['Small number of held-out recording groups; not a classroom or school generalization claim.',
                 'Targets are clip-level labels, not exact per-second activities or individual engagement.','Probabilities are uncalibrated classifier outputs.']}
    (output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return report


def load_head(directory):
    directory=Path(directory)
    config_path=directory/'model.json'
    if config_path.exists():
        config=json.loads(config_path.read_text(encoding='utf-8-sig'))
        with (directory/'model.npz').open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if config.get('weights_sha256') != digest:raise ValueError('Model checksum does not match training metadata')
    with np.load(directory/'model.npz',allow_pickle=False) as data:
        return {key:data[key].copy() for key in data.files}


def predict_features(features, head):
    x=np.asarray(features,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=len(head['mean']) or not np.isfinite(x).all():raise ValueError('Features must be finite and match model dimension')
    z=np.clip((x-head['mean'])/head['std'],-20,20)
    return _sigmoid(z@head['weight'].T+head['bias'])


def validate_cache(values, manifest):
    if manifest.get('target_mapping_version') != TARGET_MAPPING_VERSION:
        raise ValueError('Stale target label mapping; refresh target cache from source headers')
    records=manifest['records']
    if values['ids'].tolist()!=[r['id'] for r in records]:raise ValueError('Cache and manifest sample mismatch')
    if values['splits'].tolist()!=[r['split'] for r in records]:raise ValueError('Cache and manifest split mismatch')
    if not np.array_equal(values['targets'],np.asarray([r['targets'] for r in records],dtype=np.float32)):
        raise ValueError('Cache and manifest target mismatch')
    grouped={}
    for row in records:
        if row['group'] in grouped and grouped[row['group']]!=row['split']:raise ValueError('Recording group crosses split partitions')
        grouped[row['group']]=row['split']


def source_role(video_path, manifest):
    with Path(video_path).open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    roles={row['split'] for row in manifest['records'] if row['video_sha256']==digest}
    if len(roles)>1:raise ValueError('Identical video crosses dataset partitions')
    return next(iter(roles),'external')


def predict_context(video_path, model_dir, device='cpu'):
    from .vision_features import FrozenVideoEncoder
    model_dir=Path(model_dir)
    config=json.loads((model_dir/'model.json').read_text(encoding='utf-8-sig'))
    manifest_path=model_dir/'manifest.json'
    role=source_role(video_path,json.loads(manifest_path.read_text(encoding='utf-8-sig'))) if manifest_path.exists() else 'unknown'
    encoder=FrozenVideoEncoder(device=device,cache_dir=model_dir/'encoder')
    feature,metadata=encoder.video(video_path,frames=config['frames_per_clip'])
    head=load_head(model_dir);probs=predict_features(feature[None],head)[0]
    labels=[{'label':str(label),'name':CLASSES.get(str(label),(str(label),[]))[0],'probability':round(float(p),6)}
            for label,p,t in zip(head['labels'],probs,head['thresholds']) if p>=t]
    labels.sort(key=lambda r:r['probability'],reverse=True)
    text='Lesson context model prediction for the full video: '+(', '.join(r['name'] for r in labels) if labels else 'No class exceeds the validation threshold')+'. This is a segment-level prediction, not second-by-second manual annotation.'
    text+=' Source partition for this segment: '+{'train':'training set; demonstration only, excluded from held-out evaluation','val':'validation set, used to select thresholds','test':'test set','external':'video outside the training manifest','unknown':'unknown'}[role]+'.'
    return [{'id':'X0001','start_sec':0.,'end_sec':metadata['duration_sec'],'text':text,'labels':labels,
             'model_id':config['model_id'],'method':'trained_context','source_split':role}]
