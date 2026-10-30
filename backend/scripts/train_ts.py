"""Create leakage-aware TS manifest, extract frozen RGB features, train real heads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learning.ts_context import CLASSES,TARGET_MAPPING_VERSION,dataset_rows,assign_group_splits,train_head,validate_cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/learning/ts-context')
    p.add_argument('--features',type=Path,help='Train from already extracted, paired feature cache')
    p.add_argument('--extract-only',action='store_true')
    p.add_argument('--refresh-targets',action='store_true',help='Re-read source headers, verify video hashes, preserve frozen RGB features')
    p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--frames',type=int,default=16)
    p.add_argument('--test-groups',nargs='+',default=['Mathematics_006','Art_001'])
    p.add_argument('--val-groups',nargs='+',default=['Chinese_001'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cache=a.features or a.output/'features.npz'
    if not a.features:
        if not a.dataset:p.error('--dataset required for extraction')
        rows=assign_group_splits(dataset_rows(a.dataset),test_groups=a.test_groups,val_groups=a.val_groups)
        from learning.vision_features import FrozenVideoEncoder
        encoder=FrozenVideoEncoder(device=a.device,cache_dir=a.output/'encoder')
        features=[];started=time.perf_counter()
        for i,row in enumerate(rows):
            feature,meta=encoder.video(a.dataset/row['video'],frames=a.frames)
            features.append(feature);row['sampling']=meta
            print(f"TS extract {i+1}/{len(rows)} {row['id']} split={row['split']}",flush=True)
        manifest={'dataset':'TS-Interact public samples','count':len(rows),'groups':sorted({r['group'] for r in rows}),
                  'labels':list(CLASSES),'input':'RGB only; annotation text excluded','records':rows,
                  'extract_elapsed_sec':time.perf_counter()-started,'frames_per_clip':a.frames,'target_mapping_version':TARGET_MAPPING_VERSION}
        (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(cache,features=np.stack(features),targets=np.array([r['targets'] for r in rows],dtype=np.float32),
                            splits=np.array([r['split'] for r in rows]),ids=np.array([r['id'] for r in rows]))
    else:
        manifest=json.loads((cache.parent/'manifest.json').read_text(encoding='utf-8-sig'))
        if a.refresh_targets:
            if not a.dataset:p.error('--dataset required with --refresh-targets')
            fresh={r['id']:r for r in dataset_rows(a.dataset)}
            for row in manifest['records']:
                new=fresh.get(row['id'])
                if not new or new['video_sha256']!=row['video_sha256']:raise ValueError('Source video changed; RGB cache must be regenerated')
                row['targets']=new['targets']
            manifest['target_mapping_version']=TARGET_MAPPING_VERSION
            with np.load(cache,allow_pickle=False) as old:cached={k:old[k].copy() for k in old.files}
            cached['targets']=np.asarray([r['targets'] for r in manifest['records']],dtype=np.float32)
            np.savez_compressed(cache,**cached)
            (cache.parent/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        if cache.parent.resolve()!=a.output.resolve():
            (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.extract_only:
        print('Saved feature cache:',cache,flush=True);return
    with np.load(cache,allow_pickle=False) as values:
        validate_cache(values,manifest)
        report=train_head(values['features'],values['targets'],values['splits'].tolist(),a.output,labels=manifest['labels'],
                          seed=a.seed,epochs=a.epochs,device=a.device)
    digest=hashlib.file_digest((a.output/'model.npz').open('rb'),'sha256').hexdigest()
    config={'model_id':'ts-context-'+digest[:12],'architecture':'frozen_mobilenet_v3_small_temporal_mean_std_linear',
            'frames_per_clip':manifest['frames_per_clip'],'weights_sha256':digest,'training_count':report['split_counts']['train'],'dataset_count':manifest['count'],
            'groups':manifest['groups'],'task':'clip-level multilabel instructional activity','input':'RGB only',
            'training_performed':True,'dataset_scope':'70 publicly accessible samples, not full 973' if manifest['count']==70 else 'local supplied samples'}
    (a.output/'model.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'model_id':config['model_id'],'device':report['device'],'gpu':report['gpu_name'],
                      'test_macro_f1':report['test']['macro_f1'],'baseline_macro_f1':report['baseline_test']['macro_f1'],
                      'elapsed_sec':report['training_elapsed_sec']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
