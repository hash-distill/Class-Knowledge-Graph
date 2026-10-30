"""Pinned, unguided Qwen ASR runner; inputs contain WAV identity only, never GT."""
from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.qwen_asr import atomic_json, build_asr, inspect_wav, sha256_file, utc_now


def nvidia_snapshot():
    query = 'index,uuid,name,memory.total,memory.used,utilization.gpu'
    r = subprocess.run(['nvidia-smi', '--query-gpu=' + query, '--format=csv,noheader,nounits'],
                       text=True, capture_output=True, timeout=15)
    return {'at': utc_now(), 'query': query, 'csv': r.stdout.strip(),
            'returncode': r.returncode, 'stderr': r.stderr.strip()}


class Monitor:
    def __init__(self, path):
        import psutil
        self.path, self.stop, self.phase = path, threading.Event(), 'before_load'
        self.process = psutil.Process(os.getpid())
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with self.path.open('w', encoding='utf-8') as stream:
            while not self.stop.is_set():
                row = nvidia_snapshot()
                row.update(phase=self.phase, measurement_scope='whole_gpu_includes_other_processes')
                info = self.process.memory_info()
                row['host_process_memory_bytes'] = {
                    key: getattr(info, key) for key in ('rss', 'vms', 'peak_wset')
                    if hasattr(info, key)}
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                self.stop.wait(.2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--sample', action='append')
    p.add_argument('--gpu', type=int)
    p.add_argument('--min-free-mib', type=int, default=14000)
    p.add_argument('--aligner-device', choices=['cuda:0', 'cpu'], default='cuda:0')
    p.add_argument('--download-only', action='store_true')
    p.add_argument('--local-models', action='store_true')
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    supplied = json.loads(Path(a.inputs).read_text(encoding='utf-8-sig'))['samples']
    allowed = {'id', 'wav_path', 'sha256', 'source_video', 'original_wav_path'}
    if any(set(row) - allowed for row in supplied):
        raise ValueError('unapproved_model_input_fields')
    if config.get('context') != '' or config.get('ground_truth_in_model_inputs') is not False:
        raise ValueError('requires_unguided_no_ground_truth_config')
    samples = [s for s in supplied if not a.sample or s['id'] in a.sample]
    if not samples:
        raise ValueError('no_samples_selected')
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache)
    os.environ.setdefault('HF_HOME', str(cache))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    before = nvidia_snapshot()
    gpus = []
    for line in before['csv'].splitlines():
        fields = [x.strip() for x in line.split(',')]
        gpus.append((int(fields[3]) - int(fields[4]), int(fields[0])))
    gpu = a.gpu if a.gpu is not None else max(gpus)[1]
    free = next(x for x, y in gpus if y == gpu)
    if free < a.min_free_mib and not a.download_only:
        raise RuntimeError('selected_gpu_has_less_than_required_free_memory')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    from huggingface_hub import snapshot_download
    models = {}
    for kind in ['model', 'aligner']:
        start = time.perf_counter()
        if a.local_models:
            location = cache / 'hub' / ('models--' + config[kind + '_id'].replace('/', '--')) / 'snapshots' / config[kind + '_revision']
            if not (location / 'config.json').is_file():
                raise ValueError('local_model_snapshot_missing')
        else:
            location = snapshot_download(config[kind + '_id'], revision=config[kind + '_revision'],
                                         cache_dir=str(cache / 'hub'), max_workers=4)
        location = str(location)
        hashes = {str(f.relative_to(location)): sha256_file(f)
                  for f in sorted(Path(location).rglob('*'))
                  if f.is_file() and '.cache' not in f.parts and f.name != '.gitattributes'}
        models[kind] = {'id': config[kind + '_id'], 'revision': config[kind + '_revision'],
                        'path': location, 'files_sha256': hashes,
                        'download_and_hash_sec': time.perf_counter() - start}
        verified_path = output / 'download_manifest.json'
        if verified_path.is_file():
            verified = json.loads(verified_path.read_text(encoding='utf-8'))[kind]
            if (verified['id'] != models[kind]['id'] or
                    verified['revision'] != models[kind]['revision'] or
                    verified['files_sha256'] != hashes or
                    verified.get('official_identity_verified') is not True):
                raise ValueError('local_snapshot_differs_from_verified_official_manifest')
            models[kind]['official_identity_verified'] = True
            models[kind]['download_manifest_sha256'] = sha256_file(verified_path)
        atomic_json(output / 'model_manifest.json', models)
    if a.download_only:
        return
    from contextlib import nullcontext
    if os.name == 'nt':
        from review.model_runtime import model_session
        lease = model_session()
    else:
        lease = nullcontext()
    with lease:
        # Recheck memory after acquiring the lease, immediately before loading.
        before = nvidia_snapshot()
        selected = [line for line in before['csv'].splitlines()
                    if int(line.split(',')[0].strip()) == gpu]
        fields = [field.strip() for field in selected[0].split(',')]
        if int(fields[3]) - int(fields[4]) < a.min_free_mib:
            raise RuntimeError('selected_gpu_has_less_than_required_free_memory')
        run_inference(a, config, samples, output, before, gpu, models)


def run_inference(a, config, samples, output, before, gpu, models):
    # This runner uses PyTorch only, including when a base environment has TF.
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    import_started = time.perf_counter()
    import torch
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
    package_import_sec = time.perf_counter() - import_started
    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'torchaudio', 'transformers', 'qwen-asr', 'accelerate',
                 'huggingface-hub', 'soundfile', 'librosa', 'numpy', 'pandas',
                 'psutil', 'nagisa', 'DyNet38']}
    manifest = {'status': 'running', 'created_at': utc_now(), 'pid': os.getpid(),
                'config': config, 'selected_physical_gpu': gpu,
                'asr_device': 'cuda:0', 'aligner_device': a.aligner_device,
                'min_free_mib_requested': a.min_free_mib,
                'gpu_lease': 'project_model_session' if os.name == 'nt' else 'remote_single_runner',
                'gpu_before': before, 'packages': versions, 'python': sys.version,
                'python_executable': sys.executable, 'python_prefix': sys.prefix,
                'python_base_prefix': sys.base_prefix, 'command_args': sys.argv[1:],
                'package_import_sec': package_import_sec,
                'optional_frameworks': {'USE_TF': '0', 'USE_FLAX': '0'},
                'platform': platform.platform(), 'models': models,
                'input_manifest_sha256': sha256_file(a.inputs),
                'script_sha256': sha256_file(__file__),
                'conversion_module_sha256': sha256_file(Path(__file__).resolve().parents[1] / 'review/qwen_asr.py'),
                'samples': [], 'ground_truth_in_model_inputs': False}
    atomic_json(output / 'run_manifest.json', manifest)
    def memory():
        torch.cuda.synchronize()
        return {'measurement_scope': 'this_torch_process_only_bytes',
                'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
                'peak_allocated': torch.cuda.max_memory_allocated(),
                'peak_reserved': torch.cuda.max_memory_reserved()}
    monitor = Monitor(output / 'monitor.jsonl')
    monitor.thread.start()
    metrics = {'before_load': memory()}
    try:
        torch.cuda.reset_peak_memory_stats()
        monitor.phase = 'asr_model_load'
        start = time.perf_counter()
        model = Qwen3ASRModel.from_pretrained(models['model']['path'],
                    dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa',
                    max_inference_batch_size=config['max_inference_batch_size'],
                    max_new_tokens=config['max_new_tokens'])
        metrics['asr_load_sec'] = time.perf_counter() - start
        metrics['after_asr_load'] = memory()
        current_raw = []
        infer = model._infer_asr
        def capture_infer(*args, **kwargs):
            start = time.perf_counter()
            result = infer(*args, **kwargs)
            torch.cuda.synchronize()
            current_raw.append({'outputs': result, 'elapsed_sec': time.perf_counter() - start})
            return result
        model._infer_asr = capture_infer
        pending = []
        for sample in samples:
            dest = output / sample['id']
            dest.mkdir(parents=True, exist_ok=True)
            audio = inspect_wav(sample['wav_path'], sample['sha256'])
            audio['original_path'] = sample.get('original_wav_path')
            current_raw.clear()
            torch.cuda.reset_peak_memory_stats()
            row_metrics = {'before_inference': memory()}
            monitor.phase = 'inference:' + sample['id']
            start = time.perf_counter()
            result = model.transcribe(audio=sample['wav_path'], language='Chinese', context='', return_time_stamps=False)[0]
            row_metrics.update(total_transcribe_sec=time.perf_counter() - start,
                               asr_generation_sec=sum(x['elapsed_sec'] for x in current_raw),
                               after_inference=memory(), duration_sec=audio['duration_sec'])
            native = dataclasses.asdict(result)
            native.update(model=models, raw_decoded_chunks=list(current_raw),
                          config=config,
                          source_audio=audio, packages=versions, metrics=row_metrics,
                          raw_output_kind='actual_model_response_no_reference_text', created_at=utc_now())
            pending.append((sample, audio, native, row_metrics))
            atomic_json(dest / 'native_asr_alignment_pending.json', native)
            print(json.dumps({'id': sample['id'], 'status': 'asr_complete_alignment_pending',
                              'asr_generation_sec': row_metrics['asr_generation_sec']}, ensure_ascii=False), flush=True)
        metrics['after_all_asr'] = memory()
        monitor.phase = 'unload_asr_before_aligner'
        del model, infer, capture_infer
        gc.collect()
        torch.cuda.empty_cache()
        metrics['after_asr_unload'] = memory()
        torch.cuda.reset_peak_memory_stats()
        monitor.phase = 'aligner_model_load'
        start = time.perf_counter()
        aligner = Qwen3ForcedAligner.from_pretrained(models['aligner']['path'],
                    dtype=torch.bfloat16 if a.aligner_device != 'cpu' else torch.float32,
                    device_map=a.aligner_device, attn_implementation='sdpa')
        metrics['aligner_load_sec'] = time.perf_counter() - start
        metrics['after_aligner_load'] = memory()
        for sample, audio, native, row_metrics in pending:
            dest = output / sample['id']
            torch.cuda.reset_peak_memory_stats()
            row_metrics['before_alignment'] = memory()
            monitor.phase = 'alignment:' + sample['id']
            start = time.perf_counter()
            aligned = aligner.align(audio=sample['wav_path'], text=native['text'], language='Chinese')[0]
            torch.cuda.synchronize()
            row_metrics['forced_alignment_sec'] = time.perf_counter() - start
            row_metrics['after_alignment'] = memory()
            row_metrics['total_transcribe_and_align_sec'] = row_metrics['total_transcribe_sec'] + row_metrics['forced_alignment_sec']
            row_metrics['real_time_factor'] = row_metrics['total_transcribe_and_align_sec'] / audio['duration_sec']
            native['time_stamps'] = dataclasses.asdict(aligned)
            native['raw_alignment_calls'] = [{'outputs': [dataclasses.asdict(aligned)],
                                               'elapsed_sec': row_metrics['forced_alignment_sec']}]
            native['metrics'] = row_metrics
            atomic_json(dest / 'native_output.json', native)
            converted = build_asr(native, audio, config, sample.get('source_video'), row_metrics)
            converted['provenance']['native_artifact_sha256'] = sha256_file(dest / 'native_output.json')
            atomic_json(dest / 'asr.json', converted)
            manifest['samples'].append({'id': sample['id'], 'status': 'complete',
                      'segments': len(converted['segments']), 'metrics': row_metrics,
                      'source_audio_sha256': audio['sha256'],
                      'asr_sha256': sha256_file(dest / 'asr.json'),
                      'native_output_sha256': sha256_file(dest / 'native_output.json')})
            atomic_json(output / 'run_manifest.json', manifest)
            print(json.dumps(manifest['samples'][-1], ensure_ascii=False), flush=True)
        manifest['status'] = 'complete'
        metrics['after_all_inference'] = memory()
    except Exception as error:
        manifest.update(status='error', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        monitor.stop.set()
        monitor.thread.join(timeout=20)
        manifest.update(finished_at=utc_now(), load_metrics=metrics,
                        gpu_after=nvidia_snapshot())
        atomic_json(output / 'run_manifest.json', manifest)
        atomic_json(output / 'metrics.json', {'load': metrics, 'samples': manifest['samples'],
                    'whole_gpu_samples': 'monitor.jsonl', 'whole_gpu_interval_requested_sec': .2,
                    'peak_scope_note': 'Torch process peaks and sampled whole-GPU usage are distinct; sampling can miss sub-interval peaks.'})


if __name__ == '__main__':
    main()
