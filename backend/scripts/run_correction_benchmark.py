"""Measure sequential, resumable correction runs on immutable lesson evidence.

Owned llama.cpp roles are switched only by the PID/start-time-checked launcher.
No gold reference is read by this inference script. GPU memory is whole-device
usage sampled by nvidia-smi; it is not per-process allocation or an exact peak.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from review.common import atomic_json, read_json, sha256
from review.correction_provider import CorrectionProvider, NATIVE_SYSTEM
from review.provider import ReviewError
from review.proofreading import METHOD
from scripts.prepare_proofreading import prepare

PROFILES = {'qwen': ROOT/'configs/local_llm.json', 'cec4': ROOT/'configs/correction_llm.json'}
SOURCE_FILES = ('asr.json', 'asr.unguided.json', 'asr.secondary.json', 'audio.wav',
                'slides.json', 'activity.json')
# English translation of the published Chinese native-task instruction.
OFFICIAL_NATIVE_SYSTEM = 'Act as a professional Chinese text correction expert. Analyze grammatical error types and reasons for correction, but output only the corrected sentence in its original language. Error types include typos, incorrect collocations, incorrect parts of speech, word order, missing components, redundant components, incorrect conjunctions, ambiguous references, semantic or logical errors, and no error.'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **values):
    print(json.dumps({'event': event, 'utc': utc_now(), **values}, ensure_ascii=False), flush=True)


class IdleSuspendLease:
    """Windows thread-scoped idle-sleep prevention; restore on every exit."""
    def __enter__(self):
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True) if os.name == 'nt' else None
        if self.kernel and not self.kernel.SetThreadExecutionState(0x80000001):
            raise OSError(ctypes.get_last_error(), 'SetThreadExecutionState failed')
        return self

    def __exit__(self, *args):
        if self.kernel:
            self.kernel.SetThreadExecutionState(0x80000000)


def hold_awake_for_process(pid):
    """Attach an idle-sleep lease to an already running own benchmark process."""
    if os.name != 'nt':
        return
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), 'Cannot watch benchmark process')
    try:
        with IdleSuspendLease():
            emit('idle_suspend_lease_attached', benchmark_pid=pid)
            kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
    finally:
        kernel.CloseHandle(handle)
    emit('idle_suspend_lease_released', benchmark_pid=pid)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def lifecycle(profile, stop=False):
    command = [shutil.which('pwsh') or 'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
               '-File', str(ROOT/'scripts/start_local_llm.ps1'), '-ConfigPath', str(profile)]
    if stop:
        command.append('-Stop')
    else:
        command += ['-Backend', 'cuda', '-WaitSeconds', '120']
    # On Windows a detached server can inherit anonymous-pipe handles. Reading
    # communicate() pipes can then wait for the long-lived server after pwsh
    # has exited. Regular log files keep launch completion independent of it.
    log_dir = ROOT/'artifacts/verification/cec4-comparison/lifecycle'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_id = uuid.uuid4().hex[:12]
    stdout_path, stderr_path = log_dir/(log_id+'.stdout.log'), log_dir/(log_id+'.stderr.log')
    with stdout_path.open('w', encoding='utf-8') as stdout, stderr_path.open('w', encoding='utf-8') as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                stdin=subprocess.DEVNULL, timeout=180)
    record = {'utc': utc_now(), 'profile': str(profile), 'stop': stop,
              'exit_code': result.returncode,
              'stdout': stdout_path.read_text(encoding='utf-8', errors='replace').strip(),
              'stderr': stderr_path.read_text(encoding='utf-8', errors='replace').strip(),
              'stdout_path': str(stdout_path), 'stderr_path': str(stderr_path)}
    emit('lifecycle', **record)
    if result.returncode:
        raise RuntimeError('Identity-checked model lifecycle failed: '+record['stderr'][-1000:])
    return record


def parse_gpu_csv(text):
    rows = list(csv.reader(text.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 8:
        raise ValueError('Expected one selected GPU and eight nvidia-smi columns')
    index, name, uuid, total, used, utilization, driver, power = [value.strip() for value in rows[0]]
    return {'index': int(index), 'name': name, 'uuid': uuid, 'memory_total_mib': int(total),
            'memory_used_mib': int(used), 'utilization_gpu_percent': int(utilization),
            'driver_version': driver, 'power_w': float(power) if power not in ('[N/A]', 'N/A') else None}


class GpuSampler:
    def __init__(self, path, interval=.2, gpu=0):
        self.path, self.interval, self.gpu = Path(path), interval, gpu
        self.phase, self.samples, self.errors = 'initial', [], []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def _sample(self):
        with self.path.open('w', encoding='utf-8') as stream:
            while not self.stopped.is_set():
                started = time.perf_counter()
                try:
                    result = subprocess.run(['nvidia-smi', '-i', str(self.gpu),
                        '--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
                    row = {'utc': utc_now(), 'monotonic_seconds': time.perf_counter(),
                           'phase': self.phase, **parse_gpu_csv(result.stdout)}
                    self.samples.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    self.errors.append({'utc': utc_now(), 'phase': self.phase, 'error': str(exc)})
                self.stopped.wait(max(0, self.interval-(time.perf_counter()-started)))

    def close(self):
        self.stopped.set(); self.thread.join(timeout=15)

    def summarize(self, phase, baseline_phase, loaded_phase, wall_time, load_time):
        generation = [row for row in self.samples if row['phase'] == phase]
        baseline = [row['memory_used_mib'] for row in self.samples if row['phase'] == baseline_phase]
        loaded = [row['memory_used_mib'] for row in self.samples if row['phase'] == loaded_phase]
        card = {key: self.samples[-1][key] for key in ('index', 'name', 'uuid', 'driver_version', 'memory_total_mib')} if self.samples else {}
        intervals = [b['monotonic_seconds']-a['monotonic_seconds'] for a, b in zip(self.samples, self.samples[1:])]
        generation_intervals = [b['monotonic_seconds']-a['monotonic_seconds'] for a, b in zip(generation, generation[1:])]
        peak = max((row['memory_used_mib'] for row in generation), default=None)
        baseline_median = statistics.median(baseline) if baseline else None
        return {'scope': 'whole_gpu_loaded_single_model_frozen_evidence_correction',
            'measurement_source': 'nvidia-smi memory.used whole selected GPU', 'gpu': card,
            'requested_sampling_interval_sec': self.interval,
            'actual_median_sampling_interval_sec': statistics.median(intervals) if intervals else None,
            'maximum_generation_sampling_gap_sec': max(generation_intervals, default=0),
            'timing_interruption_detected': max(generation_intervals, default=0) > 5,
            'timing_valid_for_latency_comparison': max(generation_intervals, default=0) <= 5,
            'sample_count': len(generation), 'baseline_sample_count': len(baseline),
            'loaded_sample_count': len(loaded), 'baseline_device_used_mib': baseline_median,
            'loaded_device_used_mib': statistics.median(loaded) if loaded else None,
            'peak_device_used_mib': peak,
            'peak_increment_over_baseline_mib': peak-baseline_median if peak is not None and baseline_median is not None else None,
            'generation_wall_time_sec': round(wall_time, 6), 'model_load_wall_time_sec': round(load_time, 6),
            'sample_file': str(self.path.resolve()), 'sampling_errors': self.errors,
            'phase': phase, 'baseline_phase': baseline_phase, 'loaded_phase': loaded_phase,
            'notice': 'Whole GPU includes display/other processes. Sampled peak can miss between-sample spikes. Frozen ASR/OCR/activity generation time excluded.'}


def source_identity(directory):
    path = Path(directory)/'asr.json'; source = read_json(path)
    if source.get('status') != 'complete':
        raise ValueError('Native benchmark requires complete frozen ASR')
    provenance = source.get('provenance', {})
    identity = {'source_asr_sha256': sha256(path)}
    if provenance.get('source_video', {}).get('sha256'):
        identity['source_video_sha256'] = provenance['source_video']['sha256']
    elif provenance.get('source_audio', {}).get('sha256'):
        identity['source_audio_sha256'] = provenance['source_audio']['sha256']
    else:
        raise ValueError('Native benchmark requires real source video/audio identity')
    return source, identity


def conditions(profile, track, max_tokens):
    config = read_json(profile)
    return {'track': track, 'guard_method': METHOD if track == 'controlled_json' else 'none_raw_native_hypothesis',
        'temperature': config['request_defaults']['temperature'], 'seed': 'not_sent_server_default',
        'n_ctx': config['server']['context_size'], 'max_tokens': max_tokens,
        'request_format': 'json_schema' if track == 'controlled_json' else 'plain_text',
        'thinking_requested': False, 'native_prompt_policy': 'same_project_generic_prompt_both_models_not_verified_official' if track == 'native_text' else None}


def native_run(directory, output, provider, model_label, profile, max_tokens, sampler, baseline_phase, loaded_phase, load_time, phase, retry_errors=False):
    source, identity = source_identity(directory)
    run_conditions = conditions(profile, 'native_text', max_tokens)
    run_conditions['native_prompt_policy'] = provider.config.get('native_system_policy', 'project_generic_prompt_not_verified_official')
    system = provider.config.get('native_system', NATIVE_SYSTEM)
    template = provider.config.get('native_prompt_template', '{text}')
    run_conditions['system_sha256'] = hashlib.sha256(system.encode()).hexdigest()
    run_conditions['prompt_sha256'] = [hashlib.sha256(template.replace('{text}', row['text']).encode()).hexdigest() for row in source['segments']]
    contract = {'source': identity, 'conditions': run_conditions, 'model_sha256': provider.config['model']['sha256']}
    previous = read_json(output) if Path(output).is_file() else {}
    if previous and previous.get('benchmark_contract') != contract:
        raise ValueError('Existing native checkpoint is from different immutable inputs/model/conditions')
    if (previous.get('result', {}).get('status') == 'complete' and
            len(previous.get('native_outputs', [])) == len(source['segments'])):
        emit('native_already_complete', model=model_label, path=str(output))
        return previous
    by_id = {row['transcript_id']: row for row in previous.get('native_outputs', [])}
    runtime_history = previous.get('runtime_attempts', [previous['runtime']] if previous.get('runtime') else [])
    attempt_id = uuid.uuid4().hex
    envelope = {'schema_version': '1.0', 'label': model_label+'-native', 'track': 'native_text',
        'conditions': run_conditions, 'benchmark_contract': contract, 'model_profile': str(Path(profile).resolve()),
        'model_profile_sha256': sha256(profile), 'model': provider.model,
        'native_outputs': [], 'started_at': previous.get('started_at', utc_now()),
        'result': {**identity, 'method': 'raw_native_same_generic_project_prompt', 'model': provider.model,
            'status': 'running', 'human_verified': False, 'provenance': {'gold_or_reference_read': False}}}
    sampler.phase = phase
    started = time.perf_counter()
    for row in source['segments']:
        cached = by_id.get(row['id'])
        if cached and (cached.get('status') == 'complete' or not retry_errors):
            envelope['native_outputs'].append(cached)
            emit('native_resumed', model=model_label, transcript_id=row['id'])
            continue
        segment = {'transcript_id': row['id'], 'original_text': row['text'],
                   'start_sec': row['start_sec'], 'end_sec': row['end_sec'], 'started_at': utc_now()}
        call_started = time.perf_counter()
        try:
            text, metadata = provider.native_correct(row['text'], max_tokens=max_tokens)
            segment.update(corrected_text=text, metadata=metadata, status='complete')
        except ReviewError as exc:
            segment.update(corrected_text=row['text'], status='failed_preserved_original',
                error={'code': exc.code, 'message': str(exc)}, metadata=exc.diagnostic)
        segment.update(completed_at=utc_now(), wall_time_sec=round(time.perf_counter()-call_started, 6))
        envelope['native_outputs'].append(segment)
        current_runtime = sampler.summarize(phase, baseline_phase, loaded_phase, time.perf_counter()-started, load_time)
        current_runtime['attempt_id'] = attempt_id
        envelope['runtime_attempts'] = [*runtime_history, current_runtime]
        envelope['runtime'] = combine_runtime_attempts(envelope['runtime_attempts'])
        atomic_json(output, envelope)
        emit('native_segment', model=model_label, phase=phase, transcript_id=row['id'], status=segment['status'],
             elapsed_sec=segment['wall_time_sec'], original=row['text'], corrected=segment['corrected_text'],
             finish_reason=segment['metadata'].get('finish_reason'))
    envelope['completed_at'] = utc_now()
    envelope['result']['status'] = 'complete' if all(row['status'] == 'complete' for row in envelope['native_outputs']) else 'partial'
    envelope['native_failure_count'] = sum(row['status'] != 'complete' for row in envelope['native_outputs'])
    current_runtime = sampler.summarize(phase, baseline_phase, loaded_phase, time.perf_counter()-started, load_time)
    current_runtime['attempt_id'] = attempt_id
    envelope['runtime_attempts'] = [*runtime_history, current_runtime]
    envelope['runtime'] = combine_runtime_attempts(envelope['runtime_attempts'])
    envelope['runtime']['resumed_segment_count'] = sum(row['transcript_id'] in by_id for row in envelope['native_outputs'])
    envelope['runtime']['total_recorded_request_wall_time_sec'] = sum(row['wall_time_sec'] for row in envelope['native_outputs'])
    atomic_json(output, envelope)
    emit('native_complete', model=model_label, path=str(output), failures=envelope['native_failure_count'])
    return envelope


def combine_runtime_attempts(attempts):
    combined = dict(attempts[-1])
    combined['attempt_count'] = len(attempts)
    combined['sample_files'] = list(dict.fromkeys(row['sample_file'] for row in attempts if row.get('sample_file')))
    peaks = [row['peak_device_used_mib'] for row in attempts if row.get('peak_device_used_mib') is not None]
    combined['peak_device_used_mib'] = max(peaks) if peaks else None
    combined['generation_wall_time_sec'] = sum(row.get('generation_wall_time_sec', 0) for row in attempts)
    combined['sample_count'] = sum(row.get('sample_count', 0) for row in attempts)
    if len(attempts) > 1:
        combined['peak_increment_over_baseline_mib'] = None
        combined['baseline_and_loaded_scope'] = 'latest_attempt_only; use runtime_attempts for each baseline'
    combined['measurement_attempt_notice'] = 'Peak is maximum observed across retained attempts; time is sum of invocation windows, with per-attempt baselines retained separately.'
    return combined


class EmptyProposalProvider:
    def status(self):
        return {'model': 'none_source_rules_only', 'source': 'deterministic_empty_proposals'}

    def complete(self, system, prompt, **kwargs):
        return {'segments': []}, self.status()


def frozen_hashes(directory):
    return {name: sha256(Path(directory)/name) for name in SOURCE_FILES if (Path(directory)/name).is_file()}


def controlled_contract(directory, profile=None):
    return {'source_files': frozen_hashes(directory), 'guard_method': METHOD,
            'guard_source_sha256': sha256(ROOT/'review/proofreading.py'),
            'prepare_source_sha256': sha256(ROOT/'scripts/prepare_proofreading.py'),
            'provider_source_sha256': sha256(ROOT/'review/provider.py'),
            'correction_provider_source_sha256': sha256(ROOT/'review/correction_provider.py'),
            'profile_sha256': sha256(profile) if profile else None}


def archive_attempt(path):
    path = Path(path)
    if path.is_file():
        archive = path.parent/'attempts'; archive.mkdir(exist_ok=True)
        target = archive/(path.stem+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+path.suffix)
        shutil.copy2(path, target)
        emit('attempt_archived', source=str(path), archive=str(target))


def generate_reports(directory, output_dir, external_dir=None):
    from scripts.compare_correction_models import build_report, write_report
    for track in ('controlled', 'native'):
        runs = [(model+'-'+track, output_dir/f'{model}-{track}.json') for model in PROFILES
                if (output_dir/f'{model}-{track}.json').is_file()]
        if runs:
            report = build_report(directory, runs, output_dir/'source-only.json')
            write_report(report, output_dir/(track+'-report'))
            emit('report_complete', track=track, path=str(output_dir/(track+'-report/comparison.md')))
    if not external_dir:
        return
    from scripts.prepare_external_correction_references import prepare_references
    prepared = prepare_references(external_dir)
    samples = read_json(Path(external_dir)/'model-inputs.json')['samples']
    reports = []
    for sample in samples:
        media = Path(external_dir)/'evaluation'/sample['id']
        runs = [(model+'-native', media/f'{model}-native.json') for model in PROFILES if (media/f'{model}-native.json').is_file()]
        if not runs:
            continue
        report = build_report(media, runs, terms_path=prepared['terminology'], reference_path=media/'reference.json')
        write_report(report, output_dir/'external-reports'/sample['id'])
        reports.append((sample['id'], report))
    summary = aggregate_external_reports(reports)
    atomic_json(output_dir/'external-summary.json', summary)
    lines = ['# Chinese-LiPS external manual-reference check', '',
             f"Sample count: {summary['sample_count']}. A small subset of the official test set; independence from model training data is unverified.", '',
             '| Output | Reference characters | Original ASR CER | Corrected CER | Repaired characters | Damaged characters | Insertion errors: before → after |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, metrics in summary['runs'].items():
        normalized = metrics['normalizations']['no_whitespace_or_punctuation']
        transition = metrics['reference_character_transitions']['no_whitespace_or_punctuation']
        lines.append(f"| {label} | {normalized['reference_characters']} | {normalized['source_cer']:.2%} | {normalized['cer']:.2%} | {transition['repaired_reference_characters']} | {transition['damaged_previously_correct_reference_characters']} | {transition['insertion_errors_before']}→{transition['insertion_errors_after']} |")
    lines += ['', 'This table removes whitespace and Unicode punctuation. See JSON for strict CER, numerators, denominators, per-sample results, and raw responses. The small sample count does not support extrapolation to the full dataset.', '']
    (output_dir/'external-summary.md').write_text('\n'.join(lines), encoding='utf-8')
    emit('external_report_complete', path=str(output_dir/'external-summary.md'))


def aggregate_external_reports(reports):
    summary = {'schema_version': '1.0', 'sample_count': len(reports), 'samples': [identifier for identifier, _ in reports],
               'reference': 'official_manual_transcription_scorer_only', 'aggregation': 'sum_character_errors_over_sum_reference_characters',
               'runs': {}}
    for identifier, report in reports:
        if not report['accuracy_evaluated']:
            raise ValueError('Cannot aggregate external CER without independent references')
        for run in report['runs']:
            lane = run['raw_model']
            if lane is None:
                continue
            entry = summary['runs'].setdefault(run['label'], {'sample_ids': [], 'native_failure_count': 0,
                'normalizations': {}, 'reference_character_transitions': {}})
            entry['sample_ids'].append(identifier)
            entry['native_failure_count'] += run['native_failure_count']
            for normalization in ('strict', 'no_whitespace_or_punctuation'):
                score = lane['reference_metrics'][normalization]
                baseline = report['source_asr']['reference_metrics'][normalization]
                total = entry['normalizations'].setdefault(normalization, {'reference_characters': 0,
                    'character_errors': 0, 'source_character_errors': 0, 'substitutions': 0, 'deletions': 0, 'insertions': 0})
                for key in ('reference_characters', 'character_errors', 'substitutions', 'deletions', 'insertions'):
                    total[key] += score[key]
                total['source_character_errors'] += baseline['character_errors']
                transition_key = 'reference_character_transitions' if normalization == 'strict' else 'normalized_reference_character_transitions'
                transition = lane[transition_key]
                transition_total = entry['reference_character_transitions'].setdefault(normalization, {})
                for key, number in transition.items():
                    if type(number) is int:
                        transition_total[key] = transition_total.get(key, 0) + number
    for entry in summary['runs'].values():
        if set(entry['sample_ids']) != set(summary['samples']):
            raise ValueError('External model runs have different sample subsets; complete paired runs before comparative aggregation')
        for total in entry['normalizations'].values():
            total['cer'] = total['character_errors']/total['reference_characters'] if total['reference_characters'] else None
            total['source_cer'] = total['source_character_errors']/total['reference_characters'] if total['reference_characters'] else None
        for total in entry['reference_character_transitions'].values():
            total['repair_rate'] = total['repaired_reference_characters']/total['initially_incorrect_reference_characters'] if total['initially_incorrect_reference_characters'] else None
            total['damage_rate'] = total['damaged_previously_correct_reference_characters']/total['initially_correct_reference_characters'] if total['initially_correct_reference_characters'] else None
    return summary


def benchmark(args):
    directory, output_dir = args.media_dir.resolve(), args.output_dir.resolve()
    if output_dir == directory or directory in output_dir.parents:
        raise ValueError('Comparison outputs must stay outside the lesson directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    before = frozen_hashes(directory)
    atomic_json(output_dir/'benchmark-source-before.json', before)
    deterministic_path = output_dir/'source-only.json'
    rule_contract = controlled_contract(directory)
    if not deterministic_path.is_file() or read_json(deterministic_path).get('benchmark_contract') != rule_contract:
        archive_attempt(deterministic_path)
        result = prepare(directory, args.video, provider=EmptyProposalProvider(),
                         output=output_dir/'_results/source-only-result.json', force=True)
        atomic_json(deterministic_path, {'track': 'deterministic', 'label': 'source-only',
            'conditions': {'track': 'deterministic', 'guard_method': METHOD}, 'runtime': {},
            'benchmark_contract': rule_contract, 'result': result})
        emit('deterministic_complete', edit_count=result['edit_count'], contributions=result['contribution_counts'])
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    sampler = GpuSampler(output_dir/f'gpu-samples-{stamp}.jsonl', args.sample_interval, args.gpu).start()
    lifecycle_records = []
    try:
        for model_label in args.models:
            profile = PROFILES[model_label]
            config = read_json(profile)
            if (config['server']['context_size'] != 24576 or config['request_defaults']['temperature'] != .1 or
                    config['request_defaults']['chat_template_kwargs'].get('enable_thinking') is not False or
                    config['request_defaults']['model'] != config['model_alias']):
                raise ValueError('Profiles must use exact declared benchmark context/temperature/thinking/model alias')
            sampler.phase = model_label+'-unload'
            for owned_profile in PROFILES.values():
                lifecycle_records.append(lifecycle(owned_profile, stop=True))
            baseline_phase, loaded_phase = model_label+'-baseline', model_label+'-loaded'
            sampler.phase = baseline_phase; time.sleep(1.2)
            sampler.phase = model_label+'-load'; started = time.perf_counter()
            lifecycle_records.append(lifecycle(profile))
            load_time = time.perf_counter()-started
            sampler.phase = loaded_phase; time.sleep(1.2)
            provider = CorrectionProvider(config_path=profile, managed_lifecycle=False,
                                          base=config['base_url'], model=config['model_alias'], timeout=args.request_timeout)
            provider.config['native_system'] = OFFICIAL_NATIVE_SYSTEM
            provider.config['native_system_policy'] = 'same_english_translation_of_upstream_recommended_system_both_models_temperature_0.1_not_upstream_deterministic_benchmark'
            controlled_path = output_dir/f'{model_label}-controlled.json'
            current_contract = controlled_contract(directory, profile)
            if args.native_only:
                emit('controlled_skipped_by_option', model=model_label)
            elif controlled_path.is_file() and not args.force_controlled:
                existing = read_json(controlled_path)
                if existing.get('benchmark_contract') != current_contract:
                    raise ValueError('Existing controlled run has different source/model profile; use separate output')
                emit('controlled_resumed', model=model_label, path=str(controlled_path))
            else:
                archive_attempt(controlled_path)
                phase = model_label+'-controlled'; sampler.phase = phase; started = time.perf_counter()
                result = prepare(directory, args.video, provider=provider,
                    output=output_dir/f'_results/{model_label}-controlled-result.json', force=True)
                runtime = sampler.summarize(phase, baseline_phase, loaded_phase, time.perf_counter()-started, load_time)
                envelope = {'schema_version': '1.0', 'label': model_label+'-controlled', 'track': 'controlled_json',
                    'conditions': conditions(profile, 'controlled_json', 5000), 'runtime': runtime,
                    'model_profile': str(profile), 'model_profile_sha256': sha256(profile),
                    'benchmark_contract': current_contract, 'request_timeout_sec': provider.timeout, 'result': result}
                atomic_json(controlled_path, envelope)
                emit('controlled_complete', model=model_label, status=result['status'], edit_count=result['edit_count'],
                     contributions=result['contribution_counts'], runtime=runtime)
            if not args.controlled_only:
                native_run(directory, output_dir/f'{model_label}-native.json', provider, model_label, profile,
                    args.native_max_tokens, sampler, baseline_phase, loaded_phase, load_time, model_label+'-native', args.retry_errors)
            if args.external_dir and not args.controlled_only:
                evaluation_dir = args.external_dir.resolve()/'evaluation'
                model_inputs = read_json(args.external_dir.resolve()/'model-inputs.json')
                for external in model_inputs['samples']:
                    external_media = evaluation_dir/external['id']
                    if not (external_media/'asr.json').is_file():
                        raise FileNotFoundError('External real ASR is not ready: '+str(external_media/'asr.json'))
                    native_run(external_media, external_media/f'{model_label}-native.json', provider,
                        model_label, profile, args.native_max_tokens, sampler, baseline_phase, loaded_phase, load_time,
                        model_label+'-external-'+external['id'], args.retry_errors)
            provider.close()
            lifecycle_records.append(lifecycle(profile, stop=True))
    finally:
        sampler.close()
        after = frozen_hashes(directory)
        audit = {'source_before': before, 'source_after': after, 'sources_unchanged': before == after,
                 'lifecycle_records': lifecycle_records, 'gpu_sampling_errors': sampler.errors,
                 'sample_file': str(sampler.path.resolve()), 'completed_at': utc_now()}
        atomic_json(output_dir/f'benchmark-audit-{stamp}.json', audit)
        if before != after:
            raise RuntimeError('Immutable lesson source bytes changed during benchmark')
    emit('benchmark_complete', source_unchanged=True, outputs=str(output_dir))
    generate_reports(directory, output_dir, args.external_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-dir', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--external-dir', type=Path)
    parser.add_argument('--models', nargs='+', choices=tuple(PROFILES), default=['qwen', 'cec4'])
    parser.add_argument('--native-max-tokens', type=int, default=1000)
    parser.add_argument('--sample-interval', type=float, default=.2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--force-controlled', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--request-timeout', type=float, default=900)
    tracks = parser.add_mutually_exclusive_group()
    tracks.add_argument('--controlled-only', action='store_true')
    tracks.add_argument('--native-only', action='store_true')
    args = parser.parse_args()
    if not (0 < args.sample_interval <= 1 and 600 <= args.native_max_tokens <= 1000):
        parser.error('Use a sampling interval in (0,1] seconds and native max_tokens between 600 and 1000')
    with IdleSuspendLease():
        benchmark(args)


if __name__ == '__main__':
    main()
