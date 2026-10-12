#!/usr/bin/env python3
"""Run FPD-Drive closed-loop evaluation on prepared HUGSIM scenes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading

import yaml

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario-root', type=Path, default=os.getenv('SCENARIO_ROOT', ROOT / 'configs/benchmark'))
    parser.add_argument('--datasets', nargs='+', default=os.getenv('DATASETS', '').split() or None)
    parser.add_argument('--gpus', default=os.getenv('GPUS', os.getenv('CUDA_VISIBLE_DEVICES', '0')))
    parser.add_argument('--workers-per-gpu', type=int, default=int(os.getenv('WORKERS_PER_GPU', '1')))
    parser.add_argument('--output-dir', type=Path, default=os.getenv('OUT_ROOT', ROOT / 'outputs/FPD_Drive'))
    parser.add_argument('--checkpoint', type=Path, default=os.getenv('FPD_DRIVE_CKPT'))
    parser.add_argument('--model-config', type=Path, default=ROOT / 'configs/evaluation/FPD_Drive.yaml')
    parser.add_argument('--timeout', type=float, default=1800, help='Maximum seconds per scenario')
    parser.add_argument('--check', action='store_true', help='Validate paths without starting evaluation')
    args = parser.parse_args()
    if args.workers_per_gpu < 1 or args.timeout <= 0:
        parser.error('workers-per-gpu and timeout must be positive')
    gpus = [item.strip() for item in args.gpus.split(',') if item.strip()]
    if not gpus or '-1' in gpus:
        parser.error('Provide at least one available GPU in --gpus')
    for variable in ('HUGSIM_DATA_ROOT', 'FPD_DRIVE_ROOT', 'NAVSIM_V1_ROOT', 'BACKBONE_WEIGHTS', 'FPD_DRIVE_PYTHON'):
        if not os.getenv(variable):
            parser.error(f'Set {variable}; see README.md')
    env = os.environ.copy()
    env.update(HUGSIM_ROOT=str(ROOT), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'sim'), env.get('PYTHONPATH', '')])
    project = Path(env['FPD_DRIVE_ROOT']).expanduser().resolve()
    checkpoint = (args.checkpoint or project / 'checkpoints/release/base_v1.pt').expanduser().resolve()
    required = [checkpoint, args.model_config, Path(env['BACKBONE_WEIGHTS']),
                Path(env['FPD_DRIVE_PYTHON']), project / 'fpd_drive/model.py',
                Path(env['NAVSIM_V1_ROOT']) / 'navsim/agents/drivoR/drivor_features.py',
                Path(env['HUGSIM_DATA_ROOT']) / '3DRealCar', args.scenario_root]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error('Missing paths:\n' + '\n'.join(missing))
    env.update(FPD_DRIVE_ROOT=str(project), FPD_DRIVE_CKPT=str(checkpoint),
               FPD_DRIVE_CFG=str(args.model_config.resolve()))
    datasets = args.datasets or [name for name in ('nuscenes', 'waymo', 'pandaset', 'kitti360')
                                 if (args.scenario_root / name).is_dir()]
    jobs = queue.Queue()
    outputs = []
    for dataset in datasets:
        for name in (f'{dataset}_base.yaml', f'{dataset}_camera.yaml', 'kinematic.yaml'):
            if not (ROOT / 'configs/sim' / name).is_file():
                parser.error(f'Missing simulator config: {name}')
        scenarios = sorted((args.scenario_root / dataset).glob('*.yaml'))
        if not scenarios:
            parser.error(f'No scenario YAMLs for {dataset} under {args.scenario_root}')
        for path in scenarios:
            cfg = yaml.safe_load(path.read_text())
            scene, mode = str(cfg['scene_name']), str(cfg['mode'])
            if any('/' in value or '\\' in value or value in ('.', '..') for value in (scene, mode)):
                parser.error(f'Invalid scene_name or mode in {path}')
            scene_root = Path(env['HUGSIM_DATA_ROOT']) / 'scenes' / dataset / scene
            for asset in ('cfg.yaml', 'scene.pth', 'ground_param.pkl'):
                if not (scene_root / asset).is_file():
                    parser.error(f'Missing scene asset: {scene_root / asset}')
            output = args.output_dir.resolve() / f'{dataset}_FPD_Drive' / f'{scene}_{mode}'
            if output in outputs:
                parser.error(f'Duplicate scenario output: {output}')
            outputs.append(output)
            if not (output / 'eval.json').is_file():
                jobs.put((dataset, path.resolve(), output))
    if not outputs:
        parser.error('No scenarios found')
    print(f'Scenarios={len(outputs)} pending={jobs.qsize()} GPUs={gpus} workers/GPU={args.workers_per_gpu}', flush=True)
    print(f'Checkpoint: {checkpoint}', flush=True)
    if args.check:
        print('Input paths checked; no evaluation started.')
        return 0

    stopped = threading.Event()
    lock = threading.Lock()
    processes = set()

    def terminate(process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def stop(signum, frame):
        stopped.set()
        with lock:
            for process in list(processes):
                terminate(process)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def worker(gpu):
        while not stopped.is_set():
            try:
                dataset, path, output = jobs.get_nowait()
            except queue.Empty:
                return
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(ROOT / 'closed_loop.py'), '--scenario_path', str(path),
                       '--base_path', str(ROOT / 'configs/sim' / f'{dataset}_base.yaml'),
                       '--camera_path', str(ROOT / 'configs/sim' / f'{dataset}_camera.yaml'),
                       '--kinematic_path', str(ROOT / 'configs/sim/kinematic.yaml'),
                       '--output_dir', str(output), '--ad_cuda', gpu]
            print(f'[GPU {gpu}] {dataset}/{path.stem}', flush=True)
            with (output / 'simulation.log').open('a') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with lock:
                    processes.add(process)
                try:
                    process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    print(f'Timeout: {path}', file=sys.stderr, flush=True)
                finally:
                    terminate(process)
                    with lock:
                        processes.discard(process)
            jobs.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus) * args.workers_per_gpu) as executor:
        workers = [executor.submit(worker, gpu) for gpu in gpus for _ in range(args.workers_per_gpu)]
        for future in workers:
            future.result()
    failed = [str(path) for path in outputs if not (path / 'eval.json').is_file()]
    if failed:
        print('Incomplete scenarios:\n' + '\n'.join(failed), file=sys.stderr)
        return 1
    subprocess.run([sys.executable, str(ROOT / 'aggregate_hugsim.py'), '--root', str(args.output_dir.resolve())], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
