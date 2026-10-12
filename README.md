# FPD-Drive evaluation on HUGSIM

Closed-loop evaluation of [FPD-Drive](https://github.com/CountNemoChan/FPD-Drive)
using the [HUGSIM](https://github.com/hyzhou404/HUGSIM) simulator.
This repository contains inference, rendering, control, and scoring code for
prepared scenes.

## Setup

Use Linux and an NVIDIA GPU. Create a simulator environment:

```bash
git clone https://github.com/CountNemoChan/hugsim-fpd-drive.git
cd hugsim-fpd-drive
conda create -n hugsim-fpd python=3.11 -y
conda activate hugsim-fpd
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation "git+https://github.com/hyzhou404/HUGSIM_splat.git"
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
python -m pip install -e ./sim
```

The CUDA extensions require a compatible CUDA toolkit and C++ compiler.
Install FPD-Drive in a separate environment following its
[inference guide](https://github.com/CountNemoChan/FPD-Drive#quick-start), then set:

```bash
export HUGSIM_DATA_ROOT=/path/to/hugsim-assets
export FPD_DRIVE_ROOT=/path/to/FPD-Drive
export FPD_DRIVE_PYTHON=/path/to/envs/fpd-drive/bin/python
export NAVSIM_V1_ROOT=/path/to/navsim-v1-with-features
export BACKBONE_WEIGHTS=/path/to/dinov2/model.safetensors
```

`HUGSIM_DATA_ROOT` contains `scenes/<dataset>/<scene>/` and `3DRealCar/`.
Each prepared scene needs `cfg.yaml`, `scene.pth`, `ground_param.pkl`, and its
associated dynamic-object/unicycle files. Obtain these assets and full scenario
lists from the [upstream HUGSIM project](https://github.com/hyzhou404/HUGSIM).
For map-aware nuScenes scenarios, set `HUGSIM_NUSCENES_MAP_ROOT` to the prepared
trajdata map cache; its default is `$HUGSIM_DATA_ROOT/nusc_map_cache`.

## Evaluate

```bash
# Check the bundled nuScenes examples without running the simulator.
python run_evaluation.py --datasets nuscenes --check

# Evaluate a full external scenario set with parallel workers.
python run_evaluation.py --scenario-root /path/to/scenarios \
  --gpus 0,1 --workers-per-gpu 2
```

The default checkpoint is `$FPD_DRIVE_ROOT/checkpoints/release/base_v1.pt` from
[Hugging Face](https://huggingface.co/countnemochan/FPD-Drive-ckpt).
Use `--checkpoint /path/to/model.pt` to select another compatible checkpoint.
HUGSIM uses **native Top1 selection, 64 candidates, 12 DDIM steps, and seed 27**;
this selection policy differs from the NAVSIM evaluation presets. Original
experiment checkpoint keys are also accepted. The historical HUGSIM result used
an epoch-5/step-12570 checkpoint; the default release checkpoint is different.

Completed scenarios are skipped on subsequent runs. Results and logs are saved
under `outputs/FPD_Drive/<dataset>_FPD_Drive/<scene>_<mode>/`.
Set `HUGSIM_SAVE_VIZ=1` to save rollout videos. Aggregate route completion (RC)
and HD-Score, both reported on a 0–100 scale, with:

```bash
python aggregate_hugsim.py --root outputs/FPD_Drive
```

The bundled YAML files are examples, not the complete 345-scenario benchmark.
A full benchmark requires its matching scene assets and scenario list.

## Acknowledgements

Thanks to [HUGSIM](https://github.com/hyzhou404/HUGSIM),
[DrivoR](https://github.com/valeoai/DrivoR), and
[NAVSIM](https://github.com/autonomousvision/navsim) for their open-source work.
The upstream HUGSIM MIT license is preserved in [LICENSE](LICENSE).
