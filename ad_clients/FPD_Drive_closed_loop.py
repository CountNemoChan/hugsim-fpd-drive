"""FPD-Drive client for HUGSIM, using the public inference package and native Top1."""
import argparse
import os
import sys
import pickle
import math
import glob
import json
from pathlib import Path

import torch
import numpy as np
import cv2
from omegaconf import OmegaConf

# HUGSIM camera names map to the four NAVSIM camera slots used by FPD-Drive.
nusc_cameras = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK"]
nuplan_cameras = ["cam_f0", "cam_r0", "cam_l0", "cam_b0"]

# HUGSIM high-level command: 0=right, 1=left, 2=forward.
# NAVSIM driving_command one-hot: index 0=left, 1=straight, 2=right (idx 3 = unknown).
# So NAVSIM index = HUGSIM2NAVSIM_CMD[hugsim_cmd].
HUGSIM2NAVSIM_CMD = [2, 0, 1]

# Keys the model consumes; moved to device (+bf16 for floats) each frame.
_CUDA_KEYS = ("image", "cam_K", "world_2_cam", "ego_status")


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def create_cam(intrinsic, l2c, image):
    from navsim.common.dataclasses import Camera
    fovx, fovy = intrinsic["fovx"], intrinsic["fovy"]
    h, w = intrinsic["H"], intrinsic["W"]
    K = np.eye(3)
    K[0, 0], K[1, 1] = fov2focal(fovx, w), fov2focal(fovy, h)
    K[0, 2], K[1, 2] = intrinsic["cx"], intrinsic["cy"]
    c2l = np.linalg.inv(l2c)
    return Camera(
        image=image,
        sensor2lidar_rotation=c2l[:3, :3],
        sensor2lidar_translation=c2l[:3, 3],
        intrinsics=K,
    )


def save_frame(raw_image, traj_poses, cam_params, save_fn):
    plan_traj = np.stack(
        [traj_poses[:, 0], 1.4 * np.ones_like(traj_poses[:, 2]), traj_poses[:, 1]],
        axis=1,
    )
    idict = cam_params["CAM_FRONT"]["intrinsic"]
    K = np.eye(4)
    K[0, 0] = fov2focal(idict["fovx"], idict["W"])
    K[1, 1] = fov2focal(idict["fovy"], idict["H"])
    K[0, 2], K[1, 2] = idict["cx"], idict["cy"]
    uvz = (K[:3, :3] @ plan_traj.T).T
    uv = (uvz[:, :2] / uvz[:, 2][:, None]).astype(int)
    images = []
    for cam in nusc_cameras:
        im = raw_image[cam]
        if cam == "CAM_FRONT":
            for p in uv:
                im = cv2.circle(im, p, radius=5, color=(192, 152, 7), thickness=-1)
        images.append(im)
    images = cv2.cvtColor(cv2.hconcat(images), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_fn + ".jpg", images)


def to_video(folder_path, fps=5):
    imgs = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    if not imgs:
        return
    frames = [cv2.imread(p) for p in imgs]
    h, w = frames[0].shape[:2]
    mp4 = os.path.join(folder_path, "video.mp4")
    if os.path.exists(mp4):
        os.remove(mp4)
    out = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def load_model(cfg_path, ckpt_path, device, dtype):
    project = Path(os.environ["FPD_DRIVE_ROOT"]).expanduser().resolve()
    navsim_root = Path(os.environ["NAVSIM_V1_ROOT"]).expanduser().resolve()
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(navsim_root))
    from fpd_drive.model import FPDDriveModel
    from fpd_drive.navsim import _load_checkpoint_state_dict
    from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder as FPDFeatureBuilder

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    if cfg.evaluation.selection_method != "native_top1":
        raise ValueError("HUGSIM uses the checkpoint-native Top1 policy")

    class HUGSIMPlanner(FPDDriveModel):
        def _score_proposals(self, features, scene_features, proposals, stage_ids):
            status = features["ego_status"][:, -1]
            ego = self.fpd_planner.hist_encoding(status)[:, None]
            batch, count = proposals.shape[:2]
            embedded = self.fpd_planner.pos_embed(proposals.reshape(batch, count, -1).detach())
            tokens = self.fpd_planner.scorer_attention(embedded, scene_features) + ego
            logits, *_ = self.fpd_planner.scorer(proposals, tokens)
            w = self.evaluation.ranking_weights
            score = (
                w.noc * logits["no_at_fault_collisions"].sigmoid().log()
                + w.dac * logits["drivable_area_compliance"].sigmoid().log()
                + w.ddc * logits["driving_direction_compliance"].sigmoid().log()
                + (w.ttc * logits["time_to_collision_within_bound"].sigmoid()
                   + w.ep * logits["ego_progress"].sigmoid()
                   + w.comfort * logits["comfort"].sigmoid()).log()
            )
            return proposals[torch.arange(batch, device=proposals.device), score.argmax(dim=1)]

    model = HUGSIMPlanner(cfg)
    state, _ = _load_checkpoint_state_dict(ckpt_path)
    weights = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        # Also accept the original HUGSIM experiment's checkpoint key schema.
        if key.startswith("drivor_model."):
            key = "fpd_planner." + key[len("drivor_model."):]
        elif key.startswith("residual_refiner."):
            key = "refiner." + key[len("residual_refiner."):]
        if key.startswith(("fpd_planner.", "refiner.")):
            weights[key] = value
    model.load_state_dict(weights, strict=True)
    model.to(device=device, dtype=dtype).eval()
    print(f"[FPD-Drive] checkpoint={ckpt_path} selection=native_top1 candidates=64 seed=27", flush=True)
    return model, FPDFeatureBuilder(cfg.model.config), cfg


@torch.inference_mode()
def main(args):
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, feature_builder, cfg = load_model(args.config, args.checkpoint, device, dtype)
    from fpd_drive.navsim import _move_features
    from navsim.common.dataclasses import AgentInput, EgoStatus, Cameras, Camera, Lidar
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "inference_config.json"), "w") as f:
        json.dump({
            "method": "FPD_Drive", "checkpoint": args.checkpoint,
            "model_config": args.config, "dtype": str(dtype),
            "trajectory_selection": "top1",
            "proposal_num": cfg.model.config.proposal_num,
            "num_inference_steps": cfg.joint.refiner.num_inference_steps,
            "eval_noise_seeds": [cfg.joint.refiner.eval_noise_seed],
            "ranking_weights": {key: cfg.evaluation.ranking_weights[key] for key in ("noc", "dac", "ddc", "ttc", "ep", "comfort")},
        }, f, indent=2)

    # One stable scenario token for the whole rollout, mirroring NAVSIM (one
    # token per scene). The DDIM sampler seeds its initial noise from this token,
    # so every frame uses the same deterministic noise realization instead of a
    # fresh one each frame -- no per-frame noise jitter in the refiner output.
    scene_tag = os.path.basename(os.path.normpath(args.output))

    obs_pipe = os.path.join(args.output, "obs_pipe")
    plan_pipe = os.path.join(args.output, "plan_pipe")
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    if not os.path.exists(plan_pipe):
        os.mkfifo(plan_pipe)
    print("[FPD_Drive] ready for receiving", flush=True)

    save_viz = os.environ.get("HUGSIM_SAVE_VIZ", "0") == "1"
    cnt = 0
    out_folder = os.path.join(args.output, "FPD_Drive")
    if save_viz:
        os.makedirs(out_folder, exist_ok=True)

    autocast = torch.amp.autocast(
        device_type="cuda", dtype=dtype, enabled=dtype != torch.float32
    )

    while True:
        with open(obs_pipe, "rb") as pipe:
            raw_data = pickle.loads(pipe.read())

        if raw_data == "Done":
            if save_viz:
                print("[FPD_Drive] done; writing video", flush=True)
                to_video(out_folder)
            else:
                print("[FPD_Drive] done", flush=True)
            return

        obs, info = raw_data

        imgs, raw_imgs = {}, {}
        for cam in nusc_cameras:
            im = obs["rgb"][cam]
            raw_imgs[cam] = im
            imgs[cam] = im

        velo, acc = np.zeros(2), np.zeros(2)
        command = np.zeros(4)
        # Use the simulator's actual high-level command instead of hard-coding straight,
        # remapping HUGSIM's convention to NAVSIM's driving_command one-hot.
        hug_cmd = int(info["command"])
        command[HUGSIM2NAVSIM_CMD[hug_cmd]] = 1

        # NAVSIM builds ego_pose as (x, y, yaw) RELATIVE to the current frame
        # (convert_absolute_to_relative_se2_array in navsim dataclasses), so the
        # current frame -- the only one FPD-Drive consumes when
        # full_history_status=False -- is (0, 0, 0). The previous
        # `OPENCV2IMU @ info["ego_pos"]` fed the ego's absolute world position (with z
        # where yaw belongs), a large OOD signal that corrupts the ego token seeding
        # every trajectory proposal.
        ego_pose = np.zeros(3, dtype=np.float32)
        yaw = -info["ego_steer"]
        fwd_v = info["ego_velo"]
        fwd_a = info["accelerate"]
        velo[0], velo[1] = fwd_v * np.cos(yaw), fwd_v * np.sin(yaw)
        acc[0], acc[1] = fwd_a * np.cos(yaw), fwd_a * np.sin(yaw)
        ego_status = EgoStatus(ego_pose, velo, acc, command)

        cameras = Cameras(*[Camera() for _ in range(8)])
        for nusc_cam, nuplan_cam in zip(nusc_cameras, nuplan_cameras):
            cp = info["cam_params"][nusc_cam]
            setattr(cameras, nuplan_cam, create_cam(cp["intrinsic"], cp["l2c"], imgs[nusc_cam]))

        lidar = Lidar(np.zeros((6, 100), dtype=np.float32))
        agent_input = AgentInput([ego_status], [cameras], [lidar])

        features = feature_builder.compute_features(agent_input)
        for k in _CUDA_KEYS:
            features[k] = features[k].unsqueeze(0)
        features = _move_features(features, device=device, dtype=dtype)

        with autocast:
            pred_traj = model(features, scenario_tokens=[scene_tag])
        pred_traj = pred_traj.detach().float().cpu().numpy()[0]  # (8, 3)

        way_points = pred_traj[:, [1, 0, 2]]
        way_points[:, 0] *= -1

        if save_viz:
            save_frame(raw_imgs, way_points, info["cam_params"], os.path.join(out_folder, f"{cnt:04d}"))

        with open(plan_pipe, "wb") as pipe:
            pipe.write(pickle.dumps(way_points[:, :2]))
        cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPD_Drive closed-loop client for HUGSIM")
    parser.add_argument("--output", type=str, required=True, help="scenario output dir (holds the pipes)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    main(parser.parse_args())
