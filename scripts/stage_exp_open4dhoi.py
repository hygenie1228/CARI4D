#!/usr/bin/env python3
"""Build exp_dir/cari4d inputs for Open4D-HOI-style wild folders.

Open4D-HOI samples do not encode a BEHAVE/InterCap camera id in the folder name.
This stage uses the full folder basename as the video prefix and reads camera
intrinsics from the sample itself, not from another dataset.

Supported intrinsic files:
- ``<exp_dir>/human/human_params_gt.npz`` with an ``intrinsics`` array
  ``[fx, fy, cx, cy]`` or ``K``.
- ``<exp_dir>/intrinsics.pkl`` or ``<exp_dir>/camera.pkl`` with either ``K`` or
  ``fx``, ``fy``, ``cx``, ``cy`` keys.
- ``<exp_dir>/intrinsics.json`` or ``<exp_dir>/camera.json`` with the same keys.

You can also pass ``--fx --fy --cx --cy`` explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import sys
import tempfile

import cv2
import h5py
import joblib
import numpy as np
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))


def write_obj_aabb_center_at_origin(src_path: str, dst_path: str) -> np.ndarray:
    vmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    vmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    with open(src_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "v":
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vmin[0] = min(vmin[0], x)
                vmin[1] = min(vmin[1], y)
                vmin[2] = min(vmin[2], z)
                vmax[0] = max(vmax[0], x)
                vmax[1] = max(vmax[1], y)
                vmax[2] = max(vmax[2], z)
    if not np.isfinite(vmin).all():
        raise RuntimeError(f"no vertex lines (v x y z) found in {src_path}")
    center = (vmin + vmax) * 0.5

    def _fmt(x: float) -> str:
        return format(float(x), ".9g")

    with open(src_path, encoding="utf-8", errors="replace") as fin, open(
        dst_path, "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "v":
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                x -= center[0]
                y -= center[1]
                z -= center[2]
                rest = parts[4:] if len(parts) > 4 else []
                if rest:
                    fout.write(
                        "v {} {} {} {}\n".format(
                            _fmt(x), _fmt(y), _fmt(z), " ".join(rest)
                        )
                    )
                else:
                    fout.write(f"v {_fmt(x)} {_fmt(y)} {_fmt(z)}\n")
            else:
                fout.write(line.rstrip("\r\n") + "\n")
    return center


def write_masks_h5_local_then_copy(
    h5_path: str,
    video_prefix: str,
    kid: int,
    d_human: str,
    d_obj: str,
) -> None:
    cap_h = cv2.VideoCapture(d_human)
    cap_o = cv2.VideoCapture(d_obj)
    try:
        T = min(
            int(cap_h.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        if T <= 0:
            raise RuntimeError("could not read mask frame counts")

        fd, tmp = tempfile.mkstemp(suffix=".h5", prefix="cari4d_masks_")
        os.close(fd)
        try:
            with h5py.File(tmp, "w") as h5:
                for i in tqdm(range(T), desc="masks -> h5"):
                    rh, fh = cap_h.read()
                    ro, fo = cap_o.read()
                    if not rh or not ro or fh is None or fo is None:
                        raise RuntimeError(f"mask videos ended at frame {i}/{T}")
                    gh = cv2.cvtColor(fh, cv2.COLOR_BGR2GRAY) > 127
                    go = cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY) > 127
                    h5.create_dataset(
                        f"{video_prefix}/{i:06d}-k{kid}.person_mask.png",
                        data=gh,
                        compression="gzip",
                        compression_opts=3,
                    )
                    h5.create_dataset(
                        f"{video_prefix}/{i:06d}-k{kid}.obj_rend_mask.png",
                        data=go,
                        compression="gzip",
                        compression_opts=3,
                    )
            if osp.isfile(h5_path):
                os.remove(h5_path)
            shutil.copy2(tmp, h5_path)
        finally:
            if osp.isfile(tmp):
                os.remove(tmp)
    finally:
        cap_h.release()
        cap_o.release()


def _k_from_mapping(data: dict, source: str) -> np.ndarray:
    if "K" in data:
        K = np.asarray(data["K"], dtype=np.float32)
    elif "camera_matrix" in data:
        K = np.asarray(data["camera_matrix"], dtype=np.float32)
    else:
        missing = [k for k in ("fx", "fy", "cx", "cy") if k not in data]
        if missing:
            raise KeyError(f"{source} missing intrinsic keys: {', '.join(missing)}")
        fx, fy, cx, cy = (float(data[k]) for k in ("fx", "fy", "cx", "cy"))
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    if K.shape != (3, 3):
        raise ValueError(f"{source} intrinsic matrix must be 3x3, got {K.shape}")
    return K.astype(np.float32)


def _k_from_npz(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "K" in data.files:
        K = np.asarray(data["K"], dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"{path} K must be 3x3, got {K.shape}")
        return K
    if "intrinsics" not in data.files:
        raise KeyError(f"{path} missing 'intrinsics'")
    intr = np.asarray(data["intrinsics"], dtype=np.float32).reshape(-1)
    if intr.shape == (9,):
        K = intr.reshape(3, 3)
    elif intr.shape == (4,):
        fx, fy, cx, cy = (float(v) for v in intr)
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        raise ValueError(f"{path} intrinsics must be length 4 or 3x3/length 9, got {intr.shape}")
    return K.astype(np.float32)


def _load_intrinsics(exp_dir: str, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    gt_npz = osp.join(exp_dir, "human", "human_params_gt.npz")
    if osp.isfile(gt_npz):
        return _k_from_npz(gt_npz), gt_npz

    explicit = [args.fx, args.fy, args.cx, args.cy]
    if any(v is not None for v in explicit):
        if not all(v is not None for v in explicit):
            raise ValueError("--fx, --fy, --cx, and --cy must be provided together")
        K = np.array(
            [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return K, "command-line fx/fy/cx/cy"

    candidates = [
        args.intrinsics,
        osp.join(exp_dir, "intrinsics.pkl"),
        osp.join(exp_dir, "camera.pkl"),
        osp.join(exp_dir, "intrinsics.json"),
        osp.join(exp_dir, "camera.json"),
    ]
    for path in candidates:
        if not path:
            continue
        path = osp.abspath(path)
        if not osp.isfile(path):
            continue
        if path.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = joblib.load(path)
        if not isinstance(data, dict):
            raise TypeError(f"{path} must contain a dict-like intrinsic payload")
        return _k_from_mapping(data, path), path

    raise FileNotFoundError(
        "Open4D-HOI intrinsics not found. Add human/human_params_gt.npz, "
        "intrinsics.pkl/json, or camera.pkl/json under the sample folder, "
        "or pass --fx --fy --cx --cy."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--view_id", type=int, default=0)
    p.add_argument("--intrinsics", type=str, default="")
    p.add_argument("--fx", type=float)
    p.add_argument("--fy", type=float)
    p.add_argument("--cx", type=float)
    p.add_argument("--cy", type=float)
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate videos/masks/hy3d aligned obj even if outputs already exist.",
    )
    args = p.parse_args()

    exp_dir = osp.abspath(args.exp_dir)
    kid = int(args.view_id)
    video_prefix = osp.basename(exp_dir.rstrip("/"))

    src_color = osp.join(exp_dir, "video.mp4")
    d_human = osp.join(exp_dir, "processed", "human_mask.mp4")
    d_obj = osp.join(exp_dir, "processed", "object_mask.mp4")
    d_bgr = osp.join(exp_dir, "processed", "depth.mp4")

    work_root = osp.join(exp_dir, "cari4d")
    videos_dir = osp.join(work_root, "videos")
    color_out = osp.join(videos_dir, f"{video_prefix}.{kid}.color.mp4")
    pkl_out = color_out.replace(".mp4", ".pkl")
    depth_out = osp.join(videos_dir, f"{video_prefix}.{kid}.depth-reg.mp4")
    masks_dir = osp.join(work_root, "masks")
    h5_path = osp.join(masks_dir, f"{video_prefix}_masks_k{kid}.h5")
    src_model = osp.join(exp_dir, "object", "model.obj")
    hy_sub = osp.join(work_root, "hy3d_staged", f"{video_prefix}_export")
    dst_align = osp.join(hy_sub, f"{video_prefix}_align.obj")
    gender_path = osp.join(work_root, "nlf_gender.txt")

    if not args.force:
        core_ok = (
            osp.isfile(color_out)
            and osp.isfile(pkl_out)
            and osp.isfile(depth_out)
            and osp.isfile(h5_path)
        )
        has_src_mesh = osp.isfile(src_model)
        align_real = osp.isfile(dst_align) and not osp.islink(dst_align)
        mesh_gate = (not has_src_mesh) or align_real
        if core_ok and mesh_gate and osp.isfile(gender_path):
            print(f"skip (outputs exist): {work_root} -- use --force to regenerate")
            return

    assert osp.isfile(src_color), src_color
    assert osp.isfile(d_human) and osp.isfile(d_obj), (d_human, d_obj)
    assert osp.isfile(d_bgr), d_bgr

    K, intr_source = _load_intrinsics(exp_dir, args)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    os.makedirs(work_root, exist_ok=True)
    intr_pack_path = osp.join(work_root, "intrinsics.pkl")
    joblib.dump(
        {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": K, "kid": kid},
        intr_pack_path,
    )
    print(f"wrote {intr_pack_path} from {intr_source}: fx={fx} fy={fy} cx={cx} cy={cy}")

    if not osp.isfile(gender_path):
        with open(gender_path, "w", encoding="utf-8") as gf:
            gf.write("male\n")
        print(f"wrote default {gender_path} (edit to female if needed)")

    os.makedirs(videos_dir, exist_ok=True)
    shutil.copy2(src_color, color_out)
    joblib.dump({"fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": K, "kid": kid}, pkl_out)
    print(f"intrinsics mirrored -> {pkl_out}")

    shutil.copy2(d_bgr, depth_out)
    print("copied", d_bgr, "->", depth_out)

    os.makedirs(masks_dir, exist_ok=True)
    write_masks_h5_local_then_copy(h5_path, video_prefix, kid, d_human, d_obj)
    print("wrote", h5_path)

    os.makedirs(hy_sub, exist_ok=True)
    if osp.isfile(src_model):
        if osp.lexists(dst_align) or osp.isfile(dst_align):
            os.remove(dst_align)
        center = write_obj_aabb_center_at_origin(src_model, dst_align)
        print(
            "hy3d template (AABB center -> origin):",
            dst_align,
            f"(shift -[{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]) from",
            src_model,
        )
    else:
        print("warning: object/model.obj missing; fp_hy3d_2dir will need cari4d/hy3d_staged layout")

    print(f"video_prefix={video_prefix} view_id={kid} staged {color_out}")


if __name__ == "__main__":
    main()
