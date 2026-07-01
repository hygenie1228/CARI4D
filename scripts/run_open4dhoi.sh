#!/usr/bin/env bash
# Wild Open4D-HOI-style folder under experiments/open4dhoi.
# Set OPEN4DHOI_VIEW to the camera/view index used in staged filenames (default 0).
# Intrinsics are read by stage_exp_open4dhoi.py from the sample's
# human/human_params_gt.npz first, then intrinsics.pkl/json, camera.pkl/json,
# or explicit OPEN4DHOI_FX/FY/CX/CY environment variables.

set -euo pipefail

# FoundationPose nvdiffrast batches pose hypotheses; default 512 can OOM on 720p wild video.
export CARI4D_NVDIFFRAST_BS="${CARI4D_NVDIFFRAST_BS:-32}"
# Penetration-loss backward is memory-heavy; keep opt batches small on 48GB GPUs.
export OPEN4DHOI_OPT_BATCH_SIZE="${OPEN4DHOI_OPT_BATCH_SIZE:-32}"
export CARI4D_PEN_CHUNK="${CARI4D_PEN_CHUNK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exp_dir="${1:-experiments/open4dhoi/ab_wheel-20250901_203935}"
OPEN4DHOI_VIEW="${OPEN4DHOI_VIEW:-0}"
cari4d="${exp_dir}/cari4d"
videos_dir="${cari4d}/videos"
masks_dir="${cari4d}/masks"
nlf_dir="${cari4d}/nlf"
nlf_ud_dir="${cari4d}/nlf-2unidepth"
fp_dir="${cari4d}/fp-hy3d3-unidepth"

video_prefix="$(basename "${exp_dir}")"
video="${videos_dir}/${video_prefix}.${OPEN4DHOI_VIEW}.color.mp4"
hy3d_mesh="${cari4d}/hy3d_staged/${video_prefix}_export/${video_prefix}_align.obj"

mkdir -p "${nlf_dir}" "${nlf_ud_dir}" "${fp_dir}"

INTRINSIC_ARGS=( )
if [[ -n "${OPEN4DHOI_INTRINSICS:-}" ]]; then
  INTRINSIC_ARGS+=( --intrinsics "${OPEN4DHOI_INTRINSICS}" )
fi
if [[ -n "${OPEN4DHOI_FX:-}" || -n "${OPEN4DHOI_FY:-}" || -n "${OPEN4DHOI_CX:-}" || -n "${OPEN4DHOI_CY:-}" ]]; then
  INTRINSIC_ARGS+=( --fx "${OPEN4DHOI_FX:?}" --fy "${OPEN4DHOI_FY:?}" --cx "${OPEN4DHOI_CX:?}" --cy "${OPEN4DHOI_CY:?}" )
fi

# Open4D-HOI folder names do not encode a reliable trailing Kinect id, so the
# whole basename remains the video prefix and camera intrinsics come from the sample.
python scripts/stage_exp_open4dhoi.py --exp_dir "${exp_dir}" --view_id "${OPEN4DHOI_VIEW}" "${INTRINSIC_ARGS[@]}"

python prep/run_nlf_sepK.py --wild_video --data_source open4dhoi -o "${nlf_dir}" --masks_root "${masks_dir}" --video "${video}" -tstart 0

python prep/align_nlf2unidepth.py --wild_video --data_source open4dhoi -tstart 0 -o "${nlf_ud_dir}" --masks_root "${masks_dir}" \
  --nlf_path "${nlf_dir}" --video "${video}"

fp_pkl="${fp_dir}/${video_prefix}_all.pkl"
OPEN4DHOI_ZFAR_FALLBACK="${OPEN4DHOI_ZFAR_FALLBACK:-15}"
ERROR_LIST="error_list.txt"

run_fp_hy3d() {
  local zfar_arg=( )
  if [[ -n "${1:-}" ]]; then
    zfar_arg=( --zfar "$1" )
  fi
  python prep/fp_hy3d_2dir.py --wild_video --data_source open4dhoi --viz_path x --vis_thres 0.5 --vis_thres2 0.5 --iou_thres2 0.3 --angular_velo 0.1 --occ_frames_allowed 30 --kid "${OPEN4DHOI_VIEW}" -tstart 0 \
    --masks_root "${masks_dir}" --hy3d_root="${cari4d}/hy3d_staged" \
    --video "${video}" -o "${fp_dir}" \
    "${zfar_arg[@]}"
}

clear_fp_partial() {
  rm -f "${fp_dir}/${video_prefix}"_*.pkl "${fp_dir}/${video_prefix}"*filter*.mp4 2>/dev/null || true
}

set +e
run_fp_hy3d
fp_rc=$?
set -e

if [[ ! -f "${fp_pkl}" ]] && grep -qxF "${video_prefix}" "${ERROR_LIST}" 2>/dev/null; then
  echo "FP failed due to small zfar for ${video_prefix}; retrying with --zfar ${OPEN4DHOI_ZFAR_FALLBACK}" >&2
  clear_fp_partial
  run_fp_hy3d "${OPEN4DHOI_ZFAR_FALLBACK}"
elif [[ "${fp_rc}" -ne 0 ]] && [[ ! -f "${fp_pkl}" ]]; then
  echo "FP failed (exit ${fp_rc}) for ${video_prefix}" >&2
  exit "${fp_rc}"
fi

if [[ ! -f "${fp_pkl}" ]]; then
  echo "Missing FP output: ${fp_pkl}" >&2
  exit 1
fi

python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-open4dhoi.json \
use_sel_view=True render_video=True identifier=_demo use_intermediate=True data_name=test-only \
wild_video=True data_source=open4dhoi \
hy3d_meshes_root="${hy3d_mesh}" \
masks_root="${masks_dir}" \
fp_root="${fp_dir}" \
nlf_root="${nlf_ud_dir}" \
video="${video}" cam_id="${OPEN4DHOI_VIEW}" \
outpath="${cari4d}/coconet"

python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300 save_name=optv2 batch_size="${OPEN4DHOI_OPT_BATCH_SIZE}" opt_rot=True \
opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=2.0 w_j2d=0.03 opt_smpl_trans=False opt_betas=False \
wild_video=True data_source=open4dhoi \
view_id="${OPEN4DHOI_VIEW}" \
pth_file="${cari4d}/coconet/cari4d-release+step031397_demo/${video_prefix}.pth" \
video_root="${videos_dir}/" \
masks_root="${masks_dir}" \
hy3d_meshes_root="${hy3d_mesh}" outpath="${cari4d}/opt"

PTH_GENDER=( )
if [[ -f "${cari4d}/nlf_gender.txt" ]]; then
  PTH_GENDER=( --gender "$(tr -d '\r\n' < "${cari4d}/nlf_gender.txt")" )
fi
python scripts/pth2npz.py --exp_dir "${exp_dir}" \
  --pth "${cari4d}/opt/cari4d-release+step031397_demo-hy3d3-optv2/${video_prefix}.pth" \
  --coconet_pth "${cari4d}/coconet/cari4d-release+step031397_demo/${video_prefix}.pth" \
  --kid "${OPEN4DHOI_VIEW}" --data_source open4dhoi --full_basename_as_prefix \
  "${PTH_GENDER[@]}"
