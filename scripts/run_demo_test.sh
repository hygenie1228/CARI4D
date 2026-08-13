#!/usr/bin/env bash
# Wild demo folder under experiments/demo_test (Open4D-HOI-style layout).
# Set DEMO_VIEW to the camera/view index used in staged filenames (default 0).
# Intrinsics and gender are read from human/human_params_init.npz when present;
# otherwise stage_exp_open4dhoi.py uses the same fallbacks as Open4D-HOI samples.

set -euo pipefail

exp_dir="${1:-experiments/demo_test/skateboard-demo_1}"
DEMO_VIEW="${DEMO_VIEW:-0}"
cari4d="${exp_dir}/cari4d"
videos_dir="${cari4d}/videos"
masks_dir="${cari4d}/masks"
nlf_dir="${cari4d}/nlf"
nlf_ud_dir="${cari4d}/nlf-2unidepth"
fp_dir="${cari4d}/fp-hy3d3-unidepth"
init_npz="${exp_dir}/human/human_params_init.npz"

video_prefix="$(basename "${exp_dir}")"
video="${videos_dir}/${video_prefix}.${DEMO_VIEW}.color.mp4"
hy3d_mesh="${cari4d}/hy3d_staged/${video_prefix}_export/${video_prefix}_align.obj"

mkdir -p "${nlf_dir}" "${nlf_ud_dir}" "${fp_dir}"

INTRINSIC_ARGS=( )
if [[ -n "${DEMO_INTRINSICS:-}" ]]; then
  INTRINSIC_ARGS+=( --intrinsics "${DEMO_INTRINSICS}" )
fi
if [[ -n "${DEMO_FX:-}" || -n "${DEMO_FY:-}" || -n "${DEMO_CX:-}" || -n "${DEMO_CY:-}" ]]; then
  INTRINSIC_ARGS+=( --fx "${DEMO_FX:?}" --fy "${DEMO_FY:?}" --cx "${DEMO_CX:?}" --cy "${DEMO_CY:?}" )
elif [[ -f "${init_npz}" ]]; then
  mapfile -t _demo_iv < <(python3 -c "
import numpy as np
i = np.load('${init_npz}')['intrinsics']
print(i[0]); print(i[1]); print(i[2]); print(i[3])
")
  INTRINSIC_ARGS+=( --fx "${_demo_iv[0]}" --fy "${_demo_iv[1]}" --cx "${_demo_iv[2]}" --cy "${_demo_iv[3]}" )
fi

python scripts/stage_exp_open4dhoi.py --exp_dir "${exp_dir}" --view_id "${DEMO_VIEW}" "${INTRINSIC_ARGS[@]}"

if [[ -f "${init_npz}" ]]; then
  python3 -c "
import numpy as np
from pathlib import Path
g = str(np.load('${init_npz}')['gender']).strip().lower()
if g not in ('male', 'female'):
    g = 'male'
Path('${cari4d}/nlf_gender.txt').write_text(g + '\n')
print(f'wrote ${cari4d}/nlf_gender.txt from human_params_init.npz: {g}')
"
fi

if [[ -n "${DEMO_HUMAN_INIT_NPZ:-}" ]]; then
  echo "Seeding NLF from human init: ${DEMO_HUMAN_INIT_NPZ}"
  python scripts/seed_nlf_from_human_init.py \
    --exp_dir "${exp_dir}" \
    --human_init_npz "${DEMO_HUMAN_INIT_NPZ}" \
    --kid "${DEMO_VIEW}"
else
  python prep/run_nlf_sepK.py --wild_video --data_source open4dhoi -o "${nlf_dir}" --masks_root "${masks_dir}" --video "${video}" -tstart 0

  python prep/align_nlf2unidepth.py --wild_video --data_source open4dhoi -tstart 0 -o "${nlf_ud_dir}" --masks_root "${masks_dir}" \
    --nlf_path "${nlf_dir}" --video "${video}"
fi

python prep/fp_hy3d_2dir.py --wild_video --data_source open4dhoi --viz_path x --vis_thres 0.5 --vis_thres2 0.5 --iou_thres2 0.3 --angular_velo 0.1 --occ_frames_allowed 30 --kid "${DEMO_VIEW}" -tstart 0 \
  --masks_root "${masks_dir}" --hy3d_root="${cari4d}/hy3d_staged" \
  --video "${video}" -o "${fp_dir}"

python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-open4dhoi.json \
use_sel_view=True render_video=True identifier=_demo use_intermediate=True data_name=test-only \
wild_video=True data_source=open4dhoi \
hy3d_meshes_root="${hy3d_mesh}" \
masks_root="${masks_dir}" \
fp_root="${fp_dir}" \
nlf_root="${nlf_ud_dir}" \
video="${video}" cam_id="${DEMO_VIEW}" \
outpath="${cari4d}/coconet"

python scripts/render_coconet_in_debug.py \
  --exp_dir "${exp_dir}" \
  --kid "${DEMO_VIEW}" \
  --out "${exp_dir}/debug.mp4"

python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300 save_name=optv2 batch_size=128 opt_rot=True \
opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=2.0 w_j2d=0.03 opt_smpl_trans=False opt_betas=False \
wild_video=True data_source=open4dhoi \
view_id="${DEMO_VIEW}" \
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
  --kid "${DEMO_VIEW}" --data_source open4dhoi --full_basename_as_prefix \
  "${PTH_GENDER[@]}"
