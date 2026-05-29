#!/usr/bin/env bash
# Wild InterCap-style folder under experiments/intercap_test (no Kinect id in basename).
# Set ICAP_VIEW to the InterCap camera index 0–5 (default 0). This replaces BEHAVE's trailing _<id>.

set -euo pipefail

exp_dir="${1:-experiments/intercap_test/10_01_Seg_0_0}"
ICAP_VIEW="${ICAP_VIEW:-0}"
cari4d="${exp_dir}/cari4d"
videos_dir="${cari4d}/videos"
masks_dir="${cari4d}/masks"
nlf_dir="${cari4d}/nlf"
nlf_ud_dir="${cari4d}/nlf-2unidepth"
fp_dir="${cari4d}/fp-hy3d3-unidepth"

video_prefix="$(basename "${exp_dir}")"
video="${videos_dir}/${video_prefix}.${ICAP_VIEW}.color.mp4"
hy3d_mesh="${cari4d}/hy3d_staged/${video_prefix}_export/${video_prefix}_align.obj"

mkdir -p "${nlf_dir}" "${nlf_ud_dir}" "${fp_dir}"

python scripts/stage_exp_intercap.py --exp_dir "${exp_dir}" --view_id "${ICAP_VIEW}"

python prep/run_nlf_sepK.py --wild_video --data_source intercap -o "${nlf_dir}" --masks_root "${masks_dir}" --video "${video}" -tstart 0

python prep/align_nlf2unidepth.py --wild_video --data_source intercap -tstart 0 -o "${nlf_ud_dir}" --masks_root "${masks_dir}" \
  --nlf_path "${nlf_dir}" --video "${video}"

# Use behave data_source so FP loads the staged HY3D OBJ via BehaveHy3D2DirFPRunner.get_template_file
python prep/fp_hy3d_2dir.py --wild_video --data_source behave --viz_path x --vis_thres 0.5 --vis_thres2 0.5 --iou_thres2 0.3 --angular_velo 0.1 --occ_frames_allowed 30 --kid "${ICAP_VIEW}" -tstart 0 \
  --masks_root "${masks_dir}" --hy3d_root="${cari4d}/hy3d_staged" \
  --video "${video}" -o "${fp_dir}"

python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-intercap.json \
use_sel_view=True render_video=True identifier=_demo use_intermediate=True data_name=test-only \
wild_video=True data_source=intercap \
hy3d_meshes_root="${hy3d_mesh}" \
masks_root="${masks_dir}" \
fp_root="${fp_dir}" \
nlf_root="${nlf_ud_dir}" \
video="${video}" cam_id="${ICAP_VIEW}" \
outpath="${cari4d}/coconet"

python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300 save_name=optv2 batch_size=128 opt_rot=True \
opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=2.0 w_j2d=0.03 opt_smpl_trans=False opt_betas=False \
wild_video=True data_source=intercap \
view_id="${ICAP_VIEW}" \
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
  --kid "${ICAP_VIEW}" --data_source intercap --full_basename_as_prefix \
  "${PTH_GENDER[@]}"
