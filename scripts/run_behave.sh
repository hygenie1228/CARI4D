#!/usr/bin/env bash
exp_dir="experiments/debug3/Date03_Sub03_backpack_back_0"
cari4d="${exp_dir}/cari4d"
videos_dir="${cari4d}/videos"
masks_dir="${cari4d}/masks"
nlf_dir="${cari4d}/nlf"
nlf_ud_dir="${cari4d}/nlf-2unidepth"
fp_dir="${cari4d}/fp-hy3d3-unidepth"

base="$(basename "$exp_dir")"
if [[ "$base" =~ ^(.+)_([0-9]+)$ ]]; then
  video_prefix="${BASH_REMATCH[1]}"
  cam_id="${BASH_REMATCH[2]}"
else
  video_prefix="$base"
  cam_id=0
fi
video="${videos_dir}/${video_prefix}.${cam_id}.color.mp4"
# AABB-centered copy from stage_exp_behave (must match --hy3d_root mesh)
hy3d_mesh="${cari4d}/hy3d_staged/${video_prefix}_export/${video_prefix}_align.obj"

set -e
mkdir -p "${nlf_dir}" "${nlf_ud_dir}" "${fp_dir}"

# # Step 1: exp_dir video.mp4 · processed/* → cari4d/videos/*.color.pkl, depth-reg, masks/*.h5, hy3d_staged
# python scripts/stage_exp_behave.py --exp_dir "${exp_dir}"

# # Step 2: run NLF
# python prep/run_nlf_sepK.py --wild_video --data_source behave -o "${nlf_dir}" --masks_root "${masks_dir}" --video "${video}" -tstart 0

# # Step 3: align NLF to unidepth prediction
# python prep/align_nlf2unidepth.py --wild_video --data_source behave -tstart 0 -o "${nlf_ud_dir}" --masks_root "${masks_dir}" \
#   --nlf_path "${nlf_dir}" --video "${video}"

# # Step 4: run FoundationPose
# python prep/fp_hy3d_2dir.py --wild_video --viz_path x --vis_thres 0.5 --vis_thres2 0.5 --iou_thres2 0.3 --angular_velo 0.1 --occ_frames_allowed 30 --kid "${cam_id}" -tstart 0 \
#   --masks_root "${masks_dir}" --hy3d_root="${cari4d}/hy3d_staged" \
#   --video "${video}" -o "${fp_dir}"

# Step 5: run CoCoNet
python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-behave.json \
use_sel_view=True render_video=True identifier=_demo use_intermediate=True data_name=test-only \
hy3d_meshes_root="${hy3d_mesh}" \
masks_root="${masks_dir}" \
fp_root="${fp_dir}" \
nlf_root="${nlf_ud_dir}" \
video="${video}" cam_id="${cam_id}" \
outpath="${cari4d}/coconet"

# Step 6: run optimization
python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300 save_name=optv2 batch_size=128 opt_rot=True \
opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=2.0 w_j2d=0.03 opt_smpl_trans=False opt_betas=False \
view_id="${cam_id}" \
pth_file="${cari4d}/coconet/cari4d-release+step031397_demo/${video_prefix}.pth" \
video_root="${videos_dir}/" \
masks_root="${masks_dir}" \
hy3d_meshes_root="${hy3d_mesh}" outpath="${cari4d}/opt"

# Step 7: refined (and CoCoNet) checkpoints -> human/object_params*.npz under exp_dir
python scripts/pth2npz.py --exp_dir "${exp_dir}" \
  --pth "${cari4d}/opt/cari4d-release+step031397_demo-hy3d3-optv2/${video_prefix}.pth" \
  --coconet_pth "${cari4d}/coconet/cari4d-release+step031397_demo/${video_prefix}.pth" \
  --kid "${cam_id}"