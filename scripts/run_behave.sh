#!/usr/bin/env bash
exp_dir="experiments/behave/Date03_Sub03_chairblack_lift_2"
cari4d="${exp_dir}/cari4d"
videos_dir="${cari4d}/videos"
masks_dir="${cari4d}/masks"
packed_dir="${cari4d}/packed"
nlf_dir="${cari4d}/nlf"
nlf_ud_dir="${cari4d}/nlf-2unidepth"
fp_dir="${cari4d}/fp-hy3d3-unidepth"

base="$(basename "$exp_dir")"
if [[ "$base" =~ ^(.+)_[0-9]+$ ]]; then
  video_prefix="${BASH_REMATCH[1]}"
else
  video_prefix="$base"
fi
video="${videos_dir}/${video_prefix}.0.color.mp4"

set -e
mkdir -p "${packed_dir}" "${nlf_dir}" "${nlf_ud_dir}" "${fp_dir}"

# Step 1: exp_dir의 video.mp4 · processed/* → cari4d/videos/*.color.pkl, depth-reg, masks/*.h5, hy3d_staged
python scripts/stage_exp_behave.py --exp_dir "${exp_dir}"

# Step 2: run NLF
python prep/run_nlf_sepK.py --wild_video --data_source behave -o "${nlf_dir}" --masks_root "${masks_dir}" --video "${video}" -tstart 0

# Step 3: align NLF to unidepth prediction
python prep/align_nlf2unidepth.py --wild_video --data_source behave -tstart 0 -o "${nlf_ud_dir}" --masks_root "${masks_dir}" \
  --nlf_path "${nlf_dir}" --video "${video}" --packed_root "${packed_dir}"

# Step 4: run FoundationPose
python prep/fp_hy3d_2dir.py --wild_video --viz_path x --vis_thres 0.5 --vis_thres2 0.5 --iou_thres2 0.3 --angular_velo 0.1 --occ_frames_allowed 30 --kid 0 -tstart 0 \
  --masks_root "${masks_dir}" --hy3d_root="${cari4d}/hy3d_staged" \
  --video "${video}" -o "${fp_dir}"

# # Step 5: run CoCoNet
python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-behave.json \
use_sel_view=True render_video=True identifier=_demo use_intermediate=True data_name=test-only \
hy3d_meshes_root=data/cari4d-demo/meshes \
masks_root="${masks_dir}" \
fp_root="${fp_dir}" \
nlf_root="${nlf_ud_dir}" \
video="${video}" cam_id=0 \
outpath="${cari4d}/coconet"

# # Step 6: run optimization
video_prefix=$(basename "$video" | cut -d. -f1)
echo $video_prefix
python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300 save_name=optv2 batch_size=192 opt_rot=True \
opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=2.0 w_j2d=0.03 opt_smpl_trans=False opt_betas=False \
pth_file="${cari4d}/coconet/cari4d-release+step031397_demo/${video_prefix}.pth" \
video_root="${videos_dir}/" \
packed_root="${packed_dir}" \
masks_root="${masks_dir}" \
hy3d_meshes_root=data/cari4d-demo/meshes outpath="${cari4d}/opt"
