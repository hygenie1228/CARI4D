video=data/cari4d-demo/behave/videos/Date03_Sub03_chairblack_debug.2.color.mp4

set -e
_base=$(basename "$video")
if [[ ! "$_base" =~ ^(.+)\.([0-9]+)\.color\.mp4$ ]]; then
  echo "demo2.sh: expected basename <prefix>.<cam_id>.color.mp4, got: ${_base}" >&2
  exit 1
fi
video_prefix="${BASH_REMATCH[1]}"
cam_id="${BASH_REMATCH[2]}"

# # # Step 1: run Unidepth estimation
# python prep/unidepth_behave.py --cameras ${cam_id} --data_source behave -o data/cari4d-demo/behave/videos/  --video ${video}

# Step 2: run NLF to estimate human pose (add flags in NLF_EXTRA, e.g. -tstart 0 --redo --nlf_debug_dump exp/debug --nlf_debug_max 64)
# python prep/run_nlf_sepK.py --data_source behave -o data/cari4d-demo/behave/nlf-smplh-gender-sepK --masks_root data/cari4d-demo/behave/masks/ --video "${video}" -tstart 0

# # # Step 3: align NLF to unidepth prediction
# python prep/align_nlf2unidepth.py --data_source behave -tstart 0 -o data/cari4d-demo/behave/nlf-smplh-gender-sepK-2unidepth --masks_root data/cari4d-demo/behave/masks \
# --nlf_path data/cari4d-demo/behave/nlf-smplh-gender-sepK --video ${video}

# # Step 4: run FoundationPose
# # -tend must stay at or below the shortest color/depth stream end (see .time.json); 47.2 overshoots this clip and breaks load_color_depth on the last frames.
# python prep/fp_hy3d_2dir.py --viz_path x --vis_thres 0.5 --vis_thres2 0.5 --iou_thres2 0.3 --angular_velo 0.1 --occ_frames_allowed 30 --kid "${cam_id}" -tstart 0   \
# --masks_root data/cari4d-demo/behave/masks/ --hy3d_root=data/cari4d-demo/meshes \
#  --video ${video} -o data/cari4d-demo/behave/fp-hy3d3-unidepth

# # # Step 5: run CoCoNet
python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-behave.json \
use_sel_view=True render_video=True identifier=_demo use_intermediate=True data_name=test-only \
hy3d_meshes_root=data/cari4d-demo/meshes \
masks_root=data/cari4d-demo/behave/masks/ \
fp_root=data/cari4d-demo/behave/fp-hy3d3-unidepth \
nlf_root=data/cari4d-demo/behave/nlf-smplh-gender-sepK-2unidepth \
video=${video} cam_id=${cam_id} \
outpath=output/coconet

# Step 6: run optimization (video_prefix must match CoCoNet .pth stem, e.g. Date03_Sub03_chairblack_debug.pth)
# echo "video=${video} video_prefix=${video_prefix} cam_id=${cam_id}"
# python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300  save_name=optv2 batch_size=192 opt_rot=True \
# opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=2.0 w_j2d=0.03 opt_smpl_trans=False opt_betas=False  \
# pth_file=output/coconet/cari4d-release+step031397_demo/${video_prefix}.pth  \
# video_root=data/cari4d-demo/behave/videos/ \
# packed_root=data/cari4d-demo/behave/packed \
# masks_root=data/cari4d-demo/behave/masks/  \
# hy3d_meshes_root=data/cari4d-demo/meshes outpath=output/opt view_id="${cam_id}"
# # Note: use batch size=64 if OOM for GPU with memory <=24GB, e.g. 4090.


# body_models path
# /home/namhj/.local/share/smplfitter/body_models