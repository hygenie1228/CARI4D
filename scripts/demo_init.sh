video=data/cari4d-demo/behave/videos/Date03_Sub03_chairblack_lift.2.color.mp4

set -e

# Step 1: run Unidepth estimation
python prep/unidepth_behave.py --cameras 2 --data_source behave -o data/cari4d-demo/behave/videos/  --video ${video}

# Step 2: run NLF to estimate human pose
python prep/run_nlf_sepK.py --data_source behave -o data/cari4d-demo/behave/nlf-smplh-gender-sepK --masks_root data/cari4d-demo/behave/masks/ --video ${video}

# Step 3: align NLF to unidepth prediction
python prep/align_nlf2unidepth.py --data_source behave -o data/cari4d-demo/behave/nlf-smplh-gender-sepK-2unidepth --masks_root data/cari4d-demo/behave/masks \
--nlf_path data/cari4d-demo/behave/nlf-smplh-gender-sepK --video ${video}

# Step 4: run FoundationPose
python prep/fp_hy3d_2dir.py --viz_path x --vis_thres 0.5 --vis_thres2 0.5 --iou_thres2 0.3 --angular_velo 0.1 --occ_frames_allowed 30   \
--masks_root data/cari4d-demo/behave/masks/ --hy3d_root=data/cari4d-demo/meshes \
 --video ${video} -o data/cari4d-demo/behave/fp-hy3d3-unidepth